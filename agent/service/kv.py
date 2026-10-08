# -*- coding: utf-8 -*-
"""KV 抽象：限流、配额、缓存、幂等锁都走这一层。

设计延续本项目的「优雅降级」风格：
- 默认 **进程内实现**（零依赖、单实例可用、CI 可跑）；
- 配置了 `SERVICE_KV_BACKEND=redis` + `SERVICE_REDIS_URL` 且装了 `redis` 包时自动切换；
- **多实例部署必须用 Redis**，否则限流/配额在每个副本里各算一份（README 已注明）。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Protocol


class KVStore(Protocol):
    def get(self, key: str) -> Optional[str]: ...
    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None: ...
    def delete(self, key: str) -> None: ...
    def incr(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int: ...
    def setnx(self, key: str, value: str, ttl: Optional[int] = None) -> bool: ...
    def backend(self) -> str: ...


class InMemoryKV:
    """进程内 KV（带 TTL 与锁）；仅用于开发与单实例部署。"""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}
        self._expire: Dict[str, float] = {}
        self._lock = threading.RLock()

    def _expired(self, key: str) -> bool:
        exp = self._expire.get(key)
        if exp is not None and exp <= time.time():
            self._data.pop(key, None)
            self._expire.pop(key, None)
            return True
        return False

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if self._expired(key):
                return None
            value = self._data.get(key)
            return None if value is None else str(value)

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        with self._lock:
            self._data[key] = value
            if ttl:
                self._expire[key] = time.time() + ttl
            else:
                self._expire.pop(key, None)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._expire.pop(key, None)

    def incr(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        with self._lock:
            self._expired(key)
            current = int(self._data.get(key, 0)) + amount
            self._data[key] = current
            if ttl and key not in self._expire:
                self._expire[key] = time.time() + ttl
            return current

    def setnx(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        with self._lock:
            if self.get(key) is not None:
                return False
            self.set(key, value, ttl)
            return True

    def backend(self) -> str:
        return "memory"

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"keys": len(self._data)}


class RedisKV:
    """Redis 实现（多实例部署用）。"""

    def __init__(self, url: str) -> None:
        import redis  # type: ignore

        self._client = redis.Redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> Optional[str]:
        return self._client.get(key)

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        self._client.set(key, value, ex=ttl)

    def delete(self, key: str) -> None:
        self._client.delete(key)

    def incr(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        pipe = self._client.pipeline()
        pipe.incrby(key, amount)
        if ttl:
            pipe.expire(key, ttl, nx=True)
        return int(pipe.execute()[0])

    def setnx(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        return bool(self._client.set(key, value, ex=ttl, nx=True))

    def backend(self) -> str:
        return "redis"

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:      # noqa: BLE001
            return False


class SqliteKV:
    """共享文件 KV：**单机多 worker** 场景下的正确选择（多进程共享同一 SQLite 文件）。

    为什么需要它：进程内实现（`InMemoryKV`）在 `uvicorn --workers N` 或起多个容器时，
    每个副本各算一份额度——限流/配额形同虚设。本实现把状态放到共享文件里，
    用 `BEGIN IMMEDIATE` + WAL 保证 `incr` 的跨进程原子性。

    生产多实例（跨机器）仍应使用 Redis；本类适合单机多进程或作为 Redis 不可用时的
    共享兜底，也用于**验证"共享 KV 才能让多副本限流正确"**这一结论。
    """

    def __init__(self, path: str = "kv_shared.db") -> None:
        import sqlite3

        self._path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30,
                                     isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT, exp REAL)")
        self._conn.commit()

    def _alive(self, exp) -> bool:
        return exp is None or float(exp) > time.time()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute("SELECT v, exp FROM kv WHERE k=?", (key,)).fetchone()
            if row is None or not self._alive(row[1]):
                return None
            return str(row[0])

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        exp = time.time() + ttl if ttl else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO kv(k,v,exp) VALUES(?,?,?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v, exp=excluded.exp",
                (key, str(value), exp))
            self._conn.commit()

    def delete(self, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM kv WHERE k=?", (key,))
            self._conn.commit()

    def incr(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """跨进程原子自增：BEGIN IMMEDIATE 抢占写锁后再读改写。"""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT v, exp FROM kv WHERE k=?",
                                         (key,)).fetchone()
                now = time.time()
                if row is None or not self._alive(row[1]):
                    value, exp = int(amount), (now + ttl if ttl else None)
                else:
                    value, exp = int(row[0]) + int(amount), row[1]
                self._conn.execute(
                    "INSERT INTO kv(k,v,exp) VALUES(?,?,?) "
                    "ON CONFLICT(k) DO UPDATE SET v=excluded.v, exp=excluded.exp",
                    (key, str(value), exp))
                self._conn.execute("COMMIT")
                return value
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def setnx(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT v, exp FROM kv WHERE k=?",
                                         (key,)).fetchone()
                if row is not None and self._alive(row[1]):
                    self._conn.execute("COMMIT")
                    return False
                exp = time.time() + ttl if ttl else None
                self._conn.execute(
                    "INSERT INTO kv(k,v,exp) VALUES(?,?,?) "
                    "ON CONFLICT(k) DO UPDATE SET v=excluded.v, exp=excluded.exp",
                    (key, str(value), exp))
                self._conn.execute("COMMIT")
                return True
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def backend(self) -> str:
        return "sqlite"

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:                                          # noqa: BLE001
            pass


def build_kv(backend: str = "", redis_url: str = "", sqlite_path: str = "") -> KVStore:
    """按配置构建 KV。

    优先级：显式 backend > redis_url > 默认进程内。
    Redis 不可用时**明确降级**为进程内实现（单实例可跑，但多副本会失效——文档已注明）。
    """
    name = (backend or "").lower()
    if name == "sqlite":
        return SqliteKV(sqlite_path or "kv_shared.db")
    want_redis = name == "redis" or bool(redis_url)
    if want_redis:
        try:
            kv = RedisKV(redis_url or "redis://127.0.0.1:6379/0")
            if isinstance(kv, RedisKV) and kv.ping():
                return kv
        except Exception:      # noqa: BLE001  包未安装或连不上 → 降级
            pass
    return InMemoryKV()
