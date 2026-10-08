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


def build_kv(backend: str = "", redis_url: str = "") -> KVStore:
    """按配置构建 KV；Redis 不可用时**明确降级**而不是崩溃。"""
    want_redis = (backend or "").lower() == "redis" or bool(redis_url)
    if want_redis:
        try:
            kv = RedisKV(redis_url or "redis://127.0.0.1:6379/0")
            if isinstance(kv, RedisKV) and kv.ping():
                return kv
        except Exception:      # noqa: BLE001  包未安装或连不上 → 降级
            pass
    return InMemoryKV()
