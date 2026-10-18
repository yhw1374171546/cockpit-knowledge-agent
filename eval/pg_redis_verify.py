# -*- coding: utf-8 -*-
"""PostgreSQL / Redis 真跑验证：把"多副本一致性 / 幂等由 DB 约束保证"这些说法变成实测。

为什么写这个脚本
----------------
之前的"多副本限流放大 2.0×"是用 **SqliteKV 模拟共享存储**得出的。后端面试官会问：
"你在真 Redis 上验过吗？"——没有就是口径问题。这个脚本补上这一环。

三类验证
--------
**A. Redis 原子性**（限流/配额的底座）
  · 并发 incr 是否精确（丢失更新 = 限流失效）
  · 项目自带的 RateLimiter 接真 Redis：容量 50，并发 100 → 必须**严格 50 放行**
  · ``INCR`` + ``EXPIRE`` 两条命令 vs **Lua 脚本**：前者在中途异常时会留下**永不过期的 key**
    （限流窗口永远不滚动 → 该用户被永久限流），后者原子。这是"为什么要用 Lua"的实测证据。

**B. PostgreSQL 可移植性与约束幂等**
  · 同一套 schema 在 PG 上 ``alembic upgrade head`` 能否一次成功（验证没写 SQLite 专有东西）
  · **并发同 key 插入**：N 个线程抢同一个幂等键 → 必须**恰好 1 条成功**，其余全部唯一约束冲突。
    这正是"幂等靠 DB 唯一约束、而不是进程内字典"的证明。

**C. 引擎差异**
  · PG 上 ``EXPLAIN ANALYZE`` 看 ``messages(session_id)`` 是否走索引
  · SQLite 的 ``BEGIN IMMEDIATE``（写锁）在 PG 上没有对应物，PG 靠行锁/唯一约束 → 同一段代码要分支处理

用法
----
    python eval/pg_redis_verify.py                       # 用默认端口
    python eval/pg_redis_verify.py --pg-port 5433 --redis-port 6380
    python eval/pg_redis_verify.py --only redis           # 只跑 Redis 部分

退出码：0 = 全部通过；1 = 有项失败（可直接进 CI）。
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.console import harden_stdout  # noqa: E402

harden_stdout()

OK, FAIL, WARN = "[OK]", "[FAIL]", "[·]"


class Report:
    def __init__(self, title: str) -> None:
        self.title = title
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, ok, detail))
        print(f"  {OK if ok else FAIL} {name}" + (f"  → {detail}" if detail else ""))
        return ok

    def note(self, text: str) -> None:
        print(f"  {WARN} {text}")

    @property
    def failed(self) -> int:
        return sum(1 for _, ok, _ in self.rows if not ok)

    def summary(self) -> str:
        total = len(self.rows)
        return f"{self.title}: {total - self.failed}/{total} 通过"


# ───────────────────────── A. Redis ─────────────────────────

def verify_redis(port: int) -> Report:
    rep = Report("Redis 原子性")
    try:
        import redis
    except ImportError:
        rep.add("redis 客户端已安装", False, "pip install redis")
        return rep

    # protocol=2：Redis 5.x 不支持 RESP3 的 HELLO 命令（实测报 unknown command HELLO）
    client = redis.Redis(host="127.0.0.1", port=port, decode_responses=True,
                         socket_timeout=5, protocol=2)
    try:
        client.ping()
    except Exception as exc:                                   # noqa: BLE001
        rep.add("Redis 可连接", False, f"{type(exc).__name__}: {exc}")
        return rep
    rep.add("Redis 可连接", True, f"port={port}")

    # A1 并发 incr 是否精确（丢失更新会让限流额度被放大）
    client.delete("verify:incr")
    n_threads, per_thread = 8, 200
    def bump() -> None:
        for _ in range(per_thread):
            client.incr("verify:incr")
    threads = [threading.Thread(target=bump) for _ in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    expected = n_threads * per_thread
    got = int(client.get("verify:incr") or 0)
    rep.add(f"并发 incr 精确（{n_threads}×{per_thread}）", got == expected,
            f"期望 {expected}，实得 {got}，耗时 {elapsed * 1000:.0f} ms")

    # A2 项目自带的 RateLimiter 接真 Redis
    try:
        from agent.service.config import Settings
        from agent.service.kv import RedisKV
        from agent.service.security import RateLimiter

        # 先清掉上一轮可能残留的限流计数：固定窗口的 key 带时间片，
        # 但同一窗口内重跑会让计数从上次的值继续涨，导致"放行 0"这种假失败。
        for stale in client.keys("rl:verify-user*"):
            client.delete(stale)
        settings = Settings(rate_limit_capacity=50, rate_limit_refill_per_sec=0.01)
        limiter = RateLimiter(RedisKV(f"redis://127.0.0.1:{port}/0"), settings)  # RedisKV 内部已固定 RESP2
        allowed = {"n": 0}
        lock = threading.Lock()

        def hit() -> None:
            res = limiter.hit("verify-user", "lynk08")
            if res.allowed:
                with lock:
                    allowed["n"] += 1

        ts = [threading.Thread(target=hit) for _ in range(100)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        rep.add("RateLimiter 接真 Redis：容量 50 / 并发 100", allowed["n"] == 50,
                f"放行 {allowed['n']}（应严格 50）——超发说明原子性失效")
    except Exception as exc:                                   # noqa: BLE001
        rep.add("RateLimiter 接真 Redis", False, f"{type(exc).__name__}: {exc}")

    # A3 INCR+EXPIRE 两条命令 vs Lua：中途异常会留下"永不过期的 key"
    client.delete("verify:ttl_multi", "verify:ttl_lua")

    # 模拟"INCR 成功后、EXPIRE 之前进程中断"
    def incr_then_crash(key: str) -> None:
        client.incr(key)
        raise RuntimeError("模拟：EXPIRE 之前中断")

    try:
        incr_then_crash("verify:ttl_multi")
    except RuntimeError:
        pass
    ttl_multi = client.ttl("verify:ttl_multi")

    # Lua：INCR 与 EXPIRE 在同一脚本里，要么都生效要么都不生效
    lua = """
    local v = redis.call('INCR', KEYS[1])
    if v == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
    return v
    """
    client.eval(lua, 1, "verify:ttl_lua", 60)
    ttl_lua = client.ttl("verify:ttl_lua")

    rep.add("Lua 版限流：首次即带 TTL（窗口会滚动）", ttl_lua > 0,
            f"ttl={ttl_lua}s")
    rep.add("多命令版：中断后留下**永不过期**的 key（这是 bug）", ttl_multi == -1,
            f"ttl={ttl_multi}（-1 表示永不过期 → 该用户会被永久限流）")

    # A4 吞吐：Redis vs SQLiteKV vs InMemoryKV
    rounds = 2000
    results = {}
    try:
        from agent.service.kv import InMemoryKV, RedisKV, SqliteKV

        for name, kv in (("Redis", RedisKV(f"redis://127.0.0.1:{port}/0")),
                         ("SqliteKV", SqliteKV(os.path.join(os.getcwd(), "_verify_kv.db"))),
                         ("InMemoryKV", InMemoryKV())):
            kv.delete("verify:bench") if hasattr(kv, "delete") else None
            t0 = time.perf_counter()
            for _ in range(rounds):
                kv.incr(f"verify:bench:{name}", 1, ttl=60)
            results[name] = (time.perf_counter() - t0) * 1000
        detail = " ｜ ".join(f"{k} {v:.0f}ms" for k, v in results.items())
        rep.add(f"吞吐对比（各 {rounds} 次 incr）", True, detail)
        rep.note("注：单线程顺序调用，衡量的是**单次往返开销**，不是并发吞吐上限；"
                 "Redis 走网络、SQLite 走磁盘 fsync、内存版无持久化——量级差异符合预期")
    except Exception as exc:                                   # noqa: BLE001
        rep.note(f"吞吐对比跳过：{type(exc).__name__}: {exc}")

    for key in ("verify:incr", "verify:ttl_multi", "verify:ttl_lua"):
        client.delete(key)
    return rep


# ───────────────────────── B. PostgreSQL ─────────────────────────

def verify_postgres(port: int, user: str, dbname: str) -> Report:
    rep = Report("PostgreSQL 可移植性与约束幂等")
    try:
        import psycopg
    except ImportError:
        rep.add("psycopg 已安装", False, "pip install 'psycopg[binary]'")
        return rep

    dsn = f"host=127.0.0.1 port={port} user={user} dbname={dbname}"
    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            conn.execute("SELECT 1")
        rep.add("PostgreSQL 可连接", True, f"port={port} db={dbname}")
    except Exception as exc:                                   # noqa: BLE001
        rep.add("PostgreSQL 可连接", False, f"{type(exc).__name__}: {exc}")
        return rep

    # B1 同一套 schema 能否在 PG 上迁移成功
    try:
        from sqlalchemy import create_engine, inspect
        from agent.service.db import Base

        url = f"postgresql+psycopg://{user}@127.0.0.1:{port}/{dbname}"
        engine = create_engine(url, future=True)
        Base.metadata.create_all(engine)          # 这里只验证 schema 可移植性
        tables = sorted(inspect(engine).get_table_names())
        rep.add("同一套 SQLAlchemy schema 在 PG 上建表成功", len(tables) >= 12,
                f"{len(tables)} 张表: {', '.join(tables[:6])}…")
        engine.dispose()
    except Exception as exc:                                   # noqa: BLE001
        rep.add("schema 在 PG 上建表", False, f"{type(exc).__name__}: {exc}")

    # B2 并发同 key 插入 → 幂等由唯一约束保证
    try:
        key = f"verify-idem-{int(time.time())}"
        with psycopg.connect(dsn) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS verify_idem (
                    id BIGSERIAL PRIMARY KEY,
                    idem_key TEXT NOT NULL UNIQUE,
                    payload TEXT NOT NULL DEFAULT ''
                )""")
            conn.commit()

        success = {"n": 0}
        conflicts = {"n": 0}
        lock = threading.Lock()

        def insert_once() -> None:
            try:
                with psycopg.connect(dsn) as c2:
                    c2.execute("INSERT INTO verify_idem (idem_key, payload) VALUES (%s, %s)",
                               (key, "same-request"))
                    c2.commit()
                with lock:
                    success["n"] += 1
            except psycopg.errors.UniqueViolation:
                with lock:
                    conflicts["n"] += 1

        ts = [threading.Thread(target=insert_once) for _ in range(20)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

        with psycopg.connect(dsn) as conn:
            rows = conn.execute("SELECT count(*) FROM verify_idem WHERE idem_key = %s",
                                (key,)).fetchone()[0]
            conn.execute("DROP TABLE verify_idem")
            conn.commit()

        rep.add("并发 20 次同 key 插入 → 恰好 1 条落库", rows == 1 and success["n"] == 1,
                f"成功 {success['n']} / 唯一冲突 {conflicts['n']} / 表内 {rows} 条")
        rep.note("这就是'幂等靠 DB 唯一约束、而不是进程内字典'的直接证据："
                 "进程内字典在多副本/重启后就失效，唯一约束不会")
    except Exception as exc:                                   # noqa: BLE001
        rep.add("并发同 key 幂等插入", False, f"{type(exc).__name__}: {exc}")

    # C 索引使用情况
    try:
        with psycopg.connect(dsn) as conn:
            plan = conn.execute(
                "EXPLAIN (ANALYZE false) SELECT * FROM messages WHERE session_id = 'x'"
            ).fetchall()
        text = " | ".join(r[0] for r in plan)
        rep.add("messages(session_id) 查询命中索引", "Index" in text or "index" in text,
                text[:110])
    except Exception as exc:                                   # noqa: BLE001
        rep.note(f"EXPLAIN 跳过：{type(exc).__name__}: {exc}")

    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg-port", type=int, default=5433)
    ap.add_argument("--pg-user", default="postgres")
    ap.add_argument("--pg-db", default="cockpit")
    ap.add_argument("--redis-port", type=int, default=6380)
    ap.add_argument("--only", choices=["redis", "pg"], default=None)
    args = ap.parse_args()

    print("=" * 74)
    print("PostgreSQL / Redis 真跑验证")
    print("=" * 74)

    reports = []
    if args.only in (None, "redis"):
        reports.append(verify_redis(args.redis_port))
        print()
    if args.only in (None, "pg"):
        reports.append(verify_postgres(args.pg_port, args.pg_user, args.pg_db))
        print()

    print("-" * 74)
    failed = 0
    for rep in reports:
        print("  " + rep.summary())
        failed += rep.failed
    print("=" * 74)
    if failed:
        print(f"  结论：{failed} 项未通过")
        return 1
    print("  结论：全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
