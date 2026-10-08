# -*- coding: utf-8 -*-
"""多副本验证：为什么「限流/配额」必须放在共享 KV 里。

做法：构造**两个独立的应用实例**（模拟两个副本），做 A/B 对照：

    对照 A：两副本各用**进程内 KV**
        → 每个副本各算一份额度，总量被放大到 2×（限流形同虚设）
    对照 B：两副本共用**同一个共享 KV**
        → 额度是全局的，额度用尽后无论打到哪个副本都会被 429

为什么不用 Redis：本机没有 Redis 服务，而 `fakeredis` 的 TCP server 在本环境会挂住。
`SqliteKV` 用的是同一套 KV 接口、同样的跨进程原子自增（`BEGIN IMMEDIATE` + WAL），
足以验证「共享 vs 不共享」这一逻辑正确性；生产**跨机器**多实例仍必须用 Redis，
这一点在报告里如实标注。

这是压测定位出的第二个瓶颈（SQLite 单写者）之外的**多实例正确性问题**：
在本机跑不出问题，一上多副本就失效。

用法：
    python eval/multi_replica_check.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient                                          # noqa: E402

from agent.service.app import create_app                                           # noqa: E402
from agent.service.config import Settings                                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "eval", "fixtures", "mini_corpus.jsonl")

CAPACITY = 6          # 每个窗口的额度（故意设小，便于对照）


def make_settings(tmpdir: str, name: str, kv_backend: str,
                  kv_sqlite_path: str = "") -> Settings:
    return Settings(
        database_url=f"sqlite:///{os.path.join(tmpdir, name + '.db')}",
        kb_path=FIXTURE, jwt_secret="multi-replica-secret-0123456789",
        auth_required=True, dev_token_enabled=True,
        kv_backend=kv_backend, kv_sqlite_path=kv_sqlite_path,
        # 窗口拉长、补充速率极低：确保在同一窗口内测完
        rate_limit_capacity=CAPACITY, rate_limit_refill_per_sec=0.01,
        quota_tokens_per_day=10 ** 8, llm_backend="planner",
        default_vehicle_model="lynk08", allowed_models=["lynk08"],
    )


def hit(client: TestClient, headers: dict, question: str = "座椅加热怎么关闭") -> int:
    return client.post("/v1/chat", json={"message": question}, headers=headers).status_code


def run_case(tmpdir: str, label: str, kv_backend: str, kv_sqlite_path: str = "") -> dict:
    """两个副本交替发请求，看**总量**是否被全局限制住。"""
    app_a = create_app(make_settings(tmpdir, "a", kv_backend, kv_sqlite_path))
    app_b = create_app(make_settings(tmpdir, "b", kv_backend, kv_sqlite_path))
    with TestClient(app_a, raise_server_exceptions=False) as ca, \
            TestClient(app_b, raise_server_exceptions=False) as cb:
        ua = ca.post("/v1/auth/token", json={"user_id": "same-user",
                                            "vehicle_model": "lynk08"}).json()["access_token"]
        ub = cb.post("/v1/auth/token", json={"user_id": "same-user",
                                            "vehicle_model": "lynk08"}).json()["access_token"]
        ha, hb = {"Authorization": f"Bearer {ua}"}, {"Authorization": f"Bearer {ub}"}
        codes = []
        for i in range(CAPACITY):                       # 交替打：A、B、A、B…
            codes.append(hit(ca if i % 2 == 0 else cb, ha if i % 2 == 0 else hb))
        for i in range(CAPACITY):
            codes.append(hit(cb if i % 2 == 0 else ca, hb if i % 2 == 0 else ha))
        backends = (ca.get("/readyz").json().get("kv_backend"),
                    cb.get("/readyz").json().get("kv_backend"))
    # 关闭共享 KV 连接：Windows 下 SQLite 文件被占用会导致临时目录清理失败
    for app in (app_a, app_b):
        kv = getattr(app.state, "kv", None)
        if kv is not None and hasattr(kv, "close"):
            kv.close()
    passed = sum(1 for c in codes if c == 200)
    limited = sum(1 for c in codes if c == 429)
    return {
        "case": label,
        "kv_backend": kv_backend,
        "kv_backends_reported": list(backends),
        "requests_total": len(codes),
        "http_200": passed,
        "http_429": limited,
        "codes": codes,
        "expected_global_limit": CAPACITY,
        "verdict": ("✅ 额度是全局的（多副本正确）" if passed <= CAPACITY + 1
                    else f"❌ 额度被放大到 {passed}（= {passed / CAPACITY:.1f}× 配置值）"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=os.path.join(ROOT, "eval", "multi_replica_metrics.json"))
    ap.add_argument("--md-out", default=os.path.join(ROOT, "eval", "multi_replica_report.md"))
    args = ap.parse_args()

    print("=" * 72)
    print(f"多副本限流验证（每窗口额度配置为 {CAPACITY}）")
    print("=" * 72)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        shared_path = os.path.join(tmp, "shared_kv.db")
        memory_case = run_case(tmp, "两副本各用进程内 KV", "memory")
        shared_case = run_case(tmp, "两副本共用 SQLite 共享 KV", "sqlite", shared_path)

    for case in (memory_case, shared_case):
        print(f"\n[{case['case']}] kv={case['kv_backend']} {case['kv_backends_reported']}")
        print(f"  {case['requests_total']} 次请求（两副本交替）→ 200: {case['http_200']}"
              f"  429: {case['http_429']}")
        print(f"  状态码: {case['codes']}")
        print(f"  判定: {case['verdict']}")

    ok = memory_case["http_200"] > CAPACITY and shared_case["http_200"] <= CAPACITY + 1
    _write_outputs(args, memory_case, shared_case, shared_path)

    print("\n" + "-" * 72)
    print("结论：", "✅ 对照成立——进程内 KV 在多副本下失效，共享 KV 才正确"
          if ok else "❌ 对照不符合预期，见上")
    return 0 if ok else 1


def _write_outputs(args, memory_case, shared_case, shared_path) -> None:
    payload = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "configured_limit_per_window": CAPACITY,
               "shared_kv": "sqlite:" + os.path.basename(shared_path),
               "cases": [memory_case, shared_case]}
    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    lines = [
        "# 多副本限流验证报告", "",
        f"- 时间：{payload['generated_at']}",
        f"- 每窗口额度配置：**{CAPACITY}**（补充速率极低，确保在同一窗口内测完）",
        "- 数据：两个独立应用实例（模拟两个副本），**共 12 次请求交替**打过去",
        "",
        "| 对照 | KV 后端 | HTTP 200 | HTTP 429 | 判定 |",
        "| --- | --- | --- | --- | --- |",
        f"| A | 进程内（memory） | **{memory_case['http_200']}** | {memory_case['http_429']} | "
        f"{memory_case['verdict']} |",
        f"| B | 共享 KV（sqlite） | **{shared_case['http_200']}** | {shared_case['http_429']} | "
        f"{shared_case['verdict']} |",
        "",
        "## 结论", "",
        f"- 对照 A：两个副本**各算一份额度**，总放行 {memory_case['http_200']} 次 "
        f"（≈ 配置值的 {memory_case['http_200'] / CAPACITY:.1f}×）——限流在多副本下形同虚设。",
        f"- 对照 B：共享 KV 后额度是**全局**的，只放行 {shared_case['http_200']} 次，"
        f"其余 {shared_case['http_429']} 次被 429 拦住。",
        "- 这就是「进程内 KV 只能单实例」的量化证据；也是 `docker-compose.yml` 里默认把",
        "  `SERVICE_KV_BACKEND` 设为 `redis` 的原因。",
        "",
        "## 局限（如实说明）", "",
        "- 本机没有 Redis 服务，且 `fakeredis` 的 TCP server 在本环境会挂住，"
        "因此用 **`SqliteKV`（同为跨进程原子自增的共享 KV）** 代替验证「共享 vs 不共享」的逻辑；",
        "  生产**跨机器**多实例必须用 Redis。",
        "- 限流用「固定窗口 + 原子自增」实现（比 Redis Lua 令牌桶简单），**窗口边界会有突发**；",
        "  更平滑的令牌桶需要在 Redis 上用 Lua 脚本保证原子性。",
        "- PostgreSQL 侧的**多副本并发写**未实测（本机无 Docker）。",
    ]
    with open(args.md_out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n已写入 {os.path.relpath(args.json_out, ROOT)} 与 "
          f"{os.path.relpath(args.md_out, ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
