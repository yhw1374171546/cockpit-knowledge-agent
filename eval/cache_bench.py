# -*- coding: utf-8 -*-
"""语义缓存接通后的一次真实收益测量（走完整 HTTP 链路，不是单元基准）。

为什么单独写这个：缓存能力早就存在于 `agent/cache.py`，也在 `eval/ttft_bench.py` 里
被基准过，但**从来没接进服务链路**——基准里的"命中 66.67%、token −66.7%"
线上一次都没发生过。这个脚本测的就是**接进去之后**的真实数字。

用法：
    python eval/cache_bench.py
    python eval/cache_bench.py --shared          # 用 SqliteKV 共享后端（多副本场景）
    python eval/cache_bench.py --json-out eval/cache_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from agent.service.app import create_app  # noqa: E402
from agent.service.config import Settings  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "eval", "fixtures", "mini_corpus.jsonl")

# 10 个互不相同、且迷你语料能答的问题（保证 miss 阶段都是"正常作答"，才能进缓存）
QUESTIONS = [
    "座椅加热怎么关闭",
    "怎么打开危险警告灯",
    "胎压报警了怎么办",
    "座椅加热怎么打开",
    "危险警告灯怎么用",
    "胎压多少算正常",
    "多媒体显示屏怎么调座椅加热",
    "前排座椅加热在哪里设置",
    "危险警告灯开关在哪里",
    "胎压低报警是什么意思",
]


def build(shared: bool, tmpdir: str):
    settings = Settings(
        database_url=f"sqlite:///{os.path.join(tmpdir, 'cache_bench.db')}",
        kb_path=FIXTURE, jwt_secret="bench-secret",
        auth_required=True, dev_token_enabled=True,
        kv_backend="sqlite" if shared else "memory",
        kv_sqlite_path=os.path.join(tmpdir, "kv.db") if shared else "",
        rate_limit_capacity=10_000, rate_limit_refill_per_sec=10_000.0,
        quota_tokens_per_day=10_000_000, llm_backend="planner",
        default_vehicle_model="lynk08", allowed_models=["lynk08", "lynk09"],
        enable_semantic_cache=True, semantic_cache_shared=shared,
    )
    app = create_app(settings)
    return app


def run_round(client, headers, questions):
    rows = []
    for question in questions:
        t0 = time.perf_counter()
        resp = client.post("/v1/chat", json={"message": question}, headers=headers)
        cost_ms = (time.perf_counter() - t0) * 1000
        data = resp.json()
        rows.append({"question": question, "status": resp.status_code,
                     "answer_status": data.get("status"), "cached": bool(data.get("cached")),
                     "ms": cost_ms, "tokens_out": data.get("tokens_out", 0),
                     "tokens_in": data.get("tokens_in", 0), "usd": data.get("cost_usd", 0.0)})
    return rows


def summarize(rows):
    return {"n": len(rows),
            "answered": sum(1 for r in rows if r["answer_status"] == "answered"),
            "hits": sum(1 for r in rows if r["cached"]),
            "p50_ms": round(statistics.median([r["ms"] for r in rows]), 2),
            "mean_ms": round(statistics.fmean([r["ms"] for r in rows]), 2),
            "tokens_in": sum(r["tokens_in"] for r in rows),
            "tokens_out": sum(r["tokens_out"] for r in rows),
            "usd": round(sum(r["usd"] for r in rows), 6)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", action="store_true", help="用 SqliteKV 共享后端")
    ap.add_argument("--json-out", default=os.path.join(ROOT, "eval", "cache_metrics.json"))
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        app = build(args.shared, tmpdir)
        with TestClient(app) as client:
            token = client.post("/v1/auth/token",
                                json={"user_id": "bench", "role": "driver",
                                      "vehicle_model": "lynk08"}).json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            print("=" * 76)
            print(f"语义缓存收益测量（HTTP 全链路，KV={'sqlite 共享' if args.shared else 'memory'})")
            print("=" * 76)
            cold = run_round(client, headers, QUESTIONS)
            warm = run_round(client, headers, QUESTIONS)
            report = app.state.semantic_cache.report()

    cold_s, warm_s = summarize(cold), summarize(warm)
    print(f"\n{'阶段':<10}{'题数':>5}{'可答':>6}{'命中':>6}{'P50(ms)':>10}"
          f"{'均值(ms)':>10}{'输出token':>11}{'成本($)':>10}")
    print("-" * 76)
    for name, s in (("冷启动", cold_s), ("命中缓存", warm_s)):
        print(f"{name:<10}{s['n']:>5}{s['answered']:>6}{s['hits']:>6}{s['p50_ms']:>10.2f}"
              f"{s['mean_ms']:>10.2f}{s['tokens_out']:>11}{s['usd']:>10.6f}")

    saved_ms = cold_s["p50_ms"] - warm_s["p50_ms"]
    saved_pct = (saved_ms / cold_s["p50_ms"] * 100) if cold_s["p50_ms"] else 0.0
    saved_tokens = cold_s["tokens_out"] - warm_s["tokens_out"]
    print("\n" + "-" * 76)
    print(f"P50 延迟：{cold_s['p50_ms']:.2f} ms → {warm_s['p50_ms']:.2f} ms"
          f"（下降 {saved_ms:.2f} ms / {saved_pct:.1f}%）")
    print(f"输出 token：{cold_s['tokens_out']} → {warm_s['tokens_out']}"
          f"（省下 {saved_tokens}，命中时归零）")
    print(f"缓存命中率（第二轮）：{warm_s['hits']}/{warm_s['n']} = "
          f"{warm_s['hits'] / max(warm_s['n'], 1) * 100:.1f}%")
    print(f"缓存内部统计：{json.dumps(report, ensure_ascii=False)}")

    payload = {"shared": bool(args.shared), "cold": cold_s, "warm": warm_s,
               "p50_saved_ms": round(saved_ms, 2), "p50_saved_pct": round(saved_pct, 1),
               "tokens_saved": saved_tokens, "cache_report": report}
    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\n已写入 {os.path.relpath(args.json_out, ROOT)}")

    ok = warm_s["hits"] >= warm_s["n"] - 1 and warm_s["tokens_out"] * 2 < max(
        cold_s["tokens_out"], 1)
    print("\n结论：", "✅ 缓存已真实生效（命中即跳过检索与大模型调用）"
          if ok else "⚠️ 命中率或 token 节省不达预期，需检查接线")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
