# -*- coding: utf-8 -*-
"""Agent 服务并发压测：产出 QPS / P95 / P99 / TTFT 分位 / 错误分类。

口径说明（非常重要，否则数字没有意义）
--------------------------------------
本机没有 GPU，也没有 vLLM 服务，因此这里测的是**服务层容量**，分两种模式：

- `--llm planner`（默认）：规则规划器，生成耗时≈0
  → 测出的是「鉴权 + 限流 + 配额 + 幂等落库 + 检索 + 多 Agent 编排」的**真实容量上限**。
- `--llm streaming`：模拟流式生成（每块有时延）
  → 测出的是「编排 + 模拟生成」的端到端表现，TTFT 包含生成首块时间。

**不要把这里的数字说成"大模型吞吐"**：真实吞吐取决于推理引擎（vLLM 的 batch 行为），
本机无法测。报告里会明确标注口径。

用法
----
    # 1) 自己起服务（推荐：脚本会拉起一个宽松限流的实例）
    python eval/load_test.py --spawn
    python eval/load_test.py --spawn --workers 4
    python eval/load_test.py --spawn --llm streaming --concurrency 1,5,10,25

    # 2) 压已有服务（注意：限流/配额会干扰结果，见 --spawn 的默认配置）
    python eval/load_test.py --url http://127.0.0.1:8077

产出：`eval/load_metrics.json` + `eval/load_report.md`
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS = ["座椅加热怎么关闭", "怎么打开危险警告灯", "胎压报警了怎么办"]


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    k = (len(data) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (k - lo)


# ── 服务拉起 ─────────────────────────────────────────────────────────

def spawn_service(port: int, workers: int, llm: str, kb: str) -> subprocess.Popen:
    env = dict(os.environ)
    env.update({
        # 压测必须放宽限流与配额，否则测到的是 429 而不是容量
        "SERVICE_RATE_LIMIT_CAPACITY": "1000000",
        "SERVICE_RATE_LIMIT_REFILL_PER_SEC": "1000000",
        "SERVICE_QUOTA_TOKENS_PER_DAY": "1000000000",
        "SERVICE_LLM_BACKEND": llm,
        "SERVICE_DATABASE_URL": f"sqlite:///{os.path.join(ROOT, '_loadtest.db')}",
        "SERVICE_JWT_SECRET": "loadtest-secret",
        "AGENT_KB_PATH": kb,
        "PYTHONIOENCODING": "utf-8",
    })
    cmd = [sys.executable, "-m", "uvicorn", "agent.service.app:app",
           "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]
    if workers > 1:
        cmd += ["--workers", str(workers)]
    print(f"[spawn] {' '.join(cmd)}  (workers={workers}, llm={llm})")
    return subprocess.Popen(cmd, env=env, cwd=ROOT)


def wait_ready(url: str, timeout: float = 90.0) -> Dict[str, Any]:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/readyz", timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:                              # noqa: BLE001
            last = str(exc)
            time.sleep(0.5)
    raise RuntimeError(f"服务未就绪（{timeout}s）：{last}")


def fetch_token(url: str, user: str) -> str:
    req = urllib.request.Request(
        f"{url}/v1/auth/token",
        data=json.dumps({"user_id": user, "vehicle_model": "lynk08"}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]


# ── 压测核心 ─────────────────────────────────────────────────────────

async def one_request(client, url: str, headers: Dict[str, str], question: str,
                      stream: bool, scenario: str = "chat") -> Dict[str, Any]:
    """发一次请求；返回延迟、TTFT（流式）、tokens 或错误分类。"""
    out: Dict[str, Any] = {"ok": False, "status": 0, "kind": ""}
    t0 = time.perf_counter()
    try:
        if scenario == "readyz":
            # 只读对照：不写库、不检索，用来判断瓶颈是否在"写路径"
            resp = await client.get(f"{url}/readyz")
            out["status"] = resp.status_code
            if resp.status_code != 200:
                out["kind"] = f"http_{resp.status_code}"
                return out
            out["ok"] = True
        elif stream:
            ttft = None
            answer_chars = 0
            async with client.stream("POST", f"{url}/v1/chat/stream",
                                     json={"message": question}, headers=headers) as resp:
                out["status"] = resp.status_code
                if resp.status_code != 200:
                    await resp.aread()
                    out["kind"] = "http_429" if resp.status_code == 429 else f"http_{resp.status_code}"
                    return out
                async for line in resp.aiter_lines():
                    if line.startswith("event: delta") or (ttft is None and line.startswith("data:")):
                        if ttft is None and line.startswith("data:"):
                            ttft = (time.perf_counter() - t0) * 1000
                    if line.startswith("data:"):
                        answer_chars += len(line)
            out.update(ok=True, ttft_ms=ttft, chars=answer_chars)
        else:
            resp = await client.post(f"{url}/v1/chat",
                                     json={"message": question}, headers=headers)
            out["status"] = resp.status_code
            if resp.status_code != 200:
                out["kind"] = "http_429" if resp.status_code == 429 else f"http_{resp.status_code}"
                return out
            body = resp.json()
            out.update(ok=True, ttft_ms=None,
                       tokens=int(body.get("tokens_in", 0)) + int(body.get("tokens_out", 0)))
    except Exception as exc:                                  # noqa: BLE001
        out["kind"] = "timeout" if "timeout" in str(exc).lower() else f"exc:{type(exc).__name__}"
    finally:
        out["latency_ms"] = (time.perf_counter() - t0) * 1000
    return out


async def run_level(url: str, token: str, concurrency: int, total: int,
                    stream: bool, llm: str, scenario: str = "chat") -> Dict[str, Any]:
    import httpx

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    limits = httpx.Limits(max_connections=concurrency + 5,
                          max_keepalive_connections=concurrency + 5)
    results: List[Dict[str, Any]] = []
    counter = {"i": 0}
    lock = asyncio.Lock()

    async with httpx.AsyncClient(limits=limits, timeout=120.0) as client:
        # 预热：把知识库检索、缓存、连接池都打热，避免把冷启动算进容量
        if scenario != "readyz":
            for q in QUESTIONS[:2]:
                await one_request(client, url, headers, q, stream=False)

        wall0 = time.perf_counter()

        async def worker():
            while True:
                async with lock:
                    if counter["i"] >= total:
                        return
                    idx = counter["i"]
                    counter["i"] += 1
                res = await one_request(client, url, headers, QUESTIONS[idx % len(QUESTIONS)],
                                        stream, scenario)
                results.append(res)

        await asyncio.gather(*[worker() for _ in range(concurrency)])
        wall = time.perf_counter() - wall0

    ok = [r for r in results if r["ok"]]
    lat = [r["latency_ms"] for r in ok]
    ttfts = [r["ttft_ms"] for r in ok if r.get("ttft_ms")]
    errors: Dict[str, int] = {}
    for r in results:
        if not r["ok"]:
            errors[r["kind"]] = errors.get(r["kind"], 0) + 1
    tokens = sum(r.get("tokens", 0) for r in ok)

    return {
        "concurrency": concurrency,
        "requests": len(results),
        "ok": len(ok),
        "error_rate": round(1 - len(ok) / max(1, len(results)), 4),
        "errors": errors,
        "wall_s": round(wall, 3),
        "qps": round(len(ok) / wall, 2) if wall > 0 else 0.0,
        "latency_ms": {
            "p50": round(percentile(lat, 0.50), 2),
            "p90": round(percentile(lat, 0.90), 2),
            "p95": round(percentile(lat, 0.95), 2),
            "p99": round(percentile(lat, 0.99), 2),
            "max": round(max(lat), 2) if lat else 0.0,
            "mean": round(statistics.fmean(lat), 2) if lat else 0.0,
        },
        "ttft_ms": ({
            "p50": round(percentile(ttfts, 0.50), 2),
            "p95": round(percentile(ttfts, 0.95), 2),
            "p99": round(percentile(ttfts, 0.99), 2),
        } if ttfts else None),
        "tokens_total": tokens,
        "tokens_per_s": round(tokens / wall, 1) if wall > 0 else 0.0,
        "llm_mode": llm,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8077")
    ap.add_argument("--spawn", action="store_true", help="自动拉起服务（推荐，会放宽限流/配额）")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--workers", type=int, default=1, help="uvicorn worker 数（对比扩容性）")
    ap.add_argument("--llm", default="planner", choices=["planner", "streaming"])
    ap.add_argument("--kb", default=os.path.join(ROOT, "eval", "fixtures", "mini_corpus.jsonl"))
    ap.add_argument("--concurrency", default="1,5,10,25,50")
    ap.add_argument("--requests", type=int, default=0,
                    help="每档请求数；0 = 自动按 20×并发 计算")
    ap.add_argument("--duration", type=float, default=0.0, help="每档持续时间（秒），>0 时按时间跑")
    ap.add_argument("--stream", action="store_true", help="压 SSE 流式端点（测 TTFT）")
    ap.add_argument("--scenario", default="chat", choices=["chat", "readyz"],
                    help="chat=完整问答（含写库与检索）；readyz=只读对照（不写库不检索）")
    ap.add_argument("--json-out", default=os.path.join(ROOT, "eval", "load_metrics.json"))
    ap.add_argument("--md-out", default=os.path.join(ROOT, "eval", "load_report.md"))
    args = ap.parse_args()

    levels = [int(x) for x in str(args.concurrency).split(",") if x.strip()]
    url = f"http://127.0.0.1:{args.port}" if args.spawn else args.url.rstrip("/")

    proc: Optional[subprocess.Popen] = None
    if args.spawn:
        proc = spawn_service(args.port, args.workers, args.llm, args.kb)
    try:
        ready = wait_ready(url)
        print(f"[ready] db={ready.get('database')} kv={ready.get('kv_backend')} "
              f"llm={ready.get('llm_backend')} kb_chunks={ready.get('kb_chunks')}")
        token = fetch_token(url, f"loadtest-{int(time.time())}")

        results = []
        for c in levels:
            total = args.requests or max(40, c * 20)
            print(f"\n[level] 并发={c} 请求={total} "
                  f"模式={'SSE' if args.stream else 'JSON'} llm={args.llm}")
            res = asyncio.run(run_level(url, token, c, total, args.stream, args.llm,
                                        args.scenario))
            results.append(res)
            lat = res["latency_ms"]
            line = (f"  QPS={res['qps']:<8} P50={lat['p50']:<7} P95={lat['p95']:<8} "
                    f"P99={lat['p99']:<8} 错误率={res['error_rate']:.1%}")
            if res["ttft_ms"]:
                line += f" TTFT_P50={res['ttft_ms']['p50']}ms"
            print(line)
            if res["errors"]:
                print(f"  错误分类：{res['errors']}")

        summary = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "endpoint": ("/readyz" if args.scenario == "readyz"
                         else ("/v1/chat/stream" if args.stream else "/v1/chat")),
            "scenario": args.scenario,
            "llm_mode": args.llm,
            "workers": args.workers,
            "kb": os.path.relpath(args.kb, ROOT),
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "levels": results,
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

        # ── Markdown 报告 ──
        lines = [
            "# 服务层并发压测报告", "",
            f"- 时间：{summary['generated_at']}",
            f"- 端点：`{summary['endpoint']}`　模式：`{args.llm}`　worker 数：{args.workers}"
            + ("　**（只读对照：不写库、不检索）**" if args.scenario == "readyz" else ""),
            f"- 语料：`{summary['kb']}`　平台：{sys.platform} / Python {summary['python']}",
            "",
            "> **口径**：本机无 GPU、无 vLLM 服务，因此这里测的是**服务层容量**"
            "（鉴权+限流+配额+幂等落库+检索+编排），"
            + ("生成耗时为**模拟**（流式块时延）。" if args.llm == "streaming"
               else "生成耗时**未计入**（规则规划器）。"),
            "> 真实大模型吞吐取决于推理引擎的 batch 行为，不能由本表推断。",
            "",
            "| 并发 | QPS | P50 (ms) | P95 (ms) | P99 (ms) | 错误率 | TTFT P50 (ms) | tokens/s |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in results:
            lat = r["latency_ms"]
            ttft = r["ttft_ms"]["p50"] if r["ttft_ms"] else "—"
            lines.append(f"| {r['concurrency']} | {r['qps']} | {lat['p50']} | {lat['p95']} "
                         f"| {lat['p99']} | {r['error_rate']:.1%} | {ttft} | {r['tokens_per_s']} |")
        lines += ["", "## 错误分类", ""]
        for r in results:
            lines.append(f"- 并发 {r['concurrency']}：{r['errors'] or '无错误'}")
        with open(args.md_out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\n已写入 {os.path.relpath(args.json_out, ROOT)} 与 "
              f"{os.path.relpath(args.md_out, ROOT)}")
        return 0
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("[spawn] 已停止压测服务")


if __name__ == "__main__":
    sys.exit(main())
