# -*- coding: utf-8 -*-
"""首字延迟（TTFT）与大模型调用量优化评测。

座舱对首字延迟极其敏感（用户说完话希望立刻有反馈），因此这里量化三件事：

1. **流式 vs 非流式**：TTFT 从"等整段生成完"降到"首个 token"；
2. **语义缓存**：高频问题命中后 TTFT ≈ 检索/生成都省掉；
3. **预取/垫话**：路由阶段并行发起检索，并立刻回一句"正在为您查询…"降低感知延迟。

关于诚实性
--------
本机没有 GPU/大模型，因此**生成耗时用 `SimulatedStreamingLLM` 模拟**（可配置 per-token 时延），
报告里同时给出"模拟生成参数"与"编排层实测耗时"两部分，二者不混淆：
- 编排层（检索 + 路由 + 缓存 + 反思）耗时是**真实测量**；
- 生成耗时是**参数化模拟**，真实值取决于推理引擎（vLLM 的 TTFT 通常 100~300ms 量级）。

用法：
    python eval/ttft_bench.py
    python eval/ttft_bench.py --ms-per-token 3 --tokens 200 --repeat 5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.cache import SemanticCache                              # noqa: E402
from agent.graph import AgentConfig, AgentGraph                     # noqa: E402
from agent.kb import KnowledgeBase                                 # noqa: E402
from agent.llm import RuleBasedPlannerLLM, SimulatedStreamingLLM   # noqa: E402
from agent.memory import ConversationMemory, VehicleProfile        # noqa: E402
from agent.obs import Tracer                                       # noqa: E402
from agent.tools import ToolRegistry                               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "eval", "ttft_metrics.json")
OUT_MD = os.path.join(ROOT, "eval", "ttft_report.md")

HOT_QUESTIONS = [
    "怎么打开危险警告灯", "座椅加热怎么关闭", "胎压报警了怎么办",
    "我这台车该保养了吗", "空调怎么开", "怎么连接蓝牙",
]
FILLER = "好的，正在为您查询手册…"


class PlannerWithStreaming(RuleBasedPlannerLLM):
    """规则规划器 + 流式生成替身：工具调用用规则，最终答案走流式生成。

    注意：替身生成的答案必须**来自检索到的证据**（这里取第一条证据的前 60 字），
    否则会被接地校验判为"无依据"而拒答，导致缓存里什么都存不下来——
    这是第一次跑 TTFT 基准时踩到的坑（缓存只有 1 条记录）。
    """

    def __init__(self, streaming_llm: SimulatedStreamingLLM, **kw):
        super().__init__(**kw)
        self.streaming = streaming_llm

    def _grounded_answer(self, messages) -> str:
        for message in reversed(list(messages)):
            if message.get("role") != "tool":
                continue
            try:
                payload = json.loads(message.get("content") or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            for item in (payload.get("data") or []):
                if isinstance(item, dict) and item.get("text"):
                    return str(item["text"])[:60]
        return "该问题需要更多信息，建议联系领克中心确认。"

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024, guided_json=None):
        response = super().chat(messages, tools=tools, temperature=temperature,
                                max_tokens=max_tokens, guided_json=guided_json)
        if response.tool_calls:
            return response
        self.streaming.answer = self._grounded_answer(messages)
        generated = self.streaming.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return type(response)(content=generated.content, usage=generated.usage)


def measure_once(kb, question: str, streaming: SimulatedStreamingLLM,
                 cache: SemanticCache, fingerprint: str,
                 use_cache: bool = True, use_stream: bool = True,
                 use_filler: bool = True) -> Dict[str, Any]:
    tracer = Tracer()
    registry = ToolRegistry(kb)
    planner = PlannerWithStreaming(streaming)
    agent = AgentGraph(planner, registry, tracer=tracer,
                       memory=ConversationMemory(profile=VehicleProfile()),
                       config=AgentConfig())

    start = time.perf_counter()
    first_byte_ms = None
    if use_filler:
        first_byte_ms = (time.perf_counter() - start) * 1000      # 垫话立即可见

    # ① 缓存查询（跳过整条链路）
    if use_cache:
        lookup = cache.get(question, fingerprint)
        if lookup.hit:
            latency = (time.perf_counter() - start) * 1000
            return {"mode": f"cache-{lookup.kind}", "ttft_ms": round(first_byte_ms or latency, 3),
                    "total_ms": round(latency, 3), "tokens": 0,
                    "stage_ms": {"cache": round(latency, 3)}}

    # ② 正常链路（检索 → 生成）
    state = agent.run(question)
    gate = state.latency_ms.get("tools", 0.0)
    if use_stream:
        # 流式：TTFT = 检索耗时 + 首 token 时延
        ttft = (first_byte_ms or 0) + gate + streaming.ms_per_token
    else:
        ttft = (first_byte_ms or 0) + state.latency_ms.get("total", 0.0)
    total = state.latency_ms.get("total", 0.0) + (first_byte_ms or 0)

    if use_cache and state.answer and state.answer != "无答案":
        cache.put(question, state.answer, state.citations, state.status, fingerprint)

    return {"mode": "stream" if use_stream else "batch",
            "ttft_ms": round(ttft, 3), "total_ms": round(total, 3),
            "tokens": state.tokens_total,
            "stage_ms": {"retrieve": round(gate, 3),
                         "orchestration": round(state.latency_ms.get("total", 0.0), 3)}}


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def p(values, q):
        if not values:
            return None
        values = sorted(values)
        return round(values[min(len(values) - 1, int(len(values) * q))], 3)
    ttfts = [r["ttft_ms"] for r in rows]
    totals = [r["total_ms"] for r in rows]
    return {"n": len(rows), "ttft_p50": p(ttfts, 0.5), "ttft_p95": p(ttfts, 0.95),
            "ttft_mean": round(statistics.mean(ttfts), 3) if ttfts else None,
            "total_p50": p(totals, 0.5), "total_p95": p(totals, 0.95),
            "tokens_mean": round(statistics.mean([r["tokens"] for r in rows]), 1) if rows else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms-per-token", type=float, default=4.0)
    ap.add_argument("--tokens", type=int, default=150)
    ap.add_argument("--repeat", type=int, default=3, help="每个问题重复次数（预热缓存）")
    args = ap.parse_args()

    print("[info] 加载知识库…", flush=True)
    kb = KnowledgeBase.load()

    def scenario(use_cache: bool, use_stream: bool, use_filler: bool) -> Dict[str, Any]:
        streaming = SimulatedStreamingLLM(answer="根据手册，按下危险警告灯按键即可开启危险警告灯。",
                                          ms_per_token=args.ms_per_token, tokens=args.tokens)
        cache = SemanticCache(threshold=0.92)
        registry = ToolRegistry(kb)
        fingerprint = SemanticCache.fingerprint(registry.telemetry)
        rows: List[Dict[str, Any]] = []
        for _ in range(args.repeat):
            for q in HOT_QUESTIONS:
                rows.append(measure_once(kb, q, streaming, cache, fingerprint,
                                         use_cache, use_stream, use_filler))
        summary = summarize(rows)
        summary["cache"] = cache.report()
        return summary

    print("[info] 四种配置对比中…", flush=True)
    batch = scenario(use_cache=False, use_stream=False, use_filler=False)
    stream = scenario(use_cache=False, use_stream=True, use_filler=False)
    stream_filler = scenario(use_cache=False, use_stream=True, use_filler=True)
    cached = scenario(use_cache=True, use_stream=True, use_filler=True)

    summary = {
        "gen_simulation": {"ms_per_token": args.ms_per_token, "tokens": args.tokens,
                           "simulated_gen_ms": args.ms_per_token * args.tokens},
        "n_questions": len(HOT_QUESTIONS), "repeat": args.repeat,
        "batch": batch, "stream": stream, "stream_filler": stream_filler, "cached": cached,
        "ttft_reduction_stream_vs_batch": round(batch["ttft_p50"] - stream["ttft_p50"], 3),
        "ttft_reduction_cached_vs_batch": round(batch["ttft_p50"] - cached["ttft_p50"], 3),
        "cache_hit_rate": cached["cache"]["hit_rate"],
        "token_reduction": round(1 - cached["tokens_mean"] / max(1e-9, batch["tokens_mean"]), 4),
    }
    json.dump(summary, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    md = [
        "# 首字延迟（TTFT）与缓存优化报告（自动生成）\n",
        f"- 热问题集：{len(HOT_QUESTIONS)} 条 × {args.repeat} 轮（缓存从冷启动到稳定）",
        f"- 生成耗时模拟参数：{args.ms_per_token} ms/token × {args.tokens} token ≈ "
        f"**{args.ms_per_token * args.tokens:.0f} ms** 整段生成时间",
        "- ⚠️ 诚实说明：**编排层（路由/检索/缓存/反思）耗时为真实测量**；"
        "**生成耗时为参数化模拟**（本机无 GPU）。真实 TTFT 取决于推理引擎，"
        "接口（`stream_chat`）已按 vLLM/OpenAI 流式协议实现，可直接替换。\n",
        "## 1. 四种配置的 TTFT\n",
        "| 配置 | TTFT P50 | TTFT P95 | 端到端 P50 | 平均 token |",
        "| --- | --- | --- | --- | --- |",
        f"| ① 非流式（基线） | {batch['ttft_p50']} ms | {batch['ttft_p95']} ms | "
        f"{batch['total_p50']} ms | {batch['tokens_mean']} |",
        f"| ② 流式输出 | {stream['ttft_p50']} ms | {stream['ttft_p95']} ms | "
        f"{stream['total_p50']} ms | {stream['tokens_mean']} |",
        f"| ③ 流式 + 垫话 | {stream_filler['ttft_p50']} ms | {stream_filler['ttft_p95']} ms | "
        f"{stream_filler['total_p50']} ms | {stream_filler['tokens_mean']} |",
        f"| ④ ③ + 语义缓存 | **{cached['ttft_p50']} ms** | **{cached['ttft_p95']} ms** | "
        f"**{cached['total_p50']} ms** | **{cached['tokens_mean']}** |\n",
        "## 2. 收益\n",
        f"- 流式 vs 非流式：TTFT 从 **{batch['ttft_p50']} ms 降到 {stream['ttft_p50']} ms**"
        f"（减少 {summary['ttft_reduction_stream_vs_batch']} ms，"
        f"{summary['ttft_reduction_stream_vs_batch'] / max(1e-9, batch['ttft_p50']):.1%}）；",
        f"- 叠加缓存后：TTFT 进一步降到 **{cached['ttft_p50']} ms**"
        f"（相对基线减少 {summary['ttft_reduction_cached_vs_batch']} ms），"
        f"缓存命中率 **{summary['cache_hit_rate']:.2%}**，"
        f"平均 token 消耗下降 **{summary['token_reduction']:.2%}**；",
        f"- 缓存统计：{json.dumps(cached['cache'], ensure_ascii=False)}\n",
        "## 3. 工程结论\n",
        "- **流式是座舱体验的必选项**：非流式下用户要等整段生成完，"
        f"本例模拟生成 {args.ms_per_token * args.tokens:.0f} ms，感知延迟差别巨大；",
        "- **语义缓存是性价比最高的一招**：座舱高频问题重复率极高，命中即省掉检索 + 生成；",
        "- **缓存键必须包含车况指纹**：否则会拿「胎压 228kPa」时的答案回复"
        "「胎压 148kPa」的新状态——这在座舱里是事故级错误（见 `agent/cache.py` 的指纹设计）；",
        "- **垫话要谨慎**：能降低感知延迟，但要在拿到结果后覆盖，避免「先答后错」。",
    ]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\n[json] {OUT_JSON}\n[md] {OUT_MD}")


if __name__ == "__main__":
    main()
