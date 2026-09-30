# -*- coding: utf-8 -*-
"""Agent 命令行入口。

用法示例
--------
# 离线演示（规则规划器，无需 GPU / 大模型，秒级返回）
python -m agent.run_agent --backend planner "靠背太热怎么办"

# 多轮对话（观察指代消解与话题延续）
python -m agent.run_agent --backend planner --multi-turn

# 接真实大模型：先在 4 卡机上起 vLLM OpenAI 兼容服务
#   bash benchmark/server.sh
# 然后：
python -m agent.run_agent --backend openai --base-url http://127.0.0.1:8000/v1 \
    --model Qwen2_7B "胎压报警了怎么办"
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from agent.graph import AgentConfig, AgentGraph, build_langgraph_app
from agent.llm import OpenAICompatibleLLM, RuleBasedPlannerLLM
from agent.memory import ConversationMemory, VehicleProfile
from agent.reflection import Reflector
from agent.tools import build_default_registry

DEMO_QUESTIONS = [
    "靠背太热怎么办",
    "怎么打开危险警告灯",
    "胎压报警了怎么办",
    "我这台车该保养了吗",
    "领克08的纯电续航是多少",
    "中国足球的队长是谁",
]


def build_agent(args) -> AgentGraph:
    registry = build_default_registry(args.kb)
    reflector = Reflector(evidence_overlap_threshold=args.gate_threshold)
    config = AgentConfig(
        max_steps=args.max_steps,
        evidence_gate=args.gate_threshold > 0,
        evidence_overlap_threshold=args.gate_threshold,
    )
    if args.backend == "openai":
        llm = OpenAICompatibleLLM(base_url=args.base_url, model=args.model)
    else:
        llm = RuleBasedPlannerLLM(overlap_threshold=args.gate_threshold)
    memory = ConversationMemory(window=6, profile=VehicleProfile(model="领克08", mileage_km=23860))
    return AgentGraph(llm, registry, reflector=reflector, memory=memory, config=config)


def print_state(state, verbose: bool = False) -> None:
    print(f"\n{'=' * 78}")
    print(f"Q: {state.question}")
    if state.resolved_question != state.question:
        print(f"   （指代消解后）: {state.resolved_question}")
    print(f"路由: {state.route}   状态: {state.status}   步数: {state.steps}   "
          f"重试: {state.retries}   总耗时: {state.latency_ms.get('total', 0):.1f} ms")
    if state.tool_calls:
        tools = "，".join(f"{t['tool']}({'ok' if t['ok'] else 'fail'}"
                          f"{',重复' if t['repeated'] else ''})" for t in state.tool_calls)
        print(f"工具调用: {tools}")
    print(f"A: {state.answer}")
    if state.citations:
        print(f"出处: {' '.join(state.citations[:4])}")
    if state.reflection:
        r = state.reflection
        print(f"反思: {r['verdict']}（接地率 {r['grounded_ratio']}，动作 {r['next_action']}）")
    if verbose:
        print("--- 轨迹 ---")
        for t in state.trace:
            print("  " + json.dumps(t, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*", help="要问的问题；不传则跑内置演示集")
    ap.add_argument("--backend", choices=["planner", "openai"], default="planner")
    ap.add_argument("--kb", default=None, help="知识库路径（默认 kb/chunks.jsonl 或 all_text.txt）")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen2_7B")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--gate-threshold", type=float, default=0.40,
                    help="证据覆盖率硬门控阈值，0 表示关闭")
    ap.add_argument("--multi-turn", action="store_true", help="使用同一份记忆连续对话")
    ap.add_argument("--langgraph", action="store_true", help="额外演示 LangGraph 编排（需已安装）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    agent = build_agent(args)
    questions = args.question or DEMO_QUESTIONS

    start = time.perf_counter()
    if args.multi_turn:
        for q in questions:
            print_state(agent.run(q), args.verbose)
    else:
        for q in questions:
            # 单轮评测：每个问题用独立记忆，避免话题串味
            agent.memory = ConversationMemory(
                window=6, profile=VehicleProfile(model="领克08", mileage_km=23860))
            print_state(agent.run(q), args.verbose)
    print(f"\n共 {len(questions)} 题，总耗时 {(time.perf_counter() - start):.2f} s")

    if args.langgraph:
        app = build_langgraph_app(agent)
        if app is None:
            print("\n[langgraph] 未安装 langgraph，跳过（pip install langgraph 后可用）",
                  file=sys.stderr)
        else:
            out = app.invoke({"question": questions[0], "resolved": questions[0]})
            print(f"\n[langgraph] {questions[0]} → {json.dumps(out, ensure_ascii=False)[:200]}")


if __name__ == "__main__":
    main()
