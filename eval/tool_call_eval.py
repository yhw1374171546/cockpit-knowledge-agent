# -*- coding: utf-8 -*-
"""Function Calling 评测：工具选择准确率、参数准确率、过度/不足调用、解析失败率。

两种后端
------
- `--backend planner`（默认，离线）：规则规划器 + **脏参数注入**。
  它衡量的是 **FC 管线健壮性**（参数修复、策略裁决、并行执行、观察回填），
  以及在规则基线下的行为一致性；**不代表真实模型的工具调用能力**。
- `--backend openai`：接真实 LLM（vLLM / 任意 OpenAI 兼容服务），
  得到的就是可以写进简历的真实 Function Calling 指标。

指标口径
------
- tool_selection_accuracy：期望工具集与实调工具集**完全一致**的比例（严格）
- tool_recall / tool_precision：按工具维度计（宽松，看漏调与多调）
- over_call_rate：`should_not_call` 类里仍然调用工具的比例（幻觉调用）
- under_call_rate：`should_call` / `multi_tool` 类里该调却没调的比例
- arg_accuracy：关键参数命中率（子串匹配，容忍等价写法）
- parse_failure_rate：参数解析失败（修复也没救回来）比例

用法：
    python eval/tool_call_eval.py
    python eval/tool_call_eval.py --dirty-ratio 0.5 --verbose
    python eval/tool_call_eval.py --backend openai --base-url http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.tool_call_set import load_cases                             # noqa: E402

from agent.executor import ToolExecutor                              # noqa: E402
from agent.guardrails import PolicyContext                           # noqa: E402
from agent.kb import KnowledgeBase                                   # noqa: E402
from agent.llm import (OpenAICompatibleLLM, RuleBasedPlannerLLM,     # noqa: E402
                       ToolCall)
from agent.obs import Tracer                                         # noqa: E402
from agent.tools import ToolRegistry                                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "eval", "tool_call_metrics.json")
OUT_MD = os.path.join(ROOT, "eval", "tool_call_report.md")

SYSTEM_TOOL_PROMPT = (
    "你是智能座舱助手。请判断是否需要调用工具：\n"
    "- 车辆功能、故障、保养、参数这类事实性问题，必须调用工具后再回答；\n"
    "- 打招呼、闲聊、域外问题、澄清类请求不要调用工具，也不要编造答案。"
)


class DirtyPlanner(RuleBasedPlannerLLM):
    """在规则规划器外面套一层"脏参数"注入，用来压测 FC 管线（修复/校验/兜底）。"""

    def __init__(self, known_tools: Sequence[str], dirty_ratio: float = 0.3, **kw):
        super().__init__(**kw)
        self.known_tools = list(known_tools)
        self.dirty_ratio = dirty_ratio
        self.random = __import__("random").Random(11)
        self.injected_dirty = 0

    def _plan(self, question: str, available: set) -> List[ToolCall]:
        calls = super()._plan(question, available)
        out: List[ToolCall] = []
        for call in calls:
            if self.random.random() < self.dirty_ratio:
                self.injected_dirty += 1
                # 制造模型的典型脏输出：代码块 + 单引号 + 尾随逗号
                raw = "```json\n" + json.dumps(call.arguments, ensure_ascii=False) \
                    .replace('"', "'") + ",\n```"
                out.append(ToolCall(call.name, {}, call.id, raw_arguments=raw))
            else:
                out.append(call)
        return out


def _parse_dirty(calls: Sequence[ToolCall], repair) -> int:
    """把 raw_arguments 重新解析回 arguments，返回**真正的**解析失败数。

    注意：无参工具（如 get_vehicle_status）的参数本来就是 `{}`，
    不能把"解析出空字典"当成解析失败——那是评测口径 bug（会把正确的空参误记为失败）。
    因此以修复器自身的失败计数增量为准。
    """
    before = getattr(repair, "parse_failures", 0)
    for call in calls:
        if call.raw_arguments:
            args, repaired = repair.parse(call.raw_arguments)
            call.arguments = args
            call.repaired = repaired
    return getattr(repair, "parse_failures", 0) - before


def run_case(case: Dict[str, Any], llm, registry: ToolRegistry, repair,
             executor: ToolExecutor) -> Dict[str, Any]:
    tracer = Tracer(enabled=False)
    messages = [{"role": "system", "content": SYSTEM_TOOL_PROMPT},
                {"role": "user", "content": case["query"]}]
    calls: List[ToolCall] = []
    parse_failures = 0
    for _ in range(3):                       # 允许一次工具结果回填后的再决策
        resp = llm.chat(messages, tools=registry.specs(), temperature=0.0, max_tokens=512)
        if not resp.tool_calls:
            break
        parse_failures += _parse_dirty(resp.tool_calls, repair)
        real = [c for c in resp.tool_calls if c.name in registry.tools]
        calls.extend(real)
        if not real:
            break
        executed = executor.execute(real, PolicyContext())
        messages.append({"role": "assistant", "content": None,
                         "tool_calls": [c.to_openai() for c in real]})
        for item in executed:
            messages.append({"role": "tool", "name": item.call.name,
                             "content": item.result.to_observation()[:800]})

    called = []
    for c in calls:
        if c.name not in called:
            called.append(c.name)
    expected = list(case["expected_tools"])

    exact = set(called) == set(expected)
    recall = (len(set(expected) & set(called)) / len(expected)) if expected else 1.0
    precision = (len(set(expected) & set(called)) / len(called)) if called else 1.0

    arg_total = arg_hit = 0
    for tool, fields in (case.get("expected_args") or {}).items():
        actual = next((c for c in calls if c.name == tool), None)
        if actual is None:
            arg_total += len(fields)
            continue
        for field, want in fields.items():
            arg_total += 1
            got = actual.arguments.get(field)
            if got is not None and (str(want) in str(got) or str(got) in str(want)):
                arg_hit += 1

    return {
        "id": case["id"], "query": case["query"], "category": case["category"],
        "expected_tools": expected, "called_tools": called,
        "exact_match": exact, "tool_recall": recall, "tool_precision": precision,
        "arg_total": arg_total, "arg_hit": arg_hit,
        "parse_failures": parse_failures,
        "over_call": case["category"] == "should_not_call" and bool(called),
        "under_call": case["category"] in ("should_call", "multi_tool") and not called,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["planner", "openai"], default="planner")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen2_5_7B_Instruct")
    ap.add_argument("--dirty-ratio", type=float, default=0.3,
                    help="规则规划器下注入脏参数的比例（压测 FC 修复链路）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("[info] 加载知识库…", flush=True)
    kb = KnowledgeBase.load()
    registry = ToolRegistry(kb)
    tracer = Tracer()
    executor = ToolExecutor(registry, tracer=tracer, parallel=True)

    if args.backend == "openai":
        llm = OpenAICompatibleLLM(base_url=args.base_url, model=args.model,
                                 known_tools=registry.names())
    else:
        llm = DirtyPlanner(registry.names(), dirty_ratio=args.dirty_ratio)
    repair = getattr(llm, "repair", None)
    if repair is None:
        from agent.llm import ToolCallRepair
        repair = ToolCallRepair(registry.names())

    cases = load_cases()
    rows = [run_case(c, llm, registry, repair, executor) for c in cases]

    n = len(rows)
    def rate(pred) -> float:
        return round(sum(1 for r in rows if pred(r)) / n, 4) if n else 0.0

    should_call = [r for r in rows if r["category"] in ("should_call", "multi_tool", "write_guard")]
    not_call = [r for r in rows if r["category"] == "should_not_call"]
    arg_total = sum(r["arg_total"] for r in rows)
    arg_hit = sum(r["arg_hit"] for r in rows)
    parse_fail = sum(r["parse_failures"] for r in rows)

    summary = {
        "backend": args.backend,
        "n_cases": n,
        "tool_selection_accuracy": rate(lambda r: r["exact_match"]),
        "tool_recall": round(sum(r["tool_recall"] for r in rows) / n, 4),
        "tool_precision": round(sum(r["tool_precision"] for r in rows) / n, 4),
        "over_call_rate": round(sum(1 for r in not_call if r["over_call"]) / len(not_call), 4)
        if not_call else 0.0,
        "under_call_rate": round(sum(1 for r in should_call if r["under_call"]) / len(should_call), 4)
        if should_call else 0.0,
        "arg_accuracy": round(arg_hit / arg_total, 4) if arg_total else None,
        "arg_samples": arg_total,
        "parse_failures": parse_fail,
        "parse_failure_rate": round(parse_fail / max(1, sum(len(r["called_tools"]) for r in rows)), 4),
        "dirty_injected": getattr(llm, "injected_dirty", 0),
        "by_category": {
            cat: round(sum(1 for r in rows if r["category"] == cat and r["exact_match"])
                       / max(1, sum(1 for r in rows if r["category"] == cat)), 4)
            for cat in sorted({r["category"] for r in rows})
        },
    }
    json.dump({"summary": summary, "rows": rows},
              open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    md = [
        "# Function Calling 评测报告（自动生成）\n",
        f"- 评测集：{n} 条（should_call / multi_tool / write_guard / should_not_call 四类）",
        f"- 后端：`{args.backend}`"
        + ("（离线规则规划器 + 脏参数注入）" if args.backend == "planner"
           else "（真实 LLM，OpenAI 兼容接口）"),
        f"- 知识库工具数：{len(registry.names())}\n",
        "## 1. 总体指标\n",
        "| 指标 | 数值 | 说明 |", "| --- | --- | --- |",
        f"| 工具选择准确率（严格） | **{summary['tool_selection_accuracy']:.2%}** | 期望工具集与实调完全一致 |",
        f"| 工具召回 / 精确 | {summary['tool_recall']:.2%} / {summary['tool_precision']:.2%} | 漏调与多调分开看 |",
        f"| 过度调用率（闲聊也查库） | {summary['over_call_rate']:.2%} | should_not_call 类，越低越好 |",
        f"| 不足调用率（该查不查） | {summary['under_call_rate']:.2%} | 事实性问题未调工具，越低越好 |",
        f"| 关键参数准确率 | {summary['arg_accuracy']:.2%} | {summary['arg_samples']} 个标注参数 |",
        f"| 参数解析失败率 | {summary['parse_failure_rate']:.2%} | 修复链路兜底后仍失败 |",
        f"| 注入脏参数条数 | {summary['dirty_injected']} | 主动构造的脏输出 |\n",
        "## 2. 分类别严格准确率\n",
        "| 类别 | 准确率 |", "| --- | --- |",
    ]
    for cat, value in summary["by_category"].items():
        md.append(f"| {cat} | {value:.2%} |")
    md += [
        "\n## 3. 口径说明（重要）\n",
        "- 离线 `planner` 后端衡量的是 **FC 管线健壮性**（参数修复、策略裁决、并行执行、"
        "观察回填）与规则基线的一致性，**不代表真实模型的工具调用能力**；",
        "- 要拿到可写进简历的真实模型指标，请在 vLLM 起服务后执行：\n",
        "```bash",
        "python -m eval.tool_call_eval --backend openai --base-url http://127.0.0.1:8000/v1",
        "```",
    ]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))

    if args.verbose:
        print("\n逐条明细：")
        for r in rows:
            flag = "OK " if r["exact_match"] else "DIFF"
            print(f"  [{flag}] {r['id']} 期望={r['expected_tools']} 实调={r['called_tools']}  {r['query'][:24]}")
    print(f"\n[json] {OUT_JSON}\n[md] {OUT_MD}")


if __name__ == "__main__":
    main()
