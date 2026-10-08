# -*- coding: utf-8 -*-
"""多 Agent 评测：Supervisor 多 Agent vs 单 Agent（全工具）对比。

为什么要这份对比
--------------
"为什么不用一个 Agent 挂更多工具？"是多 Agent 方案必被追问的问题。
这里用同一批样本、同一套护栏与 trace，把两边放在一起量：

| 维度 | 单 Agent | 多 Agent |
|---|---|---|
| 工具作用域 | 全部 5 个工具 | 专家白名单（physical 隔离） |
| 越界工具调用 | 可能发生 | 结构上不可能 |
| 多意图覆盖 | 单次检索/单条回答 | 拆解 + 并行专家 |
| 安全合规 | 无最终把关 | 安全评审 Agent（停驶指令/许可性表述清理） |
| 冲突处理 | 无仲裁 | 显式优先级仲裁 + 升级人工 |
| 延迟 | 串行 | 并行 fan-out（I/O 型工具才有收益） |

用法：
    python eval/multiagent_eval.py                    # 离线规则替身（默认，CI 基线）
    python eval/multiagent_eval.py --provider deepseek   # 真实云端模型
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.llm_backend import (add_llm_args, backend_fields, cost_line,  # noqa: E402
                              llm_id, llm_label, make_llm)
from eval.multiagent_set import TELEMETRY_PROFILES, load_cases    # noqa: E402

from agent.kb import KnowledgeBase                                # noqa: E402
from agent.llm import redact                                      # noqa: E402
from agent.memory import ConversationMemory, VehicleProfile       # noqa: E402
from agent.obs import Tracer                                      # noqa: E402
from agent.orchestrator import Supervisor                         # noqa: E402
from agent.protocols import SAFETY_DIRECTIVES, risk_at_least      # noqa: E402
from agent.tools import ToolRegistry                              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "eval", "multiagent_metrics.json")
OUT_MD = os.path.join(ROOT, "eval", "multiagent_report.md")


def run_multi(kb, case: Dict, tracer: Tracer, llm=None) -> Dict[str, Any]:
    sup = Supervisor(kb, llm, tracer=tracer, telemetry=TELEMETRY_PROFILES[case["telemetry"]])
    state = sup.run(case["query"], memory=ConversationMemory(profile=VehicleProfile()))
    agents = {r.agent for r in state.results}
    answer = state.answer or ""
    directive = any(d in answer for d in SAFETY_DIRECTIVES)
    scope_calls = [c["tool"] for s in [sup.specialists[r.agent] for r in state.results
                                       if r.agent in sup.specialists]
                   for c in [] ]  # 占位：多 Agent 的工具调用在子 Agent 内部，用白名单判定
    allowed = {t for r in state.results if r.agent in sup.specialists
               for t in sup.specialists[r.agent].allowed_tools}
    return {
        "id": case["id"], "kind": case["kind"], "query": case["query"],
        "routing_ok": agents == set(case["expect_agents"]),
        "agents": sorted(agents),
        "answered": bool(answer.strip()) and answer.strip() != "无答案",
        "directive": directive,
        "conflict_detected": any(c.kind == "safety_override" for c in state.conflicts),
        "conflicts": [c.to_dict() for c in state.conflicts],
        "verdict": state.verdict.verdict if state.verdict else None,
        "answer": answer[:200],
        "tokens": state.tokens_total,
        "latency_ms": state.latency_ms.get("total", 0.0),
        "dispatch_ms": state.latency_ms.get("dispatch", 0.0),
        "n_tool_calls": sum(len(r.policy_flags) for r in state.results),
        "expert_tools": sorted(allowed),
    }


def run_single(kb, case: Dict, tracer: Tracer, llm=None) -> Dict[str, Any]:
    """单 Agent 基线：同一个 LLM 后端 + 全部 5 个工具 + 同样的护栏与 trace。"""
    sup = Supervisor(kb, llm, tracer=tracer, telemetry=TELEMETRY_PROFILES[case["telemetry"]])
    state = sup.run_baseline(case["query"], memory=ConversationMemory(profile=VehicleProfile()))
    answer = state.answer or ""
    called = [t["tool"] for t in state.tool_calls]
    return {
        "id": case["id"], "kind": case["kind"],
        "tools_called": sorted(set(called)),
        "answered": bool(answer.strip()) and answer.strip() != "无答案",
        "directive": any(d in answer for d in SAFETY_DIRECTIVES),
        "conflict_detected": False,           # 单 Agent 没有仲裁层
        "verdict": None,                      # 单 Agent 没有安全评审
        "answer": answer[:200],
        "tokens": state.tokens_total,
        "latency_ms": state.latency_ms.get("total", 0.0),
    }


def pct(rows: Sequence[Dict], pred) -> float:
    return round(sum(1 for r in rows if pred(r)) / len(rows), 4) if rows else 0.0


def main():
    ap = argparse.ArgumentParser()
    add_llm_args(ap)
    ap.add_argument("--json-out", default=OUT_JSON)
    ap.add_argument("--md-out", default=OUT_MD)
    args = ap.parse_args()

    print("[info] 加载知识库…", flush=True)
    kb = KnowledgeBase.load()
    cases = load_cases()
    tracer = Tracer()
    print(f"[info] 样本 {len(cases)} 条（single/multi/safety/conflict 四类）")

    if args.provider == "planner":
        # 默认离线基线：llm=None，专家各自用 ScopedPlanner（行为与 CI 基线完全一致）
        llm = None
    else:
        try:
            llm = make_llm(args, known_tools=ToolRegistry(kb).names())
        except RuntimeError as exc:
            print(f"[错误] {redact(exc)}")
            return 2
    print(f"[info] LLM 后端: {llm_id(args)} —— {llm_label(args)}")

    multi = [run_multi(kb, c, tracer, llm) for c in cases]
    single = [run_single(kb, c, tracer, llm) for c in cases]

    safety_cases = [c for c in cases if c["kind"] in ("safety", "conflict")]
    conflict_cases = [c for c in cases if c["expect_conflict"]]
    multi_cases = [c for c in cases if c["kind"] == "multi"]
    m_safety = [r for r, c in zip(multi, cases) if c["kind"] in ("safety", "conflict")]
    s_safety = [r for r, c in zip(single, cases) if c["kind"] in ("safety", "conflict")]
    m_conf = [r for r, c in zip(multi, cases) if c["expect_conflict"]]
    m_multi = [r for r, c in zip(multi, cases) if c["kind"] == "multi"]
    m_all = multi
    s_all = single

    summary = {
        "n_cases": len(cases),
        "multi": {
            "routing_accuracy": pct([r for r, c in zip(multi, cases) if c["kind"] in ("multi", "conflict")],
                                    lambda r: r["routing_ok"]),
            "answer_rate": pct(m_all, lambda r: r["answered"]),
            "safety_directive_rate": pct(m_safety, lambda r: r["directive"]),
            "conflict_detection_rate": pct(m_conf, lambda r: r["conflict_detected"]),
            "review_rate": pct(m_all, lambda r: r["verdict"] in ("approve", "revise", "block")),
            "avg_latency_ms": round(statistics.mean([r["latency_ms"] for r in m_all]), 2),
            "avg_dispatch_ms": round(statistics.mean([r["dispatch_ms"] for r in m_all]), 2),
            "avg_tokens": round(statistics.mean([r["tokens"] for r in m_all]), 1),
        },
        "single": {
            "answer_rate": pct(s_all, lambda r: r["answered"]),
            "safety_directive_rate": pct(s_safety, lambda r: r["directive"]),
            "conflict_detection_rate": 0.0,
            "review_rate": 0.0,
            "avg_latency_ms": round(statistics.mean([r["latency_ms"] for r in s_all]), 2),
            "avg_tokens": round(statistics.mean([r["tokens"] for r in s_all]), 1),
        },
    }
    # 本次使用的后端标识与成本账本（离线替身为 planner / $0）
    summary.update(backend_fields(args, llm))
    json.dump({"summary": summary, "multi": multi, "single": single},
              open(args.json_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    md = [
        "# 多 Agent vs 单 Agent 对比评测（自动生成）\n",
        f"- 样本：{len(cases)} 条（单意图 8 / 多意图 8 / 安全关键 4 / 冲突 4）",
        f"- 模型后端：`{llm_id(args)}` —— {llm_label(args)}",
        cost_line(summary),
        "- 两个系统使用**同一个 LLM 后端、同一套护栏与 trace**，唯一差别是编排方式\n",
        "## 1. 核心对比\n",
        "| 指标 | 单 Agent（全工具） | 多 Agent（Supervisor+专家） |",
        "| --- | --- | --- |",
        f"| 路由准确率（多意图/冲突类） | 不适用（无路由） | "
        f"**{summary['multi']['routing_accuracy']:.2%}** |",
        f"| 答案产出率 | {summary['single']['answer_rate']:.2%} | "
        f"{summary['multi']['answer_rate']:.2%} |",
        f"| **安全指令合规率**（critical 场景必须给停驶指令） | "
        f"{summary['single']['safety_directive_rate']:.2%} | "
        f"**{summary['multi']['safety_directive_rate']:.2%}** |",
        f"| 冲突检测率 | {summary['single']['conflict_detection_rate']:.2%} | "
        f"**{summary['multi']['conflict_detection_rate']:.2%}** |",
        f"| 安全评审覆盖率 | {summary['single']['review_rate']:.2%} | "
        f"**{summary['multi']['review_rate']:.2%}** |",
        f"| 单题平均耗时 | {summary['single']['avg_latency_ms']} ms | "
        f"{summary['multi']['avg_latency_ms']} ms（其中并行派发 "
        f"{summary['multi']['avg_dispatch_ms']} ms） |",
        f"| 单题平均 token | {summary['single']['avg_tokens']} | "
        f"{summary['multi']['avg_tokens']} |\n",
        "## 2. 工具作用域（最本质的差别）\n",
        "| 专家 | 可见工具白名单 |", "| --- | --- |",
    ]
    sup = Supervisor(kb)
    for name, tools in sup.stats()["specialists"].items():
        md.append(f"| {name} | {', '.join(tools)} |")
    md += [
        f"\n单 Agent 基线一次性暴露全部 {len(sup.registry.names())} 个工具；"
        "多 Agent 通过 `ToolRegistry.subset()` 做物理隔离——"
        "**车控专家根本没有 `create_service_order` 这个工具，也就不可能误下预约单**。\n",
        "## 3. 逐条结果（多 Agent）\n",
        "| ID | 类型 | 期望专家 | 实际专家 | 路由 | 冲突 | 评审 | 安全指令 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r, c in zip(multi, cases):
        md.append(f"| {r['id']} | {r['kind']} | {','.join(c['expect_agents'])} | "
                  f"{','.join(r['agents'])} | {'✅' if r['routing_ok'] else '❌'} | "
                  f"{'✅' if r['conflict_detected'] else '—'} | {r['verdict']} | "
                  f"{'✅' if r['directive'] else '—'} |")
    md += [
        "\n## 4. 结论与代价（如实说明）\n",
        "- **安全合规（口径已更新）**：单 Agent 的合规率为 "
        f"{summary['single']['safety_directive_rate']:.2%}，多 Agent 为 "
        f"**{summary['multi']['safety_directive_rate']:.2%}**。\n"
        "  ⚠️ 这个指标**不再是多 Agent 的增量来源**：单 Agent 链路已加入"
        "「安全硬门控」（critical 场景强制前置停驶/联系中心指令，见 "
        "`agent/protocols.py::ensure_safety_directive`），因此两者都能达到 100%。\n"
        "  换句话说，**安全兜底可以是单 Agent 的一个确定性规则，不必依赖多 Agent**——"
        "把这一点如实写出来，比拿一个已经被单 Agent 追平的指标当卖点更可信。",
        f"- **冲突仲裁仍是单 Agent 完全没有的能力**：多 Agent 检出率 "
        f"{summary['multi']['conflict_detection_rate']:.2%}（单 Agent "
        f"{summary['single']['conflict_detection_rate']:.2%}），并把"
        "「手册通用建议 vs 实时严重风险」显式裁决为安全优先；",
        f"- **代价**：多 Agent 的 token 消耗为 "
        f"{summary['multi']['avg_tokens']} vs 单 Agent {summary['single']['avg_tokens']}"
        "（专家各自带上下文与系统提示）；延迟上多意图请求可并行派发，"
        "但单意图请求会因多一层编排略慢；",
        "- **适用判断**：座舱这类「有写操作 + 有安全红线 + 有实时数据」的场景值得上多 Agent；"
        "纯问答场景单 Agent 更省。",
    ]
    open(args.md_out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\n[json] {args.json_out}\n[md] {args.md_out}")


if __name__ == "__main__":
    main()
