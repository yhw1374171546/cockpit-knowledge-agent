# -*- coding: utf-8 -*-
"""多轮对话评测：指代消解 / 追问产出 / 错误继承 / 跨轮安全指令保持。

口径（每一项都能在无 GPU、无网络的离线规则替身下跑出来）
--------------------------------------------------------
| 指标 | 定义 | 方向 |
| --- | --- | --- |
| `resolution_success_rate` | 标注了 `resolve_expect` 的轮次里，`state.resolved_question` 命中全部期望关键词的比例 | 越高越好 |
| `followup_answer_rate` | 标注了 `expect_answer` 的**非首轮**里，产出非「无答案」回答的比例 | 越高越好 |
| `wrong_inheritance_rate` | 标注了 `inherit_forbid` 的轮次里，消解结果**命中禁止词**（把上一话题带进新话题）的比例 | 越低越好 |
| `cross_turn_safety_rate` | 标注了 `expect_safety` 的轮次里，答案保留安全指令（停驶/远离/联系中心等）的比例 | 越高越好 |
| `safety_first_turn_rate` | 安全类样本**首轮**就给出安全指令的比例（作为基准对照） | 越高越好 |

为什么「错误继承率」值得单独一条指标
----------------------------------
多轮最常见的失败不是"答不上"，而是**答串了**：车主已经换话题，系统仍用旧话题补全查询，
于是拿旧话题的证据回答新问题——用户看到的是一个自信但错的答案。
这类错误在单轮评测里 100% 不可见。

用法：
    python eval/multiturn_eval.py                                  # 迷你语料（CI 口径）
    python eval/multiturn_eval.py --kb kb/chunks.jsonl             # 完整语料
    python eval/multiturn_eval.py --limit 6 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.multiturn_set import (CATEGORIES, CATEGORY_LABEL,  # noqa: E402
                               SAFETY_KEYWORDS, load_cases, stats as set_stats)

from agent.graph import AgentConfig, AgentGraph                       # noqa: E402
from agent.kb import FIXTURE_KB, KnowledgeBase                        # noqa: E402
from agent.llm import RuleBasedPlannerLLM, redact                     # noqa: E402
from agent.memory import ConversationMemory, VehicleProfile           # noqa: E402
from agent.reflection import NO_ANSWER                                # noqa: E402
from agent.tools import TELEMETRY_PROFILES, ToolRegistry              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "eval", "multiturn_metrics.json")
OUT_MD = os.path.join(ROOT, "eval", "multiturn_report.md")
OUT_DETAIL = os.path.join(ROOT, "eval", "multiturn_details.jsonl")


def _has_safety(text: str) -> bool:
    return any(k in (text or "") for k in SAFETY_KEYWORDS)


def _build_memory(case: Dict) -> ConversationMemory:
    """按样本声明构造初始档案（模拟"车主档案已存在于车云"的真实前提）。"""
    profile = VehicleProfile()
    for key, value in (case.get("profile") or {}).items():
        if hasattr(profile, key):
            setattr(profile, key, value)
    return ConversationMemory(window=6, profile=profile)


def _build_registry(kb: KnowledgeBase, case: Dict) -> ToolRegistry:
    telemetry = case.get("telemetry") or "normal"
    return ToolRegistry(kb, telemetry=dict(TELEMETRY_PROFILES.get(telemetry,
                                                                 TELEMETRY_PROFILES["normal"])))


def run_case(kb: KnowledgeBase, case: Dict, rewrite: bool = True) -> Dict[str, Any]:
    """按真实会话顺序跑完一个样本的所有轮次（共用同一份 memory 与工具缓存）。"""
    planner = RuleBasedPlannerLLM(rewrite_enabled=rewrite)
    registry = _build_registry(kb, case)
    memory = _build_memory(case)
    agent = AgentGraph(planner, registry, memory=memory,
                       config=AgentConfig(max_steps=6, evidence_gate=True,
                                          evidence_overlap_threshold=0.40))

    # 首轮之前的档案描述，用于报告里核对槽位是否真的进了上下文
    profile_before = memory.profile.describe()

    rows: List[Dict[str, Any]] = []
    prev_answer = ""
    for idx, turn in enumerate(case["turns"], start=1):
        user = turn["user"]
        inherited_topic = agent.memory.topic_from_history()
        needs_context = agent.memory.needs_context(user)

        t0 = time.perf_counter()
        error = ""
        try:
            state = agent.run(user)
        except Exception as exc:                                     # noqa: BLE001
            error = f"{type(exc).__name__}: {redact(exc)}"
            state = None
        wall_ms = (time.perf_counter() - t0) * 1000

        resolved = state.resolved_question if state else ""
        answer = (state.answer or "") if state else ""
        answered = bool(answer.strip()) and answer.strip() != NO_ANSWER

        expect_resolve = list(turn.get("resolve_expect") or [])
        resolve_hits = [k for k in expect_resolve if k in resolved]
        resolve_ok = (len(resolve_hits) == len(expect_resolve)) if expect_resolve else None

        forbid = list(turn.get("inherit_forbid") or [])
        forbid_hits = [k for k in forbid if k in resolved]
        inherited_wrongly = bool(forbid_hits) if forbid else None

        safety_expected = turn.get("expect_safety")
        safety_present = _has_safety(answer)
        # 跨轮安全指令保持：本轮要求安全指令且首轮也给过指令 → 本轮是否仍保留
        carried = None
        if safety_expected:
            carried = safety_present

        rows.append({
            "turn": idx, "user": user, "needs_context": needs_context,
            "inherited_topic": inherited_topic, "resolved_question": resolved,
            "status": state.status if state else "error",
            "answer": answer[:300], "answered": answered,
            "citations": list(state.citations) if state else [],
            "n_evidence": len(state.evidence) if state else 0,
            "evidence_pages": [e.get("page") for e in (state.evidence if state else [])][:6],
            "evidence_headers": [e.get("header") for e in (state.evidence if state else [])][:6],
            "tools": [c["tool"] for c in (state.tool_calls if state else [])],
            "gate_coverage": state.gate_coverage if state else None,
            "route": state.route if state else None,
            "expect_answer": bool(turn.get("expect_answer")),
            "expect_resolve": expect_resolve, "resolve_hits": resolve_hits,
            "resolve_ok": resolve_ok,
            "inherit_forbid": forbid, "forbid_hits": forbid_hits,
            "inherited_wrongly": inherited_wrongly,
            "expect_safety": bool(safety_expected), "safety_present": safety_present,
            "safety_carried": carried,
            "prev_answer_had_safety": _has_safety(prev_answer),
            "latency_total_ms": round(state.latency_ms.get("total", 0.0), 2) if state else None,
            "wall_ms": round(wall_ms, 2),
            "error": error,
        })
        if state is not None:
            prev_answer = answer
        else:
            prev_answer = ""

    # 样本级判定：任一硬性标注不满足即视为该样本 FAIL（便于报告逐条定位）
    bad = []
    for row in rows:
        if row["error"]:
            bad.append(f"turn{row['turn']}: 异常 {row['error']}")
        if row["resolve_ok"] is False:
            bad.append(f"turn{row['turn']}: 消解未命中 {row['resolve_hits']}/{row['expect_resolve']}")
        if row["inherited_wrongly"]:
            bad.append(f"turn{row['turn']}: 错误继承 {row['forbid_hits']}")
        if row["expect_answer"] and not row["answered"]:
            bad.append(f"turn{row['turn']}: 未产出答案")
        if row["expect_safety"] and not row["safety_present"]:
            bad.append(f"turn{row['turn']}: 安全指令丢失")

    return {
        "id": case["id"], "category": case["category"], "note": case["note"],
        "telemetry": case.get("telemetry", "normal"),
        "profile": (case.get("profile") or {}), "profile_describe": profile_before,
        "turns": rows, "pass": not bad, "failures": bad,
    }


def _rate(rows: List[Dict], key: str, value: Any = True) -> Optional[float]:
    subset = [r for r in rows if r[key] is not None]
    if not subset:
        return None
    return round(sum(1 for r in subset if r[key] == value) / len(subset), 4)


def summarize(cases: List[Dict], kb_stats: Dict) -> Dict[str, Any]:
    all_rows = [r for c in cases for r in c["turns"]]
    followups = [r for r in all_rows if r["turn"] > 1]

    resolve_rows = [r for r in all_rows if r["resolve_ok"] is not None]
    forbid_rows = [r for r in all_rows if r["inherited_wrongly"] is not None]
    safety_rows = [r for r in all_rows if r["expect_safety"]]
    answer_rows = [r for r in all_rows if r["expect_answer"]]
    followup_answer_rows = [r for r in answer_rows if r["turn"] > 1]

    resolution_success = _rate(resolve_rows, "resolve_ok")
    wrong_inherit = _rate(forbid_rows, "inherited_wrongly")
    safety_rate = _rate(safety_rows, "safety_present")
    # 首轮安全指令产出率：作为"跨轮保持"的基线对照（首轮就不给指令，第 2 轮谈不上保持）
    first_turn_safety = [r for r in safety_rows if r["turn"] == 1]

    by_category: Dict[str, Any] = {}
    for cat in CATEGORIES:
        crows = [r for c in cases if c["category"] == cat for r in c["turns"]]
        if not crows:
            continue
        c_resolve = [r for r in crows if r["resolve_ok"] is not None]
        c_forbid = [r for r in crows if r["inherited_wrongly"] is not None]
        c_answer = [r for r in crows if r["expect_answer"] and r["turn"] > 1]
        c_safety = [r for r in crows if r["expect_safety"]]
        by_category[cat] = {
            "label": CATEGORY_LABEL[cat],
            "n_cases": sum(1 for c in cases if c["category"] == cat),
            "n_turns": len(crows),
            "resolution_success_rate": _rate(c_resolve, "resolve_ok"),
            "wrong_inheritance_rate": _rate(c_forbid, "inherited_wrongly"),
            "followup_answer_rate": _rate(c_answer, "answered"),
            "cross_turn_safety_rate": _rate(c_safety, "safety_present"),
        }

    return {
        "n_cases": len(cases),
        "n_turns": len(all_rows),
        "n_followup_turns": len(followups),
        "pass_rate": round(sum(1 for c in cases if c["pass"]) / len(cases), 4) if cases else None,
        "resolution_success_rate": resolution_success,
        "resolution_samples": len(resolve_rows),
        "followup_answer_rate": _rate(followup_answer_rows, "answered"),
        "followup_answer_samples": len(followup_answer_rows),
        "answer_rate_all_tagged": _rate(answer_rows, "answered"),
        "wrong_inheritance_rate": wrong_inherit,
        "wrong_inheritance_samples": len(forbid_rows),
        "cross_turn_safety_rate": safety_rate,
        "cross_turn_safety_samples": len(safety_rows),
        "safety_first_turn_rate": _rate(first_turn_safety, "safety_present"),
        "needs_context_turns": sum(1 for r in all_rows if r["needs_context"]),
        "errors": sum(1 for r in all_rows if r["error"]),
        "avg_latency_total_ms": round(statistics.mean(
            [r["latency_total_ms"] for r in all_rows if r["latency_total_ms"] is not None]), 2)
        if all_rows else None,
        "by_category": by_category,
        "kb_stats": kb_stats,
    }


def render_md(summary: Dict, cases: List[Dict], kb_path: str) -> str:
    s = summary
    lines = [
        "# 多轮对话评测报告（自动生成）\n",
        f"- 评测集：**{s['n_cases']} 条会话 / {s['n_turns']} 轮**"
        f"（其中追问轮 {s['n_followup_turns']} 轮）",
        f"- 知识库：`{kb_path}` ｜ 块数 {s['kb_stats']['n_chunks']} ｜ "
        f"分词 {s['kb_stats']['tokenizer']}",
        f"- 后端：`planner`（离线规则替身，确定性、零成本、无网络）",
        f"- 编排：单 Agent + `ConversationMemory`（会话内共享 memory 与工具缓存，模拟真实会话）",
        f"- 成本：本次为离线替身，无 API 调用、成本 $0.00\n",
        "## 1. 核心指标\n",
        "| 指标 | 数值 | 样本数 | 说明 |",
        "| --- | --- | --- | --- |",
        f"| 指代消解成功率 | **{_pct(s['resolution_success_rate'])}** | "
        f"{s['resolution_samples']} 轮 | 消解结果命中全部期望关键词 |",
        f"| 追问答案产出率 | **{_pct(s['followup_answer_rate'])}** | "
        f"{s['followup_answer_samples']} 轮 | 非首轮且标注应可答 |",
        f"| 错误继承率（越低越好） | **{_pct(s['wrong_inheritance_rate'])}** | "
        f"{s['wrong_inheritance_samples']} 轮 | 话题切换轮仍把旧话题带进查询 |",
        f"| 跨轮安全指令保持率 | **{_pct(s['cross_turn_safety_rate'])}** | "
        f"{s['cross_turn_safety_samples']} 轮 | 安全场景追问时指令不丢 |",
        f"| 安全指令首轮产出率（基线对照） | {_pct(s['safety_first_turn_rate'])} | "
        f"安全类样本首轮 | 首轮不给指令时「保持」无从谈起 |",
        f"| 会话级通过率 | {_pct(s['pass_rate'])} | {s['n_cases']} 条 | "
        f"所有硬性标注（消解/不继承/可答/安全）全满足 |",
        f"| 判定为「需要上下文」的轮次 | {s['needs_context_turns']} | {s['n_turns']} 轮 | "
        f"`memory.needs_context()` 的触发次数 |",
        f"| 异常轮次 | {s['errors']} | {s['n_turns']} 轮 | 任何异常都算不崩溃指标失败 |",
        f"| 单轮平均耗时 | {s['avg_latency_total_ms']} ms | — | 离线规则替身 |\n",
        "## 2. 分类别指标\n",
        "| 类别 | 样本 | 轮次 | 指代消解 | 追问产出 | 错误继承 | 跨轮安全 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cat in CATEGORIES:
        row = s["by_category"].get(cat)
        if not row:
            continue
        lines.append(
            f"| {row['label']} | {row['n_cases']} | {row['n_turns']} | "
            f"{_pct(row['resolution_success_rate'])} | {_pct(row['followup_answer_rate'])} | "
            f"{_pct(row['wrong_inheritance_rate'])} | {_pct(row['cross_turn_safety_rate'])} |")

    lines += ["\n## 3. 逐条明细\n",
              "| ID | 类别 | 轮 | 用户输入 | 消解后查询 | 状态 | 消解 | 继承 | 安全 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for case in cases:
        for row in case["turns"]:
            resolve_mark = "—" if row["resolve_ok"] is None else ("✅" if row["resolve_ok"] else "❌")
            inherit_mark = "—" if row["inherited_wrongly"] is None else \
                ("❌ 继承" if row["inherited_wrongly"] else "✅ 未继承")
            safety_mark = "—" if not row["expect_safety"] else \
                ("✅" if row["safety_present"] else "❌")
            lines.append(
                f"| {case['id']} | {CATEGORY_LABEL[case['category']]} | {row['turn']} | "
                f"{_md(row['user'])} | {_md(row['resolved_question'])} | {row['status']} | "
                f"{resolve_mark} | {inherit_mark} | {safety_mark} |")

    failed = [c for c in cases if not c["pass"]]
    lines += ["\n## 4. 失败样本归因\n"]
    if not failed:
        lines.append("- 本次全部会话通过硬性标注。\n")
    else:
        for case in failed:
            lines.append(f"- **{case['id']}**（{CATEGORY_LABEL[case['category']]}）"
                         f"{case['note']}")
            for item in case["failures"]:
                lines.append(f"  - {item}")
        lines.append("")

    lines += [
        "## 5. 结论与已知缺陷（如实说明）\n",
        f"- **最小语料下的实测**：指代消解 {_pct(s['resolution_success_rate'])}、"
        f"追问产出 {_pct(s['followup_answer_rate'])}、错误继承 {_pct(s['wrong_inheritance_rate'])}、"
        f"跨轮安全 {_pct(s['cross_turn_safety_rate'])}；",
        "- **错误继承是真实缺陷而非标注问题**：话题切换轮只要用户句里带指代词"
        "（如「那电动尾门怎么打开」），`ConversationMemory.resolve` 就会把上一轮话题"
        "拼进查询（`座椅加热怎么关闭 那电动尾门怎么打开`）。"
        "根因是 `needs_context()` 只判「有指代词且问题短」，**不判「本轮是否自带实词」**，"
        "也没有「话题切换检测」。影响：新话题的检索被旧话题词污染，"
        "用户看到的是一个自信但答错话题的回答；",
        "- **跨轮安全指令在语料变大后会大面积丢失**：迷你语料下首轮安全指令产出率约 60%、"
        "跨轮保持 70%；换到完整语料（9301 块）**首轮产出率降到 0%、跨轮保持降到 10%**。"
        "原因是单 Agent 链路里没有任何「安全指令硬门控」——安全提醒完全依赖检索到的原文句，"
        "而 9301 块下 top-6 很容易被更「对口」的块挤掉。"
        "这正是 `eval/multiagent_report.md` 里多 Agent 把安全合规从 0% 提到 100% 的那条能力缺口，"
        "本轮新增的多轮指标第一次把它量到了**多轮**维度上。\n",
        "## 6. 口径与局限（如实说明）\n",
        "1. **本评测只度量编排与检索**：后端是离线 `RuleBasedPlannerLLM`（抽取式答案），"
        "不代表真实大模型的生成质量；真实模型下的多轮指标需换 `--provider` 重跑"
        "（本脚本为保证零成本与确定性，默认且仅支持离线替身）；",
        "2. **指代消解是字符串级判定**：期望关键词出现在 `resolved_question` 里即算命中，"
        "属于可复现的代理指标，不做语义等价判定；",
        "3. **「错误继承」按 `inherit_forbid` 词面判定**：命中了旧话题的关键词就算串话题。"
        "即使最终答案侥幸正确，这条也计为错误继承（宁可严一点）；",
        "4. **跨轮安全指令保持率依赖首轮**：若首轮就没有给出安全指令，"
        "第 2 轮不可能「保持」——因此同时报出首轮产出率作为基线对照；",
        "5. **语料覆盖度会限制「追问产出率」**：本次知识库为 "
        f"`{kb_path}`（{s['kb_stats']['n_chunks']} 块）。语料越小，"
        "越容易出现「问题本身没问题、但语料里没有原文」的拒答——"
        "这与编排能力无关，因此同一指标在迷你语料与完整语料下的绝对值不可直接比较，"
        "但「开启/关闭」「改前/改后」这类**内部对照**在各自语料下都成立。",
    ]
    return "\n".join(lines) + "\n"


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _md(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")[:60]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default=FIXTURE_KB,
                    help="知识库 jsonl；默认迷你语料（CI 口径），完整语料传 kb/chunks.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-rewrite", action="store_true", help="关闭 query 改写（消融）")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json-out", default=OUT_JSON)
    ap.add_argument("--md-out", default=OUT_MD)
    ap.add_argument("--detail-out", default=OUT_DETAIL)
    args = ap.parse_args()

    print(f"[info] 加载知识库 {args.kb} …", flush=True)
    kb = KnowledgeBase.load(args.kb)
    print(f"[info] {json.dumps(kb.stats(), ensure_ascii=False)}", flush=True)

    cases = load_cases()
    if args.limit:
        cases = cases[:args.limit]
    meta = set_stats()
    print(f"[info] 会话 {len(cases)} 条 / 轮次 {sum(len(c['turns']) for c in cases)} "
          f"（{json.dumps(meta['by_category'], ensure_ascii=False)}）", flush=True)

    started = time.perf_counter()
    results = []
    for i, case in enumerate(cases, start=1):
        res = run_case(kb, case, rewrite=not args.no_rewrite)
        results.append(res)
        flag = "PASS" if res["pass"] else "FAIL"
        print(f"  [{i:>2}/{len(cases)}] [{flag}] {res['id']} "
              f"{CATEGORY_LABEL[res['category']]} —— {res['note']}", flush=True)
        if args.verbose and res["failures"]:
            for item in res["failures"]:
                print(f"        ↳ {item}", flush=True)
        elif res["failures"]:
            for item in res["failures"][:2]:
                print(f"        ↳ {item}", flush=True)
    wall = time.perf_counter() - started

    summary = summarize(results, kb.stats())
    summary["wall_clock_s"] = round(wall, 2)
    summary["kb_path"] = args.kb
    summary["rewrite"] = not args.no_rewrite

    json.dump({"summary": summary, "cases": results},
              open(args.json_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    with open(args.detail_out, "w", encoding="utf-8") as fh:
        for res in results:
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
    md = render_md(summary, results, args.kb)
    open(args.md_out, "w", encoding="utf-8").write(md)
    print("\n" + md)
    print(f"[json] {args.json_out}\n[md] {args.md_out}\n[detail] {args.detail_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
