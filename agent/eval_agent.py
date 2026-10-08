# -*- coding: utf-8 -*-
"""Agent 侧评测：在 103 题单轮测试集上度量编排与检索行为。

度量口径（都能在无 GPU 环境下真实跑出来）
----------------------------------------
1. **工具选择**：路由分布、平均工具调用数、重复调用抑制命中数；
2. **证据覆盖（context recall）**：Agent 交给答案环节的证据对 gold 关键词的覆盖率，
   与基线「BM25 召回上下文 0.8314 / 双路+精排上下文 0.9059」直接可比；
3. **编排治理**：平均步数、步数上限命中率、无进展终止率（防死循环）；
4. **拒答**：2 条负样本的拒答准确率 + 101 条可答题的误杀率，并扫描门控阈值找最佳工作点；
5. **开销**：单题平均总耗时与分段耗时（LLM 决策 / 工具 / 反思），即编排层额外开销。

用法：
    python -m agent.eval_agent                 # 全量 103 题（离线规则替身，默认）
    python -m agent.eval_agent --limit 20      # 快速验证
    python -m agent.eval_agent --no-rewrite    # 关闭查询改写做消融
    python -m agent.eval_agent --provider deepseek   # 换成真实云端模型
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.llm_backend import (add_llm_args, backend_fields, cost_line,  # noqa: E402
                              llm_id, llm_label, make_llm)

from agent.graph import AgentConfig, AgentGraph              # noqa: E402
from agent.kb import KnowledgeBase                           # noqa: E402
from agent.llm import redact                                 # noqa: E402
from agent.memory import ConversationMemory, VehicleProfile   # noqa: E402
from agent.reflection import NO_ANSWER, Reflector            # noqa: E402
from agent.tools import ToolRegistry                         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(ROOT, "data", "gold.json")
BASELINE = os.path.join(ROOT, "eval", "metrics.json")


def context_keyword_recall(evidence: List[Dict], keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    text = "\n".join(e.get("text", "") for e in evidence)
    return sum(1 for w in keywords if w in text) / len(keywords)


def load_cases(limit: int = 0):
    gold = json.load(open(GOLD, encoding="utf-8"))
    cases = []
    for i, g in enumerate(gold):
        answer = (g.get("answer") or "").strip()
        cases.append({
            "idx": i,
            "question": g.get("question", ""),
            "keywords": g.get("keywords", []) or [],
            "is_negative": answer == NO_ANSWER,
        })
    return cases[:limit] if limit else cases


def make_agent(kb, threshold: float, rewrite: bool, llm) -> AgentGraph:
    registry = ToolRegistry(kb)
    planner = llm
    # 规则替身用 overlap_threshold / rewrite_enabled 控制决策；真实模型没有这两个旋钮
    if hasattr(planner, "overlap_threshold"):
        planner.overlap_threshold = threshold
    if hasattr(planner, "rewrite_enabled"):
        planner.rewrite_enabled = rewrite
    reflector = Reflector(evidence_overlap_threshold=threshold)
    config = AgentConfig(max_steps=6, evidence_gate=threshold > 0,
                         evidence_overlap_threshold=threshold)
    return AgentGraph(planner, registry, reflector=reflector,
                      memory=ConversationMemory(window=6,
                                                profile=VehicleProfile(model="领克08",
                                                                       mileage_km=23860)),
                      config=config)


def run_suite(kb, cases, threshold: float, rewrite: bool = True, llm=None) -> Dict:
    agent = make_agent(kb, threshold, rewrite, llm)
    rows = []
    started = time.perf_counter()
    for case in cases:
        agent.registry.reset_cache()
        agent.memory = ConversationMemory(window=6,
                                          profile=VehicleProfile(model="领克08",
                                                                 mileage_km=23860))
        state = agent.run(case["question"])
        recall = context_keyword_recall(state.evidence, case["keywords"]) \
            if not case["is_negative"] else None
        refused = (state.answer or "").strip() == NO_ANSWER
        rows.append({
            "idx": case["idx"], "question": case["question"], "negative": case["is_negative"],
            "status": state.status, "route": state.route, "steps": state.steps,
            "retries": state.retries, "refused": refused,
            "n_tools": len(state.tool_calls),
            "tools": [t["tool"] for t in state.tool_calls],
            "repeated": sum(1 for t in state.tool_calls if t.get("repeated")),
            "n_evidence": len(state.evidence),
            "context_recall": recall,
            "gate_coverage": state.gate_coverage,
            "latency_total": state.latency_ms.get("total", 0.0),
            "latency_llm": state.latency_ms.get("llm", 0.0),
            "latency_tools": state.latency_ms.get("tools", 0.0),
            "latency_reflection": state.latency_ms.get("reflection", 0.0),
        })
    elapsed = time.perf_counter() - started

    negatives = [r for r in rows if r["negative"]]
    positives = [r for r in rows if not r["negative"]]
    recalls = [r["context_recall"] for r in positives if r["context_recall"] is not None]

    def m(values):
        return round(statistics.mean(values), 4) if values else None

    summary = {
        "n": len(rows), "n_negative": len(negatives), "n_positive": len(positives),
        "threshold": threshold, "rewrite": rewrite,
        "refusal_accuracy": f"{sum(1 for r in negatives if r['refused'])}/{len(negatives)}"
        if negatives else "n/a",
        "false_refusal": f"{sum(1 for r in positives if r['refused'])}/{len(positives)}",
        "context_recall": m(recalls),
        "context_recall_full": m([1.0 if r["context_recall"] >= 1.0 else 0.0 for r in positives
                                  if r["context_recall"] is not None]),
        "avg_steps": m([r["steps"] for r in rows]),
        "avg_tools": m([r["n_tools"] for r in rows]),
        "repeated_calls": sum(r["repeated"] for r in rows),
        "max_steps_hits": sum(1 for r in rows if r["status"] == "max_steps"),
        "avg_latency_total_ms": m([r["latency_total"] for r in rows]),
        "avg_latency_llm_ms": m([r["latency_llm"] for r in rows]),
        "avg_latency_tools_ms": m([r["latency_tools"] for r in rows]),
        "avg_latency_reflection_ms": m([r["latency_reflection"] for r in rows]),
        "route_distribution": {k: sum(1 for r in rows if r["route"] == k)
                               for k in sorted({r["route"] for r in rows})},
        "tool_distribution": {},
        "wall_clock_s": round(elapsed, 2),
        "questions_per_second": round(len(rows) / elapsed, 2) if elapsed else None,
    }
    tool_dist: Dict[str, int] = {}
    for r in rows:
        for t in r["tools"]:
            tool_dist[t] = tool_dist.get(t, 0) + 1
    summary["tool_distribution"] = tool_dist
    return {"summary": summary, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    add_llm_args(ap)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--sweep", default="0.30,0.40,0.45,0.50,0.55,0.60",
                    help="门控阈值扫描列表（逗号分隔）；传空字符串则只跑 --threshold")
    ap.add_argument("--no-rewrite", action="store_true", help="关闭查询改写（消融）")
    ap.add_argument("--json-out", default=os.path.join(ROOT, "eval", "agent_metrics.json"))
    ap.add_argument("--md-out", default=os.path.join(ROOT, "eval", "agent_report.md"))
    ap.add_argument("--detail-out", default=os.path.join(ROOT, "eval", "agent_details.jsonl"))
    args = ap.parse_args()

    print("[info] 加载知识库…")
    kb = KnowledgeBase.load()
    print(f"[info] 知识库: {json.dumps(kb.stats(), ensure_ascii=False)}")

    try:
        # 整个评测（阈值扫描 + 消融）复用同一个 LLM 实例：成本账本才能累计成总花费
        llm = make_llm(args)
    except RuntimeError as exc:
        print(f"[错误] {redact(exc)}")
        return 2
    print(f"[info] LLM 后端: {llm_id(args)} —— {llm_label(args)}")

    cases = load_cases(args.limit)
    print(f"[info] 测试集 {len(cases)} 题（负样本 {sum(1 for c in cases if c['is_negative'])}）")

    thresholds = [float(x) for x in args.sweep.split(",") if x.strip()] if args.sweep else [args.threshold]
    sweep = []
    main_result = None
    for t in thresholds:
        res = run_suite(kb, cases, t, rewrite=not args.no_rewrite, llm=llm)
        s = res["summary"]
        sweep.append({k: s[k] for k in ("threshold", "refusal_accuracy", "false_refusal",
                                        "context_recall", "avg_steps", "avg_tools")})
        print(f"  threshold={t:<5} 拒答={s['refusal_accuracy']:<5} 误杀={s['false_refusal']:<8} "
              f"证据覆盖={s['context_recall']}  步数={s['avg_steps']}  工具={s['avg_tools']}")
        if main_result is None or t == args.threshold:
            main_result = res

    # 查询改写消融
    rewrite_ablation = None
    if not args.no_rewrite:
        res_no = run_suite(kb, cases, args.threshold, rewrite=False, llm=llm)
        rewrite_ablation = {
            "with_rewrite": main_result["summary"]["context_recall"],
            "without_rewrite": res_no["summary"]["context_recall"],
            "with_rewrite_false_refusal": main_result["summary"]["false_refusal"],
            "without_rewrite_false_refusal": res_no["summary"]["false_refusal"],
        }

    baseline = None
    if os.path.exists(BASELINE):
        b = json.load(open(BASELINE, encoding="utf-8"))
        cs = b.get("context_stats", {})

        def rnd(v):
            return round(v, 4) if isinstance(v, (int, float)) else v

        baseline = {
            "pipeline_rerank_context_recall": rnd(cs.get("answer_7", {}).get("avg_keyword_recall")),
            "pipeline_bm25_context_recall": rnd(cs.get("answer_6", {}).get("avg_keyword_recall")),
            "pipeline_answer_scores": {k: rnd(v.get("score"))
                                       for k, v in b.get("variants", {}).items()},
        }

    out = {"summary": main_result["summary"], "sweep": sweep,
           "rewrite_ablation": rewrite_ablation, "baseline_pipeline": baseline,
           "kb_stats": kb.stats()}
    # 本次使用的后端标识与成本账本（离线替身为 planner / $0）
    out["summary"].update(backend_fields(args, llm))
    json.dump(out, open(args.json_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    with open(args.detail_out, "w", encoding="utf-8") as f:
        for row in main_result["rows"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    s = main_result["summary"]
    lines = ["# Agent 侧评测报告（自动生成）\n",
             f"- 知识库：{s['n']} 题测试 ｜ 块数 {kb.stats()['n_chunks']} ｜ "
             f"分词后端 {kb.stats()['tokenizer']} ｜ 向量路 {'开启' if kb.stats()['has_vector'] else '未开启'}",
             f"- 编排：原生状态机（Router→Planner→Tool→Reflect→Answer），max_steps=6",
             f"- 模型后端：`{llm_id(args)}` —— {llm_label(args)}",
             cost_line(s) + "\n",
             "## 1. 总体指标\n",
             "| 指标 | 数值 | 对照 |", "| --- | --- | --- |",
             f"| 拒答准确率（负样本） | {s['refusal_accuracy']} | 原链路 0/2（信号未接入答案侧） |",
             f"| 可答题误杀率 | {s['false_refusal']} | — |",
             f"| 证据关键词覆盖率（context recall） | {s['context_recall']} | "
             f"基线双路+精排 {baseline['pipeline_rerank_context_recall'] if baseline else 'n/a'} / "
             f"基线 BM25 {baseline['pipeline_bm25_context_recall'] if baseline else 'n/a'} |",
             f"| 平均交互步数 | {s['avg_steps']} | — |",
             f"| 平均工具调用数 | {s['avg_tools']} | — |",
             f"| 重复调用被抑制 | {s['repeated_calls']} 次 | — |",
             f"| 步数上限命中 | {s['max_steps_hits']} 题 | 0 表示没有死循环 |",
             f"| 单题平均耗时 | {s['avg_latency_total_ms']} ms | "
             f"其中 LLM 决策 {s['avg_latency_llm_ms']} / 工具 {s['avg_latency_tools_ms']} / "
             f"反思 {s['avg_latency_reflection_ms']} |",
             f"| 吞吐 | {s['questions_per_second']} 题/秒 | 无 GPU 离线规则规划器 |\n",
             "## 2. 门控阈值扫描\n",
             "| 阈值 | 拒答准确率 | 误杀率 | 证据覆盖率 | 平均步数 |",
             "| --- | --- | --- | --- | --- |"]
    for row in sweep:
        lines.append(f"| {row['threshold']} | {row['refusal_accuracy']} | {row['false_refusal']} | "
                     f"{row['context_recall']} | {row['avg_steps']} |")
    if rewrite_ablation:
        lines += ["\n## 3. 查询改写消融（口语 → 手册术语）\n",
                  "| 配置 | 证据关键词覆盖率 | 可答题误杀率 |", "| --- | --- | --- |",
                  f"| 开启改写 | {rewrite_ablation['with_rewrite']} | "
                  f"{rewrite_ablation['with_rewrite_false_refusal']} |",
                  f"| 关闭改写 | {rewrite_ablation['without_rewrite']} | "
                  f"{rewrite_ablation['without_rewrite_false_refusal']} |"]
    lines += ["\n## 4. 路由与工具分布\n",
              f"- 路由分布：{json.dumps(s['route_distribution'], ensure_ascii=False)}",
              f"- 工具分布：{json.dumps(s['tool_distribution'], ensure_ascii=False)}"]
    md = "\n".join(lines)
    open(args.md_out, "w", encoding="utf-8").write(md + "\n")
    print("\n" + md)
    print(f"\n[json] {args.json_out}\n[md] {args.md_out}\n[detail] {args.detail_out}")


if __name__ == "__main__":
    main()
