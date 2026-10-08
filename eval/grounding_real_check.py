# -*- coding: utf-8 -*-
"""A3 的决定性验证：真实模型下被「误杀」的问题，换了语义接地后还被拒答吗？

背景：真实模型跑 103 题时，误杀从 1/101 涨到 **23/101**——
23 条都检到了 ≥6 条证据、门控覆盖率 ≥0.9，却因为 `Reflector` 的逐句**字符重叠**
接地校验对生成式答案过苛而被拒答。

本脚本只重跑**当时被误杀的那批题**（约 23 题，成本 ¥0.3 以内），对比拒答数：
- 修复前：23/23 被拒（由构造方式决定）
- 修复后：应当大幅下降；
- 同时**必须**回归检查：真正的域外问题仍然要拒答（否则就是把误杀换成了漏答）。

用法：
    python eval/grounding_real_check.py                  # 默认 deepseek
    python eval/grounding_real_check.py --limit 10       # 先小规模试
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import AgentConfig, AgentGraph                                   # noqa: E402
from agent.kb import KnowledgeBase                                                # noqa: E402
from agent.llm import build_llm, load_dotenv, redact                              # noqa: E402
from agent.memory import ConversationMemory, VehicleProfile                       # noqa: E402
from agent.reflection import Reflector                                            # noqa: E402
from agent.tools import build_default_registry                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETAILS = os.path.join(ROOT, "eval", "agent_details_deepseek.jsonl")
KB_PATH = os.path.join(ROOT, "kb", "chunks.jsonl")


def load_targets(limit: int = 0, include_negative: bool = True):
    """从真实模型明细里取出「有证据却被拒答」的题目（=误杀），以及域外题（用于回归）。"""
    false_refusals, out_of_domain = [], []
    with open(DETAILS, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            refused = bool(row.get("refused")) or row.get("status") == "refused"
            if not refused:
                continue
            item = {"idx": row.get("idx"), "question": row.get("question"),
                    "negative": bool(row.get("negative")),
                    "n_evidence": row.get("n_evidence"), "gate": row.get("gate_coverage")}
            if item["negative"]:
                out_of_domain.append(item)
            else:
                false_refusals.append(item)
    if limit:
        false_refusals = false_refusals[:limit]
    return false_refusals, (out_of_domain if include_negative else [])


def build_agent(llm, kb) -> AgentGraph:
    registry = build_default_registry(KB_PATH, kb=kb)
    if hasattr(llm, "set_known_tools"):
        llm.set_known_tools(registry.names())
    config = AgentConfig()
    return AgentGraph(llm, registry,
                      reflector=Reflector(evidence_overlap_threshold=0.40),
                      memory=ConversationMemory(profile=VehicleProfile(model="领克08")),
                      config=config)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（试跑用）")
    ap.add_argument("--model", default="", help="覆盖默认模型")
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--json-out", default=os.path.join(ROOT, "eval", "grounding_real_metrics.json"))
    ap.add_argument("--md-out", default=os.path.join(ROOT, "eval", "grounding_real_report.md"))
    args = ap.parse_args()

    load_dotenv(os.path.join(ROOT, ".env"))
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("[跳过] 未配置 DEEPSEEK_API_KEY（本脚本需要真实模型才能验证误杀）")
        return 0

    targets, negatives = load_targets(args.limit)
    if not targets:
        print("[跳过] 明细文件里没有「有证据却被拒答」的样本")
        return 0

    print("=" * 78)
    print(f"A3 验证：重跑真实模型下被误杀的 {len(targets)} 题（另含 {len(negatives)} 条域外回归）")
    print("=" * 78)

    kb = KnowledgeBase.load(KB_PATH)
    print(f"[setup] 知识库 {len(kb.chunks)} 块")

    llm = build_llm(provider=args.provider, model=args.model)
    print(f"[setup] 后端 {llm.provider}/{llm.model}，思考模式={llm.thinking or '默认'}")

    results = []
    t0 = time.time()
    for i, item in enumerate(targets, 1):
        agent = build_agent(llm, kb)
        try:
            state = agent.run(item["question"])
            status, answer = state.status, (state.answer or "")
        except Exception as exc:                                          # noqa: BLE001
            status, answer = "error", redact(exc)
        still_refused = status == "refused" or answer.strip() == "无答案"
        results.append({**item, "status": status, "still_refused": still_refused,
                        "answer_head": answer[:60]})
        flag = "仍拒答 ❌" if still_refused else "已放行 ✅"
        print(f"  [{i:>2}/{len(targets)}] {flag}  {item['question'][:36]}"
              f"  (证据 {item['n_evidence']} 条)")

    neg_results = []
    for item in negatives:
        agent = build_agent(llm, kb)
        try:
            state = agent.run(item["question"])
            status, answer = state.status, (state.answer or "")
        except Exception as exc:                                          # noqa: BLE001
            status, answer = "error", redact(exc)
        refused = status == "refused" or answer.strip() == "无答案"
        neg_results.append({**item, "status": status, "refused": refused})
        print(f"  [域外回归] {'仍正确拒答 ✅' if refused else '误答了 ❌'}  "
              f"{item['question'][:36]}")

    cost = llm.cost_report()
    before = len(targets)
    after = sum(1 for r in results if r["still_refused"])
    kept = sum(1 for r in neg_results if r["refused"])

    print("\n" + "-" * 78)
    print(f"误杀：{before}/{before} → **{after}/{before}**（下降 {before - after} 题）")
    if neg_results:
        print(f"域外回归：{kept}/{len(neg_results)} 仍正确拒答")
    print(f"成本：{cost['calls']} 次调用 / 输入 {cost['prompt_tokens']} / "
          f"输出 {cost['completion_tokens']} / ${cost['cost_usd']:.4f}")
    print(f"耗时：{time.time() - t0:.0f}s")

    _write_outputs(args, results, neg_results, cost, before, after)
    ok = after < before and (not neg_results or kept == len(neg_results))
    return 0 if ok else 1


def _write_outputs(args, results, neg_results, cost, before, after) -> None:
    payload = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "false_refusals_before": before, "false_refusals_after": after,
               "negatives_kept": sum(1 for r in neg_results if r["refused"]),
               "negatives_total": len(neg_results), "cost": cost,
               "false_refusal_rows": results, "negative_rows": neg_results}
    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    lines = [
        "# A3 真实模型验证：语义接地是否消掉误杀", "",
        f"- 时间：{payload['generated_at']}",
        "- 数据来源：`eval/agent_details_deepseek.jsonl`（真实模型跑 103 题时**被误杀**的那批）",
        "- 判定口径：重跑同一问题，`status == refused` 或答案为「无答案」即仍被误杀",
        "",
        "## 结果", "",
        f"| 指标 | 修复前 | 修复后 |",
        "| --- | --- | --- |",
        f"| 误杀题数 | **{before}/{before}** | **{after}/{before}** |",
        f"| 域外问题仍正确拒答 | — | "
        f"**{payload['negatives_kept']}/{payload['negatives_total']}** |",
        "",
        f"成本：{cost['calls']} 次调用 / 输入 {cost['prompt_tokens']} tokens / "
        f"输出 {cost['completion_tokens']} tokens / ${cost['cost_usd']:.4f}",
        "",
        "## 逐题结果", "",
        "| # | 问题 | 证据数 | 门控覆盖 | 修复后状态 | 是否仍拒答 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for i, row in enumerate(results, 1):
        lines.append(f"| {i} | {row['question'][:40]} | {row.get('n_evidence')} | "
                     f"{row.get('gate')} | {row['status']} | "
                     f"{'仍拒答' if row['still_refused'] else '已放行'} |")
    if neg_results:
        lines += ["", "## 域外回归（必须仍拒答，否则是拿误杀换漏答）", "",
                  "| 问题 | 状态 | 是否拒答 |", "| --- | --- | --- |"]
        for row in neg_results:
            lines.append(f"| {row['question'][:40]} | {row['status']} | "
                         f"{'✅' if row['refused'] else '❌'} |")
    lines += [
        "", "## 口径说明", "",
        "- 这里的「修复前 23/23」是由**构造方式**决定的：样本本身就取自当时被拒答的题目；",
        "  因此本表的正确读法是「这 23 条误杀中有多少被消掉」，而不是「整体误杀率」。",
        "- 整体误杀率需要重跑完整 103 题（约 ¥0.7）才能给出，本脚本为控制成本只重跑误杀子集。",
    ]
    with open(args.md_out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n已写入 {os.path.relpath(args.json_out, ROOT)} 与 {os.path.relpath(args.md_out, ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
