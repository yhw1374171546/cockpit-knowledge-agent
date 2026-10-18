# -*- coding: utf-8 -*-
"""可回答性评测：度量"证据没回答问题却照样作答"的假阳性率。

    python eval/answerability_eval.py            # 全量
    python eval/answerability_eval.py -v         # 逐题明细
    python eval/answerability_eval.py --json results/answerability.json

为什么这个指标重要
------------------
「车内有冰箱吗」这类问题，检索没错、接地没错（答案就是原文），
但**没回答问题**。它不会被"防幻觉"的任何手段抓到，也不会被拒答率覆盖。
在加可回答性校验之前，这批题几乎是 **100% 假阳性**——
用户问"有没有冰箱"，系统回一段"如何给冰箱充电"。

判定口径
--------
| expect | 期望 status | 说明 |
| --- | --- | --- |
| limited | limited | 应给"手册未说明 + 相关原文"的限定回答 |
| answered | answered | 手册有明确说明，正常作答 |
| refused | refused | 手册没有，应拒答 |

- **假阳性率 FP**：should_not_answer（limited+refused）里被判成 answered 的比例
  —— 这是本集的**主指标**，越低越好；
- **正确限定率**：limited 期望里确实变成 limited 的比例；
- **过度限定率**：answered 期望里被限定的比例（把该答的也限定了，同样算错）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.console import harden_stdout  # noqa: E402

harden_stdout()

from eval.existence_set import EXISTENCE_CASES, KB_FULL, is_runnable  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--no-answerability", action="store_true",
                    help="关掉可回答性校验，用来看『改造前 vs 改造后』的对比")
    args = ap.parse_args()

    if not is_runnable():
        print(f"[SKIP] 缺少完整语料 {KB_FULL}（未入库），本集跳过。")
        return 0

    from agent.graph import AgentConfig, AgentGraph
    from agent.kb import KnowledgeBase
    from agent.llm import RuleBasedPlannerLLM
    from agent.tools import ToolRegistry

    kb = KnowledgeBase.load(KB_FULL)
    config = AgentConfig()
    config.enable_answerability = not args.no_answerability
    agent = AgentGraph(RuleBasedPlannerLLM(), ToolRegistry(kb), config=config)

    print("=" * 78)
    print(f"可回答性评测 · {len(kb.chunks)} 块 · 可回答性校验 "
          f"{'关闭（对照组）' if args.no_answerability else '开启'}")
    print("=" * 78)

    rows = []
    fp = fp_total = 0
    qual = qual_total = 0
    over = over_total = 0

    for question, expect, note in EXISTENCE_CASES:
        state = agent.run(question)
        status = state.status
        verdict = (state.answerability or {}).get("verdict", "-")
        ok = status == expect

        if expect in ("limited", "refused"):
            fp_total += 1
            if status == "answered":
                fp += 1
        if expect == "limited":
            qual_total += 1
            if status == "limited":
                qual += 1
        if expect == "answered":
            over_total += 1
            if status == "limited":
                over += 1

        rows.append({"question": question, "expect": expect, "status": status,
                     "answerability": verdict, "ok": ok, "note": note,
                     "answer": (state.answer or "")[:120]})

        if args.verbose or not ok:
            flag = "[OK]" if ok else "[FAIL]"
            print(f"  {flag} {question:<22} 期望={expect:<8} 实际={status:<16} 判据={verdict}")
            if not ok:
                print(f"        {note}")
                print(f"        答案: {rows[-1]['answer']}")

    total = len(rows)
    passed = sum(1 for r in rows if r["ok"])
    fp_rate = fp / fp_total if fp_total else 0.0
    qual_rate = qual / qual_total if qual_total else 0.0
    over_rate = over / over_total if over_total else 0.0

    print("-" * 78)
    print(f"  逐题通过          {passed}/{total}  ({passed / total:.1%})")
    print(f"  假阳性率 FP       {fp}/{fp_total}  ({fp_rate:.1%})   ← 主指标，越低越好")
    print(f"  正确限定率        {qual}/{qual_total}  ({qual_rate:.1%})")
    print(f"  过度限定率        {over}/{over_total}  ({over_rate:.1%})   ← 把该答的限定了也算错")
    print("=" * 78)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"total": total, "passed": passed,
                       "false_positive_rate": round(fp_rate, 4),
                       "qualified_rate": round(qual_rate, 4),
                       "over_limit_rate": round(over_rate, 4),
                       "answerability_enabled": not args.no_answerability,
                       "rows": rows}, fh, ensure_ascii=False, indent=2)
        print(f"  结果已写入 {args.json}")

    # 阈值：假阳性率必须为 0，且逐题通过率 >= 85%
    if fp_rate > 0.0 or passed / total < 0.85:
        print("  结论：未达标（要求 FP=0 且通过率>=85%）")
        return 1
    print("  结论：达标")
    return 0


if __name__ == "__main__":
    sys.exit(main())
