# -*- coding: utf-8 -*-
"""answer_runner：答案层指标——记忆上下文 + LLM 回答 + 题型化 judge（需要真实模型）。

与检索层（runner.py）的分工：检索层测"记忆有没有记对/召回对"（零成本），
本层测"模型能不能用记忆答对"（含 ABS 拒答题）。两层都过，才能把失败归因到
"记忆"还是"生成"。

judge 启用前置（调研口径）：meta-eval 与人工标注一致率 ≥90%（30 题/题型）。
`--meta-out` 导出标注文件，人工填 `human_label` 后用 `judge.meta_eval_report`
算一致率——达标前，本层结果只能当"方向性数据"。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.llm_backend import add_llm_args, backend_fields, cost_line, make_llm  # noqa: E402
from eval.memory_backends import LocalEmbedder, make_backend  # noqa: E402
from eval.memory_bench.judge import judge  # noqa: E402
from eval.memory_bench.runner import ingest  # noqa: E402
from eval.memory_bench.schema import Sample, load_jsonl  # noqa: E402

DEFAULT_JSONL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_v0.1.jsonl")

ANSWER_PROMPT = """你是座舱车主助手。只根据【跨会话记忆】回答车主的问题。
规则：
1. 记忆里有的事实，直接、简洁地回答（中文）；
2. 记忆里**没有**的，必须回答「抱歉，我没有相关记录」，严禁编造；
3. 问"改没改/先后顺序"时，按记忆里的历史如实说明。
【跨会话记忆】
{context}

车主问题：{question}"""


def answer(llm: Any, context: str, question: str) -> str:
    resp = llm.chat([{"role": "user", "content": ANSWER_PROMPT.format(
        context=context or "（无记忆）", question=question)}],
        temperature=0.0, max_tokens=200)
    return (resp.content or "").strip()


def run_samples(samples: List[Sample], llm: Any, embedder: Any = None) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            path = tmp.name
        backend = make_backend("v2", path, embedder)
        try:
            ingest(backend, sample)
            for q in sample.questions:
                ctx = backend.context_text(q.text)
                ans = answer(llm, ctx, q.text)
                verdict = judge(llm, q.qtype, q.text, q.gold, ans, extra_note=q.note)
                rows.append({"sid": sample.sid, "qid": q.qid, "qtype": q.qtype,
                             "question": q.text, "gold": q.gold, "answer": ans[:300],
                             "verdict": verdict["verdict"], "reason": verdict["raw"][:200]})
        finally:
            backend.close()
            try:
                os.unlink(path)
            except OSError:
                pass
    by_type: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        t = by_type.setdefault(r["qtype"], {"n": 0, "yes": 0})
        t["n"] += 1
        t["yes"] += 1 if r["verdict"] == "yes" else 0
    for t, v in by_type.items():
        v["rate"] = round(v["yes"] / max(1, v["n"]), 4)
    total_n = len(rows)
    total_yes = sum(1 for r in rows if r["verdict"] == "yes")
    return {"samples": len(samples), "by_type": by_type,
            "overall": {"n": total_n, "yes": total_yes,
                        "rate": round(total_yes / max(1, total_n), 4)},
            "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description="答案层评测（记忆上下文 + LLM 回答 + judge）")
    add_llm_args(ap)
    ap.add_argument("--jsonl", default=DEFAULT_JSONL)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个车主（省钱验链路）")
    ap.add_argument("--json-out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "answer_report.json"))
    ap.add_argument("--md-out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "answer_report.md"))
    ap.add_argument("--meta-out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "meta_eval_annotation.jsonl"),
                    help="meta-eval 人工标注文件（human_label 留空待填）")
    args = ap.parse_args()

    if getattr(args, "provider", "planner") == "planner":
        print("[!] 答案层需要真实模型：--provider deepseek")
        return 1
    llm = make_llm(args)
    samples = load_jsonl(args.jsonl)
    if args.limit:
        samples = samples[:args.limit]
    print(f"[info] 跑 {len(samples)} 个车主 × 6 题（答案 + judge 各一次模型调用）")

    res = run_samples(samples, llm, embedder=LocalEmbedder())
    fields = backend_fields(args, llm)
    print(f"[结果] 总准确率 {res['overall']['rate']:.1%} ｜ 题型 "
          + " ".join(f"{k}={v['rate']:.0%}" for k, v in res["by_type"].items()))
    print(cost_line(fields))

    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump({**res, "backend": fields["backend"], "cost_usd": fields["cost_usd"]},
                  fh, ensure_ascii=False, indent=2)
    with open(args.meta_out, "w", encoding="utf-8") as fh:
        for r in res["rows"]:
            fh.write(json.dumps({"qid": r["qid"], "qtype": r["qtype"],
                                 "question": r["question"], "gold": r["gold"],
                                 "answer": r["answer"], "judge_verdict": r["verdict"],
                                 "human_label": ""}, ensure_ascii=False) + "\n")
    print(f"[json] {args.json_out}\n[meta] {args.meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
