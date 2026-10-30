# -*- coding: utf-8 -*-
"""runner：检索层指标（零成本、可进 CI）——gold 事实召回率，按题型分层，v1 vs v2 对照。

与答案层（agent 回答 + judge）的分工：本层测"记忆系统有没有把该记的记对、该召回的召回"，
答案层测"模型能不能用这些记忆答对"。两层都做，才能把失败归因到"记忆"还是"生成"。

口径：
- IE/MR/KU：召回上下文（context_text）里出现 gold 值 → hit（归一化后子串匹配）；
- TR：`backend.history("usual_center")` 含全部演化值（v1 的 overwrite 丢历史 → 如实失败）；
- ABS：检索层不适用（拒答是答案层行为）；
- 红线注入（constraint）：对任意查询（"今天天气怎么样"）红线都必须出现在上下文——
  这是 v2 的"红线常注入"设计，v1 无此概念（如实失败）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.memory_backends import MemoryBackend, make_backend  # noqa: E402
from eval.memory_bench.profiles import build_cards  # noqa: E402  (profiles 在 generate 用)
from eval.memory_bench.schema import Sample, load_jsonl, stats  # noqa: E402

DEFAULT_JSONL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_v0.1.jsonl")


def _norm(text: str) -> str:
    """归一化：去空白 + 去掉动力形式后缀（v1 存"领克08"，gold 写"领克08 EM-P"）。"""
    return re.sub(r"\s+", "", (text or "")).replace("EM-P", "").replace("EM‑P", "")


def value_hit(gold: str, ctx: str) -> bool:
    return _norm(gold) in _norm(ctx)


def ingest(backend: MemoryBackend, sample: Sample) -> None:
    """把 7 轮对话喂进记忆系统（工单确认轮按服务层 on_event 的语义写情景记忆）。"""
    for turn in sample.turns:
        if turn.speaker == "user":
            backend.extract_and_remember(turn.text)
        else:
            m = re.search(r"单号 (\S+)，更换机油机滤，(\S+) 到 (\S+)，", turn.text)
            if m:
                backend.remember_episode(
                    f"已建工单 {m.group(1)}：更换机油机滤，{m.group(2)} 到 {m.group(3)}",
                    session_id=f"{sample.sid}:s2")


def evaluate_sample(backend: MemoryBackend, sample: Sample) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    center_golds = [f["value"] for f in sample.gold_facts if f["key"] == "usual_center"]
    for q in sample.questions:
        if q.qtype == "ABS":
            continue                                    # 检索层不适用
        if q.qtype == "TR":
            hist = backend.history("usual_center")
            hit = bool(center_golds) and set(_norm(v) for v in center_golds) <= {_norm(h) for h in hist}
            row = {"sid": sample.sid, "qid": q.qid, "qtype": q.qtype,
                   "question": q.text, "gold": q.gold, "hit": hit, "clean": hit}
        else:
            ctx = backend.context_text(q.text)
            hit = value_hit(q.gold, ctx)
            # KU 干净率：最新值在**画像区**里，且旧值不在画像区。
            # 注意：只查"相关历史"之前的画像区——历史工单在旧门店办理是合法历史信息，
            # 不该算脏（mem0 语义档没有分区，整段都是平铺事实 → 旧值混在里面即算脏）。
            clean = hit
            if q.qtype == "KU" and len(center_golds) >= 2:
                profile_ctx = ctx.split("相关历史", 1)[0]
                clean = hit and not value_hit(center_golds[0], profile_ctx)
            row = {"sid": sample.sid, "qid": q.qid, "qtype": q.qtype,
                   "question": q.text, "gold": q.gold, "hit": hit, "clean": clean}
        rows.append(row)
    return rows


def run_recall(samples: List[Sample], profile: str,
               embedder: Any = None, llm: Any = None) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    constraint_hits = 0
    # 红线注入检查：三个无关查询的上下文**都必须**含红线——v2 是设计保证（无条件注入），
    # mem0 语义档靠相似度运气（偶尔命中），v1 无此概念。
    NEUTRAL_QUERIES = ("今天天气怎么样", "讲个笑话", "你好")
    for sample in samples:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            path = tmp.name
        backend = make_backend(profile, path, embedder, llm=llm)
        try:
            ingest(backend, sample)
            rows.extend(evaluate_sample(backend, sample))
            redline = sample.profile_card.get("redline", "")
            if redline and all(
                    value_hit(redline, backend.context_text(q)) for q in NEUTRAL_QUERIES):
                constraint_hits += 1
        finally:
            backend.close()
            try:
                os.unlink(path)
            except OSError:
                pass

    by_type: Dict[str, Dict[str, Any]] = {}
    ku_rows: List[Dict[str, Any]] = []
    for r in rows:
        t = by_type.setdefault(r["qtype"], {"n": 0, "hit": 0})
        t["n"] += 1
        t["hit"] += 1 if r["hit"] else 0
        if r["qtype"] == "KU":
            ku_rows.append(r)
    for t, v in by_type.items():
        v["rate"] = round(v["hit"] / max(1, v["n"]), 4)
    ku_clean = round(sum(1 for r in ku_rows if r["clean"]) / max(1, len(ku_rows)), 4)
    total = {"n": len(rows), "hit": sum(1 for r in rows if r["hit"])}
    return {"profile": profile, "samples": len(samples),
            "by_type": by_type,
            "overall": {"n": total["n"], "hit": total["hit"],
                        "rate": round(total["hit"] / max(1, total["n"]), 4)},
            "ku_clean": {"n": len(ku_rows), "rate": ku_clean},
            "constraint_injection": {"n": len(samples), "hit": constraint_hits,
                                     "rate": round(constraint_hits / max(1, len(samples)), 4)},
            "rows": rows}


def render_md(results: Dict[str, Dict[str, Any]]) -> str:
    lines = ["# 记忆检索层 benchmark（自动生成）\n",
             "> 口径：gold 事实召回率（IE/MR/KU=上下文含 gold 值；TR=演化史完整；红线=无条件注入）。",
             "> 生成：`python eval/memory_bench/generate.py`（程序化、确定性、零成本）；",
             "> 重跑：`python eval/memory_bench/runner.py --profile both`。\n",
             "## 1. 总体\n",
             "| 后端 | 样本 | 题目 | 命中 | 召回率 | 红线注入 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for profile, res in results.items():
        lines.append(f"| {profile} | {res['samples']} | {res['overall']['n']} | "
                     f"{res['overall']['hit']} | **{res['overall']['rate']:.1%}** | "
                     f"{res['constraint_injection']['rate']:.1%} |")
    lines += ["\n## 2. 按题型\n",
              "| 后端 | IE | MR | KU | KU干净 | TR |",
              "| --- | --- | --- | --- | --- | --- |"]
    for profile, res in results.items():
        bt = res["by_type"]
        row = [f"{bt.get(t, {}).get('rate', 0):.1%}" for t in ("IE", "MR", "KU")]
        row.append(f"{res['ku_clean']['rate']:.1%}")
        row.append(f"{bt.get('TR', {}).get('rate', 0):.1%}")
        lines.append("| " + profile + " | " + " | ".join(row) + " |")
    lines += ["\n## 3. 口径说明\n",
              "- TR：v1 的 overwrite 不保留旧值，`history()` 如实返回空 → TR 必然失败；"
              "这是 v1/v2 的已知差异，不是评测 bug。",
              "- KU 干净率：最新值在上下文且**旧值不在**。ADD-only 后端（mem0 语义档）"
              "会把新旧值一起返回 → 干净率低，答案层会被旧值干扰。",
              "- mem0 语义档 = 按 Mem0 公开语义的本地复现（ADD-only + 无业务 schema +"
              " 无冲突消解 + threshold=0.1），**不是 Mem0 产品调用**——用来回答"
              "'如果开箱用会发生什么'。",
              "- ABS（拒答）与答案层指标见 `answer_runner.py` / `judge.py`，检索层不适用。"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="记忆检索层 benchmark（离线、CI 可跑）")
    ap.add_argument("--jsonl", default=DEFAULT_JSONL, help="评测集路径")
    ap.add_argument("--profile", choices=["v1", "v2", "mem0", "both"], default="both")
    ap.add_argument("--llm-extract", action="store_true",
                    help="v2 用 LLM 结构化抽取（v0.2 润色集必开；需要 --provider）")
    ap.add_argument("--provider", default="deepseek", help="--llm-extract 时生效")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个样本（快速验证）")
    ap.add_argument("--embedding", action="store_true",
                    help="v2 启用 ONNX 嵌入（默认词面兜底；嵌入需 407MB 模型）")
    ap.add_argument("--json-out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "recall_report.json"))
    ap.add_argument("--md-out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "recall_report.md"))
    args = ap.parse_args()

    samples = load_jsonl(args.jsonl)
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        print(f"[!] 评测集为空：{args.jsonl}（先跑 generate.py）")
        return 2
    print(f"[info] 评测集 {stats(samples)['samples']} 样本 / {stats(samples)['turns']} 轮")

    from eval.memory_backends import LocalEmbedder
    embedder = LocalEmbedder() if args.embedding else None
    llm = None
    if args.llm_extract:
        from eval.llm_backend import make_llm
        llm = make_llm(args)

    profiles = ["v1", "mem0", "v2"] if args.profile == "both" else [args.profile]
    results: Dict[str, Dict[str, Any]] = {}
    for profile in profiles:
        res = run_recall(samples, profile, embedder, llm=llm)
        results[profile] = res
        print(f"[{profile}] 总召回 {res['overall']['rate']:.1%} ｜ 题型 "
              + " ".join(f"{k}={v['rate']:.0%}" for k, v in res["by_type"].items())
              + f" ｜ 红线注入 {res['constraint_injection']['rate']:.1%}")

    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump({"samples": len(samples), "results": results}, fh,
                  ensure_ascii=False, indent=2, default=str)
    with open(args.md_out, "w", encoding="utf-8") as fh:
        fh.write(render_md(results))
    print(f"[json] {args.json_out}\n[md] {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
