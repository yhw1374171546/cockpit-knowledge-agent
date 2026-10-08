# -*- coding: utf-8 -*-
"""口语化检索评测：口语问法的命中率 / 覆盖率 + 「开启 vs 关闭 query 改写」对照。

为什么单独做这一套
----------------
现有 103 题全部是手册术语（"危险警告灯""座椅加热"），**用术语查术语必然命中**，
因此它既高估了线上真实体验，也让「query 改写有没有用」这个实验永远测不出增益。
本脚本用 46 条真实口语问法（`eval/colloquial_set.py`）回答两个问题：
1. 口语问法在现网检索里到底能不能命中对应知识块？
2. 打开 `RuleBasedPlannerLLM._normalize_query`（口语→术语映射）之后，
   命中率是会上升、不变，还是**下降**？

指标口径
-------
| 指标 | 定义 |
| --- | --- |
| `hit_rate` | top_k 命中块里出现至少一个 `expect_keywords` 锚点的比例（主指标） |
| `hit_at_1_rate` | 同一判定，但只看 top-1 |
| `coverage` | 口语问法的实词字符被命中块文本覆盖的平均比例（连续值，抗单点波动） |
| `rewrite_applied_rate` | 现有改写表**真正改动了 query** 的比例（解释增益来源） |
| `regression_rate` | 关闭改写能命中、开启改写反而命不中的比例（改写的风险面） |

用法：
    python eval/colloquial_eval.py                                  # 迷你语料（CI 口径）
    python eval/colloquial_eval.py --kb kb/chunks.jsonl             # 完整语料
    python eval/colloquial_eval.py --top-k 6 --with-agent           # 追加端到端答案产出率
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

from eval.colloquial_set import load_cases, stats as set_stats   # noqa: E402

from agent.graph import AgentConfig, AgentGraph                  # noqa: E402
from agent.kb import FIXTURE_KB, KnowledgeBase                   # noqa: E402
from agent.llm import RuleBasedPlannerLLM                        # noqa: E402
from agent.memory import ConversationMemory, VehicleProfile      # noqa: E402
from agent.reflection import NO_ANSWER                           # noqa: E402
from agent.tools import ToolRegistry                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "eval", "colloquial_metrics.json")
OUT_MD = os.path.join(ROOT, "eval", "colloquial_report.md")
OUT_DETAIL = os.path.join(ROOT, "eval", "colloquial_details.jsonl")

_STOP = set("的了吗呢么怎如何是有什么可以请问一下我你他这那个哪些为对能会要怎样也就都很太老")


def content_chars(text: str) -> str:
    """取实词字符（去停用字），用于覆盖率计算。"""
    return "".join(c for c in (text or "") if "\u4e00" <= c <= "\u9fff" and c not in _STOP)


def coverage(query: str, texts: List[str]) -> float:
    chars = set(content_chars(query))
    if not chars:
        return 1.0
    joined = "\n".join(texts)
    return sum(1 for c in chars if c in joined) / len(chars)


def evaluate_case(kb: KnowledgeBase, planner: RuleBasedPlannerLLM,
                  case: Dict, top_k: int, rewrite: bool) -> Dict:
    query = case["colloquial"]
    search_query = planner._normalize_query(query) if rewrite else query
    hits = kb.search(search_query, top_k=top_k)
    texts = [h.text for h in hits]
    keywords = case["expect_keywords"]

    hit_any = any(any(k in t for k in keywords) for t in texts)
    hit_top1 = bool(texts) and any(k in texts[0] for k in keywords)
    first_rank = None
    for rank, text in enumerate(texts, start=1):
        if any(k in text for k in keywords):
            first_rank = rank
            break

    return {
        "id": case["id"], "colloquial": query, "manual_term": case["manual_term"],
        "expect_keywords": keywords, "rewrite_terms": case["rewrite_terms"], "note": case["note"],
        "search_query": search_query, "rewrite_changed": search_query != query,
        "n_hits": len(hits), "hit": hit_any, "hit_top1": hit_top1, "first_rank": first_rank,
        "coverage": round(coverage(query, texts), 4),
        "search_coverage": round(coverage(search_query, texts), 4),
        "top_citation": hits[0].citation if hits else "",
        "top_header": hits[0].header if hits else None,
        "top_score": round(hits[0].score, 3) if hits else None,
        "top_text": (hits[0].text[:60] if hits else ""),
        "citations": [h.citation for h in hits],
    }


def run_agent_support(kb: KnowledgeBase, cases: List[Dict], rewrite: bool,
                      top_k: int) -> Dict[str, Any]:
    """可选的端到端检查：口语问题走完整 Agent 链路的答案产出率。"""
    planner = RuleBasedPlannerLLM(rewrite_enabled=rewrite)
    registry = ToolRegistry(kb)
    agent = AgentGraph(planner, registry,
                       memory=ConversationMemory(profile=VehicleProfile()),
                       config=AgentConfig(max_steps=6, evidence_gate=True,
                                          evidence_overlap_threshold=0.40))
    rows = []
    for case in cases:
        agent.registry.reset_cache()
        agent.memory = ConversationMemory(profile=VehicleProfile())
        state = agent.run(case["colloquial"])
        answer = (state.answer or "").strip()
        rows.append({
            "id": case["id"], "colloquial": case["colloquial"],
            "answered": bool(answer) and answer != NO_ANSWER,
            "status": state.status, "gate_coverage": state.gate_coverage,
            "answer": answer[:180], "citations": state.citations[:3],
        })
    return {
        "answer_rate": round(sum(1 for r in rows if r["answered"]) / len(rows), 4) if rows else None,
        "rows": rows,
    }


def _rate(rows: List[Dict], key: str) -> Optional[float]:
    if not rows:
        return None
    return round(sum(1 for r in rows if r[key]) / len(rows), 4)


def merge_pairs(rows_rewrite: List[Dict], rows_plain: List[Dict]) -> Dict[str, Any]:
    """把「开启改写 / 关闭改写」两组逐条结果合并，并算出增益/回归清单。

    独立成函数（而不是留在 `main` 里）是为了让 `eval/eval_extra_smoke.py`
    能复用同一套口径做冒烟，避免两处各写一份统计而漂移。
    """
    merged, regressions, gains = [], [], []
    for r_on, r_off in zip(rows_rewrite, rows_plain):
        item = dict(r_on)
        item["rewrite"] = {k: v for k, v in r_on.items()
                           if k in ("search_query", "hit", "hit_top1", "first_rank",
                                    "coverage", "search_coverage", "n_hits")}
        item["no_rewrite"] = {k: v for k, v in r_off.items()
                              if k in ("search_query", "hit", "hit_top1", "first_rank",
                                       "coverage", "search_coverage", "n_hits")}
        merged.append(item)
        if r_off["hit"] and not r_on["hit"]:
            regressions.append(r_on["id"])
        if r_on["hit"] and not r_off["hit"]:
            gains.append(r_on["id"])
    return {"merged": merged, "gain_cases": gains, "regression_cases": regressions}


def summarize(rows: List[Dict]) -> Dict[str, Any]:
    """单组（开启或关闭改写）的指标。"""
    return {
        "n": len(rows),
        "hit_rate": _rate(rows, "hit"),
        "hit_at_1_rate": _rate(rows, "hit_top1"),
        "coverage": round(statistics.mean([r["coverage"] for r in rows]), 4) if rows else None,
        "search_coverage": round(statistics.mean([r["search_coverage"] for r in rows]), 4)
        if rows else None,
        "mean_first_rank": round(statistics.mean(
            [r["first_rank"] for r in rows if r["first_rank"]]), 3)
        if any(r["first_rank"] for r in rows) else None,
        "n_zero_hit": sum(1 for r in rows if r["n_hits"] == 0),
        "rewrite_changed": sum(1 for r in rows if r["rewrite_changed"]),
    }


def compare(rows_rewrite: List[Dict], rows_plain: List[Dict]) -> Dict[str, Any]:
    """两组对照的完整结论（含 Δ、增益/回归样本）。"""
    pair = merge_pairs(rows_rewrite, rows_plain)
    with_rw = summarize(rows_rewrite)
    no_rw = summarize(rows_plain)
    delta = {
        "hit_rate_delta": round((with_rw["hit_rate"] or 0) - (no_rw["hit_rate"] or 0), 4),
        "coverage_delta": round((with_rw["coverage"] or 0) - (no_rw["coverage"] or 0), 4),
        "search_coverage_delta": round((with_rw["search_coverage"] or 0)
                                       - (no_rw["search_coverage"] or 0), 4),
        "gain_cases": pair["gain_cases"], "regression_cases": pair["regression_cases"],
    }
    return {"with_rewrite": with_rw, "without_rewrite": no_rw, "delta": delta,
            "merged": pair["merged"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default=FIXTURE_KB,
                    help="知识库 jsonl；默认迷你语料（CI 口径），完整语料传 kb/chunks.jsonl")
    ap.add_argument("--top-k", type=int, default=6, help="检索返回条数（与 AgentConfig.top_k 一致）")
    ap.add_argument("--with-agent", action="store_true",
                    help="追加端到端答案产出率（较慢，默认关闭）")
    ap.add_argument("--limit", type=int, default=0)
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
    planner_on = RuleBasedPlannerLLM(rewrite_enabled=True)
    planner_off = RuleBasedPlannerLLM(rewrite_enabled=False)

    print(f"[info] 口语样本 {len(cases)} 条，top_k={args.top_k}，开始双组对照…", flush=True)
    started = time.perf_counter()
    rows_rewrite, rows_plain = [], []
    for i, case in enumerate(cases, start=1):
        r_on = evaluate_case(kb, planner_on, case, args.top_k, True)
        r_off = evaluate_case(kb, planner_off, case, args.top_k, False)
        rows_rewrite.append(r_on)
        rows_plain.append(r_off)
        if args.verbose:
            print(f"  [{i:>2}/{len(cases)}] {case['id']} {case['colloquial']}", flush=True)
            print(f"        改写后查询={r_on['search_query']!r} 命中={r_on['hit']} "
                  f"rank={r_on['first_rank']} cov={r_on['coverage']}", flush=True)
            print(f"        原始查询  ={r_off['search_query']!r} 命中={r_off['hit']} "
                  f"rank={r_off['first_rank']} cov={r_off['coverage']}", flush=True)
        else:
            print(f"  [{i:>2}/{len(cases)}] {case['id']} 改写={r_on['hit']} "
                  f"原始={r_off['hit']}  {case['colloquial'][:22]}", flush=True)

    cmp_result = compare(rows_rewrite, rows_plain)
    merged = cmp_result["merged"]
    with_rw = cmp_result["with_rewrite"]
    no_rw = cmp_result["without_rewrite"]
    delta = cmp_result["delta"]

    # 按「改写是否真的命中映射词」分组，解释增益从哪来
    rw_applicable = [r for r, c in zip(rows_rewrite, cases) if c["rewrite_terms"]]
    rw_other = [r for r, c in zip(rows_rewrite, cases) if not c["rewrite_terms"]]
    by_group = {
        "rewrite_mapped": {"n": len(rw_applicable), "with_rewrite": summarize(rw_applicable)},
        "rewrite_unmapped": {"n": len(rw_other), "with_rewrite": summarize(rw_other)},
    }

    agent_support = None
    if args.with_agent:
        print("[info] 端到端（Agent 全链路）对照…", flush=True)
        on = run_agent_support(kb, cases, True, args.top_k)
        off = run_agent_support(kb, cases, False, args.top_k)
        agent_support = {
            "with_rewrite": {"answer_rate": on["answer_rate"], "rows": on["rows"]},
            "without_rewrite": {"answer_rate": off["answer_rate"], "rows": off["rows"]},
        }

    wall = time.perf_counter() - started
    summary = {
        "n_cases": len(cases), "top_k": args.top_k, "kb_path": args.kb,
        "kb_stats": kb.stats(), "wall_clock_s": round(wall, 2),
        "with_rewrite": with_rw, "without_rewrite": no_rw, "delta": delta,
        "by_group": by_group, "set_stats": set_stats(),
        "agent_support": agent_support,
    }
    json.dump({"summary": summary, "cases": merged},
              open(args.json_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    with open(args.detail_out, "w", encoding="utf-8") as fh:
        for row in merged:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    md = render_md(summary, merged)
    open(args.md_out, "w", encoding="utf-8").write(md)
    print("\n" + md)
    print(f"[json] {args.json_out}\n[md] {args.md_out}\n[detail] {args.detail_out}")
    return 0


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.2%}"


def _pct4(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def render_md(summary: Dict, merged: List[Dict]) -> str:
    w, n, d = summary["with_rewrite"], summary["without_rewrite"], summary["delta"]
    verdict = ("**有增益**" if (d["hit_rate_delta"] or 0) > 0
               else "**没测出增益**" if (d["hit_rate_delta"] or 0) == 0 else "**有负增益（改写有害）**")
    regs = d["regression_cases"] or []
    gains = d["gain_cases"] or []
    lines = [
        "# 口语化问法检索评测报告（自动生成）\n",
        f"- 测试集：**{summary['n_cases']} 条口语问法**，覆盖 "
        f"{summary['set_stats']['n_manual_terms']} 条手册术语",
        f"- 知识库：`{summary['kb_path']}` ｜ 块数 {summary['kb_stats']['n_chunks']} ｜ "
        f"分词 {summary['kb_stats']['tokenizer']}",
        f"- 检索：BM25（top_k={summary['top_k']}），两组的唯一差别是 query 改写开关",
        f"- 后端：`planner`（离线规则替身），零成本、确定性\n",
        "## 1. 核心结论\n",
        f"**query 改写（口语→术语映射）在本次实验里：{verdict}**，"
        f"命中率 {_pct(n['hit_rate'])} → {_pct(w['hit_rate'])}"
        f"（Δ {d['hit_rate_delta']:+.4f}）；"
        f"实词覆盖率 {_pct4(n['coverage'])} → {_pct4(w['coverage'])}"
        f"（Δ {d['coverage_delta']:+.4f}）。\n",
        f"- 改写**新增命中**：{gains or '无'}；改写**导致丢命中**：{regs or '无'}。",
        "- 注意：`hit_rate`（top_k 里是否出现锚点词）比 `coverage`（实词覆盖率）"
        "**更接近「能不能答出来」**，但也更受候选池大小影响；"
        "两个指标给出不同符号时，以 `hit_rate` 为主、`coverage` 为辅解释。\n",
        "## 2. 两组对照\n",
        "| 指标 | 关闭改写 | 开启改写 | Δ |", "| --- | --- | --- | --- |",
        f"| 命中率（top_k 出现锚点） | **{_pct(n['hit_rate'])}** | **{_pct(w['hit_rate'])}** | "
        f"{d['hit_rate_delta']:+.4f} |",
        f"| 命中率（仅看 top-1） | {_pct(n['hit_at_1_rate'])} | {_pct(w['hit_at_1_rate'])} | "
        f"{(w['hit_at_1_rate'] or 0) - (n['hit_at_1_rate'] or 0):+.4f} |",
        f"| 实词覆盖率（原口语问法） | {_pct4(n['coverage'])} | {_pct4(w['coverage'])} | "
        f"{d['coverage_delta']:+.4f} |",
        f"| 实词覆盖率（实际检索 query） | {_pct4(n['search_coverage'])} | "
        f"{_pct4(w['search_coverage'])} | {d['search_coverage_delta']:+.4f} |",
        f"| 首次命中平均排名 | {n['mean_first_rank']} | {w['mean_first_rank']} | — |",
        f"| 零命中条数 | {n['n_zero_hit']} | {w['n_zero_hit']} | "
        f"{w['n_zero_hit'] - n['n_zero_hit']:+d} |",
        f"| 实际被改写改动的 query | 0 | {w['rewrite_changed']} | — |\n",
        "## 3. 增益来源归因\n",
        f"- 测试集里**改写表能命中映射词**的样本：{summary['by_group']['rewrite_mapped']['n']} 条，"
        f"这部分命中率 {_pct(summary['by_group']['rewrite_mapped']['with_rewrite']['hit_rate'])}；",
        f"- 改写表**完全不涉及**的样本：{summary['by_group']['rewrite_unmapped']['n']} 条，"
        f"命中率 {_pct(summary['by_group']['rewrite_unmapped']['with_rewrite']['hit_rate'])}；",
        f"- 改写**新增命中**的样本：{d['gain_cases'] or '无'}；",
        f"- 改写**导致丢命中**（回归）的样本：{d['regression_cases'] or '无'}。",
        "- 回归是怎么发生的（完整语料实测）：`怎么关`→`关闭`、`打不开`→`无法开启` "
        "这类**把口语动词换成手册近义词**的映射，在 9301 块的候选池里会引入大量同词噪声——"
        "例如 `倒车雷达关闭` 命中了一堆「开启/关闭氛围灯」的块，"
        "而原始口语 `倒车雷达怎么关` 反而因为「倒车雷达」三字共现把正确块排进 top-6。"
        "**结论：改写的收益与风险都随候选池增大而放大，需要用真实语料标定，不能想当然**。\n",
    ]
    if summary.get("agent_support"):
        a_on = summary["agent_support"]["with_rewrite"]["answer_rate"]
        a_off = summary["agent_support"]["without_rewrite"]["answer_rate"]
        lines += ["## 4. 端到端答案产出率（`--with-agent`）\n",
                  "| 配置 | 答案产出率 |", "| --- | --- |",
                  f"| 关闭改写 | {_pct(a_off)} |", f"| 开启改写 | {_pct(a_on)} |",
                  f"\nΔ = {(a_on or 0) - (a_off or 0):+.4f}\n"]
        sec = 5
    else:
        sec = 4
    lines += [
        f"## {sec}. 逐条明细\n",
        "| ID | 口语问法 | 对应手册术语 | 改写后 query | 关闭改写 | 开启改写 | 排名 | 覆盖率(开) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in merged:
        lines.append(
            f"| {row['id']} | {row['colloquial']} | {row['manual_term']} | "
            f"{row['rewrite']['search_query']} | "
            f"{'✅' if row['no_rewrite']['hit'] else '❌'} | "
            f"{'✅' if row['rewrite']['hit'] else '❌'} | "
            f"{row['rewrite']['first_rank'] or '—'} | {row['rewrite']['coverage']:.4f} |")
    lines += [
        f"\n## {sec + 1}. 口径与局限（如实说明）\n",
        "1. **命中判定是词面锚点**：`expect_keywords` 出现在 top_k 命中块里即算命中，"
        "不做语义判定。它可能把「检索到了但答不到点子上」算成命中，"
        "也可能把「换了说法但语义正确」算成未命中——因此同时给出连续值 `coverage` 作为补充；",
        "2. **现有改写表非常小**：`agent/llm.py::_normalize_query` 只有 "
        f"{len(_REWRITE_HINT)} 组映射（靠背太热/屁股热/怎么关/打不开/亮黄灯/空调不凉/没电了/刹车），"
        "覆盖不到「双闪」「引擎盖」「雨刮器」「倒车雷达」这些同样是高频口语的词。"
        "本实验的结论只针对**现有改写表**，不能外推成「query 改写没用」；",
        "3. **两组共用同一分词器与同一索引**：差异只来自 query 文本，"
        "排除了索引重建、随机性等干扰（完全确定性，可复跑）；",
        "4. **完整语料下命中率显著低于迷你语料（本次实测见两组报告对照）**："
        "9301 块的候选池远大于 24 块，top_k 里被近义块挤掉的概率高得多。"
        "两个语料下的绝对值不能互相比较，"
        "但「开启 vs 关闭改写」的内部对照在各自语料下都由同一套确定性流程给出；",
    ]
    return "\n".join(lines) + "\n"


_REWRITE_HINT = ("靠背太热", "靠背发烫", "屁股热", "怎么关", "打不开", "亮黄灯",
                 "空调不凉", "没电了", "刹车")


if __name__ == "__main__":
    sys.exit(main())
