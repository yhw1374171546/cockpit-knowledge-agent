# -*- coding: utf-8 -*-
"""补充评测的 CI 冒烟：用**迷你语料**把本轮新增的 5 项评测全部跑通并检查关键不变量。

为什么单独一个文件
----------------
`eval/ci_smoke.py` 覆盖的是原有资产（单 Agent / 多 Agent / 注入 / 缓存），
本轮新增的 5 项评测（多轮、口语化、边界、引用准召、反馈闭环）需要各自的线上回归检查点。
为不动 `eval/ci_smoke.py`（避免与并行改动冲突）与 `eval/thresholds.json` / `eval/ci_gate.py`
（由项目负责人统一维护），本脚本把「能跑通 + 关键不变量不被破坏」收敛到一处。

检查项（全部离线、确定性、零成本）
--------------------------------
1. 多轮评测集规模与结构（≥30 条、六类齐全、轮次统计自洽）；
2. 多轮评测可跑通，且**关键指标不为 0**（指代消解成功、追问产出、无异常）；
3. 口语化测试集规模（≥40 条）与标注完整性（每条都有手册术语与锚点关键词）；
4. 口语化检索两组的命中率都能算出来（改写开关不改变可执行性）；
5. 边界用例 ≥25 条、**不崩溃率 100%**、写操作 0 次被静默执行；
6. 引用准召能算出精确率/召回率，且**没有误杀**（假阳性率 0）；
7. 反馈闭环在**无数据**时优雅输出，`--demo-seed` 时能产出 badcase。
8. 完整语料没有被误加载（迷你语料 24 块）——防止 CI 意外去读 9301 块的真实语料。

用法：
    python eval/eval_extra_smoke.py
    python eval/eval_extra_smoke.py --json-out eval/eval_extra_smoke_report.json
退出码：0 = 全部通过；1 = 有检查项失败
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.citation_eval import (_LABEL_CASES, label_case, run_live,  # noqa: E402
                                summarize as cite_summarize)
from eval.colloquial_eval import compare as cq_compare, evaluate_case  # noqa: E402
from eval.colloquial_set import load_cases as load_cq, stats as cq_stats   # noqa: E402
from eval.edge_cases import _CASES as EDGE_CASES, summarize as edge_summarize  # noqa: E402
from eval.edge_cases import _case_ok                                          # noqa: E402
from eval.feedback_loop import aggregate, attribute, collect                  # noqa: E402
from eval.multiturn_eval import run_case, summarize                           # noqa: E402
from eval.multiturn_set import CATEGORIES, load_cases, stats as mt_stats      # noqa: E402

from agent.kb import FIXTURE_KB, KnowledgeBase                                # noqa: E402
from agent.llm import RuleBasedPlannerLLM                                     # noqa: E402
from agent.reflection import Reflector                                        # noqa: E402

OUT = os.path.join(ROOT, "eval", "eval_extra_smoke_report.json")
SMOKE_MT_LIMIT = 8          # 多轮：取前 8 条（覆盖 6 类中的前几类，够冒烟）
SMOKE_CQ_LIMIT = 12         # 口语：取前 12 条
SMOKE_EDGE_LIMIT = 16       # 边界：取前 16 条
SMOKE_LIVE_LIMIT = 6        # 引用：真实链路取前 6 个问题


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default=FIXTURE_KB,
                    help="知识库 jsonl；默认迷你语料（CI 口径），完整语料传 kb/chunks.jsonl")
    ap.add_argument("--json-out", default=OUT)
    args = ap.parse_args()

    checks: List[Dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})
        print(f"  [{'✅' if ok else '❌'}] {name} {detail}", flush=True)
        return bool(ok)

    started = time.perf_counter()
    print("=" * 72)
    print("补充评测冒烟（迷你语料，离线、确定性、零成本）")
    print("=" * 72)

    print("\n[0] 语料与依赖")
    kb = KnowledgeBase.load(args.kb)
    is_fixture = os.path.abspath(args.kb) == os.path.abspath(FIXTURE_KB)
    record("知识库加载", len(kb.chunks) >= 10, f"{len(kb.chunks)} 块（{args.kb}）")
    record("CI 未误加载完整语料", (not is_fixture) or len(kb.chunks) <= 100,
           f"块数 {len(kb.chunks)}；CI 默认口径必须用 24 块迷你语料")
    record("语料带页码元数据", kb.stats()["with_page_meta"] == len(kb.chunks),
           f"{kb.stats()['with_page_meta']}/{len(kb.chunks)}")

    print("\n[1] 多轮评测集结构")
    ms = mt_stats()
    record("多轮样本数 ≥30", ms["n_cases"] >= 30, f"{ms['n_cases']} 条 / {ms['n_turns']} 轮")
    missing_cat = [c for c in CATEGORIES if ms["by_category"].get(c, 0) == 0]
    record("六类样本齐全", not missing_cat, f"缺失={missing_cat or '无'}")
    record("标注了消解/继承/安全期望",
           ms["n_resolve_expect"] > 0 and ms["n_inherit_forbid"] > 0 and ms["n_expect_safety"] > 0,
           f"消解{ms['n_resolve_expect']} 继承{ms['n_inherit_forbid']} 安全{ms['n_expect_safety']}")

    print("\n[2] 多轮评测可跑通")
    all_mt = load_cases()
    # 冒烟要覆盖到「跨轮安全」类（安全样本排在集合末尾），否则这条检查会假失败
    smoke_mt = all_mt[:SMOKE_MT_LIMIT]
    smoke_mt += [c for c in all_mt if c["category"] == "safety"][:2]
    results = [run_case(kb, c) for c in smoke_mt]
    msum = summarize(results, kb.stats())
    record("多轮无异常", msum["errors"] == 0, f"异常 {msum['errors']} 轮 / {msum['n_turns']} 轮")
    record("指代消解成功率 > 0", (msum["resolution_success_rate"] or 0) > 0,
           f"{msum['resolution_success_rate']}（样本 {msum['resolution_samples']}）")
    record("追问答案产出率 > 0", (msum["followup_answer_rate"] or 0) > 0,
           f"{msum['followup_answer_rate']}")
    record("跨轮安全指标准有样本", msum["cross_turn_safety_samples"] > 0,
           f"{msum['cross_turn_safety_samples']} 轮（含 {sum(1 for c in smoke_mt if c['category'] == 'safety')} 条安全样本）")

    print("\n[3] 口语化测试集结构")
    cqs = cq_stats()
    cq_cases = load_cq()
    record("口语样本数 ≥40", cqs["n_cases"] >= 40, f"{cqs['n_cases']} 条")
    bad = [c["id"] for c in cq_cases
           if not c.get("manual_term") or not c.get("expect_keywords") or not c.get("colloquial")]
    record("每条都有术语+锚点标注", not bad, f"缺标注={bad or '无'}")

    print("\n[4] 口语化检索（开/关改写两组）")
    sub = cq_cases[:SMOKE_CQ_LIMIT]
    on = [evaluate_case(kb, RuleBasedPlannerLLM(rewrite_enabled=True), c, 6, True) for c in sub]
    off = [evaluate_case(kb, RuleBasedPlannerLLM(rewrite_enabled=False), c, 6, False) for c in sub]
    cq_cmp = cq_compare(on, off)
    s_on, s_off = cq_cmp["with_rewrite"], cq_cmp["without_rewrite"]
    record("两组命中率都算得出", s_on["hit_rate"] is not None and s_off["hit_rate"] is not None,
           f"改写 {s_on['hit_rate']} / 原始 {s_off['hit_rate']}")
    record("覆盖率是连续值", 0.0 <= (s_on["coverage"] or 0) <= 1.0, f"{s_on['coverage']}")

    print("\n[5] 边界与长尾")
    edge_rows = []
    from agent.guardrails import InjectionDetector                    # noqa: E402
    from eval.edge_cases import run_case as edge_run                   # noqa: E402
    detector = InjectionDetector()
    for case in EDGE_CASES[:SMOKE_EDGE_LIMIT]:
        edge_rows.append(edge_run(kb, case, detector))
    esum = edge_summarize(edge_rows, kb.stats())
    record("边界用例集 ≥25", len(EDGE_CASES) >= 25, f"{len(EDGE_CASES)} 条")
    record("不崩溃率 100%", esum["no_crash_rate"] == 1.0,
           f"{esum['no_crash_rate']}（崩溃 {esum['n_crashes']}：{esum['crash_ids'] or '无'}）")
    record("写操作 0 次被静默执行", esum["write_orders_silently_created"] == 0,
           f"{esum['write_orders_silently_created']} 次")
    record("响应时间有上界统计", esum["latency_max_ms"] is not None,
           f"p50={esum['latency_p50_ms']} p95={esum['latency_p95_ms']} max={esum['latency_max_ms']} ms")

    print("\n[6] 引用准召")
    reflector = Reflector()
    label_rows = [label_case(c, reflector, kb) for c in _LABEL_CASES]
    live_rows = run_live(kb, ["危险警告灯怎么打开", "座椅加热怎么关闭", "胎压报警了怎么办",
                              "保养周期是多久", "蓝牙怎么连接手机", "无线充电怎么开启"][:SMOKE_LIVE_LIMIT])
    csum = cite_summarize(label_rows, live_rows, kb.stats())
    record("引用精确率算得出", csum["citation_precision"] is not None,
           f"precision={csum['citation_precision']}")
    record("引用召回率算得出", csum["citation_recall"] is not None,
           f"recall={csum['citation_recall']}（TP={csum['tp']} FN={csum['fn']}）")
    record("没有误杀合法引用（假阳性率 0）", csum["false_positive_rate"] == 0.0,
           f"FP={csum['fp']} 假阳性率={csum['false_positive_rate']}")
    record("标注单元与标注一一对齐",
           all(r["label_mismatch"] is None for r in label_rows),
           f"错位={[r['id'] for r in label_rows if r['label_mismatch']] or '无'}")
    record("真实链路引用都能对上证据", (csum["live"]["citation_precision_live"] or 0) >= 0.99,
           f"{csum['live']['citation_precision_live']}")

    print("\n[7] 反馈闭环")
    tmpdir = tempfile.mkdtemp(prefix="dsh_eval_fb_")
    empty_db = os.path.join(tmpdir, "empty.db")
    collected = collect(empty_db, -1, 0)
    record("无数据时优雅输出", (not collected["exists"]) and collected["rows"] == [],
           f"exists={collected['exists']} rows={len(collected['rows'])}")
    demo_db = os.path.join(tmpdir, "demo.db")
    from eval.feedback_loop import seed_demo                          # noqa: E402
    seed_demo(demo_db, n=6)
    demo_rows = collect(demo_db, -1, 0)["rows"]
    for row in demo_rows:
        row["attribution"] = attribute(row)
    demo_agg = aggregate(demo_rows)
    record("--demo-seed 造数可读", len(demo_rows) == 6, f"{len(demo_rows)} 条负反馈")
    record("归因聚合非空", len(demo_agg["by_failure"]) >= 3,
           json.dumps(demo_agg["by_failure"], ensure_ascii=False))

    wall = time.perf_counter() - started
    failed = [c for c in checks if not c["ok"]]
    json.dump({"checks": checks, "failed": len(failed), "n_checks": len(checks),
               "wall_clock_s": round(wall, 2), "kb": FIXTURE_KB,
               "kb_chunks": len(kb.chunks), "offline": True, "cost_usd": 0.0},
              open(args.json_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n" + "-" * 72)
    print(f"补充冒烟结果：{len(checks) - len(failed)}/{len(checks)} 通过"
          f"　（{wall:.1f}s，离线零成本）　→ {'通过' if not failed else '未通过'}")
    if failed:
        for item in failed:
            print(f"  ❌ {item['check']}：{item['detail']}")
    print(f"[json] {args.json_out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
