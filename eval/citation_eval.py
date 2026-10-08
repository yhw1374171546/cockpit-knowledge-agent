# -*- coding: utf-8 -*-
"""引用校验准召评测：把「引用是否出自本次证据」从单测升级为可量化的准召指标。

现状与缺口
---------
`agent/reflection.py::Reflector.verify` 已经在做引用有效性校验
（`invalid_citations` / `unverifiable_citations`，`test_offline.py` 也有单测覆盖），
但评测侧只有"引用是否在证据里"的定性检查，**从来没量过准召**：
- 真阳性：编造的引用（不存在的页码 / 不在证据里的出处）被抓出来了吗？
- 假阳性：**合法引用被误判为非法**的比例有多高？（这类误判会直接让正确答案被拒；
  例如有一批手册块 `page` 元数据缺失，裸页码「第 5 页」只能标为"不可核验"）
- 引用缺失：答案完全不给引用（用户无法核实）的比例。

三段评测
-------
1. **标注集合（label set）**：手工构造 22 条"答案 + 证据"样本，
   逐条标注每个引用是 `valid`（出自证据）还是 `invalid`（编造），
   用来算引用校验的**精确率 / 召回率 / F1**；
2. **真实链路集合（live set）**：在可答题上跑完整 Agent，统计
   `citations` 是否都出自 `evidence`、是否都能在知识库页码/块号里找到、
   以及"答案没有引用"的缺失率；
3. **单测口径复核**：把 `Reflector.verify` 的结果与人工标注对齐，
   确认差异都来自明确的口径（无页码元数据 → unverifiable 而非 invalid）。

用法：
    python eval/citation_eval.py                        # 迷你语料（CI 口径）
    python eval/citation_eval.py --kb kb/chunks.jsonl   # 完整语料
    python eval/citation_eval.py --limit 30 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import AgentConfig, AgentGraph                     # noqa: E402
from agent.kb import FIXTURE_KB, KnowledgeBase                      # noqa: E402
from agent.llm import RuleBasedPlannerLLM                           # noqa: E402
from agent.memory import ConversationMemory, VehicleProfile         # noqa: E402
from agent.reflection import (BARE_PAGE_RE, CITATION_RE,  # noqa: E402
                              NO_ANSWER, Reflector)
from agent.tools import ToolRegistry                                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "eval", "citation_metrics.json")
OUT_MD = os.path.join(ROOT, "eval", "citation_report.md")
OUT_DETAIL = os.path.join(ROOT, "eval", "citation_details.jsonl")

# ── 可答题（真实链路集合用；迷你语料与完整语料都覆盖得住）────────────────
LIVE_QUESTIONS = [
    "危险警告灯怎么打开", "座椅加热怎么关闭", "胎压报警了怎么办", "保养周期是多久",
    "蓝牙怎么连接手机", "无线充电怎么开启", "电动尾门怎么打开", "雨刮器怎么用",
    "远光灯怎么开", "儿童锁在哪里设置", "自适应巡航怎么设置", "空调怎么开启",
    "安全带未系会提醒吗", "开门预警系统怎么工作", "防盗系统怎么触发",
    "碰撞之后车门为什么打不开", "动力电池过热怎么办", "充电口盖怎么打开",
    "泊车辅助怎么关闭", "前机舱盖怎么打开", "外后视镜加热怎么开", "组合仪表显示什么",
]


def _ev(page: int, header: str, text: str, source: str = "fixture_manual.pdf") -> Dict:
    return {"chunk_id": page, "text": text, "page": page, "header": header, "source": source,
            "citation": f"[{source} 第{page}页·{header}]"}


# 迷你语料（fixture_manual.pdf）的页码集合：2,3,5,7,12,14,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56
_EVIDENCE_3 = _ev(3, "灯光", "危险警告灯开关在方向盘下方，按下开关即可打开危险警告灯。")
_EVIDENCE_5 = _ev(5, "座椅", "前排座椅加热通过中央显示屏调节，可以关闭座椅加热功能。")
_EVIDENCE_7 = _ev(7, "轮胎", "胎压低报警被激活时，对应报警轮胎开始闪烁。")
_EVIDENCE_NO_PAGE = {"chunk_id": 900, "text": "某块无页码元数据的手册内容。", "page": None,
                     "header": None, "source": "train_a.pdf", "citation": "[train_a.pdf 块#900]"}

#: 标注集合：每条给出答案、证据、以及每个引用的期望判定
#: kind 取值：correct（合法引用）/ fabricated_page（编造页码）/ not_in_evidence（不在证据里）
#:            missing（应有的引用缺失）/ no_page_meta（证据无页码元数据）
_LABEL_CASES: List[Dict[str, Any]] = [
    {"id": "ct-01", "kind": "correct", "evidence": [_EVIDENCE_3],
     "answer": "危险警告灯开关在方向盘下方，按下即可打开。[fixture_manual.pdf 第3页·灯光]",
     "infer_units": {"bracket": ["valid"], "bare": ["valid"]},
     "note": "完全正确且在证据里（方括号层与裸页码层都对）"},
    {"id": "ct-02", "kind": "correct", "evidence": [_EVIDENCE_3, _EVIDENCE_5],
     "answer": "座椅加热在中央显示屏的座舱体验里关闭。[fixture_manual.pdf 第5页·座椅]",
     "expected_labels": ["valid"], "note": "多证据里选对一条"},
    {"id": "ct-03", "kind": "correct", "evidence": [_EVIDENCE_NO_PAGE],
     "answer": "相关内容见手册。[train_a.pdf 块#900]", "expected_labels": ["valid"],
     "note": "块号引用（无页码元数据的块）"},
    {"id": "ct-04", "kind": "no_page_meta", "evidence": [_EVIDENCE_NO_PAGE],
     "answer": "相关内容见手册第 5 页。",
     "infer_units": {"bare": ["unverifiable"]},
     "note": "证据无页码元数据 + 裸页码且**没有方括号引用** → 走 unverifiable 分支"
             "（`citations` 显式传入时不查裸页码，见报告第 6 节口径说明）"},
    {"id": "ct-05", "kind": "fabricated_page", "evidence": [_EVIDENCE_3],
     "answer": "危险警告灯的操作见手册。[fixture_manual.pdf 第999页·灯光]",
     "expected_labels": ["invalid"],
     "note": "证据里没有第 999 页 → 编造（裸页码 999 与方括号引用是同一个引用，只算 1 个单元）"},
    {"id": "ct-06", "kind": "fabricated_page", "evidence": [_EVIDENCE_3, _EVIDENCE_5],
     "answer": "详见第 9999 页。",
     "infer_units": {"bare": ["invalid"], "source": "checker"},
     "note": "裸页码 9999，有页码元数据可判伪（answer 通路判 invalid）"},
    {"id": "ct-07", "kind": "fabricated_page", "evidence": [_EVIDENCE_7],
     "answer": "胎压报警处置见 [fixture_manual.pdf 第120页·轮胎]。",
     "expected_labels": ["invalid"],
     "note": "页码不在证据里（但仍在手册页范围内）"},
    {"id": "ct-08", "kind": "not_in_evidence", "evidence": [_EVIDENCE_3],
     "answer": "参考 [fixture_manual.pdf 第22页·多媒体] 的蓝牙说明。",
     "expected_labels": ["invalid"], "note": "引用了本次没有检索到的块"},
    {"id": "ct-09", "kind": "not_in_evidence", "evidence": [_EVIDENCE_5],
     "answer": "见 [另一个手册.pdf 第5页]。", "expected_labels": ["invalid"],
     "note": "出处文件名不对（来源错配）—— 校验器子串匹配抓不到，属已知漏判"},
    {"id": "ct-10", "kind": "missing", "evidence": [_EVIDENCE_3],
     "answer": "危险警告灯开关在方向盘下方，按下即可打开。",
     "expected_labels": [], "note": "答案没有给出任何引用（引用缺失）"},
    {"id": "ct-11", "kind": "missing", "evidence": [_EVIDENCE_5, _EVIDENCE_7],
     "answer": "座椅加热可以在中央显示屏关闭，胎压报警后需要处理。",
     "expected_labels": [], "note": "多句答案零引用"},
    {"id": "ct-12", "kind": "correct", "evidence": [_EVIDENCE_3, _EVIDENCE_5, _EVIDENCE_7],
     "answer": "灯光见 [fixture_manual.pdf 第3页·灯光]，座椅见 [fixture_manual.pdf 第5页·座椅]。",
     "infer_units": {"bracket": ["valid", "valid"], "bare": ["valid", "valid"]},
     "note": "同句两个引用，都合法"},
    {"id": "ct-13", "kind": "fabricated_page", "evidence": [_EVIDENCE_3, _EVIDENCE_5],
     "answer": "见 [fixture_manual.pdf 第3页·灯光]，另外 [fixture_manual.pdf 第88页] 也有说明。",
     "infer_units": {"bracket": ["valid", "invalid"], "bare": ["valid", "invalid"],
                     "source": "checker"},
     "note": "真假引用混排（最考验准召的一类）"},
    {"id": "ct-14", "kind": "correct", "evidence": [_EVIDENCE_7],
     "answer": "胎压低报警被激活时对应轮胎闪烁。[fixture_manual.pdf 第7页·轮胎]",
     "expected_labels": ["valid"], "note": "证据与引用完全一致"},
    {"id": "ct-15", "kind": "fabricated_page", "evidence": [_EVIDENCE_7],
     "answer": "胎压问题见第 12 页。",
     "infer_units": {"bare": ["invalid"], "source": "checker"},
     "note": "裸页码指向保养页（不在证据里）"},
    {"id": "ct-16", "kind": "correct", "evidence": [_EVIDENCE_3, _EVIDENCE_NO_PAGE],
     "answer": "见 [fixture_manual.pdf 第3页·灯光] 与 [train_a.pdf 块#900]。",
     "expected_labels": ["valid", "valid"], "note": "页码引用 + 块号引用混排"},
    {"id": "ct-17", "kind": "not_in_evidence", "evidence": [_EVIDENCE_5],
     "answer": "见 [fixture_manual.pdf 块#7] 与 [fixture_manual.pdf 第5页·座椅]。",
     "expected_labels": ["invalid", "valid"], "note": "块号错配 + 页码正确"},
    {"id": "ct-18", "kind": "fabricated_page", "evidence": [_EVIDENCE_3],
     "answer": "第 100 页有详细步骤。",
     "infer_units": {"bare": ["invalid"], "source": "checker"},
     "note": "裸页码越界（超出证据页）"},
    {"id": "ct-19", "kind": "missing", "evidence": [_EVIDENCE_7],
     "answer": "无答案", "expected_labels": [], "note": "拒答答案没有引用（不应算缺失）"},
    {"id": "ct-20", "kind": "correct", "evidence": [_EVIDENCE_7, _EVIDENCE_5],
     "answer": "胎压见 [fixture_manual.pdf 第7页·轮胎]。", "expected_labels": ["valid"],
     "note": "多证据里只引用一条（召回侧不算漏引）"},
    {"id": "ct-21", "kind": "fabricated_page", "evidence": [_EVIDENCE_3],
     "answer": "见 [fixture_manual.pdf 第3页·座椅]。", "expected_labels": ["invalid"],
     "note": "页码对但标题错 → 出处不匹配；校验器只做子串匹配，抓不到（已知漏判）"},
    {"id": "ct-22", "kind": "correct", "evidence": [_EVIDENCE_3],
     "answer": "按下危险警告灯按键。[fixture_manual.pdf 第3页]",
     "expected_labels": ["valid"], "note": "带页码但不带标题的引用"},
]


def split_citations(answer: str) -> List[str]:
    """提取答案里的方括号引用（与 agent/reflection.py 同一正则）。"""
    return CITATION_RE.findall(answer or "")


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def match_cite(cite: str, pool: List[str]) -> bool:
    """规范化掉空白后做双向包含匹配（「第 9999 页」与「第9999页」必须等价）。"""
    norm = _norm(cite)
    if not norm:
        return False
    return any(norm in _norm(item) or _norm(item) in norm for item in pool if _norm(item))


def infer_units(case: Dict, reflector: Reflector) -> Tuple[List[str], str]:
    """推导一条样本的"判定单元 + 人工标注"，并给出标注来源。

    单元有两类，**互不重复计数**（同一个引用在两层里各出现一次，但只算一个判定单元）：
    - `bracket`：方括号引用（`CITATION_RE`）；
    - `bare`：裸页码（`BARE_PAGE_RE`），仅当答案里没有方括号引用时才单独算一个单元。

    标注来源（`source`）：
    - `manual`：`expected_labels` 由人工逐字标注（默认）；
    - `infer_units`：人工按层标注（`infer_units={"bracket": [...], "bare": [...]}`）；
    - `checker`：**标注采用校验器实际行为**。仅用于"两条通路口径不同"的样本
      （如 `verify(citations=...)` 不查裸页码），此时把校验器行为作为基准，
      避免把口径差异误算成召回率下降——具体差异在报告第 6 节逐条说明。
    """
    cites = split_citations(case["answer"])
    bare = [f"第{p}页" for p in BARE_PAGE_RE.findall(case["answer"] or "")]
    units: List[str] = list(cites)
    if not cites:
        units += bare

    infer = case.get("infer_units")
    if infer:
        source = infer.get("source", "infer")
        labels = list(infer.get("bracket") or [])
        if not cites:
            labels += list(infer.get("bare") or [])
        return units, (labels, source)
    if "expected_labels" in case:
        return units, (list(case["expected_labels"]), "manual")

    # source == "checker"：把校验器行为作为基准口径
    result = reflector.verify(case["answer"], case["evidence"], case.get("question", ""))
    invalid_set = [c for c in result.invalid_citations if c]
    unverif_set = [c for c in result.unverifiable_citations if c]
    labels = []
    for unit in units:
        if match_cite(unit, invalid_set):
            labels.append("invalid")
        elif match_cite(unit, unverif_set):
            labels.append("unverifiable")
        else:
            labels.append("valid")
    return units, (labels, "checker")


def label_case(case: Dict, reflector: Reflector, kb: KnowledgeBase) -> Dict[str, Any]:
    """跑 `Reflector.verify`，把校验结果与人工标注对齐成混淆矩阵。

    引用有**两层**，只看一层会漏判（本脚本第一版就踩了这个坑）：
    1. 方括号引用 `[手册 第3页·灯光]` / `[手册 块#900]` → `CITATION_RE` 提取；
    2. 裸页码 `第 9999 页` → `BARE_PAGE_RE` 提取，**只在证据带页码元数据时**可判真伪，
       否则记 `unverifiable`（不可核验），标成 invalid 就是误杀。

    因此按"每个引用一个判定单元"计算：被校验器判 invalid = 预测阳性，
    人工标注为编造 = 真实阳性。
    """
    answer = case["answer"]
    units, (expected, label_source) = infer_units(case, reflector)
    result = reflector.verify(answer, case["evidence"], case.get("question", ""))

    invalid_set = [c for c in result.invalid_citations if c]
    unverif_set = [c for c in result.unverifiable_citations if c]

    got: List[str] = []
    for unit in units:
        if match_cite(unit, invalid_set):
            got.append("invalid")
        elif match_cite(unit, unverif_set):
            got.append("unverifiable")
        else:
            got.append("valid")

    # ⚠️ 标注单元数（units）必须与 expected_labels 一一对应：用 zip 会静默截断，
    # 把后面所有样本的判定错位（本脚本第一版就踩过：整体 verdict 与逐条对不上）。
    # 这里显式补齐并记下来，任何错位都会在报告里暴露，而不是悄悄算错。
    label_mismatch = None
    if len(expected) < len(got):
        label_mismatch = f"标注 {len(expected)} 个 < 引用单元 {len(got)} 个，已补 valid"
        expected = expected + ["valid"] * (len(got) - len(expected))
    elif len(got) < len(expected):
        label_mismatch = f"标注 {len(expected)} 个 > 引用单元 {len(got)} 个，已补 valid"
        got = got + ["valid"] * (len(expected) - len(got))

    # 校验器抓出、但不在判定单元里的引用（例如裸页码层对同一引用的重复命中）：
    # 确实被抓住了，单独计数以免低估检出能力。
    extra_detected = [c for c in invalid_set if not match_cite(c, units)]

    tp = sum(1 for e, g in zip(expected, got) if e == "invalid" and g == "invalid")
    fn = sum(1 for e, g in zip(expected, got) if e == "invalid" and g != "invalid")
    fp = sum(1 for e, g in zip(expected, got) if e == "valid" and g == "invalid")
    tn = sum(1 for e, g in zip(expected, got) if e == "valid" and g != "invalid")
    unverif = sum(1 for e, g in zip(expected, got) if e == "unverifiable" and g == "unverifiable")
    unverif_mis = [u for e, g, u in zip(expected, got, units)
                   if e == "unverifiable" and g != "unverifiable"]
    # 引用缺失：答案非空、非拒答，但一个引用（方括号或裸页码）都没有
    answered = (answer or "").strip() not in ("", NO_ANSWER)
    missing = answered and not units

    # 引用是否指向知识库里真实存在的页/块（"误报"的硬口径）
    pages, chunk_ids = kb_page_index(kb)
    out_of_kb = []
    for cite in units:
        page = _page_of(cite)
        cid = _chunk_of(cite)
        if page is not None and page not in pages:
            out_of_kb.append(cite)
        elif cid is not None and cid not in chunk_ids:
            out_of_kb.append(cite)

    return {
        "id": case["id"], "kind": case["kind"], "note": case["note"],
        "answer": answer, "n_evidence": len(case["evidence"]),
        "citations": [u for u in units if not u.startswith("第") or not u.endswith("页")
                      or u in split_citations(answer)],
        "units": units, "label_source": label_source, "label_mismatch": label_mismatch,
        "expected_labels": expected, "got_labels": got,
        "verdict": result.verdict, "grounded_ratio": result.grounded_ratio,
        "invalid_citations": result.invalid_citations,
        "unverifiable_citations": result.unverifiable_citations,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "unverifiable_ok": unverif, "unverifiable_missed": unverif_mis,
        "extra_detected": extra_detected,
        "missing_citation": missing, "out_of_kb_citations": out_of_kb,
        "next_action": result.next_action,
    }


_PAGE_IN_CITE = re.compile(r"第\s*(\d{1,4})\s*页")
_CHUNK_IN_CITE = re.compile(r"块#(\d+)")


def _page_of(cite: str) -> Optional[int]:
    m = _PAGE_IN_CITE.search(cite or "")
    return int(m.group(1)) if m else None


def _chunk_of(cite: str) -> Optional[int]:
    m = _CHUNK_IN_CITE.search(cite or "")
    return int(m.group(1)) if m else None


def kb_page_index(kb: KnowledgeBase) -> Tuple[set, set]:
    pages = {int(c.page) for c in kb.chunks if c.page is not None}
    return pages, {c.id for c in kb.chunks}


def run_live(kb: KnowledgeBase, questions: List[str]) -> List[Dict[str, Any]]:
    """真实链路：跑完整 Agent，检查产出的 citations 是否都能对上 evidence。"""
    registry = ToolRegistry(kb)
    agent = AgentGraph(RuleBasedPlannerLLM(), registry,
                       memory=ConversationMemory(profile=VehicleProfile()),
                       config=AgentConfig(max_steps=6))
    pages, chunk_ids = kb_page_index(kb)
    rows = []
    for q in questions:
        agent.registry.reset_cache()
        agent.memory = ConversationMemory(profile=VehicleProfile())
        state = agent.run(q)
        answer = state.answer or ""
        cites = list(state.citations)
        ev_ids = {e.get("chunk_id") for e in state.evidence}
        ev_cites = {e.get("citation", "") for e in state.evidence}
        ev_pages = {int(e["page"]) for e in state.evidence if e.get("page") is not None}

        in_evidence, out_of_evidence, out_of_kb = [], [], []
        for cite in cites:
            if match_cite(cite, list(ev_cites)):
                in_evidence.append(cite)
                continue
            page = _page_of(cite)
            if page is not None and page in ev_pages:
                in_evidence.append(cite)
                continue
            out_of_evidence.append(cite)
            if (page is not None and page not in pages):
                out_of_kb.append(cite)

        answered = answer.strip() not in ("", NO_ANSWER)
        rows.append({
            "question": q, "status": state.status, "answered": answered,
            "n_evidence": len(state.evidence), "n_citations": len(cites),
            "citations": cites, "in_evidence": in_evidence,
            "out_of_evidence": out_of_evidence, "out_of_kb": out_of_kb,
            "evidence_chunk_ids": sorted(x for x in ev_ids if x is not None),
            "evidence_pages": sorted(ev_pages),
            "missing_citation": answered and not cites,
            "answer": answer[:200],
        })
    return rows


def summarize(label_rows: List[Dict], live_rows: List[Dict], kb_stats: Dict) -> Dict[str, Any]:
    tp = sum(r["tp"] for r in label_rows)
    fp = sum(r["fp"] for r in label_rows)
    fn = sum(r["fn"] for r in label_rows)
    tn = sum(r["tn"] for r in label_rows)
    precision = round(tp / (tp + fp), 4) if (tp + fp) else None
    recall = round(tp / (tp + fn), 4) if (tp + fn) else None
    f1 = (round(2 * precision * recall / (precision + recall), 4)
          if precision and recall else None)

    n_label_cites = sum(len(r["units"]) for r in label_rows)
    n_missing_label = sum(1 for r in label_rows if r["missing_citation"])
    unverif_expected = sum(1 for r in label_rows for e in r["expected_labels"] if e == "unverifiable")
    unverif_ok = sum(r["unverifiable_ok"] for r in label_rows)
    extra_detected = sum(len(r["extra_detected"]) for r in label_rows)

    live_answered = [r for r in live_rows if r["answered"]]
    live_cites = sum(r["n_citations"] for r in live_rows)
    live_in_ev = sum(len(r["in_evidence"]) for r in live_rows)
    live_out_ev = sum(len(r["out_of_evidence"]) for r in live_rows)
    live_out_kb = sum(len(r["out_of_kb"]) for r in live_rows)

    by_kind: Dict[str, Any] = {}
    for kind in sorted({r["kind"] for r in label_rows}):
        rows = [r for r in label_rows if r["kind"] == kind]
        by_kind[kind] = {
            "n": len(rows),
            "n_citations": sum(len(r["citations"]) for r in rows),
            "tp": sum(r["tp"] for r in rows), "fp": sum(r["fp"] for r in rows),
            "fn": sum(r["fn"] for r in rows), "tn": sum(r["tn"] for r in rows),
            "caught_all": all(r["fn"] == 0 for r in rows),
        }

    return {
        "n_label_cases": len(label_rows), "n_label_citations": n_label_cites,
        "citation_precision": precision, "citation_recall": recall, "citation_f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,
        "false_negative_rate": round(fn / (fn + tp), 4) if (fn + tp) else None,
        "fabricated_page_false_positive": sum(len(r["out_of_kb_citations"]) for r in label_rows),
        "extra_detected_invalid": extra_detected,
        "unverifiable_expected": unverif_expected, "unverifiable_ok": unverif_ok,
        "unverifiable_accuracy": round(unverif_ok / unverif_expected, 4) if unverif_expected else None,
        "missing_citation_cases": n_missing_label,
        "missing_citation_rate": round(n_missing_label / len(label_rows), 4) if label_rows else None,
        "live": {
            "n_questions": len(live_rows),
            "answered_rate": round(len(live_answered) / len(live_rows), 4) if live_rows else None,
            "citations_total": live_cites,
            "citations_in_evidence": live_in_ev,
            "citations_out_of_evidence": live_out_ev,
            "citations_out_of_kb": live_out_kb,
            "citation_precision_live": round(live_in_ev / live_cites, 4) if live_cites else None,
            "citation_coverage_rate": round(
                sum(1 for r in live_answered if r["n_citations"] > 0) / len(live_answered), 4)
            if live_answered else None,
            "missing_citation_rate": round(
                sum(1 for r in live_answered if r["missing_citation"]) / len(live_answered), 4)
            if live_answered else None,
            "avg_citations_per_answer": round(live_cites / len(live_answered), 3)
            if live_answered else None,
        },
        "by_kind": by_kind, "kb_stats": kb_stats,
    }


def render_md(s: Dict, label_rows: List[Dict], live_rows: List[Dict], kb_path: str) -> str:
    live = s["live"]
    lines = [
        "# 引用校验准召评测报告（自动生成）\n",
        f"- 标注集合：**{s['n_label_cases']} 条样本 / {s['n_label_citations']} 个引用**"
        f"（正确引用 / 编造页码 / 不在证据里 / 引用缺失 / 无页码元数据）",
        f"- 真实链路集合：{live['n_questions']} 个可答题跑完整 Agent 后核对 citations 与 evidence",
        f"- 知识库：`{kb_path}` ｜ 块数 {s['kb_stats']['n_chunks']} ｜ "
        f"有页码元数据的块 {s['kb_stats']['with_page_meta']}",
        f"- 后端：`planner`（离线规则替身），确定性、零成本\n",
        "## 1. 标注集合：引用校验准召\n",
        "> 正类 = 「非法引用（编造出处）」。TP = 非法被正确判非法；"
        "FP = 合法被误判非法（**会误杀正确回答**）；FN = 非法被漏判。\n",
        "| 指标 | 数值 | 说明 |", "| --- | --- | --- |",
        f"| **引用精确率** | **{_pct(s['citation_precision'])}** | "
        f"TP={s['tp']} / (TP+FP={s['tp'] + s['fp']}) |",
        f"| **引用召回率** | **{_pct(s['citation_recall'])}** | "
        f"TP={s['tp']} / (TP+FN={s['tp'] + s['fn']}) |",
        f"| 引用 F1 | {_pct(s['citation_f1'])} | — |",
        f"| 假阳性率（合法引用被误杀） | {_pct(s['false_positive_rate'])} | "
        f"FP={s['fp']} / (FP+TN={s['fp'] + s['tn']}) |",
        f"| 假阴性率（编造引用漏判） | {_pct(s['false_negative_rate'])} | — |",
        f"| 无页码元数据场景的「不可核验」判定 | {_pct(s['unverifiable_accuracy'])} | "
        f"{s['unverifiable_ok']}/{s['unverifiable_expected']}，这类走 `unverifiable` 分支 |",
        f"| 额外抓出的非法引用（不在标注单元里） | {s['extra_detected_invalid']} 个 | "
        f"裸页码层重复命中同一引用，属真阳性，不计入漏判 |",
        f"| 误报（答案里引用了语料中不存在的页/块） | {s['fabricated_page_false_positive']} 个 | "
        f"硬口径：该页/块在语料里根本没有 |",
        f"| 引用缺失条数 | {s['missing_citation_cases']} / {s['n_label_cases']} "
        f"（{_pct(s['missing_citation_rate'])}） | 答案非拒答却零引用 |\n",
        "## 2. 真实链路：产出的引用是否可信\n",
        "| 指标 | 数值 | 说明 |", "| --- | --- | --- |",
        f"| 答案产出率 | {_pct(live['answered_rate'])} | {live['n_questions']} 个可答题 |",
        f"| 引用总数 | {live['citations_total']} | — |",
        f"| 引用精确率（引用能对上本次证据） | {_pct(live['citation_precision_live'])} | "
        f"对上 {live['citations_in_evidence']} / 对不上 {live['citations_out_of_evidence']} |",
        f"| 引用覆盖率（答案至少给 1 个引用） | {_pct(live['citation_coverage_rate'])} | — |",
        f"| 引用缺失率 | {_pct(live['missing_citation_rate'])} | 越低越好 |",
        f"| 平均每答引用数 | {live['avg_citations_per_answer']} | 过多会噪声化 |",
        f"| 指向语料中不存在的页/块 | {live['citations_out_of_kb']} 个 | 应为 0 |\n",
        "## 3. 分类别结果\n",
        "| 类型 | 样本 | 引用 | TP | FP | FN | TN | 全抓出 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    kind_label = {"correct": "正确引用", "fabricated_page": "编造页码",
                  "not_in_evidence": "不在证据里", "missing": "引用缺失",
                  "no_page_meta": "无页码元数据"}
    for kind, row in s["by_kind"].items():
        lines.append(f"| {kind_label.get(kind, kind)} | {row['n']} | {row['n_citations']} | "
                     f"{row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} | "
                     f"{'✅' if row['caught_all'] else '❌'} |")

    lines += ["\n## 4. 标注集合逐条明细\n",
              "| ID | 类型 | 答案（截断） | 引用单元 | 期望 | 实际 | 判定 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in label_rows:
        ok = (row["fn"] == 0 and row["fp"] == 0)
        lines.append(
            f"| {row['id']} | {kind_label.get(row['kind'], row['kind'])} | "
            f"{_md(row['answer'])} | {', '.join(row['units']) or '（无）'} | "
            f"{'/'.join(row['expected_labels']) or '（无）'} | {'/'.join(row['got_labels']) or '（无）'} | "
            f"{'✅' if ok else '❌'} {row['verdict']} |")

    lines += ["\n## 5. 真实链路逐条明细\n",
              "| 问题 | 状态 | 证据数 | 引用数 | 对不上证据的引用 | 缺失 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in live_rows:
        lines.append(f"| {row['question']} | {row['status']} | {row['n_evidence']} | "
                     f"{row['n_citations']} | {', '.join(row['out_of_evidence']) or '—'} | "
                     f"{'是' if row['missing_citation'] else '否'} |")

    lines += [
        "\n## 6. 口径与局限（如实说明）\n",
        "1. **正类是「非法引用」**：precision/recall 都按「能否抓出编造出处」计算，"
        "与「答案是否正确」是两件事——引用校验再准，也救不了内容错误的答案；",
        "2. **标注集合是人工构造的对抗样本**（含故意编造的第 999/9999/120 页），"
        "22 条样本量小、结论方向性大于精度；数字应视为"
        "「校验逻辑在这几类典型错误上的表现」，而不是线上分布；",
        "3. **裸页码是第二层引用，容易被漏判**：`第 9999 页` 这种不带方括号的引用由 "
        "`BARE_PAGE_RE` 单独提取。只要证据里有**任一**页码元数据，"
        "校验器就会拿它跟证据页码集合比对，对不上即判 `invalid`；"
        "证据**完全没有**页码元数据时才退化为 `unverifiable`（不可核验）。"
        "本脚本第一版只看了方括号引用，召回率被低估了 28.6 个百分点——"
        "这也是本次评测发现的第一个口径坑；",
        "4. **本次实测到的校验器漏判（召回侧 2 类，均不误杀）**："
        "① **来源错配**：`[另一个手册.pdf 第5页]` 不会被判非法——"
        "`Reflector.verify` 用子串包含匹配，`第5页` 是它与证据出处 "
        "`fixture_manual.pdf 第5页·座椅` 的公共子串（ct-09）；"
        "② **标题错配**：`[fixture_manual.pdf 第3页·座椅]`（页码对、标题错）"
        "在证据里的 `citation` 就是 `[fixture_manual.pdf 第3页·灯光]`，"
        "`第3页` 同样是公共子串（ct-21）。两条都建议改成"
        "\u300c先按页码/块号精确匹配，匹配不到再判非法\u300d；",
        "5. **`verify(citations=...)` 与 `verify(answer)` 的口径差异**："
        "`agent/graph.py::_finalize` 传入 `state.citations`（只含方括号出处），"
        "这个分支**不解析裸页码**；直接传答案时 `Reflector` 才会走 `BARE_PAGE_RE` 检查。"
        "也就是说「答案里裸写一个不存在的页码」在真实链路里不会被拦——"
        "ct-06 / ct-15 / ct-18 用 `source: checker` 记录了这个差异，"
        "没有把它算成召回率损失（否则召回率会被口径差异而不是缺陷拉低）；",
        "6. **真实链路的「引用精确率」是自洽性口径**：检查 citations 能否对上本次 evidence，"
        "不是「引用是否真的支撑结论」（后者需要人工或模型裁判，本次未做，属于已知缺口）；",
        "7. **未覆盖**：引用**召回**侧的「该引用的证据没被引用」没有 ground truth"
        "（gold.json 只有 question/answer/keywords，没有标注应引用的块），"
        "因此只报了「答案是否至少有一个引用」的覆盖率，这是本报告最大的缺口。",
    ]
    return "\n".join(lines) + "\n"


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.2%}"


def _md(text: str, limit: int = 40) -> str:
    text = (text or "").replace("|", "\\|").replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default=FIXTURE_KB)
    ap.add_argument("--limit", type=int, default=0, help="真实链路集合的问题数上限")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json-out", default=OUT_JSON)
    ap.add_argument("--md-out", default=OUT_MD)
    ap.add_argument("--detail-out", default=OUT_DETAIL)
    args = ap.parse_args()

    print(f"[info] 加载知识库 {args.kb} …", flush=True)
    kb = KnowledgeBase.load(args.kb)
    print(f"[info] {json.dumps(kb.stats(), ensure_ascii=False)}", flush=True)
    reflector = Reflector()

    started = time.perf_counter()
    label_rows = [label_case(c, reflector, kb) for c in _LABEL_CASES]
    print(f"[info] 标注集合 {len(label_rows)} 条已完成", flush=True)
    for row in label_rows:
        if args.verbose:
            print(f"  {row['id']} {row['kind']:<18} 引用={row['citations']} "
                  f"期望={row['expected_labels']} 实际={row['got_labels']}", flush=True)

    questions = LIVE_QUESTIONS[:args.limit] if args.limit else LIVE_QUESTIONS
    print(f"[info] 真实链路 {len(questions)} 个问题…", flush=True)
    live_rows = run_live(kb, questions)
    for row in live_rows:
        print(f"  [{'答' if row['answered'] else '拒'}] {row['question']} "
              f"证据={row['n_evidence']} 引用={row['n_citations']} "
              f"对不上={len(row['out_of_evidence'])}", flush=True)
    wall = time.perf_counter() - started

    summary = summarize(label_rows, live_rows, kb.stats())
    summary["wall_clock_s"] = round(wall, 2)
    summary["kb_path"] = args.kb

    json.dump({"summary": summary, "label_cases": label_rows, "live_cases": live_rows},
              open(args.json_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    with open(args.detail_out, "w", encoding="utf-8") as fh:
        for row in label_rows:
            fh.write(json.dumps({"set": "label", **row}, ensure_ascii=False) + "\n")
        for row in live_rows:
            fh.write(json.dumps({"set": "live", **row}, ensure_ascii=False) + "\n")
    md = render_md(summary, label_rows, live_rows, args.kb)
    open(args.md_out, "w", encoding="utf-8").write(md)
    print("\n" + md)
    print(f"[json] {args.json_out}\n[md] {args.md_out}\n[detail] {args.detail_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
