# -*- coding: utf-8 -*-
"""反思层：引用校验 + 证据充分性判定 + 拒答决策。

座舱场景的核心风险是「自信地编造」。原实现只在 Prompt 里写「无法回答就说无答案」，
属于**软约束**；本模块把它变成**硬校验**：
1. 把答案拆成句子，逐句检查是否有检索证据支撑（字符级覆盖 + 关键实体命中）；
2. 校验引用是否真的来自本次检索结果（防止模型编页码）；
3. 证据不足时给出明确动作：重检索（一次）/ 降级为「无答案」。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

NO_ANSWER = "无答案"
# 只匹配「像引用」的方括号内容（含页码 / 块号 / 文件名），避免把 JSON 数组括号误判为引用
CITATION_RE = re.compile(r"\[([^\[\]]{1,80}?(?:第\s*\d+\s*页|块#\d+|\.pdf|手册)[^\[\]]{0,40}?)\]")
# 无括号的裸页码引用，例如「第 9999 页」
BARE_PAGE_RE = re.compile(r"第\s*(\d{1,4})\s*页")


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"[。！？!?\n]", text or "")
    return [p.strip() for p in parts if len(p.strip()) >= 4]


def char_coverage(query: str, text: str, stop: Optional[set] = None) -> float:
    """query 中实词字符被 text 覆盖的比例（轻量、无需模型）。"""
    stop = stop or set("的了吗呢么怎如何是有什么可以请问一下我你他这那个哪些为对能会要怎样也就都很")
    chars = {c for c in query if "\u4e00" <= c <= "\u9fff" and c not in stop}
    if not chars:
        chars = {c for c in query if "\u4e00" <= c <= "\u9fff"}
    if not chars:
        return 1.0
    return sum(1 for c in chars if c in text) / len(chars)


@dataclass
class ReflectionResult:
    verdict: str                       # grounded | partial | ungrounded
    grounded_ratio: float
    unsupported_sentences: List[str] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    unverifiable_citations: List[str] = field(default_factory=list)
    next_action: str = "accept"        # accept | retry | refuse
    detail: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {"verdict": self.verdict, "grounded_ratio": round(self.grounded_ratio, 4),
                "unsupported": self.unsupported_sentences,
                "invalid_citations": self.invalid_citations,
                "unverifiable_citations": self.unverifiable_citations,
                "next_action": self.next_action, "detail": self.detail}


class Reflector:
    def __init__(self, min_grounded_ratio: float = 0.6, min_sentence_support: float = 0.55,
                 evidence_overlap_threshold: float = 0.12):
        self.min_grounded_ratio = min_grounded_ratio
        self.min_sentence_support = min_sentence_support
        self.evidence_overlap_threshold = evidence_overlap_threshold

    # ── 证据充分性（回答前） ──
    def question_coverage(self, question: str, evidence_text: str) -> float:
        """问题实词被单条证据覆盖的比例（硬门控使用的核心指标）。"""
        return char_coverage(question, evidence_text or "")

    def evidence_sufficient(self, question: str, evidence_texts: Sequence[str],
                            threshold: Optional[float] = None) -> bool:
        if not evidence_texts:
            return False
        limit = self.evidence_overlap_threshold if threshold is None else threshold
        best = max(char_coverage(question, t) for t in evidence_texts)
        return best >= limit

    # ── 引用校验 + 答案接地校验（回答后） ──
    def verify(self, answer: str, evidence: Sequence[Dict], question: str = "",
               citations: Optional[Sequence[str]] = None) -> ReflectionResult:
        if not answer or answer.strip() == NO_ANSWER:
            return ReflectionResult("ungrounded", 0.0, [], [], "refuse",
                                    {"reason": "answer_is_no_answer"})

        evidence_texts = [e.get("text", "") for e in evidence if e.get("text")]
        joined = "\n".join(evidence_texts)

        sentences = split_sentences(answer)
        unsupported = []
        for sent in sentences:
            support = max((char_coverage(sent, t) for t in evidence_texts), default=0.0)
            if support < self.min_sentence_support:
                unsupported.append(sent)
        ratio = 1.0 - (len(unsupported) / len(sentences)) if sentences else 0.0

        # 引用有效性：引用的出处必须能在本次证据里找到
        valid_cites = set()
        known_pages = set()
        for e in evidence:
            valid_cites.add(e.get("citation", ""))
            if e.get("page") is not None:
                known_pages.add(int(e["page"]))
                valid_cites.add(f"第{e['page']}页")

        invalid: List[str] = []
        unverifiable: List[str] = []
        for cite in (citations if citations is not None else CITATION_RE.findall(answer)):
            if cite and not any(cite in v or v in cite for v in valid_cites):
                invalid.append(cite)

        # 无括号的裸页码（如「第9999页」）：有页码元数据时可判定真伪，无元数据时只能标记为不可核验
        for raw_page in BARE_PAGE_RE.findall(answer or ""):
            page_no = int(raw_page)
            label = f"第{page_no}页"
            if not known_pages:
                if label not in unverifiable:
                    unverifiable.append(label)
            elif page_no not in known_pages:
                if label not in invalid:
                    invalid.append(label)

        if ratio >= self.min_grounded_ratio and not invalid:
            verdict, action = "grounded", "accept"
        elif ratio >= 0.34 and not invalid:
            verdict, action = "partial", "retry"
        else:
            verdict, action = "ungrounded", "refuse"

        return ReflectionResult(verdict, ratio, unsupported, invalid, unverifiable,
                                action,
                                {"n_sentences": len(sentences),
                                 "n_evidence": len(evidence_texts),
                                 "pages_known": sorted(known_pages),
                                 "question_coverage": round(char_coverage(question, joined), 4)
                                 if question else None})

    # ── 拒答决策（软/硬约束结合） ──
    def should_refuse(self, question: str, evidence: Sequence[Dict],
                      reflection: Optional[ReflectionResult] = None,
                      retrieval_score: Optional[float] = None,
                      score_threshold: Optional[float] = None) -> bool:
        if not evidence:
            return True
        if score_threshold is not None and retrieval_score is not None \
                and retrieval_score < score_threshold:
            return True
        if reflection is not None and reflection.verdict == "ungrounded":
            return True
        return False
