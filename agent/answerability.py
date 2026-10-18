# -*- coding: utf-8 -*-
"""可回答性校验（answerability）：接地只保证"忠于证据"，不保证"回答了问题"。

为什么需要这一层
----------------
线上真实案例：「**车内有冰箱吗**」——检索命中第 129 页 12V 电源那段：

    车载12V电源通过 12 V 电源插座，可以给电子设备（例如：音乐播放器、车载冰箱和移动手机）充电。

- **检索没错**：`车载冰箱` 在 9,301 块里**只出现在这一处**；
- **接地没错**：答案就是原文，忠实度 100%；
- **但它没回答问题**：原文说的是"**可以给**车载冰箱**充电**"，即冰箱是**用户自带的受电设备**，
  而不是"本车**配备**冰箱"。

现有三层护栏全部放行（注入检测 / 证据门控 / 接地校验），因为它们回答的是
"内容里有没有攻击""有没有证据""答案是否忠于证据"——**没有一层回答"证据是否回答了问题"**。

一句话：**相关 ≠ 可回答；忠实 ≠ 相关。**

核心判据：X 是"配置物"还是"受电对象"
------------------------------------
| 证据片段 | 谓词类别 | X 的角色 | 能否回答"有 X 吗" |
| --- | --- | --- | --- |
| 本车**配备**车载冰箱 | CONFIG | 配置宾语 | ✅ |
| 后排**设有**独立空调 | CONFIG | 配置宾语 | ✅ |
| 12V 插座可以**给**车载冰箱**充电** | CONSUME | 受电对象 | ❌ |
| 车辆可以作为电源**对**微波炉**供电** | CONSUME | 受益者 | ❌ |
| 请勿将**手机**放在充电板上 | WARNING | 警示对象 | ❌ |
| 例如：音乐播放器、**车载冰箱** | EXAMPLE | 举例项 | ❌ |

⚠️ 注意：**不能只看"例如/比如"这种引导词**。实测「车内有微波炉吗」的证据是
"车辆可以作为电源对照明、小功率烤箱、微波炉等家用电器供电"——**没有引导词**，
所以第一版靠引导词的规则漏掉了。必须判**谓词语义**，不能判**引导词**。

覆盖的问句类型
--------------
1. **存在性**（有没有 X / 是否配备 X）→ 需要 CONFIG 语境，否则给限定回答；
2. **能力**（支持 X 吗 / 能不能 X）→ 同上，但允许"支持/具备"类谓词；
3. **范围**（全系都有吗 / 所有车型都支持吗）→ 证据含"选装/高配/部分"则必须限定；
4. **前提式**（为什么 X 之后可以 Y）→ 只做**标记**不改答案（前提是否成立需要更强的蕴含判定，
   当前只如实标注，不假装解决——见 `premise_flagged`）。

刻意不做的
----------
- 不引入 NLI 模型（无 GPU、也不想为一个规则够用的场景加依赖）；
- 不做完整的语义角色标注，只做**谓词窗口内的词表分类**——覆盖手册语料足够，
  但**边界必须说清**：这是启发式，不是蕴含推理。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ── 谓词词表：按"X 在句子里的角色"分类 ──

# CONFIG：X 是被配置/提供的东西 → 可以回答"有"
CONFIG_WORDS = (
    "配备", "标配", "选装", "选配", "设有", "装有", "配有", "置有", "带有",
    "提供", "具备", "支持", "内含", "内置", "位于", "安装在", "布置在",
    "可享受", "可选",
)

# CONSUME：X 是受电/受益/承载对象 → **不能**回答"有"
CONSUME_PATTERNS = (
    r"给[^，。；]{0,12}?(?:充电|供电|加热|降温|使用|连接)",
    r"为[^，。；]{0,12}?(?:充电|供电|提供|加热)",
    r"对[^，。；]{0,14}?(?:充电|供电|放电)",
    r"(?:可以|可|能)(?:给|为|对)",
    r"连接(?:到|至)?",
    r"放置在", r"放入", r"插入", r"接入",
    r"作为电源",
)

# WARNING：警示语气，不是配置声明
WARNING_WORDS = ("请勿", "切勿", "警告", "注意", "严禁", "不得", "禁止")

# EXAMPLE：举例引导（辅助信号，**单独不可靠**，只用于降权）
EXAMPLE_LEADS = ("例如", "比如", "诸如", "如：", "如:", "像是", "像")

# SCOPE：范围限定词（问句侧 vs 证据侧）
SCOPE_ALL = ("全系", "所有车型", "全部车型", "每个配置", "都", "均", "一律")
SCOPE_LIMITED = ("选装", "选配", "高配", "低配", "部分车型", "某些车型", "视配置",
                 "依配置", "仅限", "仅", "取决于")

# ── 问句类型 ──
Q_EXISTENCE = "existence"
Q_CAPABILITY = "capability"
Q_SCOPE = "scope"
Q_PREMISE = "premise"
Q_OPERATIONAL = "operational"        # 怎么用/怎么关 → 不需要存在性判定

# ── 判定结果 ──
ANSWERABLE = "answerable"
NEEDS_QUALIFICATION = "needs_qualification"
NOT_ANSWERABLE = "not_answerable"

# 存在性问句的典型问法（顺序重要：先匹配更具体的）
_EXISTENCE_PATTERNS = (
    r"(?:本车|这款车|这车|车辆|车内|车上|车里)?\s*(?:有没有|有无|是否有)\s*(?P<entity>[^吗嘛?？，。]{1,18})",
    r"(?:本车|这款车|这车|车辆)?\s*(?:有)\s*(?P<entity>[^吗嘛?？，。]{1,18})\s*(?:吗|嘛)",
    r"(?:是否)?(?:配备|标配|选装|选配|装有|设有|带有)\s*(?P<entity>[^吗嘛?？，。]{1,18})",
)
_CAPABILITY_PATTERNS = (
    r"(?:是否)?(?:支持|可以|能不能|能|能否)\s*(?P<entity>[^吗嘛?？，。]{1,24})",
)
_SCOPE_PATTERNS = (
    r"(?P<scope>全系|所有车型|全部车型|每个配置|都|均)",
    r"(?P<entity>[^吗嘛?？，。]{1,24})(?:吗|嘛)",
)
_PREMISE_PATTERNS = (
    r"为什么[^，。?？]{0,20}(?:可以|能|会|要)",
    r"既然[^，。?？]{0,20}(?:那|为什么)",
)

# 问句里要剥掉的口语成分（否则"冰箱啊"会被当成实体）
_STRIP_TAIL = re.compile(r"[啊呀呢吧哈嘛吗么？?。，,！!\s]+$")
_STRIP_HEAD = re.compile(r"^(?:本车|这车|车辆|车内|车上|车里|前排|后排)")


@dataclass
class PredicateHit:
    """目标实体在证据里的**一次出现**及其谓词上下文判定。"""
    role: str                 # config | consume | warning | example | neutral
    snippet: str              # 谓词窗口（便于解释"为什么这么判"）
    citation: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "snippet": self.snippet, "citation": self.citation}


@dataclass
class AnswerabilityResult:
    verdict: str                              # answerable | needs_qualification | not_answerable
    question_type: str                        # existence | capability | scope | premise | operational
    entity: str = ""
    reason: str = ""
    hits: List[PredicateHit] = field(default_factory=list)
    has_related_evidence: bool = False         # 是否检索到了"提到它"的证据（决定能否给限定回答）
    evidence_text: str = ""                    # 供限定回答引用（被提到的那句）
    evidence_citation: str = ""
    premise_flagged: bool = False              # 前提式问句：只标记，不改答案
    scope_mismatch: bool = False

    @property
    def ok(self) -> bool:
        return self.verdict == ANSWERABLE

    @property
    def needs_qualification(self) -> bool:
        return self.verdict == NEEDS_QUALIFICATION

    def to_dict(self) -> Dict[str, object]:
        return {"verdict": self.verdict, "question_type": self.question_type,
                "entity": self.entity, "reason": self.reason,
                "has_related_evidence": self.has_related_evidence,
                "evidence_citation": self.evidence_citation,
                "premise_flagged": self.premise_flagged,
                "scope_mismatch": self.scope_mismatch,
                "hits": [h.to_dict() for h in self.hits[:4]]}


# ── 证据适配：既支持 Evidence 对象，也支持 dict / 纯字符串（便于单测） ──

def _text_of(item: object) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("text", ""))
    return str(getattr(item, "text", ""))


def _citation_of(item: object) -> str:
    if isinstance(item, dict):
        return str(item.get("citation", ""))
    return str(getattr(item, "citation", ""))


# ── 问句解析 ──

def classify_question(question: str) -> Tuple[str, str, bool]:
    """返回 (问句类型, 目标实体, 是否是范围全域问法)。"""
    q = (question or "").strip()
    if not q:
        return Q_OPERATIONAL, "", False

    # 前提式优先：它问的是"为什么/既然"，不是"有没有"
    for pat in _PREMISE_PATTERNS:
        if re.search(pat, q):
            return Q_PREMISE, "", False

    wants_all = any(word in q for word in SCOPE_ALL)

    # 操作类（怎么/如何/步骤）优先排除：这类问题不需要存在性判定
    if re.search(r"(怎么|怎样|如何|步骤|方法|设置|打开|关闭|调节|取消|重置)", q) \
            and not re.search(r"(有没有|有无|是否有|支持|配备)", q):
        return Q_OPERATIONAL, "", False

    for pat in _EXISTENCE_PATTERNS:
        m = re.search(pat, q)
        if m:
            entity = _clean_entity(m.group("entity"))
            if entity:
                return (Q_SCOPE if wants_all else Q_EXISTENCE), entity, wants_all

    for pat in _CAPABILITY_PATTERNS:
        m = re.search(pat, q)
        if m:
            entity = _clean_entity(m.group("entity"))
            if entity:
                return (Q_SCOPE if wants_all else Q_CAPABILITY), entity, wants_all

    return Q_OPERATIONAL, "", False


def _clean_entity(raw: str) -> str:
    text = _STRIP_TAIL.sub("", (raw or "").strip())
    text = _STRIP_HEAD.sub("", text)
    # 去掉动词残留（"支持座椅通风" → "座椅通风" 已被 pattern 吃掉，这里兜底）
    text = re.sub(r"^(?:有|是|带|配|装|设)", "", text)
    return text.strip()


# ── 谓词判定 ──

def judge_occurrence(text: str, entity: str, citation: str = "",
                     window: int = 24) -> Optional[PredicateHit]:
    """判定实体在文本中**首次出现**处的谓词角色。"""
    pos = text.find(entity)
    if pos < 0:
        return None
    before = text[max(0, pos - window):pos]
    after = text[pos + len(entity): pos + len(entity) + window]
    around = before + entity + after

    # 顺序：CONFIG 最强（明确配置声明），其次 CONSUME/WARNING，最后 EXAMPLE 只作降权
    if any(word in before for word in CONFIG_WORDS):
        return PredicateHit("config", (before[-12:] + entity), citation)
    for pattern in CONSUME_PATTERNS:
        if re.search(pattern, around):
            return PredicateHit("consume", (before[-12:] + entity + after[:10]), citation)
    if any(word in around for word in WARNING_WORDS):
        return PredicateHit("warning", (before[-12:] + entity), citation)
    if any(lead in before for lead in EXAMPLE_LEADS):
        return PredicateHit("example", (before[-12:] + entity), citation)
    return PredicateHit("neutral", (before[-12:] + entity), citation)


def _pick_evidence_sentence(text: str, entity: str) -> str:
    """从证据里挑出**包含实体**的那一句，供限定回答引用（不要整段丢给用户）。"""
    # 先按句末标点切，再退回按逗号
    for sep in ("。", "；", "\n"):
        for piece in text.split(sep):
            if entity in piece:
                sentence = piece.strip()
                if len(sentence) > 8:
                    return sentence if len(sentence) <= 90 else sentence[:88] + "…"
    pos = text.find(entity)
    start = max(0, pos - 30)
    snippet = text[start: pos + len(entity) + 30].strip()
    return snippet if len(snippet) <= 90 else snippet[:88] + "…"


# ── 主入口 ──

def check_answerability(question: str, evidence: Sequence[object]) -> AnswerabilityResult:
    """判断"证据是否回答了问题"。

    - `answerable`：证据里有明确的支持性表述，正常回答；
    - `needs_qualification`：证据提到了它，但只是**举例/受电/警示**语境 → 必须给限定回答；
    - `not_answerable`：证据里根本没有它 → 交给拒答逻辑（本模块不改这种情况）。
    """
    qtype, entity, wants_all = classify_question(question)
    if qtype == Q_OPERATIONAL:
        return AnswerabilityResult(ANSWERABLE, qtype, entity,
                                   reason="非存在性/能力问句，不做可回答性约束")
    if qtype == Q_PREMISE:
        return AnswerabilityResult(NEEDS_QUALIFICATION, qtype, entity,
                                   reason="前提式问句：前提是否成立需更强判定，当前只做标记",
                                   premise_flagged=True)

    texts = [(_text_of(item), _citation_of(item)) for item in evidence]
    if not any(entity in text for text, _ in texts):
        return AnswerabilityResult(NOT_ANSWERABLE, qtype, entity,
                                   reason=f"证据中没有提到「{entity}」",
                                   has_related_evidence=False)

    hits: List[PredicateHit] = []
    evidence_text = ""
    evidence_citation = ""
    for text, citation in texts:
        hit = judge_occurrence(text, entity, citation)
        if hit is None:
            continue
        hits.append(hit)
        if not evidence_text:
            evidence_text = _pick_evidence_sentence(text, entity)
            evidence_citation = citation
        if hit.role == "config":
            break                      # 出现明确配置声明即可判定可回答

    roles = {h.role for h in hits}
    config_hits = [h for h in hits if h.role == "config"]

    # 范围一致性：问"全系都有吗"、证据却写"选装/高配/部分" → 必须限定
    scope_mismatch = False
    if wants_all:
        joined = " ".join(text for text, _ in texts)
        scope_mismatch = (not config_hits) or any(w in joined for w in SCOPE_LIMITED)

    if config_hits and not scope_mismatch:
        return AnswerabilityResult(ANSWERABLE, qtype, entity,
                                   reason=f"证据中有明确配置声明（{config_hits[0].snippet}）",
                                   hits=hits, has_related_evidence=True,
                                   evidence_text=evidence_text,
                                   evidence_citation=evidence_citation)

    if scope_mismatch:
        reason = f"问句要求「全系/都」，证据只覆盖部分配置（如{'/'.join(w for w in SCOPE_LIMITED if w in ' '.join(t for t, _ in texts)) or '选装'}）"
    elif roles <= {"example"}:
        reason = f"「{entity}」只出现在举例语境，不能作为配置依据"
    elif "consume" in roles:
        reason = f"「{entity}」是受电/承载对象（{hits[0].snippet}），不是本车配置"
    elif "warning" in roles:
        reason = f"「{entity}」只出现在警示语境，不是配置声明"
    else:
        reason = f"证据提到「{entity}」但没有配置性表述"

    return AnswerabilityResult(NEEDS_QUALIFICATION, qtype, entity, reason=reason,
                               hits=hits, has_related_evidence=True,
                               evidence_text=evidence_text,
                               evidence_citation=evidence_citation,
                               scope_mismatch=scope_mismatch)


# ── 限定回答模板 ──
#
# 设计原则：**不要简单拒答**。用户问"有没有冰箱"，最有用的回答是
#   ① 明确告诉他没有依据说"有"；② 把他真正看到的那句原文给他，并说明它的含义。
# 拒答只会让用户觉得助手什么都不知道。

def build_qualified_answer(result: AnswerabilityResult) -> str:
    """把"不可直接回答"变成**如实限定**的回答（含出处）。"""
    entity = result.entity or "该配置"
    cite = f"（{result.evidence_citation}）" if result.evidence_citation else ""

    if result.scope_mismatch:
        head = f"手册没有说明「{entity}」是全系标准配置。"
    elif result.question_type == Q_CAPABILITY:
        head = f"手册没有明确说明本车支持「{entity}」。"
    elif result.question_type == Q_PREMISE:
        head = "这个问题基于一个前提，手册里没有找到该前提的依据，因此下面的信息仅供参考。"
    else:
        head = f"手册没有说明本车配备「{entity}」。"

    if result.evidence_text:
        body = f"手册里相关的一句是{cite}：「{result.evidence_text}」"
        if result.scope_mismatch:
            # 范围类：证据自己在说"选装/高配/部分"，直接点出来比"没说明"更有用
            limited = [w for w in SCOPE_LIMITED if w in result.evidence_text]
            hint = "、".join(limited[:2]) if limited else "部分配置"
            body += f"——这是{hint}配置，不是全系标准配置。"
        elif result.hits:
            role = result.hits[0].role
            if role == "consume":
                body += "——这是在说它可以被车载电源供电，并不代表车内配有该设备。"
            elif role == "example":
                body += "——这只是举例，并不是配置说明。"
            elif role == "warning":
                body += "——这是安全提示，不是配置说明。"
            else:
                body += "——这句话没有直接说明是否配备。"
        return head + body

    return head + "建议以车辆配置表或经销商信息为准。"
