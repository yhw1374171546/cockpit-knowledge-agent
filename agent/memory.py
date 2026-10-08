# -*- coding: utf-8 -*-
"""会话记忆层：多轮上下文、车辆档案、指代消解。

解决原 RAG 链路「单轮无状态」的问题：
- 车主问「那它怎么关？」这类省略/指代问题，原链路会直接检索失败；
- 车主的车型、里程、所在城市等档案信息无法注入，导致每次都要重复说明。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PRONOUNS = ("它", "他", "这个", "那个", "这样", "该功能", "此功能", "上述", "刚才", "那")

# 「强指代」：一旦出现，几乎一定在回指上文话题（「那」不算，见下）
STRONG_PRONOUNS = ("它", "他", "她", "这个", "那个", "这样", "那样",
                   "该功能", "此功能", "上述", "刚才")

# 延续标记：出现即认为本轮在**延续**上文（而不是另起话题）。
# 「这/此/本」是指示代词（「这车的轮胎规格」＝指上一轮提到的车）；
# 「我/咱」把问题锚定在用户自身情境上，通常是追问。
CONTINUATION_MARKERS = STRONG_PRONOUNS + ("这", "此", "本", "我", "咱", "我们", "咱们", "我的")

# 场景/条件词：这类词是对上文的**限定**（「那晚上也有效吗」），不构成新话题
SCENARIO_WORDS = ("晚上", "白天", "夜间", "夜里", "早上", "高速", "高速上", "市区", "堵车",
                  "雨天", "下雨", "雪天", "下雪", "冬天", "夏天", "冷车", "热车",
                  "上坡", "下坡", "停车", "行驶中", "倒车", "充电时", "长途")

# 功能词/疑问词/语气词：抽"实词"时先去掉它们
FUNCTION_WORDS = (
    "怎么", "如何", "什么", "为什么", "是否", "可以", "能不能", "有没有", "要不要",
    "请问", "一下", "的话", "怎么办", "咋弄", "咋办", "多少", "哪个", "哪些",
    "该", "那", "这个", "那个", "它", "他", "她", "此", "上述", "刚才",
    "的", "了", "吗", "呢", "吧", "啊", "呀", "哦", "嘛", "算", "呢",
    "还有", "有", "是", "在", "要", "能", "会", "就", "也", "都", "还", "又", "再",
    "我", "你", "咱", "自己", "别人", "个", "先", "后",
    "方式", "方法", "办法", "意思", "情况", "时候", "问题", "回事", "用途", "作用",
)


def _tokenize(text: str) -> List[str]:
    """分词（有 jieba 用 jieba，否则退化为按标点切分）。

    为什么要分词而不是按字粘接：按字粘接会造出「先充电」这种假实词，
    进而把「那我先充个电可以吗」误判成话题切换（实际是追问）。
    """
    raw = (text or "").strip()
    if not raw:
        return []
    try:
        import jieba  # noqa: PLC0415  延迟导入：没有 jieba 时也能跑

        return [t for t in jieba.cut(raw) if t.strip()]
    except Exception:                                          # noqa: BLE001
        return [t for t in re.split(r"[^\u4e00-\u9fffA-Za-z0-9]+", raw) if t.strip()]


@dataclass
class VehicleProfile:
    model: Optional[str] = None
    mileage_km: Optional[int] = None
    vin: Optional[str] = None
    city: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def describe(self) -> str:
        parts = []
        if self.model:
            parts.append(f"车型 {self.model}")
        if self.mileage_km is not None:
            parts.append(f"里程 {self.mileage_km} km")
        if self.city:
            parts.append(f"城市 {self.city}")
        return "；".join(parts) if parts else "（暂无车辆档案）"


@dataclass
class Turn:
    role: str
    content: str
    tool_names: List[str] = field(default_factory=list)


class ConversationMemory:
    """短窗口记忆 + 轻量槽位抽取 + 指代消解。"""

    def __init__(self, window: int = 6, profile: Optional[VehicleProfile] = None):
        self.window = window
        self.turns: List[Turn] = []
        self.profile = profile or VehicleProfile()
        self.current_topic: Optional[str] = None

    # ── 写入 ──
    def add_user(self, text: str) -> None:
        text = (text or "").strip()
        self.turns.append(Turn("user", text))
        self._extract_slots(text)

    def add_assistant(self, text: str, tool_names: Optional[List[str]] = None) -> None:
        self.turns.append(Turn("assistant", text or "", tool_names or []))

    def set_topic(self, topic: str) -> None:
        self.current_topic = (topic or "").strip() or None

    # ── 读取 ──
    def recent(self, n: Optional[int] = None) -> List[Turn]:
        return self.turns[-(n or self.window):]

    def history_text(self, n: Optional[int] = None) -> str:
        return "\n".join(f"{t.role}: {t.content}" for t in self.recent(n))

    # ── 槽位抽取 ──
    def _extract_slots(self, text: str) -> None:
        m = re.search(r"(领克|lynk\s*&?\s*co)\s*(0?[1-9])", text, re.I)
        if m:
            self.profile.model = f"领克{int(m.group(2)):02d}"
        m = re.search(r"(\d+(?:\.\d+)?)\s*(万)?\s*(?:公里|km|KM|千米)", text)
        if m:
            value = float(m.group(1)) * (10000 if m.group(2) else 1)
            self.profile.mileage_km = int(value)
        for city in ("上海", "北京", "深圳", "广州", "杭州", "成都", "合肥"):
            if city in text:
                self.profile.city = city
                break

    # ── 指代消解 ──
    def needs_context(self, question: str) -> bool:
        """问题过短或含指代词 → 需要借助上文补全。自包含的问题必须原样检索。"""
        q = (question or "").strip()
        if len(q) <= 6:
            return True
        return any(p in q for p in PRONOUNS) and len(q) <= 14

    def topic_from_history(self) -> str:
        """从最近一轮用户提问推断话题（`current_topic` 未被显式设置时的兜底）。

        修复的缺陷：`needs_context()` 判定"需要上下文"，但 `resolve()` 依赖
        `self.current_topic`——而它在纯离线链路里**从未被设置**，于是指代消解静默失效
        （解析结果等于原句，检索不到 → 拒答）。二者是隐性契约，一旦一端没人调用就会
        出现"看起来支持多轮、实际不支持"的假象。
        """
        for turn in reversed(self.turns):
            if turn.role == "user" and turn.content.strip():
                return turn.content.strip()
        return self.current_topic

    def resolve(self, question: str) -> str:
        """把省略/指代问题补全成可检索的独立查询。

        这里修掉了一个真实缺陷（评测集 `inherit_forbid` 标出的「错误继承」）：
        旧逻辑只要「含指代词且句子短」就把上一轮话题拼进来，
        于是话题切换轮（如上一轮聊「座椅加热关闭」、这一轮问「那电动尾门怎么打开」）
        会被拼成 `座椅加热怎么关闭 那电动尾门怎么打开` —— 新话题的检索被旧话题词污染，
        用户拿到的是一个自信但答错话题的回答。

        新逻辑区分两种情况：
        - **真指代**（出现强指代词，或本轮没有自带实词）→ 拼上话题，帮助检索；
        - **话题切换**（没有强指代词，且本轮自带了一个 ≥3 字的、与话题不重叠的实词）
          → 视为换了话题，**原样检索**。
        「那」被刻意排除在强指代之外：它既可以是代词（「那它怎么关闭」），
        也可以是话语标记（「那电动尾门怎么打开」）。
        """
        q = (question or "").strip()
        topic = self.current_topic or self.topic_from_history()
        if not self.needs_context(q) or not topic:
            return q
        if topic in q:
            return q
        if self.switches_topic(q, topic):
            return q
        return f"{topic} {q}"

    # ── 实词与话题切换判定 ──
    @staticmethod
    def content_phrases(text: str) -> List[str]:
        """抽实词短语：分词 → 去功能词/指代词/单字 → 合并相邻实词（长度 ≥3 才算）。"""
        kept: List[Optional[str]] = []
        for token in _tokenize(text):
            tk = token.strip()
            if (not tk or tk in FUNCTION_WORDS or tk in CONTINUATION_MARKERS
                    or len(tk) < 2):
                kept.append(None)                 # 断开，避免跨功能词粘接
            else:
                kept.append(tk)
        phrases: List[str] = []
        current = ""
        for item in kept:
            if item is None:
                if current:
                    phrases.append(current)
                    current = ""
            else:
                current += item
        if current:
            phrases.append(current)
        return [p for p in phrases if len(p) >= 3]

    @classmethod
    def content_words(cls, text: str) -> List[str]:
        """保留旧接口（部分调用方按"词"使用）。"""
        return cls.content_phrases(text)

    def switches_topic(self, question: str, topic: str) -> bool:
        """本轮是否引入了**新话题**（而不是在追问旧话题）。

        判定条件（都满足才算切换）：
        1. 没有任何**延续标记**（它/这个/该功能/这/此/本/我…）——「那」不算，
           因为它既可以是代词（「那它怎么关闭」）也可以是话语标记（「那电动尾门怎么打开」）；
        2. 本轮自带一个与话题**不重叠**的实词短语（长度 ≥3）；
        3. 该短语不是单纯的**场景/条件词**（「那晚上也有效吗」是对上文的限定，不是新话题）。
        """
        q = question or ""
        if any(marker in q for marker in CONTINUATION_MARKERS):
            return False
        topic_blob = "".join(self.content_phrases(topic)) + (topic or "")
        # 词级重叠也算延续：避免「标准胎压」vs「胎压报警」这种部分重叠被误判为切换
        topic_tokens = {t for t in _tokenize(topic) if len(t) >= 2}
        for phrase in self.content_phrases(q):
            if phrase in topic_blob:
                continue                          # 与话题重叠 → 延续
            shared = [t for t in _tokenize(phrase) if len(t) >= 2 and t in topic_tokens]
            if shared:
                continue                          # 有共同实词 → 延续
            if any(word in SCENARIO_WORDS for word in _tokenize(phrase)):
                continue                          # 只是限定条件 → 延续
            return True
        return False

    def system_context(self) -> str:
        lines = [f"车主档案：{self.profile.describe()}"]
        if self.current_topic:
            lines.append(f"当前话题：{self.current_topic}")
        history = self.history_text(4)
        if history:
            lines.append("最近对话：\n" + history)
        return "\n".join(lines)
