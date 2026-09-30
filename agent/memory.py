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

    def resolve(self, question: str) -> str:
        """把省略/指代问题补全成可检索的独立查询。"""
        q = (question or "").strip()
        if not self.needs_context(q) or not self.current_topic:
            return q
        if self.current_topic in q:
            return q
        return f"{self.current_topic} {q}"

    def system_context(self) -> str:
        lines = [f"车主档案：{self.profile.describe()}"]
        if self.current_topic:
            lines.append(f"当前话题：{self.current_topic}")
        history = self.history_text(4)
        if history:
            lines.append("最近对话：\n" + history)
        return "\n".join(lines)
