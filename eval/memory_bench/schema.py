# -*- coding: utf-8 -*-
"""memory_bench：座舱长期记忆专用 benchmark 的样本 schema 与 JSONL IO。

设计对齐（调研结论的直接落地，出处见 docs/业界长期记忆方案调研.md）：
- **样本构造**：画像卡 → 跨会话对话（带时间戳与会话号）→ gold 事实标注
  （LongMemEval 的"证据语句 + 位置标注"思想：每条 gold 事实记录它出自哪一轮）；
- **题型**：五能力六题型（LongMemEval 五能力到座舱的映射）——
  IE 信息抽取 / MR 跨会话推理 / KU 知识更新 / TR 时间推理 / ABS 拒答（假前提）；
- **版本化**：JSONL 带版本号，指标对比必须同版本（防"偷偷换题"）。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

BENCH_VERSION = "v0.1"

QUESTION_TYPES = ("IE", "MR", "KU", "TR", "ABS")

#: 六题型的判定口径（judge 与检索层共用）
TYPE_NOTES = {
    "IE": "直接事实回忆：回答含 gold 即对",
    "MR": "跨会话推理：需要把工单/情景与画像拼起来回答",
    "KU": "知识更新：必须答最新值；回答同时含旧值和新值、只要新值正确也判对",
    "TR": "时间推理：事实演化顺序（旧值→新值）必须正确",
    "ABS": "拒答：记忆里没有该事实，必须明确表示不知道；任何具体编造都判错",
}


@dataclass
class Turn:
    ts: str                 # 日期 YYYY-MM-DD（时间戳：时间推理的依据）
    session: int            # 会话号（跨会话边界）
    speaker: str            # user | assistant
    text: str


@dataclass
class Question:
    qid: str
    qtype: str              # IE | MR | KU | TR | ABS
    text: str
    gold: str               # 期望答案要点（judge 的判定依据）
    note: str = ""


@dataclass
class Sample:
    sid: str
    profile_card: Dict[str, Any]              # 画像卡（生成对话的依据）
    turns: List[Turn] = field(default_factory=list)
    gold_facts: List[Dict[str, Any]] = field(default_factory=list)
    # gold_facts 元素：{key, value, source, turn}——turn 为该事实出自第几轮（位置标注）
    questions: List[Question] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"sid": self.sid, "profile_card": self.profile_card,
                "turns": [asdict(t) for t in self.turns],
                "gold_facts": self.gold_facts,
                "questions": [asdict(q) for q in self.questions]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sample":
        return cls(sid=data["sid"], profile_card=data["profile_card"],
                   turns=[Turn(**t) for t in data["turns"]],
                   gold_facts=list(data["gold_facts"]),
                   questions=[Question(**q) for q in data["questions"]])


def save_jsonl(samples: List[Sample], path: str, version: str = BENCH_VERSION) -> int:
    """写评测集（带版本头注释？JSONL 不支持注释——版本写进每行 meta 由 loader 校验）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            row = s.to_dict()
            row["bench_version"] = version
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(samples)


def load_jsonl(path: str) -> List[Sample]:
    samples: List[Sample] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            samples.append(Sample.from_dict(json.loads(line)))
    return samples


def stats(samples: List[Sample]) -> Dict[str, Any]:
    turns = sum(len(s.turns) for s in samples)
    questions = sum(len(s.questions) for s in samples)
    by_type: Dict[str, int] = {}
    for s in samples:
        for q in s.questions:
            by_type[q.qtype] = by_type.get(q.qtype, 0) + 1
    return {"samples": len(samples), "turns": turns, "questions": questions,
            "by_type": by_type, "sessions": sum(len({t.session for t in s.turns})
                                                for s in samples)}
