# -*- coding: utf-8 -*-
"""画像事实存储：SQLite 双时态表（借 Zep 的 t_valid / t_invalid）+ events 审计。

与 v1（agent/longterm.py）的关系
--------------------------------
v1 冻结为基线（对比矩阵 B 档），本模块是 v2 的存储层。差异：
1. **双时态**：每个事实有 t_valid（生效时间）与 t_invalid（失效时间，NULL=仍有效）。
   旧值被覆盖时**保留在库里、标 t_invalid**——支持"当时 vs 现在"的演化查询，
   而不是 v1 的"整条 overwrite 丢信息"（这是调研里 Zep Edge Invalidation 的直接落地）；
2. **来源与置信度列**：冲突消解（consolidate 层）需要按来源优先级裁决；
3. **审计**：沿用 v1 的 events 表模式（写入/覆盖/拒绝/失效全程留痕，座舱可审计要求）。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from agent.guardrails.policy import mask_sensitive
from agent.profile.schema import spec_for


@dataclass
class ProfileFact:
    user_id: str
    layer: str
    key: str
    value: str
    source: str
    confidence: float
    t_valid: float
    t_invalid: Optional[float]          # None = 仍有效（借 Zep 的事实有效区间）
    created_at: float
    updated_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id, "layer": self.layer, "key": self.key,
            "value": self.value, "source": self.source,
            "confidence": round(self.confidence, 3),
            "t_valid": self.t_valid, "t_invalid": self.t_invalid,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ProfileFact":
        return cls(user_id=str(row["user_id"]), layer=str(row["layer"]),
                   key=str(row["key"]), value=str(row["value"]),
                   source=str(row["source"]), confidence=float(row["confidence"]),
                   t_valid=float(row["t_valid"]),
                   t_invalid=None if row["t_invalid"] is None else float(row["t_invalid"]),
                   created_at=float(row["created_at"]), updated_at=float(row["updated_at"]))


class ProfileStore:
    """SQLite 支撑的画像事实库（双时态 + 审计）。"""

    def __init__(self, path: str = ":memory:", clock: Optional[Callable[[], float]] = None):
        self.path = path
        self._now = clock or time.time
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:                      # :memory: 不支持 WAL
            pass
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS profile_facts (
                user_id TEXT NOT NULL,
                layer TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                t_valid REAL NOT NULL,
                t_invalid REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (user_id, layer, key, t_valid)
            );
            CREATE INDEX IF NOT EXISTS idx_facts_current
                ON profile_facts(user_id, layer, key, t_valid DESC);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT, layer TEXT, key TEXT, action TEXT,
                old_value TEXT, new_value TEXT, reason TEXT, at REAL
            );
            """
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "ProfileStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ── 读 ──
    def current(self, user_id: str, key: str) -> Optional[ProfileFact]:
        """当前有效的事实（t_invalid IS NULL 的最新一条）。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM profile_facts WHERE user_id=? AND key=? "
                "AND t_invalid IS NULL ORDER BY t_valid DESC LIMIT 1",
                (user_id, key)).fetchone()
        return ProfileFact.from_row(row) if row else None

    def history(self, user_id: str, key: str, limit: int = 50) -> List[ProfileFact]:
        """演化史（含已失效的旧值）——"当时 vs 现在"的查询基础。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM profile_facts WHERE user_id=? AND key=? "
                "ORDER BY t_valid DESC LIMIT ?", (user_id, key, limit)).fetchall()
        return [ProfileFact.from_row(r) for r in rows]

    # ── 写（供 consolidate 层调用；业务代码不直接碰这三件套）──
    def insert(self, fact: ProfileFact) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO profile_facts (user_id, layer, key, value, source, confidence,"
                " t_valid, t_invalid, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (fact.user_id, fact.layer, fact.key, fact.value, fact.source,
                 float(fact.confidence), fact.t_valid, fact.t_invalid,
                 fact.created_at, fact.updated_at))

    def invalidate(self, fact: ProfileFact, reason: str, at: float) -> None:
        """把一条仍有效的事实标 t_invalid（保留历史，不物理删除）。"""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE profile_facts SET t_invalid=?, updated_at=? "
                "WHERE user_id=? AND key=? AND t_valid=? AND t_invalid IS NULL",
                (at, at, fact.user_id, fact.key, fact.t_valid))
        self._event(fact.user_id, fact.layer, fact.key, "invalidate",
                    fact.value, None, reason, at)

    def bump(self, fact: ProfileFact, confidence: float, reason: str, at: float) -> None:
        """置信度回升（同一事实被再次确认——复述是强信号）。"""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE profile_facts SET confidence=?, updated_at=? "
                "WHERE user_id=? AND key=? AND t_valid=?",
                (float(confidence), at, fact.user_id, fact.key, fact.t_valid))
        self._event(fact.user_id, fact.layer, fact.key, "reinforce",
                    fact.value, fact.value, reason, at)

    def _event(self, user_id: str, layer: str, key: str, action: str,
               old_value: Optional[str], new_value: Optional[str],
               reason: str, at: float) -> None:
        self._conn.execute(
            "INSERT INTO events (user_id, layer, key, action, old_value, new_value,"
            " reason, at) VALUES (?,?,?,?,?,?,?,?)",
            (user_id, layer, key, action, old_value, new_value, reason, at))

    # ── 观测 ──
    def stats(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            facts = self._conn.execute(
                "SELECT COUNT(*) AS n FROM profile_facts WHERE user_id=? "
                "AND t_invalid IS NULL", (user_id,)).fetchone()
            events = self._conn.execute(
                "SELECT COUNT(*) AS n FROM events WHERE user_id=?", (user_id,)).fetchone()
        return {"active_facts": int(facts["n"]), "events": int(events["n"])}

    def events(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT layer, key, action, old_value, new_value, reason, at FROM events"
                " WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()
        return [dict(r) for r in rows]


def _append_value(current: str, new_item: str) -> str:
    """append_only 字段：去重追加（保养轨迹只增不改）。"""
    try:
        items = json.loads(current)
        if not isinstance(items, list):
            items = [current]
    except (json.JSONDecodeError, TypeError):
        items = [current] if current else []
    if new_item not in items:
        items.append(new_item)
    return json.dumps(items, ensure_ascii=False)


def make_fact(user_id: str, key: str, value: str, source: str,
              confidence: float, now: float) -> ProfileFact:
    """按 schema 组装一条事实（层与默认置信度来自 FieldSpec）。"""
    spec = spec_for(key)
    return ProfileFact(user_id=user_id, layer=spec.layer, key=key,
                       value=mask_sensitive(value)[:200], source=source,
                       confidence=float(confidence), t_valid=now, t_invalid=None,
                       created_at=now, updated_at=now)
