# -*- coding: utf-8 -*-
"""forget：画像事实的四种遗忘/清理路径（v1 maintain() 的 v2 版）。

1. **expire（TTL 到期）**：字段 TTL 到期 → 标 t_invalid（保留历史，可查"当时"）；
2. **cap_active（容量上限）**：每用户活跃事实超限 → 淘汰（置信度低、更新旧者先）——
   对应 Letta"避免过度写入"的工程化版本；
3. **explicit_forget（显式遗忘）**："忘掉我说的 X" → 当前事实标失效，理由 user_forget；
4. **erase_user（合规删除）**：硬删除该用户全部事实与审计（被遗忘权），不留历史。
"""

from __future__ import annotations

from typing import Dict

from agent.profile.schema import spec_for
from agent.profile.store import ProfileFact, ProfileStore

DAY = 86400.0


def expire(store: ProfileStore, user_id: str = "default",
           now: float | None = None) -> int:
    """TTL 到期的事实标 t_invalid（借 Zep：失效不是删除，历史可查）。"""
    now = store._now() if now is None else now
    n = 0
    with store._lock, store._conn:
        rows = store._conn.execute(
            "SELECT * FROM profile_facts WHERE user_id=? AND t_invalid IS NULL",
            (user_id,)).fetchall()
        for row in rows:
            fact = ProfileFact.from_row(row)
            spec = spec_for(fact.key)
            if spec.ttl_days and now - fact.updated_at > spec.ttl_days * DAY:
                store._conn.execute(
                    "UPDATE profile_facts SET t_invalid=?, updated_at=? "
                    "WHERE user_id=? AND key=? AND t_valid=?",
                    (now, now, user_id, fact.key, fact.t_valid))
                store._event(user_id, fact.layer, fact.key, "expire",
                             fact.value, None, f"TTL>{spec.ttl_days}d", now)
                n += 1
    return n


def cap_active(store: ProfileStore, user_id: str = "default",
               max_facts: int = 100, now: float | None = None) -> int:
    """容量上限：活跃事实超限时先淘汰低置信度、再淘汰久未更新的。"""
    now = store._now() if now is None else now
    with store._lock, store._conn:
        rows = store._conn.execute(
            "SELECT * FROM profile_facts WHERE user_id=? AND t_invalid IS NULL "
            "ORDER BY confidence ASC, updated_at ASC", (user_id,)).fetchall()
    excess = len(rows) - max(0, max_facts)
    evicted = 0
    for row in rows[:max(0, excess)]:
        fact = ProfileFact.from_row(row)
        with store._lock, store._conn:
            store._conn.execute(
                "UPDATE profile_facts SET t_invalid=?, updated_at=? "
                "WHERE user_id=? AND key=? AND t_valid=?",
                (now, now, user_id, fact.key, fact.t_valid))
        store._event(user_id, fact.layer, fact.key, "capacity_evict",
                     fact.value, None, f"活跃事实超 {max_facts}", now)
        evicted += 1
    return evicted


def explicit_forget(store: ProfileStore, user_id: str, key: str,
                    now: float | None = None) -> bool:
    """用户显式遗忘（"忘掉我说的常去门店"）→ 当前事实标失效。返回是否有事实被遗忘。"""
    now = store._now() if now is None else now
    current = store.current(user_id, key)
    if current is None:
        return False
    store.invalidate(current, "user_forget", now)
    return True


def erase_user(store: ProfileStore, user_id: str) -> Dict[str, int]:
    """合规删除（被遗忘权）：硬删除该用户全部事实与审计，不留任何历史。"""
    with store._lock, store._conn:
        facts = store._conn.execute(
            "DELETE FROM profile_facts WHERE user_id=?", (user_id,)).rowcount
        events = store._conn.execute(
            "DELETE FROM events WHERE user_id=?", (user_id,)).rowcount
    return {"facts_deleted": int(facts or 0), "events_deleted": int(events or 0)}


def maintain(store: ProfileStore, user_id: str = "default",
             max_facts: int = 100, now: float | None = None) -> Dict[str, int]:
    """定期维护入口（v1 maintain() 的 v2 版）：TTL 到期 + 容量上限。"""
    return {"expired": expire(store, user_id, now=now),
            "evicted": cap_active(store, user_id, max_facts=max_facts, now=now)}
