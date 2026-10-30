# -*- coding: utf-8 -*-
"""forget 测试：TTL 到期 / 容量淘汰 / 显式遗忘 / 合规删除。"""

from __future__ import annotations

from agent.profile.consolidate import MemoryCandidate, consolidate
from agent.profile.forget import cap_active, erase_user, expire, explicit_forget, maintain
from agent.profile.store import ProfileStore

DAY = 86400.0


def test_expire_marks_invalid_but_keeps_history():
    with ProfileStore(":memory:") as store:
        consolidate(MemoryCandidate("usual_center", "徐汇店", "user_explicit", 0.85),
                    store, user_id="u1", now=1000.0)
        assert expire(store, "u1", now=1000.0 + 91 * DAY) == 1
        assert store.current("u1", "usual_center") is None
        assert len(store.history("u1", "usual_center")) == 1   # 历史保留（借 Zep）


def test_expire_skips_never_expiring_fields():
    with ProfileStore(":memory:") as store:
        consolidate(MemoryCandidate("vehicle_model", "领克08", "user_explicit", 0.95),
                    store, user_id="u1", now=1000.0)
        assert expire(store, "u1", now=1000.0 + 365 * DAY) == 0    # 静态档案不衰减


def test_cap_evicts_low_confidence_first():
    with ProfileStore(":memory:") as store:
        for i in range(3):
            consolidate(MemoryCandidate(f"pref:功能{i}", f"值{i}", "user_explicit", 0.6 + i * 0.1),
                        store, user_id="u1", now=1000.0, allow_inferred=True)
        assert cap_active(store, "u1", max_facts=2, now=2000.0) == 1
        assert store.current("u1", "pref:功能0") is None            # 置信度最低者被淘汰


def test_explicit_forget():
    with ProfileStore(":memory:") as store:
        consolidate(MemoryCandidate("usual_center", "徐汇店", "user_explicit", 0.85),
                    store, user_id="u1", now=1000.0)
        assert explicit_forget(store, "u1", "usual_center") is True
        assert store.current("u1", "usual_center") is None
        assert explicit_forget(store, "u1", "usual_center") is False   # 已遗忘


def test_erase_user_hard_deletes_everything():
    with ProfileStore(":memory:") as store:
        consolidate(MemoryCandidate("city", "上海", "user_explicit", 0.9),
                    store, user_id="u1", now=1000.0)
        res = erase_user(store, "u1")
        assert res["facts_deleted"] == 1 and res["events_deleted"] >= 1
        assert store.history("u1", "city") == []                # 被遗忘权：不留历史


def test_maintain_runs_expire_and_cap():
    with ProfileStore(":memory:") as store:
        consolidate(MemoryCandidate("usual_center", "徐汇店", "user_explicit", 0.85),
                    store, user_id="u1", now=1000.0)
        stats = maintain(store, "u1", max_facts=0, now=1000.0 + 91 * DAY)
        assert stats["expired"] == 1
