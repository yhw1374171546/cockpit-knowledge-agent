# -*- coding: utf-8 -*-
"""consolidate 冲突矩阵测试：insert / reinforce / overwrite / append / reject。"""

from __future__ import annotations

from agent.profile.consolidate import MemoryCandidate, consolidate
from agent.profile.store import ProfileStore


def cand(key: str, value: str, source: str = "user_explicit") -> MemoryCandidate:
    return MemoryCandidate(key=key, value=value, source=source,
                           confidence=0.9, evidence_span="", layer="")


def test_insert_when_empty():
    with ProfileStore(":memory:") as store:
        res = consolidate(cand("usual_center", "领克中心(徐汇店)"), store, user_id="u1", now=1000.0)
        assert res.action == "insert"
        assert store.current("u1", "usual_center").value == "领克中心(徐汇店)"


def test_reinforce_when_same_value():
    with ProfileStore(":memory:") as store:
        consolidate(cand("usual_center", "徐汇店"), store, user_id="u1", now=1000.0)
        res = consolidate(cand("usual_center", "徐汇店"), store, user_id="u1", now=2000.0)
        assert res.action == "reinforce"
        assert len(store.history("u1", "usual_center")) == 1


def test_drift_overwrite_same_source_new_wins():
    """偏好漂移：同优先级（user_explicit vs user_explicit）→ 新信息优先（借 Zep 事务线语义）。"""
    with ProfileStore(":memory:") as store:
        consolidate(cand("usual_center", "徐汇店"), store, user_id="u1", now=1000.0)
        res = consolidate(cand("usual_center", "杭州西湖店"), store, user_id="u1", now=2000.0)
        assert res.action == "overwrite"
        assert store.current("u1", "usual_center").value == "杭州西湖店"
        hist = store.history("u1", "usual_center")
        assert [f.value for f in hist] == ["杭州西湖店", "徐汇店"]   # 旧值保留失效


def test_inferred_cannot_override_explicit():
    with ProfileStore(":memory:") as store:
        consolidate(cand("usual_center", "徐汇店", "user_explicit"), store, user_id="u1", now=1000.0)
        res = consolidate(cand("usual_center", "浦东店", "inferred"), store,
                          user_id="u1", now=2000.0, allow_inferred=True)
        assert res.action == "reject"
        assert store.current("u1", "usual_center").value == "徐汇店"


def test_explicit_can_override_inferred():
    with ProfileStore(":memory:") as store:
        consolidate(cand("usual_center", "浦东店", "inferred"), store,
                    user_id="u1", now=1000.0, allow_inferred=True)
        res = consolidate(cand("usual_center", "徐汇店", "user_explicit"), store,
                          user_id="u1", now=2000.0)
        assert res.action == "overwrite"
        assert store.current("u1", "usual_center").value == "徐汇店"


def test_order_system_is_authoritative_for_lifecycle():
    with ProfileStore(":memory:") as store:
        consolidate(cand("next_due", "24000km", "user_explicit"), store, user_id="u1", now=1000.0)
        res = consolidate(cand("next_due", "20000km", "order_system"), store,
                          user_id="u1", now=2000.0)
        assert res.action == "overwrite"       # 工单系统（rank 3）覆盖口述（rank 2）
        assert store.current("u1", "next_due").value == "20000km"


def test_inferred_default_rejected_without_flag():
    with ProfileStore(":memory:") as store:
        res = consolidate(cand("city", "上海", "inferred"), store, user_id="u1", now=1000.0)
        assert res.action == "reject"          # 只记明示：推断默认不落库


def test_append_only_field_never_overwrites():
    with ProfileStore(":memory:") as store:
        consolidate(cand("maintenance_history", "更换机油机滤", "order_system"),
                    store, user_id="u1", now=1000.0)
        res = consolidate(cand("maintenance_history", "更换刹车片", "order_system"),
                          store, user_id="u1", now=2000.0)
        assert res.action == "append"
        assert "更换机油机滤" in store.current("u1", "maintenance_history").value
        assert "更换刹车片" in store.current("u1", "maintenance_history").value


def test_reject_unknown_source():
    with ProfileStore(":memory:") as store:
        res = consolidate(cand("usual_center", "徐汇店", "gossip"), store, user_id="u1", now=1000.0)
        assert res.action == "reject"
        assert "允许来源" in res.reason


def test_reject_override_leaves_audit_trail():
    with ProfileStore(":memory:") as store:
        consolidate(cand("usual_center", "徐汇店"), store, user_id="u1", now=1000.0)
        consolidate(cand("usual_center", "浦东店", "inferred"), store,
                    user_id="u1", now=2000.0, allow_inferred=True)
        events = store.events("u1")
        assert any(e["action"] == "reject_override" for e in events)
        assert any(e["old_value"] == "徐汇店" for e in events)   # 拒绝也留痕（可审计）
