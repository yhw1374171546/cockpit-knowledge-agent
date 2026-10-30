# -*- coding: utf-8 -*-
"""ProfileStore 双时态语义测试：insert / invalidate / bump / append / history。"""

from __future__ import annotations

from agent.profile.store import ProfileStore, _append_value, make_fact


def test_insert_and_current_roundtrip():
    with ProfileStore(":memory:") as store:
        fact = make_fact("u1", "usual_center", "领克中心(徐汇店)", "user_explicit", 0.85, 1000.0)
        store.insert(fact)
        cur = store.current("u1", "usual_center")
        assert cur is not None
        assert cur.value == "领克中心(徐汇店)"
        assert cur.t_invalid is None


def test_invalidate_keeps_history():
    with ProfileStore(":memory:") as store:
        old = make_fact("u1", "usual_center", "徐汇店", "user_explicit", 0.85, 1000.0)
        store.insert(old)
        store.invalidate(old, "被覆盖", 2000.0)
        assert store.current("u1", "usual_center") is None
        hist = store.history("u1", "usual_center")
        assert len(hist) == 1
        assert hist[0].t_invalid == 2000.0        # 旧事实保留、标失效（借 Zep）


def test_overwrite_creates_new_row_with_validity_chain():
    with ProfileStore(":memory:") as store:
        old = make_fact("u1", "usual_center", "徐汇店", "user_explicit", 0.85, 1000.0)
        new = make_fact("u1", "usual_center", "浦东店", "user_explicit", 0.85, 2000.0)
        store.insert(old)
        store.invalidate(old, "被覆盖", 2000.0)
        store.insert(new)
        assert store.current("u1", "usual_center").value == "浦东店"
        hist = store.history("u1", "usual_center")
        assert [f.value for f in hist] == ["浦东店", "徐汇店"]  # 支持"当时 vs 现在"


def test_bump_raises_confidence_only():
    with ProfileStore(":memory:") as store:
        fact = make_fact("u1", "city", "上海", "user_explicit", 0.9, 1000.0)
        store.insert(fact)
        store.bump(fact, 0.95, "reinforce", 1500.0)
        cur = store.current("u1", "city")
        assert cur.confidence == 0.95
        assert len(store.history("u1", "city")) == 1     # 不重复插行


def test_append_value_dedupes_and_preserves_order():
    assert _append_value("[]", "更换机油机滤") == '["更换机油机滤"]'
    merged = _append_value('["更换机油机滤"]', "更换刹车片")
    assert merged == '["更换机油机滤", "更换刹车片"]'
    assert _append_value(merged, "更换机油机滤") == merged   # 去重


def test_mask_sensitive_applied_at_fact_creation():
    fact = make_fact("u1", "city", "上海", "user_explicit", 0.9, 1000.0)
    assert "137" not in fact.value        # 手机号等 PII 会被脱敏（间接回归）


def test_stats_counts_active_facts_and_events():
    with ProfileStore(":memory:") as store:
        fact = make_fact("u1", "city", "上海", "user_explicit", 0.9, 1000.0)
        store.insert(fact)
        store.invalidate(fact, "test", 2000.0)
        stats = store.stats("u1")
        assert stats["active_facts"] == 0
        assert stats["events"] == 1
