# -*- coding: utf-8 -*-
"""ProfilePipeline 集成测试：v1 LongTermMemory 同款接口语义 + 漂移/否定闭环。"""

from __future__ import annotations

from agent.profile.pipeline import ProfilePipeline
from agent.profile.store import ProfileStore


def test_extract_remember_and_recall_closed_loop():
    with ProfileStore(":memory:") as store:
        pipe = ProfilePipeline(store, user_id="u1")
        results = pipe.extract_and_remember("我住在上海，我一般去领克中心(徐汇店)，以后都去这家")
        assert {r.action for r in results} == {"insert"}
        ctx = pipe.recall("帮我约个保养", user_id="u1")
        assert any(i.key == "usual_center" and i.text == "领克中心(徐汇店)" for i in ctx.items)


def test_drift_second_utterance_wins():
    with ProfileStore(":memory:") as store:
        pipe = ProfilePipeline(store, user_id="u1")
        pipe.extract_and_remember("我一般去领克中心(徐汇店)，以后都去这家")
        results = pipe.extract_and_remember("不对，我一般去领克中心(杭州西湖店)")
        assert any(r.action == "overwrite" for r in results)
        assert pipe.preference("u1", "usual_center") == "领克中心(杭州西湖店)"
        # 旧值保留在历史里（"当时 vs 现在"）
        assert [f.value for f in store.history("u1", "usual_center")] == \
            ["领克中心(杭州西湖店)", "领克中心(徐汇店)"]


def test_negation_correction_updates_store():
    with ProfileStore(":memory:") as store:
        pipe = ProfilePipeline(store, user_id="u1")
        pipe.extract_and_remember("我一般去领克中心(徐汇店)，以后都去这家")
        pipe.extract_and_remember("不是那个店，是浦东金桥店")
        assert pipe.preference("u1", "usual_center") == "浦东金桥店"


def test_defaults_from_preferences_fills_tool_defaults():
    with ProfileStore(":memory:") as store:
        pipe = ProfilePipeline(store, user_id="u1")
        pipe.extract_and_remember("我一般去领克中心(徐汇店)，以后都去这家")
        assert pipe.defaults_from_preferences("u1") == {"center": "领克中心(徐汇店)"}


def test_fingerprint_changes_when_memory_changes():
    with ProfileStore(":memory:") as store:
        pipe = ProfilePipeline(store, user_id="u1")
        before = pipe.fingerprint("u1")
        pipe.extract_and_remember("我一般去领克中心(徐汇店)，以后都去这家")
        after = pipe.fingerprint("u1")
        assert before != after                          # 缓存 scope 必须随记忆失效


def test_maintain_runs_without_crash():
    with ProfileStore(":memory:") as store:
        pipe = ProfilePipeline(store, user_id="u1")
        pipe.extract_and_remember("我一般去领克中心(徐汇店)，以后都去这家")
        assert pipe.maintain("u1", max_facts=1) == {"expired": 0, "evicted": 0}


def test_red_line_always_enters_context():
    with ProfileStore(":memory:") as store:
        pipe = ProfilePipeline(store, user_id="u1")
        pipe.extract_and_remember("记住，以后千万别给我关闭车身稳定系统")
        text = pipe.context_text("今天天气怎么样", user_id="u1")
        assert "红线" in text and "车身稳定系统" in text
