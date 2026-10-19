# -*- coding: utf-8 -*-
"""长期记忆测试：写入策略、冲突覆盖、遗忘、脱敏、召回相关性、注入位。

    python agent/tests/test_memory_longterm.py

这里的断言刻意都落在**策略**上，而不是"能存取"上：
记忆错了比没记更糟，所以"不该记的没记""该覆盖的覆盖了""不该注入的没注入"才是重点。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.longterm import LongTermMemory, MemoryConfig  # noqa: E402
from agent.memory import ConversationMemory, VehicleProfile  # noqa: E402

DAY = 86400.0


class FakeClock:
    """可控时钟：遗忘/衰减这类逻辑必须有确定性测试，不能靠 sleep。"""

    def __init__(self, now: float = 1_760_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, days: float = 0.0, hours: float = 0.0) -> None:
        self.now += days * DAY + hours * 3600.0


class TestWritePolicy(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.mem = LongTermMemory(config=MemoryConfig(), clock=self.clock)

    def tearDown(self):
        self.mem.close()

    def test_explicit_city_is_written(self):
        written = self.mem.extract_and_remember("我住在上海，以后保养方便点", user_id="u1")
        self.assertIn("city=上海", written)
        self.assertEqual(self.mem.stats("u1")["semantic"], 1)

    def test_no_write_without_explicit_cue(self):
        # "在上海出差"不是常住地：高精度规则宁可漏记，也不写错
        written = self.mem.extract_and_remember("我这两天在上海出差，车放家里", user_id="u1")
        self.assertEqual(written, [])
        self.assertEqual(self.mem.stats("u1")["semantic"], 0)

    def test_inferred_is_rejected_by_default(self):
        ok = self.mem.remember_preference("city", "北京", user_id="u1", source="inferred")
        self.assertFalse(ok)
        self.assertEqual(self.mem.stats("u1")["semantic"], 0)

    def test_inferred_allowed_only_when_asked(self):
        ok = self.mem.remember_preference("city", "北京", user_id="u1", source="inferred",
                                          allow_inferred=True)
        self.assertTrue(ok)

    def test_vehicle_model_is_normalized(self):
        written = self.mem.extract_and_remember("我的车是领克 8", user_id="u1")
        self.assertIn("vehicle_model=领克08", written)

    def test_preference_feature_and_action(self):
        written = self.mem.extract_and_remember("以后座椅加热都关掉", user_id="u1")
        self.assertIn("pref:座椅加热=关闭", written)

    def test_empty_values_are_ignored(self):
        self.assertFalse(self.mem.remember_preference("city", "   ", user_id="u1"))


class TestConflictAndAudit(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.mem = LongTermMemory(clock=self.clock)

    def tearDown(self):
        self.mem.close()

    def test_overwrite_keeps_old_value_in_events(self):
        self.mem.remember_preference("city", "上海", user_id="u1")
        self.clock.advance(days=1)
        self.assertTrue(self.mem.remember_preference("city", "北京", user_id="u1"))
        events = self.mem.events("u1")
        overwrite = [e for e in events if e["action"] == "overwrite"]
        self.assertEqual(len(overwrite), 1)
        self.assertEqual(overwrite[0]["old_value"], "上海")
        self.assertEqual(overwrite[0]["new_value"], "北京")

    def test_explicit_cannot_be_overwritten_by_inferred(self):
        self.mem.remember_preference("city", "上海", user_id="u1")
        ok = self.mem.remember_preference("city", "深圳", user_id="u1",
                                          source="inferred", allow_inferred=True)
        self.assertFalse(ok)
        self.assertTrue(any(e["action"] == "reject_override" for e in self.mem.events("u1")))
        self.assertIn("上海", self.mem.recall("帮我约保养", user_id="u1").to_prompt())

    def test_reinforce_raises_confidence(self):
        self.mem.remember_preference("city", "上海", user_id="u1", confidence=0.5)
        self.clock.advance(days=20)
        self.mem.remember_preference("city", "上海", user_id="u1", confidence=0.9)
        item = self.mem.recall("保养预约", user_id="u1").items[0]
        self.assertEqual(item.meta["age_days"], 0)          # updated_at 被刷新
        self.assertGreaterEqual(item.meta["confidence"], 0.9)


class TestForgetting(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.mem = LongTermMemory(config=MemoryConfig(ttl_days_episodic=30), clock=self.clock)

    def tearDown(self):
        self.mem.close()

    def test_episode_expires_after_ttl(self):
        self.mem.remember_episode("胎压报警，建议到店检查", user_id="u1", session_id="s1")
        self.assertTrue(self.mem.recall("胎压报警", user_id="u1"))
        self.clock.advance(days=31)
        self.assertFalse(self.mem.recall("胎压报警", user_id="u1"))
        stats = self.mem.maintain(user_id="u1")
        self.assertEqual(stats["expired_episodes"], 1)

    def test_semantic_decays_below_threshold(self):
        self.mem.remember_preference("city", "上海", user_id="u1", confidence=0.9)
        self.clock.advance(days=70)          # 0.9 * 0.98^70 ≈ 0.22 < 0.25
        self.assertFalse(self.mem.recall("帮我约保养", user_id="u1"))
        stats = self.mem.maintain(user_id="u1")
        self.assertGreaterEqual(stats["purged_semantics"], 1)
        self.assertEqual(self.mem.stats("u1")["semantic"], 0)

    def test_capacity_evicts_oldest(self):
        mem = LongTermMemory(config=MemoryConfig(max_episodes_per_user=3), clock=self.clock)
        try:
            for i in range(5):
                self.clock.advance(hours=1)
                mem.remember_episode(f"第{i}次咨询座椅加热", user_id="u1", session_id=f"s{i}")
            stats = mem.maintain(user_id="u1")
            self.assertEqual(stats["evicted_episodes"], 2)
            self.assertEqual(mem.stats("u1")["episodic"], 3)
        finally:
            mem.close()


class TestRecall(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.mem = LongTermMemory(clock=self.clock)
        self.mem.remember_preference("city", "上海", user_id="u1", label="常住城市")
        self.mem.remember_preference("usual_center", "领克中心(浦东金桥店)", user_id="u1",
                                     label="惯用门店")
        self.mem.remember_preference("pref:座椅加热", "关闭", user_id="u1", label="常用功能·座椅加热")
        self.mem.remember_episode("胎压报警，建议到店检查并创建工单 SO123", user_id="u1",
                                  session_id="s1")

    def tearDown(self):
        self.mem.close()

    def test_service_query_recalls_city_and_center(self):
        context = self.mem.recall("帮我约个保养", user_id="u1")
        keys = {item.key for item in context.items}
        self.assertTrue({"city", "usual_center"} <= keys, keys)

    def test_unrelated_query_injects_nothing(self):
        context = self.mem.recall("今天天气怎么样", user_id="u1")
        self.assertEqual(context.items, [])
        self.assertEqual(context.to_prompt(), "")

    def test_false_inject_is_zero_for_tire_question(self):
        # 胎压问题只该召回情景记忆，不该把"门店/城市"塞进去
        context = self.mem.recall("胎压报警了怎么办", user_id="u1")
        kinds = {item.kind for item in context.items}
        self.assertEqual(kinds, {"episodic"})

    def test_preference_recalled_by_feature_name(self):
        context = self.mem.recall("座椅加热怎么关", user_id="u1")
        self.assertTrue(any(item.key == "pref:座椅加热" for item in context.items))

    def test_episodic_recall_by_topic(self):
        context = self.mem.recall("还是那个胎压问题", user_id="u1")
        episodic = [i for i in context.items if i.kind == "episodic"]
        self.assertEqual(len(episodic), 1)
        self.assertIn("SO123", episodic[0].text)

    def test_prompt_has_conflict_guidance(self):
        prompt = self.mem.recall("帮我约个保养", user_id="u1").to_prompt()
        self.assertIn("【跨会话记忆】", prompt)
        self.assertIn("以本次信息为准", prompt)

    def test_user_isolation(self):
        self.assertEqual(self.mem.recall("帮我约个保养", user_id="u2").items, [])

    def test_pii_is_masked_before_storage(self):
        self.mem.remember_episode("车主电话 13812345678，约了下周三", user_id="u3", session_id="s9")
        context = self.mem.recall("车主电话", user_id="u3")
        self.assertTrue(context.items)
        self.assertNotIn("13812345678", context.to_prompt())

    def test_duplicate_episode_is_ignored(self):
        first = self.mem.remember_episode("同一件事", user_id="u1", session_id="s2", dedupe_key="k1")
        second = self.mem.remember_episode("同一件事", user_id="u1", session_id="s2", dedupe_key="k1")
        self.assertIsNotNone(first)
        self.assertIsNone(second)


class TestPromptInjection(unittest.TestCase):
    def test_conversation_memory_embeds_longterm_block(self):
        memory = ConversationMemory(profile=VehicleProfile(model="领克08"))
        memory.add_user("帮我约保养")
        memory.add_assistant("好的")
        before = memory.system_context()
        self.assertNotIn("跨会话记忆", before)
        memory.attach_longterm("【跨会话记忆】\n- 车主偏好：常住城市=上海")
        after = memory.system_context()
        self.assertIn("跨会话记忆", after)
        self.assertIn("上海", after)

    def test_empty_longterm_keeps_prompt_unchanged(self):
        memory = ConversationMemory()
        memory.attach_longterm("")
        self.assertNotIn("跨会话记忆", memory.system_context())


class TestSwitchOff(unittest.TestCase):
    """AB 对照的"关掉"一侧：不挂长期记忆时，事实根本不在上下文里。"""

    def test_without_memory_fact_is_absent(self):
        memory = ConversationMemory(profile=VehicleProfile())
        self.assertNotIn("浦东金桥", memory.system_context())

    def test_with_memory_fact_is_present(self):
        clock = FakeClock()
        mem = LongTermMemory(clock=clock)
        try:
            mem.remember_preference("usual_center", "领克中心(浦东金桥店)", user_id="u1")
            text = mem.context_text("帮我约个保养", user_id="u1")
            memory = ConversationMemory()
            memory.attach_longterm(text)
            self.assertIn("浦东金桥", memory.system_context())
        finally:
            mem.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
