# -*- coding: utf-8 -*-
"""Agent 离线单元测试（零依赖，stdlib unittest 即可运行）。

    python -m unittest discover -s agent/tests -v
    python agent/tests/test_offline.py            # 也可以直接跑

覆盖点：工具 schema 合法性、检索质量、写操作确认、重复调用抑制、
步数上限与无进展终止、拒答门控、引用校验、记忆与指代消解、LangGraph 适配器可用性。
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.graph import AgentConfig, AgentGraph, build_langgraph_app     # noqa: E402
from agent.kb import KnowledgeBase, Tokenizer                            # noqa: E402
from agent.llm import LLMResponse, RuleBasedPlannerLLM, ScriptedLLM, ToolCall  # noqa: E402
from agent.memory import ConversationMemory, VehicleProfile              # noqa: E402
from agent.reflection import Reflector                                   # noqa: E402
from agent.tools import ToolRegistry, build_default_registry             # noqa: E402

KB = None


def setUpModule():
    global KB
    KB = KnowledgeBase.load()
    assert KB.chunks, "知识库为空，请确认 all_text.txt 或 kb/chunks.jsonl 存在"


def fresh_registry() -> ToolRegistry:
    return ToolRegistry(KB)


def fresh_agent(llm=None, config=None) -> AgentGraph:
    return AgentGraph(llm or RuleBasedPlannerLLM(), fresh_registry(),
                      reflector=Reflector(), config=config or AgentConfig())


# ── 1. 知识库与检索 ──────────────────────────────────────────────────


class TestKnowledgeBase(unittest.TestCase):
    def test_stats(self):
        stats = KB.stats()
        self.assertEqual(stats["n_chunks"], 8785)
        self.assertGreater(stats["total_chars"], 5_000_000)

    def test_retrieval_hits_expected_topic(self):
        """检索到的内容必须与问题同主题。"""
        cases = {
            "怎么打开危险警告灯": "危险警告灯",
            "座椅加热": "座椅",
            "胎压": "胎压",
            "保养周期": "保养",
        }
        for query, expect in cases.items():
            with self.subTest(query=query):
                hits = KB.search(query, top_k=3)
                self.assertTrue(hits, f"{query} 无召回结果")
                self.assertTrue(any(expect in h.text for h in hits),
                                f"{query} 的 Top-3 中没有出现「{expect}」")

    def test_bigram_tokenizer_fallback(self):
        tokens = Tokenizer(prefer_jieba=False).cut("座椅加热")
        self.assertTrue(all(isinstance(t, str) and t for t in tokens))
        self.assertIn("座椅", tokens)

    def test_empty_query_returns_nothing(self):
        self.assertEqual(KB.search("", top_k=3), [])


# ── 2. 工具层 ────────────────────────────────────────────────────────


class TestTools(unittest.TestCase):
    def setUp(self):
        self.reg = fresh_registry()

    def test_schema_is_openai_function_calling_shape(self):
        specs = self.reg.specs()
        self.assertGreaterEqual(len(specs), 5)
        for spec in specs:
            self.assertEqual(spec["type"], "function")
            fn = spec["function"]
            self.assertIn("name", fn)
            self.assertIn("description", fn)
            params = fn["parameters"]
            self.assertEqual(params["type"], "object")
            self.assertIn("properties", params)
            for required in params.get("required", []):
                self.assertIn(required, params["properties"],
                              f"{fn['name']} 的 required 字段缺少 schema 定义")
            json.dumps(spec)  # 必须可序列化，才能直接 POST 给 vLLM

    def test_search_manual_returns_citations(self):
        res = self.reg.call("search_manual", {"query": "座椅加热", "top_k": 3})
        self.assertTrue(res.ok)
        self.assertEqual(len(res.data), 3)
        self.assertTrue(all("citation" in item for item in res.data))

    def test_search_manual_bad_params_returns_structured_error(self):
        res = self.reg.call("search_manual", {"query": "座椅", "top_k": 999})
        self.assertTrue(res.ok)  # top_k 会被夹到上限而非报错
        self.assertLessEqual(len(res.data), 10)

    def test_unknown_tool_returns_hint(self):
        res = self.reg.call("no_such_tool", {})
        self.assertFalse(res.ok)
        self.assertIn("可用工具", res.hint or "")

    def test_write_action_requires_confirmation(self):
        res = self.reg.call("create_service_order",
                            {"item": "更换机油机滤", "preferred_date": "2025-06-18"})
        self.assertFalse(res.ok)
        self.assertTrue(res.meta.get("needs_confirmation"))
        action_key = res.meta["action_key"]
        self.reg.confirm(action_key)
        res2 = self.reg.call("create_service_order",
                             {"item": "更换机油机滤", "preferred_date": "2025-06-18"})
        self.assertTrue(res2.ok)
        self.assertIn("预约单号", res2.data)

    def test_repeated_call_is_deduped(self):
        first = self.reg.call("get_vehicle_status", {})
        second = self.reg.call("get_vehicle_status", {})
        self.assertFalse(first.repeated)
        self.assertTrue(second.repeated)
        self.assertEqual(self.reg.cache_stats()["repeated_calls"], 1)

    def test_maintenance_plan_advances_with_mileage(self):
        low = self.reg.call("get_maintenance_plan", {"mileage_km": 3000}).data
        high = self.reg.call("get_maintenance_plan", {"mileage_km": 45000}).data
        self.assertEqual(low["本次对应里程"], None)
        self.assertEqual(high["本次对应里程"], 40000)


# ── 3. 记忆 ──────────────────────────────────────────────────────────


class TestMemory(unittest.TestCase):
    def test_slot_extraction(self):
        mem = ConversationMemory()
        mem.add_user("我的领克08跑了2.5万公里，在上海")
        self.assertEqual(mem.profile.model, "领克08")
        self.assertEqual(mem.profile.mileage_km, 25000)
        self.assertEqual(mem.profile.city, "上海")

    def test_pronoun_resolution_uses_topic(self):
        mem = ConversationMemory()
        mem.set_topic("座椅加热")
        self.assertEqual(mem.resolve("那它怎么关"), "座椅加热 那它怎么关")

    def test_self_contained_question_is_not_rewritten(self):
        mem = ConversationMemory()
        mem.set_topic("座椅加热")
        self.assertEqual(mem.resolve("胎压报警了怎么办"), "胎压报警了怎么办")


# ── 4. 反思与拒答 ────────────────────────────────────────────────────


class TestReflection(unittest.TestCase):
    def setUp(self):
        self.r = Reflector()

    def test_grounded_answer_accepted(self):
        evidence = [{"text": "危险警告灯开关在方向盘下方，按下开关即可打开危险警告灯。",
                     "citation": "[train_a.pdf 块#211]"}]
        res = self.r.verify("危险警告灯开关在方向盘下方，按下开关即可打开危险警告灯。", evidence)
        self.assertEqual(res.verdict, "grounded")
        self.assertEqual(res.next_action, "accept")

    def test_fabricated_answer_rejected(self):
        evidence = [{"text": "危险警告灯开关在方向盘下方。", "citation": "[train_a.pdf 块#211]"}]
        res = self.r.verify("本车配备航空级钛合金防撞梁，碰撞时自动弹出降落伞。", evidence)
        self.assertEqual(res.verdict, "ungrounded")
        self.assertEqual(res.next_action, "refuse")

    def test_invalid_citation_detected(self):
        """有页码元数据时，编造的页码必须被判定为无效引用。"""
        evidence = [{"text": "危险警告灯开关在方向盘下方。",
                     "citation": "[train_a.pdf 第12页]", "page": 12}]
        res = self.r.verify("检查第9999页可知危险警告灯开关在方向盘下方。", evidence)
        self.assertIn("第9999页", " ".join(res.invalid_citations))
        self.assertEqual(res.next_action, "refuse")

    def test_page_citation_without_metadata_is_unverifiable(self):
        """知识库没有页码元数据时，页码只能标记为「不可核验」，不应直接拒答。"""
        evidence = [{"text": "危险警告灯开关在方向盘下方。", "citation": "[train_a.pdf 块#211]"}]
        res = self.r.verify("危险警告灯开关在方向盘下方（第12页）。", evidence)
        self.assertEqual(res.invalid_citations, [])
        self.assertIn("第12页", res.unverifiable_citations)
        self.assertEqual(res.next_action, "accept")

    def test_json_brackets_are_not_treated_as_citation(self):
        evidence = [{"text": "保养项目包含更换机油机滤。", "citation": "[train_a.pdf 块#1]"}]
        res = self.r.verify('保养项目：["更换机油机滤", "更换空气滤芯"]。', evidence)
        self.assertEqual(res.invalid_citations, [])

    def test_evidence_sufficiency(self):
        self.assertFalse(self.r.evidence_sufficient("中国足球的队长是谁", []))
        evidence = ["危险警告灯开关在方向盘下方，按下开关即可打开危险警告灯。"]
        self.assertTrue(self.r.evidence_sufficient("怎么打开危险警告灯", evidence))


# ── 5. 编排与治理 ────────────────────────────────────────────────────


class TestAgentGraph(unittest.TestCase):
    def test_answers_manual_question_with_citation(self):
        st = fresh_agent().run("怎么打开危险警告灯")
        self.assertEqual(st.status, "answered")
        self.assertTrue(st.citations)
        self.assertIn("危险警告灯", st.answer)

    def test_refuses_off_topic_question(self):
        st = fresh_agent().run("中国足球的队长是谁")
        self.assertEqual(st.status, "refused")
        self.assertEqual(st.answer, "无答案")

    def test_multi_tool_for_vehicle_status_question(self):
        st = fresh_agent().run("胎压报警了怎么办")
        tools = {t["tool"] for t in st.tool_calls}
        self.assertIn("get_vehicle_status", tools)
        self.assertIn("search_manual", tools)
        self.assertEqual(st.status, "answered")

    def test_write_action_returns_needs_confirmation(self):
        llm = ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall("create_service_order",
                                             {"item": "更换机油机滤",
                                              "preferred_date": "2025-06-18"})]),
        ])
        st = fresh_agent(llm).run("帮我预约到店保养")
        self.assertEqual(st.status, "needs_confirmation")
        self.assertIn("确认", st.answer)

    def test_max_steps_governance(self):
        """LLM 一直要求调用工具 → 必须被步数上限截断，不能死循环。"""
        class LoopingLLM(RuleBasedPlannerLLM):
            def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024):
                self.turn += 1
                return LLMResponse(tool_calls=[ToolCall("search_manual",
                                                        {"query": f"座椅{i}", "top_k": 2})]
                                   if False else
                                   [ToolCall("search_manual", {"query": f"座椅{self.turn}"})])

        config = AgentConfig(max_steps=4)
        st = fresh_agent(LoopingLLM(), config).run("座椅加热")
        self.assertLessEqual(st.steps, 4)
        self.assertIn(st.status, ("answered", "refused", "max_steps"))

    def test_no_progress_terminates(self):
        """重复调用同一工具同一参数 → 无进展检测生效，立即收敛。"""
        class StuckLLM(RuleBasedPlannerLLM):
            def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024):
                return LLMResponse(tool_calls=[ToolCall("get_vehicle_status", {})])

        st = fresh_agent(StuckLLM(), AgentConfig(max_steps=8)).run("胎压怎么样")
        self.assertLess(st.steps, 8, "无进展检测未生效，步数未被压缩")
        self.assertIn(st.status, ("answered", "refused"))

    def test_gate_can_be_disabled(self):
        config = AgentConfig(evidence_gate=False)
        st = fresh_agent(RuleBasedPlannerLLM(), config).run("中国足球的队长是谁")
        self.assertNotEqual(st.trace[0].get("reason"), "evidence_gate_disabled")

    def test_langgraph_adapter_optional(self):
        app = build_langgraph_app(fresh_agent())
        if app is None:
            self.skipTest("未安装 langgraph，适配器返回 None（预期行为）")
        out = app.invoke({"question": "怎么打开危险警告灯", "resolved": "怎么打开危险警告灯"})
        self.assertIn("answer", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
