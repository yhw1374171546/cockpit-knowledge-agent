# -*- coding: utf-8 -*-
"""Function Calling 管线测试：参数修复、工具名纠正、策略裁决、幂等、并行执行。

    python agent/tests/test_fc_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.executor import ToolExecutor                                  # noqa: E402
from agent.guardrails import PolicyContext                               # noqa: E402
from agent.kb import KnowledgeBase                                       # noqa: E402
from agent.llm import (SimulatedToolCallLLM, ToolCall, ToolCallRepair,   # noqa: E402
                       estimate_tokens)
from agent.obs import Tracer                                             # noqa: E402
from agent.tools import ToolRegistry                                     # noqa: E402

KB = None
TOOLS = ["search_manual", "get_vehicle_status", "lookup_vehicle_spec",
         "get_maintenance_plan", "create_service_order"]


def setUpModule():
    global KB
    KB = KnowledgeBase.load()


class TestToolCallRepair(unittest.TestCase):
    def setUp(self):
        self.r = ToolCallRepair(TOOLS)

    def test_clean_json_needs_no_repair(self):
        args, repaired = self.r.parse('{"query": "座椅加热", "top_k": 6}')
        self.assertFalse(repaired)
        self.assertEqual(args["top_k"], 6)
        self.assertEqual(self.r.stats()["repairs"], 0)

    def test_code_fence_is_stripped(self):
        raw = '```json\n{"query": "座椅加热", "top_k": 6}\n```'
        args, repaired = self.r.parse(raw)
        self.assertTrue(repaired)
        self.assertEqual(args.get("query"), "座椅加热")

    def test_single_quotes_and_trailing_comma(self):
        args, repaired = self.r.parse("{'query': '座椅加热', 'top_k': 6,}")
        self.assertTrue(repaired)
        self.assertEqual(args.get("query"), "座椅加热")
        self.assertEqual(args.get("top_k"), 6)

    def test_truncated_json_is_closed(self):
        args, repaired = self.r.parse('{"query": "座椅加热"')
        self.assertTrue(repaired)
        self.assertEqual(args.get("query"), "座椅加热")

    def test_lenient_key_value_form(self):
        args, repaired = self.r.parse("query:座椅加热, top_k:6")
        self.assertTrue(repaired)
        self.assertEqual(args.get("query"), "座椅加热")
        self.assertEqual(args.get("top_k"), 6)

    def test_garbage_is_counted_as_parse_failure(self):
        args, repaired = self.r.parse("完全不是结构化参数的文本@@@")
        self.assertTrue(repaired)
        self.assertEqual(args, {})
        self.assertEqual(self.r.stats()["parse_failures"], 1)

    def test_dict_passthrough(self):
        args, repaired = self.r.parse({"query": "胎压"})
        self.assertFalse(repaired)
        self.assertEqual(args["query"], "胎压")

    def test_tool_name_fuzzy_fix(self):
        self.assertEqual(self.r.fix_name("search_manual"), "search_manual")
        self.assertEqual(self.r.fix_name("search_manuals"), "search_manual")
        self.assertEqual(self.r.fix_name("searchManual"), "search_manual")
        self.assertEqual(self.r.fix_name("get_vehicle_statuss"), "get_vehicle_status")
        # 完全不相关的名字不应被强行改写
        self.assertEqual(self.r.fix_name("launch_rocket"), "launch_rocket")

    def test_stats_accumulate(self):
        self.r.parse("{'a': 1,}")
        self.r.parse("bad text @@")
        stats = self.r.stats()
        self.assertEqual(stats["repairs"], 1)
        self.assertEqual(stats["parse_failures"], 1)


class TestSimulatedToolCallLLM(unittest.TestCase):
    def test_emits_openai_style_tool_calls(self):
        llm = SimulatedToolCallLLM([
            {"tool": "search_manual", "arguments": {"query": "座椅加热"}},
            {"content": "最终答案"},
        ])
        resp1 = llm.chat([{"role": "user", "content": "x"}], tools=[{"function": {"name": n}} for n in TOOLS])
        self.assertTrue(resp1.wants_tool)
        self.assertEqual(resp1.tool_calls[0].name, "search_manual")
        resp2 = llm.chat([{"role": "user", "content": "x"}], tools=[])
        self.assertFalse(resp2.wants_tool)
        self.assertEqual(resp2.content, "最终答案")

    def test_dirty_output_still_yields_correct_args(self):
        llm = SimulatedToolCallLLM([
            {"tool": "search_manual", "arguments": {"query": "座椅加热", "top_k": 6}},
        ], dirty_ratio=1.0)
        resp = llm.chat([{"role": "user", "content": "x"}],
                        tools=[{"function": {"name": n}} for n in TOOLS])
        call = resp.tool_calls[0]
        self.assertTrue(call.repaired, "脏输出应被标记为已修复")
        self.assertEqual(call.arguments.get("query"), "座椅加热")
        self.assertEqual(call.arguments.get("top_k"), 6)

    def test_hallucinated_tool_name_is_corrected(self):
        llm = SimulatedToolCallLLM([{"tool": "search_manuals", "arguments": {"query": "胎压"}}])
        resp = llm.chat([{"role": "user", "content": "x"}],
                        tools=[{"function": {"name": n}} for n in TOOLS])
        self.assertEqual(resp.tool_calls[0].name, "search_manual")


class TestToolExecutor(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry(KB)
        self.tracer = Tracer()

    def test_policy_blocks_unconfirmed_write(self):
        ex = ToolExecutor(self.registry, tracer=self.tracer)
        call = ToolCall("create_service_order",
                        {"item": "更换机油机滤", "preferred_date": "2025-06-18"})
        out = ex.execute([call], PolicyContext())
        self.assertTrue(out[0].blocked)
        self.assertIn("确认", out[0].reason)
        self.assertEqual(ex.stats()["blocked"], 1)

    def test_injection_risk_blocks_write_tool(self):
        ex = ToolExecutor(self.registry, tracer=self.tracer)
        call = ToolCall("create_service_order",
                        {"item": "更换机油机滤", "preferred_date": "2025-06-18"})
        ctx = PolicyContext(injection_risk=5, injection_suspicious=True)
        self.registry.confirm("create_service_order:更换机油机滤:2025-06-18")
        out = ex.execute([call], ctx)
        self.assertTrue(out[0].blocked)
        self.assertIn("注入", out[0].reason)

    def test_read_tool_executes_and_records_trace(self):
        ex = ToolExecutor(self.registry, tracer=self.tracer)
        out = ex.execute([ToolCall("search_manual", {"query": "座椅加热", "top_k": 3})])
        self.assertTrue(out[0].result.ok)
        self.assertGreaterEqual(len(out[0].result.data), 1)
        self.assertIn("tool.search_manual", [s.name for s in self.tracer.spans])

    def test_idempotent_write_is_not_repeated(self):
        ex = ToolExecutor(self.registry, tracer=self.tracer)
        self.registry.confirm("create_service_order:更换机油机滤:2025-06-18")
        call = ToolCall("create_service_order",
                        {"item": "更换机油机滤", "preferred_date": "2025-06-18"})
        first = ex.execute([call], PolicyContext())
        self.assertTrue(first[0].result.ok)
        order_no = first[0].result.data["预约单号"]
        # 第二次提交同一请求（模拟重试）→ 必须复用首次结果，不能重复下单
        second = ex.execute([call], PolicyContext())
        self.assertTrue(second[0].idempotent_reuse)
        self.assertEqual(second[0].result.data["预约单号"], order_no)
        self.assertEqual(ex.stats()["idempotent_reuse"], 1)

    def test_parallel_execution_with_simulated_io(self):
        """并行收益需要 I/O 型工具才能体现，这里注入模拟网络时延来量化（报告中标注为模拟）。"""
        ex = ToolExecutor(self.registry, tracer=self.tracer, parallel=True,
                          mock_io_latency_ms=40)
        calls = [ToolCall("search_manual", {"query": f"座椅{i}", "top_k": 2}) for i in range(4)]
        ex.execute(calls)
        stats = ex.stats()
        self.assertEqual(stats["calls"], 4)
        self.assertLess(stats["wall_clock_ms"], stats["serial_estimate_ms"] * 0.8,
                        "4 个 I/O 工具并行后墙钟时间应明显小于串行估算")

    def test_budget_records_tool_calls(self):
        from agent.guardrails import Budget, BudgetTracker
        budget = BudgetTracker(Budget(max_tool_calls=10))
        ex = ToolExecutor(self.registry, tracer=self.tracer)
        ex.execute([ToolCall("search_manual", {"query": "胎压", "top_k": 2})], budget=budget)
        self.assertEqual(budget.tool_calls, 1)
        self.assertEqual(budget.tool_calls_by_name.get("search_manual"), 1)

    def test_unknown_tool_returns_structured_error(self):
        ex = ToolExecutor(self.registry, tracer=Tracer(enabled=False))
        out = ex.execute([ToolCall("no_such_tool", {})])
        self.assertFalse(out[0].result.ok)
        self.assertIn("未知工具", out[0].result.error)


class TestTokenEstimate(unittest.TestCase):
    def test_chinese_and_english(self):
        self.assertGreater(estimate_tokens("胎压低报警被激活时，对应报警轮胎开始闪烁。"), 5)
        self.assertGreater(estimate_tokens("search the manual"), 3)
        self.assertEqual(estimate_tokens(""), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
