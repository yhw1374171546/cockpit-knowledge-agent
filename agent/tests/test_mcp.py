# -*- coding: utf-8 -*-
"""MCP 接入层测试：协议、白名单、跨进程确认、trace 透传、编排层端到端。

    python agent/tests/test_mcp.py

覆盖的四类"真问题"（不是能调通就算过）：
1. **协议**：JSON-RPC 分帧、坏输入不把服务端打挂、通知不产生响应、未知方法/工具的错误码；
2. **白名单**：子 Agent 拿不到写操作工具，而且被拦住时**服务端根本没收到请求**（不是"服务端拒了"）；
3. **写操作确认跨进程**：未确认 → 返回 action_key；确认 → 才执行；本地镜像供执行器前置判断；
4. **编排层透明**：`AgentGraph` 与 `Supervisor` 都不知道工具在另一个进程里，端到端仍然答得出带出处。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.graph import AgentConfig, AgentGraph  # noqa: E402
from agent.kb import KnowledgeBase  # noqa: E402
from agent.llm import RuleBasedPlannerLLM  # noqa: E402
from agent.mcp.client import McpClient  # noqa: E402
from agent.mcp.protocol import (  # noqa: E402
    CONTROL_TOOLS,
    FORBIDDEN,
    PROTOCOL_VERSION,
    McpError,
    McpTimeout,
)
from agent.mcp.registry import McpToolRegistry  # noqa: E402
from agent.mcp.scope import issue_scope_token  # noqa: E402
from agent.obs import Tracer  # noqa: E402
from agent.orchestrator import Supervisor  # noqa: E402
from agent.repair_advisor import RepairAdvisor  # noqa: E402
from agent.tools import ToolRegistry  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PATH = os.path.join(ROOT, "eval", "fixtures", "mini_corpus.jsonl")
SERVER_ARGS = ["--kb", KB_PATH, "--telemetry", "normal"]
TIMEOUT = 30.0


def make_client(**kwargs) -> McpClient:
    kwargs.setdefault("timeout", TIMEOUT)
    return McpClient(args=SERVER_ARGS, **kwargs).start()


def raw_server() -> subprocess.Popen:
    env = dict(os.environ)
    prefix = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONPATH"] = prefix
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.Popen(                                            # noqa: S603
        [sys.executable, "-m", "agent.mcp.server", *SERVER_ARGS],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8", bufsize=1, cwd=ROOT, env=env,
    )


class TestProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = make_client()

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_initialize_handshake(self):
        self.assertEqual(self.client.server_info.get("name"), "cockpit-knowledge-mcp")
        self.assertEqual(self.client.protocol_version, PROTOCOL_VERSION)

    def test_ping(self):
        self.assertTrue(self.client.ping())

    def test_tools_list_order_and_schema(self):
        tools = self.client.list_tools()
        self.assertEqual(
            [t["name"] for t in tools],
            ["search_manual", "get_vehicle_status", "lookup_vehicle_spec",
             "get_maintenance_plan", "create_service_order",
             "get_my_orders", "reschedule_order", "cancel_order"],
        )
        for tool in tools:
            self.assertEqual(tool["inputSchema"]["type"], "object")
            self.assertTrue(tool["description"])
            self.assertIn("readOnlyHint", tool["annotations"])

    def test_control_tools_are_hidden_from_tools_list(self):
        names = {t["name"] for t in self.client.list_tools()}
        self.assertEqual(names & set(CONTROL_TOOLS), set())
        # 但客户端仍能调用控制面（模型看不到 ≠ 客户端不能用）
        payload = self.client.call_tool("get_telemetry", {})
        self.assertTrue(payload["structuredContent"]["ok"])
        self.assertTrue(payload["_meta"]["control"])

    def test_schema_parity_with_local_registry(self):
        """MCP 的 inputSchema 与本地 Function Calling schema 必须逐字段一致（同一份定义）。"""
        local = ToolRegistry(KnowledgeBase.load(KB_PATH))
        local_specs = {s["function"]["name"]: s["function"] for s in local.specs()}
        remote = self.client.list_tools()
        self.assertEqual(len(remote), len(local_specs))
        for tool in remote:
            function = local_specs[tool["name"]]
            self.assertEqual(tool["description"], function["description"], tool["name"])
            self.assertEqual(tool["inputSchema"], function["parameters"], tool["name"])

    def test_unknown_method_error_code(self):
        with self.assertRaises(McpError) as ctx:
            self.client.request("tools/dance", {})
        self.assertEqual(ctx.exception.code, -32601)

    def test_unknown_tool_error_code(self):
        with self.assertRaises(McpError) as ctx:
            self.client.call_tool("rm_rf", {})
        self.assertEqual(ctx.exception.code, -32602)
        self.assertIn("search_manual", json.dumps(ctx.exception.data, ensure_ascii=False))


class TestWireProtocol(unittest.TestCase):
    """直接对着管道收发，验证"坏输入不会把服务端打挂"这类线上问题。"""

    def setUp(self):
        self.proc = raw_server()
        assert self.proc.stdin is not None and self.proc.stdout is not None

    def tearDown(self):
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:                                               # noqa: BLE001
            self.proc.kill()
        for stream in (self.proc.stdout, self.proc.stderr):
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except Exception:                                           # noqa: BLE001
                pass

    def _send(self, payload: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(payload + "\n")
        self.proc.stdin.flush()

    def _read(self) -> dict:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        self.assertTrue(line, "服务端没有响应就退出了")
        return json.loads(line)

    def test_malformed_json_then_recover(self):
        self._send("这不是 JSON")
        response = self._read()
        self.assertEqual(response["error"]["code"], -32700)
        self._send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}))
        self.assertEqual(self._read()["result"], {})

    def test_notifications_produce_no_response(self):
        self._send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}))
        self.assertIn("result", self._read())
        # 通知没有 id，必须**不产生响应**；下一条读到的应该是 id=2 的 ping
        self._send(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self._send(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}))
        self.assertEqual(self._read()["id"], 2)


class TestTimeout(unittest.TestCase):
    def test_call_timeout_is_bounded(self):
        """服务端卡住时，客户端必须在 deadline 内返回错误，而不是把整个 Agent 拖死。"""
        script = (
            "import json, sys, time\n"
            "for line in sys.stdin:\n"
            "    msg = json.loads(line)\n"
            "    if 'id' in msg and msg.get('method') == 'initialize':\n"
            "        print(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': {\n"
            "            'protocolVersion': '2024-11-05', 'capabilities': {'tools': {}},\n"
            "            'serverInfo': {'name': 'stub'}}}), flush=True)\n"
            "    # tools/call 故意不回，模拟服务端卡死\n"
        )
        handle = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8")
        handle.write(script)
        handle.close()
        client = McpClient(command=[sys.executable, handle.name], timeout=1.0).start()
        try:
            started = time.monotonic()
            with self.assertRaises(McpTimeout):
                client.call_tool("search_manual", {"query": "座椅加热"})
            self.assertLess(time.monotonic() - started, 5.0)
            self.assertEqual(client.timed_out, 1)
        finally:
            client.close()
            os.unlink(handle.name)


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.client = make_client()
        self.registry = McpToolRegistry(self.client)

    def tearDown(self):
        self.client.close()

    def test_search_manual_roundtrip(self):
        result = self.registry.call("search_manual", {"query": "座椅加热怎么关闭", "top_k": 3})
        self.assertTrue(result.ok, result.error)
        self.assertTrue(result.data)
        self.assertIn("page", result.data[0])
        self.assertEqual(result.meta.get("transport"), "mcp")
        self.assertIsNotNone(result.meta.get("server_ms"))

    def test_repeated_call_is_marked_by_server_cache(self):
        first = self.registry.call("get_vehicle_status", {})
        second = self.registry.call("get_vehicle_status", {})
        self.assertFalse(first.repeated)
        self.assertTrue(second.repeated)
        self.assertEqual(self.registry.cache_stats()["repeated_calls"], 1)

    def test_whitelist_blocks_write_tool_without_touching_server(self):
        scoped = self.registry.subset(["search_manual", "get_vehicle_status"], "vehicle_control")
        self.assertEqual([s["function"]["name"] for s in scoped.specs()],
                         ["search_manual", "get_vehicle_status"])
        before = self.client.request_count
        result = scoped.call("create_service_order",
                             {"item": "更换机油机滤", "preferred_date": "2025-06-18"})
        self.assertFalse(result.ok)
        self.assertIn("白名单", result.error or "")
        self.assertEqual(self.client.request_count, before,
                         "白名单必须挡在客户端，不能让请求到达服务端")

    def test_whitelist_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            self.registry.subset(["no_such_tool"])

    def test_telemetry_property(self):
        telemetry = self.registry.telemetry
        self.assertIn("胎压_kPa", telemetry)
        self.assertIn("里程_km", telemetry)

    def test_reset_cache_clears_local_trace(self):
        self.registry.call("get_vehicle_status", {})
        self.assertGreater(self.registry.cache_stats()["total_calls"], 0)
        self.registry.reset_cache()
        self.assertEqual(self.registry.cache_stats()["total_calls"], 0)


class TestWriteConfirmation(unittest.TestCase):
    ARGS = {"item": "更换机油机滤", "preferred_date": "2025-06-18"}

    def setUp(self):
        self.client = make_client()
        self.registry = McpToolRegistry(self.client)

    def tearDown(self):
        self.client.close()

    def test_unconfirmed_write_returns_action_key(self):
        result = self.registry.call("create_service_order", dict(self.ARGS))
        self.assertFalse(result.ok)
        self.assertTrue(result.meta.get("needs_confirmation"))
        self.assertIn("action_key", result.meta)
        self.assertNotIn(result.meta["action_key"], self.registry.confirmed_actions)

    def test_confirm_then_execute(self):
        pending = self.registry.call("create_service_order", dict(self.ARGS))
        action_key = pending.meta["action_key"]
        self.registry.confirm(action_key)
        # 本地镜像是给 ToolExecutor._is_confirmed() 做前置判断用的
        self.assertIn(action_key, self.registry.confirmed_actions)
        done = self.registry.call("create_service_order", dict(self.ARGS))
        self.assertTrue(done.ok, done.error)
        self.assertIn("预约单号", done.data)

    def test_confirm_is_shared_with_subset(self):
        pending = self.registry.call("create_service_order", dict(self.ARGS))
        action_key = pending.meta["action_key"]
        scoped = self.registry.subset(["create_service_order"], "service_advisor")
        scoped.confirm(action_key)
        self.assertIn(action_key, self.registry.confirmed_actions)


class TestOrchestrationOverMcp(unittest.TestCase):
    """编排层不感知工具在另一个进程：这是"MCP 是接入层替换"的核心证据。"""

    def setUp(self):
        self.client = make_client()
        self.registry = McpToolRegistry(self.client)

    def tearDown(self):
        self.client.close()

    def test_agent_graph_runs_over_mcp(self):
        graph = AgentGraph(RuleBasedPlannerLLM(), self.registry, config=AgentConfig())
        state = graph.run("座椅加热怎么关闭")
        self.assertEqual(state.status, "answered")
        self.assertTrue(state.answer)
        self.assertTrue(state.citations)
        self.assertIn("search_manual", [item["tool"] for item in self.registry.trace])

    def test_supervisor_runs_two_experts_on_one_server(self):
        supervisor = Supervisor(None, registry=self.registry, tracer=Tracer())
        before = self.client.request_count
        state = supervisor.run("胎压报警了怎么办，另外座椅加热怎么关闭")
        self.assertGreater(self.client.request_count, before)
        agents = {result.agent for result in state.results}
        self.assertTrue({"vehicle_control", "manual_expert"} <= agents, agents)
        self.assertNotIn("create_service_order",
                         supervisor.specialists["vehicle_control"].registry.names())


class TestTracePassthrough(unittest.TestCase):
    def setUp(self):
        self.client = make_client()
        self.registry = McpToolRegistry(self.client)

    def tearDown(self):
        self.client.close()

    def test_trace_id_is_echoed_by_server(self):
        payload = self.client.call_tool("get_vehicle_status", {}, trace_id="trace-abc")
        self.assertEqual(payload["_meta"]["trace_id"], "trace-abc")

    def test_registry_records_client_and_server_latency(self):
        self.registry.call("get_vehicle_status", {})
        entry = self.registry.trace[-1]
        self.assertTrue(str(entry["trace_id"]).startswith("mcp-"))
        self.assertIsNotNone(entry["server_ms"])
        self.assertEqual(entry["transport"], "mcp")


class TestOrdersOverMcp(unittest.TestCase):
    """工单工具跨进程可用，且**用户隔离靠 `_meta.user_id` 随调用传递**。"""

    def setUp(self):
        self.client = make_client()
        self.alice = McpToolRegistry(self.client, user_id="alice")
        self.bob = McpToolRegistry(self.client, user_id="bob")

    def tearDown(self):
        self.client.close()

    def _create(self, registry, date="2099-06-18"):
        args = {"item": "更换机油机滤", "preferred_date": date, "center": "浦东店"}
        pending = registry.call("create_service_order", dict(args))
        self.assertTrue(pending.meta.get("needs_confirmation"), pending.error)
        registry.confirm(pending.meta["action_key"])
        return registry.call("create_service_order", dict(args))

    def test_orders_are_created_and_listed_over_mcp(self):
        created = self._create(self.alice)
        self.assertTrue(created.ok, created.error)
        listed = self.alice.call("get_my_orders", {})
        self.assertTrue(listed.ok, listed.error)
        self.assertEqual(len(listed.data["工单"]), 1)
        self.assertEqual(listed.data["工单"][0]["预约单号"], created.data["预约单号"])

    def test_other_user_cannot_see_or_cancel_the_order(self):
        created = self._create(self.alice)
        order_id = created.data["预约单号"]
        self.assertEqual(self.bob.call("get_my_orders", {}).data["工单"], [])
        pending = self.bob.call("cancel_order", {"order_id": order_id})
        self.bob.confirm(pending.meta["action_key"])
        denied = self.bob.call("cancel_order", {"order_id": order_id})
        self.assertFalse(denied.ok)
        self.assertEqual(denied.meta.get("code"), "not_found")
        # 原主人的单子还在
        self.assertEqual(self.alice.call("get_my_orders", {}).data["工单"][0]["状态"], "booked")


class TestScopeToken(unittest.TestCase):
    """服务端鉴权：token 决定能看/能调哪些工具，越权**在服务端**被拒（不是客户端白名单）。"""

    SECRET = "test-scope-secret"
    READONLY = ("search_manual", "get_maintenance_plan", "get_my_orders", "get_vehicle_status")

    def _client(self, token: str = ""):
        return McpClient(args=[*SERVER_ARGS, "--scope-secret", self.SECRET],
                         timeout=TIMEOUT, scope_token=token)

    def test_missing_token_is_rejected(self):
        with self.assertRaises(McpError) as ctx:
            self._client("").start()
        self.assertEqual(ctx.exception.code, FORBIDDEN)

    def test_bad_signature_is_rejected(self):
        token = issue_scope_token("repair", self.READONLY, "wrong-secret")
        with self.assertRaises(McpError) as ctx:
            self._client(token).start()
        self.assertEqual(ctx.exception.code, FORBIDDEN)

    def test_expired_token_is_rejected(self):
        token = issue_scope_token("repair", self.READONLY, self.SECRET, ttl_s=-10)
        with self.assertRaises(McpError) as ctx:
            self._client(token).start()
        self.assertEqual(ctx.exception.code, FORBIDDEN)

    def test_readonly_scope_filters_tools(self):
        client = self._client(issue_scope_token("repair", self.READONLY, self.SECRET)).start()
        try:
            self.assertEqual({t["name"] for t in client.list_tools()}, set(self.READONLY))
            self.assertEqual(client.scope_info.get("scope_subject"), "repair")
        finally:
            client.close()

    def test_write_call_is_rejected_by_server(self):
        client = self._client(issue_scope_token("repair", self.READONLY, self.SECRET)).start()
        try:
            before = client.request_count
            with self.assertRaises(McpError) as ctx:
                client.call_tool("create_service_order",
                                 {"item": "更换机油机滤", "preferred_date": "2099-06-18"})
            self.assertEqual(ctx.exception.code, FORBIDDEN)
            self.assertGreater(client.request_count, before, "请求必须到达服务端才被拒")
        finally:
            client.close()

    def test_repair_advisor_answers_through_scope(self):
        client = self._client(issue_scope_token("repair", self.READONLY, self.SECRET)).start()
        try:
            advisor = RepairAdvisor(client)
            self.assertEqual(set(advisor.tools), set(self.READONLY))
            result = advisor.ask("座椅加热怎么关闭")
            self.assertEqual(result["status"], "answered")
            self.assertTrue(result["citations"])
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
