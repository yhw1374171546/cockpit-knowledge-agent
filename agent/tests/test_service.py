# -*- coding: utf-8 -*-
"""服务层接口测试：鉴权、租户、限流、配额、幂等、SSE 流式、写操作确认、审计、指标。

运行：
    python agent/tests/test_service.py

设计说明：
- 用 **内置迷你语料**（`eval/fixtures/mini_corpus.jsonl`）+ **SQLite** 跑，保证 CI 可执行；
- 写操作路径用**脚本化 LLM**（离线规则规划器不会主动下预约单），
  这样才能真正压到「确认 → 幂等键 → 落库 → 审计」这条链路上。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi.testclient import TestClient                                   # noqa: E402

from agent.llm import LLMBackend, LLMResponse, ToolCall                     # noqa: E402
from agent.service.app import _load_history, create_app                     # noqa: E402
from agent.service.config import Settings                                   # noqa: E402
from agent.service.db import SessionRow, ToolCallRow                        # noqa: E402
from agent.service.security import IdempotencyStore                         # noqa: E402
from agent.tools import ToolRegistry, build_default_registry                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(ROOT, "eval", "fixtures", "mini_corpus.jsonl")


class ScriptedWriteLLM(LLMBackend):
    """脚本化后端：先请求写操作，拿到结果后给出最终答复。用于覆盖写操作链路。"""

    name = "scripted-write"

    def __init__(self, item: str = "更换机油机滤", date: str = "2025-06-18"):
        self.item, self.date = item, date
        self.turns = 0

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024, guided_json=None):
        self.turns += 1
        if not any(m.get("role") == "tool" for m in messages):
            return LLMResponse(tool_calls=[ToolCall("create_service_order",
                                                    {"item": self.item,
                                                     "preferred_date": self.date})])
        return LLMResponse(content=f"已按您的要求处理：{self.item}。")


def make_settings(tmpdir: str, **overrides) -> Settings:
    base = dict(
        database_url=f"sqlite:///{os.path.join(tmpdir, 'service_test.db')}",
        kb_path=FIXTURE,
        jwt_secret="test-secret",
        auth_required=True,
        dev_token_enabled=True,
        kv_backend="memory",
        rate_limit_capacity=1000,
        rate_limit_refill_per_sec=1000.0,
        quota_tokens_per_day=10_000_000,
        llm_backend="planner",
        default_vehicle_model="lynk08",
        allowed_models=["lynk08", "lynk09"],
    )
    base.update(overrides)
    return Settings(**base)


class ServiceTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.settings = make_settings(self._tmp.name)
        self.app = create_app(self.settings)
        self.client_ctx = TestClient(self.app)
        self.client = self.client_ctx.__enter__()
        self.token = self._token("u1", "driver", "lynk08")
        self.h = self._auth_headers(self.token)

    def tearDown(self):
        try:
            self.client_ctx.__exit__(None, None, None)
        finally:
            self._tmp.cleanup()

    # ── 工具方法 ──
    def _token(self, user_id: str, role: str = "driver", model: str = "lynk08") -> str:
        resp = self.client.post("/v1/auth/token",
                                json={"user_id": user_id, "role": role, "vehicle_model": model})
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()["access_token"]

    @staticmethod
    def _auth_headers(token: str, extra: dict = None) -> dict:
        headers = {"Authorization": f"Bearer {token}"}
        headers.update(extra or {})
        return headers

    def chat(self, message: str, headers: dict = None, **body):
        payload = {"message": message}
        payload.update(body)
        return self.client.post("/v1/chat", json=payload, headers=headers or self.h)


class TestOpsEndpoints(ServiceTestBase):
    def test_healthz(self):
        resp = self.client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_readyz_reports_dependencies(self):
        resp = self.client.get("/readyz")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["database"])
        self.assertEqual(body["kv_backend"], "memory")
        self.assertGreater(body["kb_chunks"], 0)

    def test_openapi_contract_generated(self):
        spec = self.client.get("/openapi.json").json()
        self.assertIn("/v1/chat", spec["paths"])
        self.assertIn("/v1/chat/stream", spec["paths"])
        self.assertEqual(spec["info"]["version"], self.settings.version)

    def test_tools_endpoint_exposes_function_schema(self):
        resp = self.client.get("/v1/tools", headers=self.h)
        self.assertEqual(resp.status_code, 200)
        names = {t["name"] for t in resp.json()}
        self.assertIn("search_manual", names)
        self.assertIn("create_service_order", names)


class TestAuthAndTenant(ServiceTestBase):
    def test_missing_token_is_401(self):
        resp = self.client.post("/v1/chat", json={"message": "座椅加热怎么关闭"})
        self.assertEqual(resp.status_code, 401)

    def test_invalid_token_is_401(self):
        resp = self.client.post("/v1/chat", json={"message": "hi"},
                                headers={"Authorization": "Bearer not-a-token"})
        self.assertEqual(resp.status_code, 401)

    def test_cross_tenant_is_403(self):
        # 令牌绑定lynk08，却请求lynk09 → 403
        resp = self.chat("座椅加热怎么关闭", self._auth_headers(self.token,
                                                             {"X-Vehicle-Model": "lynk09"}))
        self.assertEqual(resp.status_code, 403)

    def test_unknown_model_is_403(self):
        # 租户标识必须 ASCII（HTTP 头限制），所以用代号而不是中文车型名
        resp = self.chat("座椅加热怎么关闭",
                         self._auth_headers(self.token, {"X-Vehicle-Model": "tesla-model3"}))
        self.assertEqual(resp.status_code, 403)

    def test_trace_id_is_echoed(self):
        resp = self.chat("座椅加热怎么关闭", self._auth_headers(self.token,
                                                             {"X-Trace-Id": "trace-abc"}))
        self.assertEqual(resp.headers.get("X-Trace-Id"), "trace-abc")
        self.assertEqual(resp.json()["trace_id"], "trace-abc")


class TestChatBasic(ServiceTestBase):
    def test_answer_with_citations(self):
        resp = self.chat("怎么打开危险警告灯")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["status"], "answered")
        self.assertIn("危险警告灯", body["answer"])
        self.assertTrue(body["citations"])
        self.assertGreater(body["tokens_in"], 0)
        self.assertGreaterEqual(body["cost_usd"], 0)

    def test_refusal_for_out_of_domain(self):
        body = self.chat("中国足球的队长是谁").json()
        self.assertEqual(body["answer"].strip(), "无答案")
        self.assertEqual(body["status"], "refused")

    def test_session_is_created_and_reused(self):
        first = self.chat("座椅加热怎么关闭").json()
        sid = first["session_id"]
        second = self.chat("危险警告灯怎么开", session_id=sid).json()
        self.assertEqual(second["session_id"], sid)
        listing = self.client.get("/v1/sessions", headers=self.h).json()
        self.assertIn(sid, [s["session_id"] for s in listing])

    def test_feedback_accepted(self):
        body = self.chat("座椅加热怎么关闭").json()
        resp = self.client.post("/v1/feedback",
                                json={"message_id": body["message_id"], "rating": 1,
                                      "comment": "有帮助"}, headers=self.h)
        self.assertEqual(resp.status_code, 201)


class TestRateLimitAndQuota(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_rate_limit_returns_429_with_retry_after(self):
        settings = make_settings(self._tmp.name, rate_limit_capacity=3,
                                 rate_limit_refill_per_sec=1.0)
        app = create_app(settings)
        with TestClient(app) as client:
            token = client.post("/v1/auth/token",
                                json={"user_id": "u1", "vehicle_model": "lynk08"}).json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            codes = [client.post("/v1/chat", json={"message": "座椅加热怎么关闭"},
                                 headers=headers).status_code for _ in range(5)]
            self.assertIn(429, codes)
            self.assertEqual(codes[:3], [200, 200, 200])
            blocked = client.post("/v1/chat", json={"message": "座椅加热怎么关闭"},
                                  headers=headers)
            self.assertEqual(blocked.status_code, 429)
            self.assertIn("Retry-After", blocked.headers)

    def test_quota_exceeded_returns_429(self):
        settings = make_settings(self._tmp.name, quota_tokens_per_day=100)
        app = create_app(settings)
        with TestClient(app) as client:
            token = client.post("/v1/auth/token",
                                json={"user_id": "u2", "vehicle_model": "lynk08"}).json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            first = client.post("/v1/chat", json={"message": "座椅加热怎么关闭"}, headers=headers)
            self.assertEqual(first.status_code, 200)
            self.assertGreater(first.json()["quota"]["used"], 100)   # 单次就超了配额
            second = client.post("/v1/chat", json={"message": "座椅加热怎么关闭"}, headers=headers)
            self.assertEqual(second.status_code, 429)
            self.assertIn("配额", second.json()["error"])


class TestIdempotency(ServiceTestBase):
    def test_replay_returns_first_response(self):
        headers = self._auth_headers(self.token, {"Idempotency-Key": "idem-1"})
        first = self.chat("座椅加热怎么关闭", headers).json()
        second = self.chat("座椅加热怎么关闭", headers).json()
        self.assertFalse(first["idempotent_replay"])
        self.assertTrue(second["idempotent_replay"])
        self.assertEqual(first["message_id"], second["message_id"])
        self.assertEqual(first["answer"], second["answer"])

    def test_same_key_different_body_is_422(self):
        headers = self._auth_headers(self.token, {"Idempotency-Key": "idem-2"})
        self.chat("座椅加热怎么关闭", headers)
        resp = self.chat("危险警告灯怎么开", headers)
        self.assertEqual(resp.status_code, 422)
        self.assertIn("不同的请求体", resp.json()["error"])

    def test_in_flight_key_is_409(self):
        headers = self._auth_headers(self.token, {"Idempotency-Key": "idem-3"})
        store = IdempotencyStore(self.app.state.db, self.settings)
        store.begin("u1", "idem-3", "/v1/chat",
                    store.hash_request({"message": "座椅加热怎么关闭", "model": "lynk08"}))
        resp = self.chat("座椅加热怎么关闭", headers)
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.headers.get("Retry-After"), "1")


class TestStreaming(ServiceTestBase):
    def setUp(self):
        super().setUp()
        # 换成"逐块流式"后端：TTFT ≈ 检索耗时 + 一块时延
        from agent.llm import StreamingRulePlanner
        self.app.state.runtime.build_llm = lambda on_delta=None: StreamingRulePlanner(
            ms_per_chunk=15.0, chunk_size=4)

    def parse_sse(self, text: str):
        events = []
        for block in text.strip().split("\n\n"):
            if not block.strip():
                continue
            event, data = None, None
            for line in block.split("\n"):
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
            events.append((event, data))
        return events

    def test_sse_event_sequence(self):
        resp = self.client.post("/v1/chat/stream", json={"message": "座椅加热怎么关闭"},
                                headers=self.h)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.headers["content-type"].startswith("text/event-stream"))
        events = self.parse_sse(resp.text)
        names = [e for e, _ in events]
        self.assertEqual(names[0], "meta")
        self.assertEqual(names[1], "stage")
        self.assertIn("delta", names)
        self.assertEqual(names[-1], "done")
        self.assertIn("citations", names)

        done = [d for e, d in events if e == "done"][0]
        deltas = [d for e, d in events if e == "delta"]
        self.assertEqual("".join(d["text"] for d in deltas), done["answer"])
        # 流式：首字延迟应显著小于总耗时
        self.assertIsNotNone(done["ttft_ms"])
        self.assertLess(done["ttft_ms"], done["latency_ms"])

    def test_stream_matches_non_stream_answer(self):
        streamed = self.parse_sse(self.client.post(
            "/v1/chat/stream", json={"message": "怎么打开危险警告灯"}, headers=self.h).text)
        done = [d for e, d in streamed if e == "done"][0]
        plain = self.chat("怎么打开危险警告灯").json()
        self.assertEqual(done["answer"], plain["answer"])

    def test_stream_deltas_are_progressive_and_ordered(self):
        """验证流式语义：多个 delta 分块下发、顺序在 done 之前、首字延迟小于总耗时。

        说明：TestClient 会把响应缓冲后再交给调用方，因此这里做**结构性**验证
        （事件顺序 + 分块数量 + TTFT < 总耗时），而不是用挂钟时间测量网络到达时刻。
        """
        resp = self.client.post("/v1/chat/stream", json={"message": "座椅加热怎么关闭"},
                                headers=self.h)
        events = self.parse_sse(resp.text)
        names = [e for e, _ in events]
        deltas = [d for e, d in events if e == "delta"]
        done = [d for e, d in events if e == "done"][0]
        self.assertGreaterEqual(len(deltas), 2, "应分多块下发")
        self.assertLess(names.index("delta"), names.index("done"))
        # 每块的耗时递增 → 说明是逐块产出而非一次性拼好
        self.assertEqual(sorted(d["elapsed_ms"] for d in deltas),
                         [d["elapsed_ms"] for d in deltas])
        self.assertLess(done["ttft_ms"], done["latency_ms"])

    def test_idempotent_stream_replay(self):
        headers = self._auth_headers(self.token, {"Idempotency-Key": "idem-stream"})
        self.chat("座椅加热怎么关闭", headers)
        resp = self.client.post("/v1/chat/stream", json={"message": "座椅加热怎么关闭"},
                                headers=headers)
        events = self.parse_sse(resp.text)
        done = [d for e, d in events if e == "done"][0]
        self.assertTrue(done["idempotent_replay"])


class TestWriteFlow(ServiceTestBase):
    def setUp(self):
        super().setUp()
        self.item, self.date = "更换机油机滤", "2025-06-18"
        self.app.state.runtime.build_llm = lambda on_delta=None: ScriptedWriteLLM(
            self.item, self.date)
        self.action_key = f"create_service_order:{self.item}:{self.date}"

    def test_write_requires_confirmation_then_succeeds_with_idempotency(self):
        # ① 未确认 → needs_confirmation
        first = self.chat("帮我预约到店保养").json()
        self.assertEqual(first["status"], "needs_confirmation")
        self.assertEqual(first["needs_confirmation"], self.item)
        self.assertTrue(any(t["blocked"] for t in first["tool_calls"]))
        self.assertIn("确认", first["tool_calls"][0]["reason"])

        # ② 确认落库
        resp = self.client.post("/v1/writes/confirm",
                                json={"action_key": self.action_key}, headers=self.h)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["confirmed"])

        # ③ 再次请求（带幂等键）→ 写操作真正执行
        headers = self._auth_headers(self.token, {"Idempotency-Key": "write-1"})
        second = self.chat("帮我预约到店保养", headers=headers).json()
        write_calls = [t for t in second["tool_calls"] if t["tool"] == "create_service_order"]
        self.assertTrue(write_calls, second)
        self.assertTrue(write_calls[0]["ok"], write_calls)
        self.assertEqual(write_calls[0]["idempotency_key"], "write-1")

    def test_confirmed_write_without_idempotency_key_is_blocked(self):
        """服务层收紧策略：确认过了，但没带幂等键 → 依然拒绝，防客户端重试重复下单。"""
        self.client.post("/v1/writes/confirm", json={"action_key": self.action_key},
                         headers=self.h)
        body = self.chat("帮我预约到店保养").json()          # 注意：没带 Idempotency-Key
        write_calls = [t for t in body["tool_calls"] if t["tool"] == "create_service_order"]
        self.assertTrue(write_calls)
        self.assertTrue(write_calls[0]["blocked"])
        self.assertIn("Idempotency-Key", write_calls[0]["reason"])

    def test_write_is_audited_and_persisted(self):
        self.client.post("/v1/writes/confirm", json={"action_key": self.action_key},
                         headers=self.h)
        headers = self._auth_headers(self.token, {"Idempotency-Key": "write-audit"})
        self.chat("帮我预约到店保养", headers=headers)
        with self.app.state.db.session() as db:
            rows = db.query(ToolCallRow).filter(ToolCallRow.tool == "create_service_order").all()
        self.assertTrue(rows)
        self.assertEqual(rows[-1].idempotency_key, "write-audit")
        self.assertTrue(rows[-1].ok)

        admin = self._token("admin", "admin", "lynk08")
        audit = self.client.get("/v1/admin/audit",
                                headers=self._auth_headers(admin)).json()
        self.assertTrue(any(a["action"] == "write_tool" for a in audit))


class TestAdmin(ServiceTestBase):
    def test_driver_cannot_read_admin(self):
        self.assertEqual(self.client.get("/v1/admin/audit", headers=self.h).status_code, 403)
        self.assertEqual(self.client.get("/v1/admin/metrics", headers=self.h).status_code, 403)

    def test_metrics_aggregation(self):
        self.chat("座椅加热怎么关闭")
        self.chat("怎么打开危险警告灯")
        admin = self._token("admin", "admin")
        metrics = self.client.get("/v1/admin/metrics",
                                  headers=self._auth_headers(admin)).json()
        self.assertGreaterEqual(metrics["messages"], 2)
        self.assertGreater(metrics["tokens_total"], 0)
        self.assertIn("failure_distribution", metrics)
        self.assertEqual(metrics["kv_backend"], "memory")


class TestRegistryReuse(ServiceTestBase):
    """锁死一个性能 bug：服务层曾每请求重建知识库索引。

    实测（9301 块）：`KnowledgeBase.load` ≈14.6 s、`build_default_registry` ≈16.3 s，
    而单次 BM25 检索只要 ≈12 ms——即每请求白花十几秒，压测 QPS 只有 0.06。
    这里用"对象同一性"来断言注册表复用了运行时缓存，而不是靠计时（CI 上计时不稳定）。
    """

    def test_registry_reuses_cached_kb(self):
        runtime = self.app.state.runtime
        kb = runtime.kb(self.settings.default_vehicle_model)
        registry = build_default_registry(self.settings.kb_path or None, kb=kb)
        self.assertIs(registry.kb, kb, "注册表没有复用缓存的知识库实例（会每请求重建索引）")

    def test_build_agent_reuses_cached_kb(self):
        runtime = self.app.state.runtime
        kb = runtime.kb(self.settings.default_vehicle_model)
        agent, _tracer, registry = runtime.build_agent(self.settings.default_vehicle_model)
        self.assertIs(registry.kb, kb, "build_agent 没有复用缓存的知识库实例")

    def test_two_registries_share_same_kb(self):
        runtime = self.app.state.runtime
        _a, _, reg_a = runtime.build_agent(self.settings.default_vehicle_model)
        _b, _, reg_b = runtime.build_agent(self.settings.default_vehicle_model)
        self.assertIs(reg_a.kb, reg_b.kb, "两次构建拿到了不同的 KB 实例（缓存未命中）")


class TestDegradationAndMetrics(unittest.TestCase):
    """故障注入 → 降级链；以及 Prometheus 指标端点。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def _client(self, **overrides):
        settings = make_settings(self._tmp.name, **overrides)
        app = create_app(settings)
        # raise_server_exceptions=False：让"未降级时的 5xx"以响应返回，而不是在测试里抛出
        ctx = TestClient(app, raise_server_exceptions=False)
        client = ctx.__enter__()
        token = client.post("/v1/auth/token",
                            json={"user_id": "u1", "vehicle_model": "lynk08"}).json()["access_token"]
        return ctx, client, {"Authorization": f"Bearer {token}"}

    def test_llm_failure_falls_back_to_offline_planner(self):
        """主推理后端全挂 → 仍返回 200，但标记 degraded（座舱不至于彻底失能）。"""
        ctx, client, headers = self._client(fault_mode="error", enable_degradation=True)
        try:
            body = client.post("/v1/chat", json={"message": "座椅加热怎么关闭"},
                               headers=headers).json()
            self.assertEqual(body["status"], "answered", body)
            self.assertTrue(body["degraded"], "没有标记降级")
            self.assertIn("注入故障", body["degrade_reason"])
        finally:
            ctx.__exit__(None, None, None)

    def test_without_degradation_fault_returns_5xx(self):
        """关掉降级链 → 注入故障应当直接失败（说明降级确实在起作用）。"""
        ctx, client, headers = self._client(fault_mode="error", enable_degradation=False)
        try:
            resp = client.post("/v1/chat", json={"message": "座椅加热怎么关闭"},
                               headers=headers)
            self.assertGreaterEqual(resp.status_code, 500)
        finally:
            ctx.__exit__(None, None, None)

    def test_metrics_endpoint_requires_privilege_by_default(self):
        ctx, client, headers = self._client()
        try:
            self.assertEqual(client.get("/metrics", headers=headers).status_code, 403)
            admin = client.post("/v1/auth/token",
                                json={"user_id": "a", "role": "admin",
                                      "vehicle_model": "lynk08"}).json()["access_token"]
            resp = client.get("/metrics", headers={"Authorization": f"Bearer {admin}"})
            self.assertEqual(resp.status_code, 200)
            text = resp.text
            for metric in ("agent_up", "agent_http_requests_total",
                           "agent_request_duration_seconds_bucket",
                           "agent_tokens_total", "agent_tool_calls_total"):
                self.assertIn(metric, text, f"指标缺失：{metric}")
        finally:
            ctx.__exit__(None, None, None)

    def test_metrics_public_mode(self):
        ctx, client, _headers = self._client(metrics_public=True)
        try:
            resp = client.get("/metrics")
            self.assertEqual(resp.status_code, 200)
            self.assertIn("agent_up", resp.text)
        finally:
            ctx.__exit__(None, None, None)

    def test_metrics_count_refusals_and_tool_calls(self):
        ctx, client, headers = self._client(metrics_public=True)
        try:
            client.post("/v1/chat", json={"message": "座椅加热怎么关闭"}, headers=headers)
            client.post("/v1/chat", json={"message": "中国足球的队长是谁"}, headers=headers)
            text = client.get("/metrics").text
            self.assertIn('agent_refusals_total{instance=', text)
            # 标签按字典序渲染，所以只断言键值对本身
            self.assertIn('tool="search_manual"', text)
            self.assertIn("agent_tokens_total", text)
        finally:
            ctx.__exit__(None, None, None)


class TestSessionIsolation(ServiceTestBase):
    def test_other_user_cannot_use_session(self):
        sid = self.chat("座椅加热怎么关闭").json()["session_id"]
        other = self._token("u2", "driver", "lynk08")
        resp = self.chat("危险警告灯怎么开", self._auth_headers(other), session_id=sid)
        self.assertEqual(resp.status_code, 403)

    def test_messages_persisted_per_session(self):
        body = self.chat("座椅加热怎么关闭").json()
        with self.app.state.db.session() as db:
            row = db.get(SessionRow, body["session_id"])
            self.assertIsNotNone(row)
            self.assertEqual(row.user_id, "u1")


class TestMultiTurnMemory(ServiceTestBase):
    """会话记忆必须**落库并回填**——否则服务层的多轮是断的。

    这是本项目的真实缺口：`sessions`/`messages` 表存了历史，但每个请求都新建
    一份空的 `ConversationMemory`，于是"它怎么关闭"这类追问无法解析指代。
    """

    def test_history_is_loaded_from_db(self):
        first = self.chat("座椅加热怎么关闭").json()
        sid = first["session_id"]
        history = _load_history(self.app.state.db, sid)
        self.assertTrue(history, "会话历史没有从数据库读出来")
        self.assertEqual(history[0][0], "user")
        self.assertIn("座椅加热", history[0][1])

    def test_pronoun_followup_resolves_with_context(self):
        """追问用代词时，应能借助历史解析指代（否则会答成域外拒答）。"""
        sid = self.chat("座椅加热怎么关闭").json()["session_id"]
        follow = self.chat("它怎么重新打开", session_id=sid).json()
        self.assertEqual(follow["session_id"], sid)
        # 有上下文时不该直接拒答
        self.assertNotEqual(follow["answer"].strip(), "无答案",
                            f"追问被拒答，说明上下文没生效：{follow}")

    def test_history_is_capped(self):
        """历史不能无限增长（成本与上下文都要控）。"""
        sid = self.chat("座椅加热怎么关闭").json()["session_id"]
        for i in range(4):
            self.chat(f"第{i}个问题：危险警告灯怎么开", session_id=sid)
        history = _load_history(self.app.state.db, sid, max_turns=3)
        self.assertLessEqual(len(history), 6, "历史轮数未被裁剪")


if __name__ == "__main__":
    unittest.main(verbosity=2)
