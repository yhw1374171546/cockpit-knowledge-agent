# -*- coding: utf-8 -*-
"""SSE 断线重连与断连取消测试（BE7）。

要证明的两件事：
1. **重连能拿到已产出的内容**：事件带 `id:`，客户端带 `Last-Event-ID` + 同一个 `X-Trace-Id`
   重连时，服务端从缓冲补发——**不重跑 Agent、不再花一次钱**；
2. **断开能取消上游**：客户端走了，Agent 在下一步之前退出，而不是默默把 token 烧完。

权衡说明（写在 agent/service/sse.py 顶部）：**"断开即取消"与"重连续跑"是冲突的**。
我选取消优先（座舱里"用户切走"比"刚断了一下"更常见），因此重连只保证**部分重放**：
缓冲区里有的事件补发，原流没跑完则补一个 `interrupted`，明确告诉客户端"要完整答案请重新发起"。

    python agent/tests/test_sse.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi.testclient import TestClient  # noqa: E402

from agent.graph import AgentConfig, AgentGraph  # noqa: E402
from agent.kb import KnowledgeBase  # noqa: E402
from agent.llm import RuleBasedPlannerLLM  # noqa: E402
from agent.service.app import create_app  # noqa: E402
from agent.service.config import Settings  # noqa: E402
from agent.service.sse import SSEReplayBuffer  # noqa: E402
from agent.tools import ToolRegistry  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(ROOT, "eval", "fixtures", "mini_corpus.jsonl")


def make_settings(tmpdir: str) -> Settings:
    return Settings(
        database_url=f"sqlite:///{os.path.join(tmpdir, 'sse.db')}",
        kb_path=FIXTURE, jwt_secret="sse-test-secret-32-bytes-long!!",
        auth_required=True, dev_token_enabled=True, kv_backend="memory",
        rate_limit_capacity=10_000, rate_limit_refill_per_sec=10_000.0,
        quota_tokens_per_day=10_000_000, llm_backend="streaming",
        llm_ms_per_token=0.2, default_vehicle_model="lynk08",
        allowed_models=["lynk08", "lynk09"],
    )


class SSEBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.app = create_app(make_settings(self._tmp.name))
        self._ctx = TestClient(self.app)
        self.client = self._ctx.__enter__()
        token = self.client.post("/v1/auth/token",
                                 json={"user_id": "u1", "role": "driver",
                                       "vehicle_model": "lynk08"}).json()["access_token"]
        self.auth = {"Authorization": f"Bearer {token}"}

    def tearDown(self):
        try:
            self._ctx.__exit__(None, None, None)
        finally:
            self._tmp.cleanup()

    def stream(self, message: str, headers: dict = None) -> str:
        h = dict(self.auth)
        h.update(headers or {})
        with self.client.stream("POST", "/v1/chat/stream",
                                json={"message": message}, headers=h) as resp:
            return "".join(resp.iter_text())


def parse_events(text: str):
    """把 SSE 文本解析成 [(id, event, data_json_str)]。"""
    import json as _json

    out, cur = [], {}
    for line in text.splitlines():
        if not line.strip():
            if cur:
                out.append((cur.get("id"), cur.get("event"), cur.get("data")))
                cur = {}
            continue
        if line.startswith("id:"):
            cur["id"] = int(line[3:].strip())
        elif line.startswith("event:"):
            cur["event"] = line[6:].strip()
        elif line.startswith("data:"):
            cur["data"] = line[5:].strip()
    if cur:
        out.append((cur.get("id"), cur.get("event"), cur.get("data")))
    return [(i, e, _json.loads(d) if d else {}) for i, e, d in out]


class TestEventIds(SSEBase):
    def test_events_are_numbered_and_retry_hint_present(self):
        text = self.stream("座椅加热怎么关闭", {"X-Trace-Id": "trace-1"})
        events = parse_events(text)
        self.assertTrue(events)
        ids = [i for i, _, _ in events]
        self.assertEqual(ids, list(range(1, len(ids) + 1)),
                         f"事件 id 不连续：{ids}")
        self.assertIn("retry: 2000", text, "缺少 retry 提示，客户端不知道多久重连")
        self.assertEqual(events[-1][1], "done")


class TestReconnectReplay(SSEBase):
    def test_reconnect_replays_remainder_without_rerunning_agent(self):
        """原流跑完后重连：从 Last-Event-ID 之后补发，且**不再执行 Agent**。"""
        from unittest import mock

        trace = "trace-reconnect"
        self.stream("座椅加热怎么关闭", {"X-Trace-Id": trace})
        self.assertTrue(self.app.state.sse_buffer.has(trace))

        calls = {"n": 0}
        original = AgentGraph.run

        def counted(self_agent, *a, **kw):
            calls["n"] += 1
            return original(self_agent, *a, **kw)

        with mock.patch.object(AgentGraph, "run", counted):
            replay = self.stream("座椅加热怎么关闭",
                                 {"X-Trace-Id": trace, "Last-Event-ID": "3"})

        events = parse_events(replay)
        self.assertEqual(calls["n"], 0, "重连触发了 Agent 重跑，白白多花一次钱")
        self.assertTrue(events, "重连没有补发任何事件")
        self.assertGreater(min(i for i, _, _ in events), 3, "补发的事件 id 应从 Last-Event-ID 之后开始")
        self.assertEqual(events[-1][1], "done", "补发应以 done 收尾")

    def test_reconnect_unknown_stream_falls_back_to_fresh(self):
        """缓冲里没有这条流（过期/换了实例）：按新请求处理，不能报错。"""
        text = self.stream("座椅加热怎么关闭",
                           {"X-Trace-Id": "trace-unknown", "Last-Event-ID": "99"})
        events = parse_events(text)
        self.assertEqual(events[-1][1], "done")
        self.assertGreaterEqual(len(events), 2)

    def test_buffer_reports_replay_stats(self):
        trace = "trace-stats"
        self.stream("座椅加热怎么关闭", {"X-Trace-Id": trace})
        self.stream("座椅加热怎么关闭", {"X-Trace-Id": trace, "Last-Event-ID": "1"})
        stats = self.app.state.sse_buffer.stats_snapshot()
        self.assertGreaterEqual(stats["replays"], 1)
        self.assertGreaterEqual(stats["replayed_events"], 1)


class TestBufferUnit(unittest.TestCase):
    """缓冲本身的容量/TTL/中断语义（不依赖 HTTP，纯逻辑）。"""

    def test_replay_from_last_event_id(self):
        buf = SSEReplayBuffer()
        buf.open("s1")
        for i in range(1, 6):
            buf.append("s1", i, f"chunk{i}")
        buf.close("s1", completed=True)
        chunks, done = buf.replay("s1", 2)
        self.assertEqual(chunks, ["chunk3", "chunk4", "chunk5"])
        self.assertTrue(done)

    def test_interrupted_stream_not_marked_done(self):
        buf = SSEReplayBuffer()
        buf.open("s2")
        buf.append("s2", 1, "partial")
        buf.close("s2", completed=False)
        chunks, done = buf.replay("s2", None)
        self.assertEqual(chunks, ["partial"])
        self.assertFalse(done, "中断的流不能被标成已完成")
        self.assertEqual(buf.stats["interrupted"], 1)

    def test_ttl_expires_stream(self):
        buf = SSEReplayBuffer(ttl_s=0.2)
        buf.open("s3")
        buf.append("s3", 1, "x")
        import time as _t

        _t.sleep(0.3)
        chunks, done = buf.replay("s3", None)
        self.assertEqual(chunks, [])
        self.assertFalse(buf.has("s3"))

    def test_max_events_caps_stream(self):
        buf = SSEReplayBuffer(max_events=3)
        buf.open("s4")
        for i in range(1, 11):
            buf.append("s4", i, f"c{i}")
        chunks, _ = buf.replay("s4", None)
        self.assertEqual(chunks, ["c8", "c9", "c10"], "没有按上限裁剪，缓冲会无限增长")

    def test_max_streams_evicts_oldest(self):
        buf = SSEReplayBuffer(max_streams=2)
        for name in ("a", "b", "c"):
            buf.open(name)
            buf.append(name, 1, name)
        self.assertFalse(buf.has("a"), "超过上限时最旧的流应被淘汰")
        self.assertTrue(buf.has("c"))


class TestAmendEvent(unittest.TestCase):
    """流式增量与最终答案不一致时，必须补一个 amend 事件。

    线上抓到的问题：问「车上有冰箱吗」，徽标显示"已限定"，
    **气泡里却还是那 12V 电源原文**——因为可回答性校验（answerability）在 `_finalize`
    里改写答案，而流式增量在那之前已经发出去了。安全硬门控追加处置指令、
    接地失败改拒答，也是同一类问题（"先说了、再改口、但没告诉客户端"）。

    刻意**不继承 SSEBase**：那个基类已经起了一个 app，复用同一临时目录会让两个 app
    的数据库/生命周期互相干扰（我第一次写这个用例就栽在这里，表现为"只有一个 delta"）。
    """

    FRIDGE_TEXT = ("车载12V电源通过 12V电源插座，可以给电子设备"
                   "（例如：音乐播放器、车载冰箱和移动手机）充电。")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _client_with_fridge(self):
        import json

        corpus = os.path.join(self._tmp.name, "fridge.jsonl")
        with open(corpus, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": 1, "text": self.FRIDGE_TEXT,
                                 "source": "train_a.pdf", "page": 128},
                                ensure_ascii=False) + "\n")
        # 语料里只有这一块：问"有冰箱吗"必然命中它，从而触发限定回答
        settings = make_settings(self._tmp.name)
        settings.kb_path = corpus
        # 关掉语义缓存：否则第二次请求命中缓存后 delta 直接来自最终答案，amend 永不触发
        settings.enable_semantic_cache = False
        return TestClient(create_app(settings))

    def _token(self, client) -> dict:
        token = client.post("/v1/auth/token",
                            json={"user_id": "u1", "role": "driver",
                                  "vehicle_model": "lynk08"}).json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_stream_emits_amend_and_done_has_final_answer(self):
        with self._client_with_fridge() as client:
            auth = self._token(client)
            with client.stream("POST", "/v1/chat/stream",
                               json={"message": "车上有冰箱吗"}, headers=auth) as resp:
                text = "".join(resp.iter_text())

        events = parse_events(text)
        names = [e[1] for e in events]
        self.assertIn("amend", names, f"没有 amend 事件：{names}")
        self.assertEqual(names[-1], "done")
        self.assertLess(names.index("amend"), names.index("citations"),
                        "amend 应在 citations 之前发出")

        amend = next(d for _id, ev, d in events if ev == "amend")
        self.assertTrue(amend["text"].startswith("手册没有说明"),
                        f"amend 内容不是限定回答：{amend['text'][:60]}")
        self.assertEqual(amend["status"], "limited")

        # 已流出的增量确实是**改写前**的原文——这正是需要 amend 的原因
        deltas = "".join(d.get("text", "") for _id, ev, d in events if ev == "delta")
        self.assertIn("车载冰箱", deltas, "增量里应包含原文（证明它先于改写）")
        self.assertNotIn("手册没有说明", deltas, "增量不该已经是限定回答")

        # done.answer 永远是权威值
        done = events[-1][2]
        self.assertTrue(done["answer"].startswith("手册没有说明"))
        self.assertEqual(done["status"], "limited")

    def test_no_amend_when_answer_unchanged(self):
        """正常问答（答案没被改写）不该多发 amend——避免无意义事件。"""
        corpus = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "eval", "fixtures",
            "mini_corpus.jsonl")
        settings = make_settings(self._tmp.name)
        settings.kb_path = corpus
        settings.enable_semantic_cache = False
        with TestClient(create_app(settings)) as client:
            auth = self._token(client)
            with client.stream("POST", "/v1/chat/stream",
                               json={"message": "座椅加热怎么关闭"}, headers=auth) as resp:
                text = "".join(resp.iter_text())
        names = [e[1] for e in parse_events(text)]
        self.assertEqual(names[-1], "done")
        self.assertNotIn("amend", names, f"不该有 amend：{names}")


class TestCancelUpstream(unittest.TestCase):
    """断连取消：Agent 必须在下一步之前退出，而不是把 token 烧完。"""

    def setUp(self):
        self.kb = KnowledgeBase.load(FIXTURE)

    def _agent(self, cancel_event=None) -> AgentGraph:
        config = AgentConfig()
        config.cancel_event = cancel_event
        return AgentGraph(RuleBasedPlannerLLM(), ToolRegistry(self.kb), config=config)

    def test_pre_cancelled_agent_exits_immediately(self):
        import threading as _t

        event = _t.Event()
        event.set()
        state = self._agent(event).run("座椅加热怎么关闭")
        self.assertEqual(state.status, "cancelled")
        self.assertEqual(state.steps, 0, "取消后仍执行了步骤")
        self.assertTrue(any(step.get("node") == "cancel" for step in state.trace),
                        "trace 里没有记录取消原因")

    def test_not_cancelled_runs_normally(self):
        import threading as _t

        state = self._agent(_t.Event()).run("座椅加热怎么关闭")
        self.assertNotEqual(state.status, "cancelled")
        self.assertGreater(state.steps, 0)

    def test_no_event_configured_is_safe(self):
        state = self._agent(None).run("座椅加热怎么关闭")
        self.assertNotEqual(state.status, "cancelled")


if __name__ == "__main__":
    unittest.main(verbosity=2)
