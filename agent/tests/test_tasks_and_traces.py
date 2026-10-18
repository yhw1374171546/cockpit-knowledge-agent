# -*- coding: utf-8 -*-
"""后台任务队列（BE3）与追踪落库/回放（BE4）测试。

BE3 要证明的：提交 → 原子认领 → 执行 → 查结果，且**失败可重试、错误必须留痕**。
BE4 要证明的：一次请求的 spans / 决策轨迹 / 成本能落库，并能按 trace_id 与会话回放。

    python agent/tests/test_tasks_and_traces.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi.testclient import TestClient  # noqa: E402

from agent.service.app import create_app  # noqa: E402
from agent.service.config import Settings  # noqa: E402
from agent.service.db import get_database, load_trace  # noqa: E402
from agent.service.tasks import (  # noqa: E402
    FAILED,
    QUEUED,
    SUCCEEDED,
    TaskWorker,
    make_noop_handler,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(ROOT, "eval", "fixtures", "mini_corpus.jsonl")


def make_settings(tmpdir: str, **overrides) -> Settings:
    base = dict(
        database_url=f"sqlite:///{os.path.join(tmpdir, 'tasks.db')}",
        kb_path=FIXTURE, jwt_secret="task-test-secret-32-bytes!!",
        auth_required=True, dev_token_enabled=True, kv_backend="memory",
        rate_limit_capacity=10_000, rate_limit_refill_per_sec=10_000.0,
        quota_tokens_per_day=10_000_000, llm_backend="planner",
        default_vehicle_model="lynk08", allowed_models=["lynk08"],
        # 测试里不让后台线程自己跑，改为手动驱动 run_once()，避免时序不稳定
        enable_task_worker=False,
    )
    base.update(overrides)
    return Settings(**base)


class QueueBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.app = create_app(make_settings(self._tmp.name))
        self._ctx = TestClient(self.app)
        self.client = self._ctx.__enter__()
        self.db = get_database(f"sqlite:///{os.path.join(self._tmp.name, 'tasks.db')}")
        self.queue = self.app.state.task_queue
        self.svc = {"Authorization": "Bearer " + self.client.post(
            "/v1/auth/token",
            json={"user_id": "svc", "role": "service",
                  "vehicle_model": "lynk08"}).json()["access_token"]}
        self.driver = {"Authorization": "Bearer " + self.client.post(
            "/v1/auth/token",
            json={"user_id": "u1", "role": "driver",
                  "vehicle_model": "lynk08"}).json()["access_token"]}

    def tearDown(self):
        try:
            self._ctx.__exit__(None, None, None)
        finally:
            self._tmp.cleanup()


class TestTaskQueue(QueueBase):
    def test_enqueue_and_manual_run_succeeds(self):
        info = self.queue.enqueue("noop", {"x": 1})
        self.assertEqual(info.status, QUEUED)
        worker = TaskWorker(self.queue, {"noop": make_noop_handler()})
        self.assertTrue(worker.run_once())
        done = self.queue.get(info.id)
        self.assertEqual(done.status, SUCCEEDED)
        self.assertEqual(done.attempts, 1)
        self.assertEqual(done.result["echo"], {"x": 1})

    def test_claim_is_atomic_single_owner(self):
        """两个 worker 不能认领同一个任务（否则索引会被建两遍）。"""
        info = self.queue.enqueue("noop", {})
        first = self.queue.claim()
        second = self.queue.claim()
        self.assertIsNotNone(first)
        self.assertIsNone(second, "同一个任务被认领了两次")
        self.assertEqual(first[0].id, info.id)

    def test_failure_is_recorded_and_retried_then_failed(self):
        """失败必须留痕，并且在上限内可重试；超过上限才置 failed。"""
        def boom(_payload):
            raise RuntimeError("模拟索引构建失败")

        info = self.queue.enqueue("noop", {})
        worker = TaskWorker(self.queue, {"noop": boom})
        # 第 1、2 次失败 → 退回 queued（可重试）
        for _ in range(2):
            self.assertTrue(worker.run_once())
            current = self.queue.get(info.id)
            self.assertEqual(current.status, QUEUED, "未到上限时应退回排队重试")
            self.assertIn("模拟索引构建失败", current.error)
        # 第 3 次失败 → 置 failed
        self.assertTrue(worker.run_once())
        final = self.queue.get(info.id)
        self.assertEqual(final.status, FAILED)
        self.assertEqual(final.attempts, 3)
        self.assertIn("RuntimeError", final.error)

    def test_unknown_kind_fails_fast(self):
        info = self.queue.enqueue("不存在的任务类型", {})
        worker = TaskWorker(self.queue, {})
        self.assertTrue(worker.run_once())
        self.assertEqual(self.queue.get(info.id).status, QUEUED)   # 未到上限 → 重试
        for _ in range(2):
            worker.run_once()
        self.assertEqual(self.queue.get(info.id).status, FAILED)
        self.assertIn("未知任务类型", self.queue.get(info.id).error)

    def test_stats_counts(self):
        self.queue.enqueue("noop", {})
        self.queue.enqueue("noop", {})
        stats = self.queue.stats()
        self.assertGreaterEqual(stats["queued"], 2)


class TestTaskApi(QueueBase):
    def test_submit_and_query_via_api(self):
        created = self.client.post("/v1/admin/tasks",
                                   json={"kind": "noop", "payload": {"n": 7}},
                                   headers=self.svc)
        self.assertEqual(created.status_code, 200, created.text)
        task_id = created.json()["id"]
        TaskWorker(self.queue, {"noop": make_noop_handler()}).run_once()
        fetched = self.client.get(f"/v1/admin/tasks/{task_id}", headers=self.svc)
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.json()["status"], SUCCEEDED)
        listed = self.client.get("/v1/admin/tasks", headers=self.svc).json()
        self.assertTrue(any(t["id"] == task_id for t in listed["tasks"]))

    def test_task_endpoints_require_service_role(self):
        self.assertEqual(self.client.post("/v1/admin/tasks",
                                          json={"kind": "noop"}, headers=self.driver
                                          ).status_code, 403)
        self.assertEqual(self.client.get("/v1/admin/tasks",
                                         headers=self.driver).status_code, 403)

    def test_unknown_task_returns_404(self):
        self.assertEqual(self.client.get("/v1/admin/tasks/nope", headers=self.svc
                                         ).status_code, 404)

    def test_kb_build_handler_registers_version(self):
        """kb_build 任务：把一份语料注册成新版本（耗时的加载不在请求线程里做）。"""
        import json as _json

        target = os.path.join(self._tmp.name, "corpus_task.jsonl")
        rows = [_json.loads(line) for line in open(FIXTURE, encoding="utf-8") if line.strip()]
        rows.append({"id": 9001, "text": "（任务版）电动尾门支持脚踢开启。", "source": "t.pdf"})
        with open(target, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(_json.dumps(row, ensure_ascii=False) + "\n")

        info = self.queue.enqueue("kb_build", {"path": target, "vehicle_model": "lynk08"})
        from agent.service.tasks import make_kb_build_handler

        worker = TaskWorker(self.queue,
                            {"kb_build": make_kb_build_handler(self.app.state.kb_registry)})
        self.assertTrue(worker.run_once())
        done = self.queue.get(info.id)
        self.assertEqual(done.status, SUCCEEDED, done.error)
        self.assertTrue(done.result["version"].startswith("v"))
        self.assertIsNotNone(self.app.state.kb_registry.get(done.result["version"]),
                             "任务执行完但注册表里没有新版本")


class TestTracePersistence(QueueBase):
    def _chat(self, message: str, session_id: str = None) -> dict:
        body = {"message": message}
        if session_id:
            body["session_id"] = session_id
        return self.client.post("/v1/chat", json=body, headers=self.driver).json()

    def test_trace_saved_and_queryable(self):
        body = self._chat("座椅加热怎么关闭")
        trace_id = body["trace_id"]
        record = load_trace(self.db, trace_id)
        self.assertIsNotNone(record, "trace 没有落库")
        self.assertEqual(record["status"], body["status"])
        self.assertEqual(record["kb_version"], body["kb_version"])
        self.assertIn("spans", record["payload"])
        self.assertIn("trace", record["payload"])
        self.assertEqual(record["payload"]["resolved_question"], "座椅加热怎么关闭")

    def test_trace_api_and_role(self):
        trace_id = self._chat("座椅加热怎么关闭")["trace_id"]
        ok = self.client.get(f"/v1/admin/traces/{trace_id}", headers=self.svc)
        self.assertEqual(ok.status_code, 200, ok.text)
        self.assertEqual(ok.json()["trace_id"], trace_id)
        self.assertEqual(self.client.get(f"/v1/admin/traces/{trace_id}",
                                         headers=self.driver).status_code, 403)
        self.assertEqual(self.client.get("/v1/admin/traces/nope", headers=self.svc
                                         ).status_code, 404)

    def test_session_replay_is_chronological(self):
        """按会话回放：时间正序，且能把成本累加出来。"""
        first = self._chat("座椅加热怎么关闭")
        session_id = first["session_id"]
        self._chat("胎压多少算正常", session_id=session_id)
        self._chat("危险警告灯怎么用", session_id=session_id)

        replay = self.client.get(f"/v1/admin/sessions/{session_id}/traces",
                                 headers=self.svc).json()
        self.assertGreaterEqual(replay["n"], 3)
        ids = [t["trace_id"] for t in replay["traces"]]
        self.assertEqual(ids[0], first["trace_id"], "回放顺序不是时间正序")
        self.assertGreaterEqual(replay["total_cost_usd"], 0.0)
        for item in replay["traces"]:
            self.assertIn("status", item)
            self.assertIn("latency_ms", item)

    def test_trace_written_in_same_transaction_as_message(self):
        """trace 与业务数据同事务：不会出现"有答案没 trace"或反过来。"""
        body = self._chat("座椅加热怎么关闭")
        record = load_trace(self.db, body["trace_id"])
        self.assertIsNotNone(record)
        self.assertEqual(record["session_id"], body["session_id"])
        self.assertEqual(record["tokens_out"], body["tokens_out"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
