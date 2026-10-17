# -*- coding: utf-8 -*-
"""数据库迁移与知识库版本持久化测试（BE5）。

三件事要证明：
1. **迁移可用**：空库 `alembic upgrade head` 能建出全部表与新增列，`downgrade` 能回滚；
2. **表真的被用起来了**：`kb_versions` 以前"定义了但零使用"，现在注册/切换会写库；
3. **跨重启恢复**：灰度做到一半重启服务，状态不能丢（否则灰度等于白做）。

    python agent/tests/test_migrations.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi.testclient import TestClient  # noqa: E402

from agent.service.app import create_app  # noqa: E402
from agent.service.config import Settings  # noqa: E402
from agent.service.db import get_database, load_kb_versions, record_kb_version  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(ROOT, "eval", "fixtures", "mini_corpus.jsonl")


def make_settings(db_path: str) -> Settings:
    return Settings(
        database_url=f"sqlite:///{db_path}", kb_path=FIXTURE,
        jwt_secret="migration-test-secret-32-bytes!", auth_required=True,
        dev_token_enabled=True, kv_backend="memory",
        rate_limit_capacity=10_000, rate_limit_refill_per_sec=10_000.0,
        quota_tokens_per_day=10_000_000, llm_backend="planner",
        default_vehicle_model="lynk08", allowed_models=["lynk08"],
    )


def alembic_available() -> bool:
    try:
        import alembic  # noqa: F401
    except Exception:                                                        # noqa: BLE001
        return False
    return True


class TestAlembicMigrations(unittest.TestCase):
    """迁移脚本本身：空库能建、能回滚、能加列。"""

    def setUp(self):
        if not alembic_available():
            self.skipTest("未安装 alembic")
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db = os.path.join(self._tmp.name, "mig.db")
        self.url = "sqlite:///" + self.db.replace("\\", "/")

    def tearDown(self):
        self._tmp.cleanup()

    def _alembic(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "alembic", "-x", f"db_url={self.url}", *args],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            env=dict(os.environ, PYTHONIOENCODING="utf-8"))

    def _tables(self):
        con = sqlite3.connect(self.db)
        try:
            return sorted(r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"))
        finally:
            con.close()

    def _columns(self, table: str):
        con = sqlite3.connect(self.db)
        try:
            return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
        finally:
            con.close()

    def test_upgrade_head_creates_all_tables(self):
        result = self._alembic("upgrade", "head")
        self.assertEqual(result.returncode, 0, result.stderr[-800:])
        tables = self._tables()
        for expected in ("sessions", "messages", "tool_calls", "audit_logs", "feedback",
                         "quotas", "idempotency_keys", "kb_versions", "confirmations",
                         "vehicles", "alembic_version"):
            self.assertIn(expected, tables, f"迁移后缺少表 {expected}")

    def test_upgrade_adds_kb_version_columns(self):
        """第二个迁移必须真的加上 version/checksum/traffic_pct。

        这三列以前加不了（没有迁移机制，`create_all` 不会改老表），
        当时的折中是往 `source_file` 里塞 "version|path"——现在应该补正式字段。
        """
        self.assertEqual(self._alembic("upgrade", "head").returncode, 0)
        cols = self._columns("kb_versions")
        for expected in ("version", "checksum", "traffic_pct"):
            self.assertIn(expected, cols, f"kb_versions 缺少列 {expected}")

    def test_downgrade_rolls_back_columns(self):
        self.assertEqual(self._alembic("upgrade", "head").returncode, 0)
        self.assertEqual(self._alembic("downgrade", "-1").returncode, 0)
        cols = self._columns("kb_versions")
        self.assertNotIn("version", cols, "回滚后仍保留新增列")

    def test_offline_sql_generation(self):
        """离线模式只生成 SQL、不连库——CI 用它快速校验迁移脚本可解析。"""
        result = self._alembic("upgrade", "head", "--sql")
        self.assertEqual(result.returncode, 0, result.stderr[-800:])
        self.assertIn("CREATE TABLE", result.stdout.upper())


class TestKbVersionPersistence(unittest.TestCase):
    """`kb_versions` 表从"零使用"变成真正的状态存储。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = os.path.join(self._tmp.name, "persist.db")

    def tearDown(self):
        self._tmp.cleanup()

    def test_record_and_load_roundtrip(self):
        db = get_database(f"sqlite:///{self.db_path}")
        db.init()
        record_kb_version(db, vehicle_model="lynk08", version="v1-abc12345",
                          path="/tmp/a.jsonl", checksum="abc12345", n_chunks=24,
                          status="active", traffic_pct=100)
        record_kb_version(db, vehicle_model="lynk08", version="v2-def67890",
                          path="/tmp/b.jsonl", checksum="def67890", n_chunks=25,
                          status="shadow", traffic_pct=30)
        rows = load_kb_versions(db)
        self.assertEqual(len(rows), 2)
        by_version = {r["version"]: r for r in rows}
        self.assertEqual(by_version["v1-abc12345"]["status"], "active")
        self.assertEqual(by_version["v2-def67890"]["traffic_pct"], 30)
        self.assertEqual(by_version["v2-def67890"]["n_chunks"], 25)

        # 同 (租户, 版本) 再写一次 → upsert，不能出现重复行
        record_kb_version(db, vehicle_model="lynk08", version="v2-def67890",
                          path="/tmp/b.jsonl", checksum="def67890", n_chunks=25,
                          status="active", traffic_pct=100)
        rows2 = load_kb_versions(db)
        self.assertEqual(len(rows2), 2, "upsert 产生了重复行")
        self.assertEqual({r["version"]: r for r in rows2}["v2-def67890"]["status"], "active")

    def test_version_state_survives_restart(self):
        """灰度做到一半重启服务：active/shadow 必须都还在。"""
        v2_path = os.path.join(self._tmp.name, "v2.jsonl")
        rows = [json.loads(line) for line in open(FIXTURE, encoding="utf-8") if line.strip()]
        rows.append({"id": 9001, "text": "（新版）电动尾门支持脚踢开启。",
                     "source": "v2.pdf", "page": 8})
        with open(v2_path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        app1 = create_app(make_settings(self.db_path))
        with TestClient(app1) as client:
            svc = {"Authorization": "Bearer " + client.post(
                "/v1/auth/token",
                json={"user_id": "svc", "role": "service",
                      "vehicle_model": "lynk08"}).json()["access_token"]}
            record = client.post("/v1/admin/kb/register",
                                 json={"path": v2_path, "vehicle_model": "lynk08"},
                                 headers=svc).json()
            client.post("/v1/admin/kb/activate",
                        json={"version": record["version"], "traffic_pct": 50,
                              "vehicle_model": "lynk08"}, headers=svc)
            first_active = app1.state.kb_registry.active_version("lynk08")
            first_shadow = app1.state.kb_registry.stats()["shadow"]
            self.assertTrue(first_shadow, "第一次启动没有进入灰度状态")

        # 用同一个数据库重新起一个 app（模拟重启）
        app2 = create_app(make_settings(self.db_path))
        with TestClient(app2) as client:
            registry = app2.state.kb_registry
            self.assertEqual(registry.active_version("lynk08"), first_active,
                             "重启后 active 版本丢了")
            self.assertEqual(registry.stats()["shadow"], first_shadow,
                             "重启后灰度状态丢了（灰度等于白做）")
            token = client.post("/v1/auth/token",
                                json={"user_id": "u1", "role": "driver",
                                      "vehicle_model": "lynk08"}).json()["access_token"]
            body = client.post("/v1/chat", json={"message": "座椅加热怎么关闭"},
                               headers={"Authorization": f"Bearer {token}"}).json()
            self.assertIn(body["kb_version"], {first_active, record["version"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
