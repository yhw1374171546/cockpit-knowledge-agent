# -*- coding: utf-8 -*-
"""服务工单测试：状态机、幂等、跨用户隔离、工具层集成、事件钩子。

    python agent/tests/test_orders.py

为什么这些断言值得写：工单是**唯一有真实状态**的业务对象。
它是长期记忆与 MCP 复用的消费者——如果状态机有洞，记忆记的"那单子"就是错的，
MCP 暴露给售后 Agent 的写操作也会误伤。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.kb import FIXTURE_KB, KnowledgeBase  # noqa: E402
from agent.orders import OrderError, OrderStore  # noqa: E402
from agent.tools import ToolRegistry  # noqa: E402

FUTURE = "2099-06-18"
PAST = "2020-01-01"
NOW = 1_760_000_000.0          # 固定时钟：不依赖真实日期


class TestOrderStore(unittest.TestCase):
    def setUp(self):
        self.store = OrderStore(clock=lambda: NOW, enforce_future_date=True)

    def tearDown(self):
        self.store.close()

    def test_create_booked_order(self):
        order = self.store.create("u1", "更换机油机滤", FUTURE, "浦东店")
        self.assertTrue(order.id.startswith("SO"))
        self.assertEqual(order.status, "booked")
        self.assertEqual(order.history[0]["action"], "created")
        self.assertFalse(order.existing)

    def test_create_is_idempotent(self):
        first = self.store.create("u1", "更换机油机滤", FUTURE, "浦东店")
        second = self.store.create("u1", "更换机油机滤", FUTURE, "浦东店")
        self.assertTrue(second.existing)
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(self.store.list("u1")), 1)

    def test_bad_date_format(self):
        with self.assertRaises(OrderError) as ctx:
            self.store.create("u1", "更换机油机滤", "6月18日", "浦东店")
        self.assertEqual(ctx.exception.code, "bad_date")

    def test_past_date_rejected_when_enforced(self):
        with self.assertRaises(OrderError) as ctx:
            self.store.create("u1", "更换机油机滤", PAST, "浦东店")
        self.assertEqual(ctx.exception.code, "past_date")

    def test_past_date_allowed_when_not_enforced(self):
        loose = OrderStore(clock=lambda: NOW)
        try:
            order = loose.create("u1", "更换机油机滤", PAST, "浦东店")
            self.assertEqual(order.date, PAST)
        finally:
            loose.close()

    def test_reschedule_updates_version_and_history(self):
        order = self.store.create("u1", "更换机油机滤", FUTURE, "浦东店")
        moved = self.store.reschedule("u1", order.id, date="2099-07-01", center="徐汇店")
        self.assertEqual(moved.date, "2099-07-01")
        self.assertEqual(moved.center, "徐汇店")
        self.assertEqual(moved.version, 2)
        self.assertEqual(moved.history[-1]["action"], "rescheduled")

    def test_reschedule_with_same_params_is_noop(self):
        order = self.store.create("u1", "更换机油机滤", FUTURE, "浦东店")
        again = self.store.reschedule("u1", order.id, date=FUTURE, center="浦东店")
        self.assertTrue(again.existing)
        self.assertEqual(again.version, 1)

    def test_cancel_is_terminal(self):
        order = self.store.create("u1", "更换机油机滤", FUTURE, "浦东店")
        self.assertEqual(self.store.cancel("u1", order.id).status, "cancelled")
        with self.assertRaises(OrderError) as ctx:
            self.store.cancel("u1", order.id)
        self.assertEqual(ctx.exception.code, "invalid_state")

    def test_cannot_reschedule_cancelled(self):
        order = self.store.create("u1", "更换机油机滤", FUTURE, "浦东店")
        self.store.cancel("u1", order.id)
        with self.assertRaises(OrderError) as ctx:
            self.store.reschedule("u1", order.id, date="2099-07-01")
        self.assertEqual(ctx.exception.code, "invalid_state")

    def test_cannot_cancel_completed(self):
        order = self.store.create("u1", "更换机油机滤", FUTURE, "浦东店")
        self.store.complete("u1", order.id)
        with self.assertRaises(OrderError):
            self.store.cancel("u1", order.id)

    def test_cross_user_is_not_found(self):
        order = self.store.create("u1", "更换机油机滤", FUTURE, "浦东店")
        with self.assertRaises(OrderError) as ctx:
            self.store.get("u2", order.id)
        self.assertEqual(ctx.exception.code, "not_found")

    def test_list_filters_and_limits(self):
        for i, day in enumerate(("2099-06-01", "2099-06-02", "2099-06-03")):
            self.store.create("u1", f"项目{i}", day, "浦东店")
        self.store.cancel("u1", self.store.list("u1")[0].id)
        self.assertEqual(len(self.store.list("u1", limit=2)), 2)
        self.assertEqual(len(self.store.list("u1", status="booked")), 2)
        self.assertEqual(len(self.store.list("u1", status="cancelled")), 1)

    def test_stats_counts_by_state(self):
        first = self.store.create("u1", "A", FUTURE, "浦东店")
        self.store.create("u1", "B", "2099-07-01", "浦东店")
        self.store.cancel("u1", first.id)
        stats = self.store.stats("u1")
        self.assertEqual((stats["booked"], stats["cancelled"]), (1, 1))


class TestOrderTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase.load(FIXTURE_KB if os.path.exists(FIXTURE_KB) else None)

    def setUp(self):
        self.events = []
        self.store = OrderStore(clock=lambda: NOW)
        self.registry = ToolRegistry(self.kb, orders=self.store, user_id="u1",
                                     defaults={"center": "领克中心(徐汇店)"},
                                     on_event=lambda action, payload: self.events.append(action))

    def tearDown(self):
        self.store.close()

    def _create(self, date=FUTURE, center=None):
        args = {"item": "更换机油机滤", "preferred_date": date}
        if center:
            args["center"] = center
        pending = self.registry.call("create_service_order", args)
        self.assertTrue(pending.meta.get("needs_confirmation"), pending.error)
        self.registry.confirm(pending.meta["action_key"])
        return self.registry.call("create_service_order", args)

    def test_order_tools_are_registered(self):
        names = set(self.registry.names())
        self.assertTrue({"get_my_orders", "reschedule_order", "cancel_order"} <= names)

    def test_created_order_is_queryable(self):
        created = self._create()
        self.assertTrue(created.ok, created.error)
        order_id = created.data["预约单号"]
        listed = self.registry.call("get_my_orders", {})
        self.assertTrue(listed.ok)
        self.assertEqual(listed.data["工单"][0]["预约单号"], order_id)

    def test_default_center_comes_from_preferences(self):
        created = self._create()
        self.assertEqual(created.data["门店"], "领克中心(徐汇店)")

    def test_explicit_center_wins_over_default(self):
        created = self._create(center="领克中心(浦东店)")
        self.assertEqual(created.data["门店"], "领克中心(浦东店)")

    def test_reschedule_requires_its_own_confirmation_key(self):
        order_id = self._create().data["预约单号"]
        pending = self.registry.call("reschedule_order",
                                     {"order_id": order_id, "new_date": "2099-07-01"})
        self.assertFalse(pending.ok)
        self.assertEqual(pending.meta["action_key"], f"reschedule_order:{order_id}:2099-07-01")
        self.registry.confirm(pending.meta["action_key"])
        moved = self.registry.call("reschedule_order",
                                   {"order_id": order_id, "new_date": "2099-07-01"})
        self.assertTrue(moved.ok, moved.error)
        self.assertEqual(moved.data["到店日期"], "2099-07-01")

    def test_cancel_then_query_status(self):
        order_id = self._create().data["预约单号"]
        pending = self.registry.call("cancel_order", {"order_id": order_id})
        self.registry.confirm(pending.meta["action_key"])
        cancelled = self.registry.call("cancel_order", {"order_id": order_id})
        self.assertTrue(cancelled.ok, cancelled.error)
        self.assertEqual(cancelled.data["状态"], "cancelled")
        listed = self.registry.call("get_my_orders", {"status": "cancelled"})
        self.assertEqual(len(listed.data["工单"]), 1)

    def test_unknown_order_returns_structured_error(self):
        pending = self.registry.call("cancel_order", {"order_id": "SO00000000"})
        self.registry.confirm(pending.meta["action_key"])
        result = self.registry.call("cancel_order", {"order_id": "SO00000000"})
        self.assertFalse(result.ok)
        self.assertEqual(result.meta.get("code"), "not_found")

    def test_events_are_emitted_for_memory(self):
        order_id = self._create().data["预约单号"]
        pending = self.registry.call("reschedule_order",
                                     {"order_id": order_id, "new_date": "2099-07-01"})
        self.registry.confirm(pending.meta["action_key"])
        self.registry.call("reschedule_order", {"order_id": order_id, "new_date": "2099-07-01"})
        self.assertEqual(self.events[:2], ["order_created", "order_rescheduled"])

    def test_scope_isolation_between_users(self):
        self._create()
        other = ToolRegistry(self.kb, orders=self.store, user_id="u2")
        listed = other.call("get_my_orders", {})
        self.assertEqual(listed.data["工单"], [])

    def test_subset_inherits_order_store_and_user(self):
        scoped = self.registry.subset(["get_my_orders", "create_service_order"], "svc")
        self.assertIs(scoped.orders, self.store)
        self.assertEqual(scoped.user_id, "u1")
        self.assertEqual(scoped.defaults["center"], "领克中心(徐汇店)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
