# -*- coding: utf-8 -*-
"""工单闭环 + 跨会话记忆：一条命令演示"问答之外，业务真的能办成"。

    python eval/order_flow.py

为什么要有这个脚本
------------------
之前 MCP 与长期记忆都是"基础设施"，没有一个真实业务对象去消费它们，所以只能自证。
这个脚本要证明的是**闭环**：

    会话 1：用户说「我一般去浦东店」+「约个保养」 → 建单（门店来自记忆，用户没说）
    会话 2（第二天、新进程）：用户说「上次那个保养改到 X 号」 → 从记忆里捞到单号 → 改期成功

顺带量化"少问几句"：记忆 ON 时不需要问单号/门店（0 次澄清），OFF 时至少要问单号（估算 1 次）、
如果门店也没记还要再问一次（估算 2 次）。

如实说明边界：本脚本驱动的是**工具链 + 记忆**，不是模型选工具——
离线规则规划器不会主动调 `reschedule_order`（那需要真模型），所以这里直接按"模型该做的调用"
去执行；指标是"信息是否可得 + 调用是否成功"，不是"模型准确率"。
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.console import harden_stdout  # noqa: E402

harden_stdout()

from agent.kb import FIXTURE_KB, KnowledgeBase  # noqa: E402
from agent.longterm import (  # noqa: E402
    LongTermMemory,
    defaults_from_preferences,
    order_event_recorder,
)
from agent.orders import OrderStore  # noqa: E402
from agent.tools import ToolRegistry  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER = "u1"
ITEM = "更换机油机滤"
FIRST_DATE = "2099-06-18"
NEW_DATE = "2099-07-01"
CENTER = "领克中心(浦东金桥店)"


def make_registry(kb: Any, store: LongTermMemory, orders: OrderStore,
                  query: str = "") -> ToolRegistry:
    return ToolRegistry(kb, orders=orders, user_id=USER,
                        defaults=defaults_from_preferences(store, USER),
                        on_event=order_event_recorder(store, USER, "flow", query))


def confirmed_call(registry: ToolRegistry, name: str, args: Dict[str, Any]) -> Any:
    """先拿 action_key 再确认——服务端/控制台做的就是这两步。"""
    pending = registry.call(name, dict(args))
    if pending.meta.get("needs_confirmation"):
        registry.confirm(pending.meta["action_key"])
        return registry.call(name, dict(args))
    return pending


def main() -> int:
    checks: List[Dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"  {'[OK]' if ok else '[FAIL]'} {name}" + (f" —— {detail}" if detail else ""))
        return ok

    kb = KnowledgeBase.load(FIXTURE_KB if os.path.exists(FIXTURE_KB) else None)
    tmp = tempfile.mkdtemp(prefix="order-flow-")
    mem_path = os.path.join(tmp, "memory.db")
    orders_path = os.path.join(tmp, "orders.db")

    print("=" * 78)
    print("工单闭环（问答 → 建单 → 跨会话查/改 → 取消）")
    print("=" * 78)

    # ── 会话 1：说偏好 + 建单 ──
    store = LongTermMemory(mem_path)
    orders = OrderStore(orders_path, enforce_future_date=True)
    session1_query = f"我一般去{CENTER}，以后保养都去这家，麻烦帮我约个保养"
    registry = make_registry(kb, store, orders, session1_query)
    store.extract_and_remember(session1_query, user_id=USER)
    record("会话 1：偏好写入长期记忆", store.preference(USER, "usual_center") == CENTER,
           f"usual_center={store.preference(USER, 'usual_center')}")

    created = confirmed_call(registry, "create_service_order",
                             {"item": ITEM, "preferred_date": FIRST_DATE})   # 故意不给 center
    order_id = (created.data or {}).get("预约单号", "") if created.ok else ""
    record("会话 1：建单成功，且门店由偏好自动填参",
           bool(created.ok) and (created.data or {}).get("门店") == CENTER,
           f"单号={order_id}，门店={(created.data or {}).get('门店')}")

    listed = registry.call("get_my_orders", {})
    record("会话内：工单可查", bool(listed.ok) and len(listed.data["工单"]) == 1,
           f"返回 {len((listed.data or {}).get('工单', []))} 条")

    # ── 会话 2：模拟"第二天、新进程"——重新打开两个库，只靠记忆恢复上下文 ──
    store.close()
    orders.close()

    store2 = LongTermMemory(mem_path)
    orders2 = OrderStore(orders_path, enforce_future_date=True)
    registry2 = make_registry(kb, store2, orders2)

    context = store2.context_text("上次那个保养我想改个日期", user_id=USER)
    recovered = re.search(r"SO\d{6,}", context)
    record("会话 2：从长期记忆里恢复单号（不需要用户再说一次）",
           bool(recovered) and (recovered.group(0) == order_id),
           f"命中 {recovered.group(0) if recovered else '无'}")

    pending = registry2.call("reschedule_order",
                             {"order_id": order_id, "new_date": NEW_DATE})
    record("会话 2：改期需要二次确认", not pending.ok and bool(pending.meta.get("needs_confirmation")),
           f"action_key={pending.meta.get('action_key')}")
    registry2.confirm(pending.meta["action_key"])
    moved = registry2.call("reschedule_order", {"order_id": order_id, "new_date": NEW_DATE})
    record("会话 2：改期成功", bool(moved.ok) and (moved.data or {}).get("到店日期") == NEW_DATE,
           f"新日期={(moved.data or {}).get('到店日期')}")

    persisted = orders2.get(USER, order_id)
    record("跨进程：状态一致（工单已落库，不是内存里的假单）",
           persisted.date == NEW_DATE and persisted.status == "booked",
           f"{persisted.summary()}")

    cancelled = confirmed_call(registry2, "cancel_order", {"order_id": order_id})
    record("会话 2：取消成功且状态落库",
           bool(cancelled.ok) and orders2.get(USER, order_id).status == "cancelled",
           f"状态={orders2.get(USER, order_id).status}")

    # ── 对照：没有长期记忆时会怎样（估算澄清轮次） ──
    store_off = LongTermMemory(os.path.join(tmp, "empty_memory.db"))
    context_off = store_off.context_text("上次那个保养我想改个日期", user_id=USER)
    order_id_off = re.search(r"SO\d{6,}", context_off)
    clarify_on = 0
    clarify_off = (1 if not order_id_off else 0) + (1 if not defaults_from_preferences(store_off, USER) else 0)
    record("对照：无记忆时单号不可得、门店也要再问（估算澄清轮次 2）",
           order_id_off is None and clarify_off == 2,
           f"ON 澄清 {clarify_on} 次 / OFF 估算 {clarify_off} 次")
    store_off.close()

    stats = orders2.stats(USER)
    passed = sum(1 for c in checks if c["ok"])
    print("-" * 78)
    print(f"合计 {passed}/{len(checks)} 项通过｜工单状态分布 {stats}｜"
          f"澄清轮次 ON {clarify_on} / OFF {clarify_off}")
    report = os.path.join(ROOT, "eval", "order_flow_report.json")
    with open(report, "w", encoding="utf-8") as fh:
        json.dump({"checks": checks, "stats": stats,
                   "clarify_rounds": {"on": clarify_on, "off_estimate": clarify_off}},
                  fh, ensure_ascii=False, indent=2)
    print(f"报告：{report}")

    store2.close()
    orders2.close()
    if passed != len(checks):
        print("[FAIL] 存在未通过项")
        return 1
    print("[OK] 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
