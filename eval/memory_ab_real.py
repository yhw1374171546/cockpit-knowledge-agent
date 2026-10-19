# -*- coding: utf-8 -*-
"""真模型的记忆答案层 AB：同一个跨会话任务，记忆 ON/OFF，比"一次办成"和"要不要追问"。

    python eval/memory_ab_real.py --provider deepseek
    python eval/memory_ab_real.py --provider deepseek --limit 1     # 先花一分钱验链路

和 `eval/memory_ab.py` 的区别
-----------------------------
`memory_ab.py` 测的是**上下文**：事实有没有进 system prompt（用 `SpyPlanner` 抓实发 messages）。
那个指标可复现，但只证明"模型看得到"。这里补**答案层**：用真模型跑同一个跨会话任务，
看它能不能**一次办成**、会不会回头问用户要已经说过的信息。

四个场景（每个都跑 ON/OFF 两遍，初始数据库完全相同，只有"有没有长期记忆"不同）
------------------------------------------------------------------------------
1. `pref_store`：昨天说过"我一般去徐汇店"，今天只说"帮我约 2099-06-18 保养"——
   ON 的门店应该填对；OFF 只能问，或者落到默认门店（错的）。
2. `order_reschedule`：昨天建过单，今天说"上次那个保养改到 2099-07-01"——
   ON 直接从记忆里拿单号；OFF 得先查工单列表。
3. `order_query`："我那个保养约的哪天？"——ON 直接答；OFF 必须查工具。
4. `order_cancel`："上次那个保养取消掉"——同上。

如实说明
--------
- 场景 1 里的"默认门店"是**桩数据**（`OrderStore.create` 的兜底门店），
  所以 OFF 侧不是一定失败，而是**门店会填错**——这恰好是记忆该解决的错；
- 写操作仍走"二次确认"：单号对、进入 `needs_confirmation` 就算"模型定位对了"，
  真正的确认由产品流程完成（不在这里替它点确认）；
- 这里只跑 4 个场景 × 2 组，**样本小**，结论是"方向性证据"，不是统计显著性。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.console import harden_stdout  # noqa: E402

harden_stdout()

from agent.graph import AgentConfig, AgentGraph  # noqa: E402
from agent.kb import FIXTURE_KB, KnowledgeBase  # noqa: E402
from agent.longterm import (  # noqa: E402
    LongTermMemory,
    defaults_from_preferences,
    order_event_recorder,
)
from agent.memory import ConversationMemory  # noqa: E402
from agent.orders import OrderStore  # noqa: E402
from agent.tools import ToolRegistry  # noqa: E402
from eval.llm_backend import (  # noqa: E402
    add_llm_args,
    backend_fields,
    cost_line,
    llm_label,
    make_llm,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER = "u1"
ITEM = "更换机油机滤"
ORDER_DATE = "2099-06-18"
ORDER_DATE2 = "2099-06-25"
NEW_DATE = "2099-07-01"
# 两家门店都故意与 OrderStore 的兜底门店（浦东金桥店）不同，这样"门店对不对"可判
CENTER = "领克中心(徐汇店)"
CENTER2 = "领克中心(杭州西湖店)"
ASK_PATTERN = re.compile(r"(请问|您(的|要|想|方便)|哪家|哪个店|哪一天|几点|需要您|告诉我|麻烦您)")
# 「在向用户要门店/单号」——这些信息昨天已经说过，ON 不该再问。
# 注意不能只匹配"门店"两个字：模型复述订单时也会出现"门店：徐汇店"（第一版就是这么误判的）。
ASK_FOR_STORE = re.compile(
    r"(指定.{0,6}(门店|店)|(提供|告诉|选择).{0,6}(门店|店)|(门店|店).{0,6}(是哪|哪个|哪家)|哪家店|哪个店)")
ASK_FOR_ORDER = re.compile(
    r"(请提供.{0,6}(单号|工单)|(单号|工单号).{0,8}(是多少|是什么|提供|告诉我)|哪一?个?单|哪个预约|哪一张工单)")


@dataclass
class Scenario:
    sid: str
    seed_text: str = ""               # 昨天说过的话（写偏好 + 情景记忆）
    need_order: bool = False          # 昨天有没有建过单
    query: str = ""                   # 今天说的话
    note: str = ""
    # 用户"已经点过确认"的写操作键（评测里预置，避免把二次确认当成失败）：
    # 可用的占位符 {item} {date} {new_date} {order_id}
    confirm_template: str = ""
    # "重复追问"的模式：这些信息昨天已经说过（存进了长期记忆），
    # ON 不该再问，OFF 只能问——这才是记忆在答案层的真实收益。
    redundant_pattern: Optional[re.Pattern[str]] = None
    check: Optional[Callable[[Dict[str, Any], OrderStore, str], bool]] = None


def _has_call(state: Dict[str, Any], tool: str, **fields: Any) -> bool:
    for call in state.get("tool_calls") or []:
        if call.get("tool") != tool:
            continue
        arguments = call.get("arguments") or {}
        if all(str(arguments.get(k)) == str(v) for k, v in fields.items()):
            return True
    return False


def _pref_store_check(center: str) -> Callable[[Dict[str, Any], OrderStore, str], bool]:
    """建出来的工单门店是否等于"昨天说过的惯用门店"。"""
    def check(state: Dict[str, Any], orders: OrderStore, order_id: str) -> bool:
        rows = orders.list(USER, limit=1)
        return bool(rows) and rows[0].center == center
    return check


def _check_reschedule(state: Dict[str, Any], orders: OrderStore, order_id: str) -> bool:
    return _has_call(state, "reschedule_order", order_id=order_id, new_date=NEW_DATE)


def _check_query(state: Dict[str, Any], orders: OrderStore, order_id: str) -> bool:
    return ORDER_DATE in str(state.get("answer") or "")


def _check_cancel(state: Dict[str, Any], orders: OrderStore, order_id: str) -> bool:
    return _has_call(state, "cancel_order", order_id=order_id)


SCENARIOS: List[Scenario] = [
    Scenario("pref_store", seed_text=f"我一般去{CENTER}，以后保养都去这家",
             query=f"帮我约 {ORDER_DATE} 到店保养，换机油机滤。我已经确认过了，直接办。",
             note="门店偏好（徐汇）",
             confirm_template="create_service_order:{item}:{date}",
             redundant_pattern=ASK_FOR_STORE,
             check=_pref_store_check(CENTER)),
    Scenario("pref_store2", seed_text=f"我一般去{CENTER2}，以后都去这家",
             query=f"帮我约 {ORDER_DATE2} 到店保养，换机油机滤。我已经确认过了，直接办。",
             note="门店偏好（杭州）",
             confirm_template="create_service_order:{item}:{date}",
             redundant_pattern=ASK_FOR_STORE,
             check=_pref_store_check(CENTER2)),
    Scenario("order_reschedule", need_order=True,
             query=f"上次那个保养我想改到 {NEW_DATE}。我已经确认过了，直接改。",
             note="跨会话改期",
             confirm_template="reschedule_order:{order_id}:{new_date}",
             redundant_pattern=ASK_FOR_ORDER,
             check=_check_reschedule),
    Scenario("order_query", need_order=True, query="我那个保养约的是哪天？",
             note="跨会话查询", check=_check_query),
    Scenario("order_cancel", need_order=True, query="上次那个保养取消掉吧，我已经确认过了。",
             note="跨会话取消", confirm_template="cancel_order:{order_id}",
             redundant_pattern=ASK_FOR_ORDER,
             check=_check_cancel),
]


def seed(mem_path: str, orders_path: str, scenario: Scenario) -> str:
    """把"昨天发生过的事"写进两个库；返回工单号（没建单则为空串）。"""
    store = LongTermMemory(mem_path)
    orders = OrderStore(orders_path, enforce_future_date=True)
    order_id = ""
    try:
        if scenario.seed_text:
            store.extract_and_remember(scenario.seed_text, user_id=USER)
            orders_defaults = defaults_from_preferences(store, USER)
        else:
            orders_defaults = {}
        if scenario.need_order:
            registry = ToolRegistry(KnowledgeBase.load(FIXTURE_KB), orders=orders,
                                    user_id=USER, defaults=orders_defaults,
                                    on_event=order_event_recorder(store, USER, "seed",
                                                                  scenario.query))
            pending = registry.call("create_service_order",
                                    {"item": ITEM, "preferred_date": ORDER_DATE,
                                     "center": orders_defaults.get("center", CENTER)})
            if pending.meta.get("needs_confirmation"):
                registry.confirm(pending.meta["action_key"])
            result = registry.call("create_service_order",
                                   {"item": ITEM, "preferred_date": ORDER_DATE,
                                    "center": orders_defaults.get("center", CENTER)})
            order_id = (result.data or {}).get("预约单号", "") if result.ok else ""
    finally:
        store.close()
        orders.close()
    return order_id


def run_case(scenario: Scenario, use_memory: bool, llm: Any, kb: Any,
             mem_path: str, orders_path: str, order_id: str = "") -> Dict[str, Any]:
    store = LongTermMemory(mem_path) if use_memory else None
    orders = OrderStore(orders_path, enforce_future_date=False)
    try:
        registry = ToolRegistry(
            kb, orders=orders, user_id=USER,
            defaults=defaults_from_preferences(store, USER) if store else {},
            on_event=order_event_recorder(store, USER, "ab", scenario.query) if store else None,
        )
        # 写操作需要车主确认：评测里把"用户已确认"预置好（两种条件完全一致），
        # 否则会把"确认流程"误判成"任务失败"。
        if scenario.confirm_template:
            action_key = scenario.confirm_template.format(
                item=ITEM, date=ORDER_DATE, new_date=NEW_DATE, order_id=order_id)
            registry.confirm(action_key)
        memory = ConversationMemory()
        if store is not None:
            context = store.context_text(scenario.query, user_id=USER)
            memory.attach_longterm(context)
        # 这是"任务完成"评测，不是 RAG 接地评测：关掉证据门控与接地拒答，
        # 否则模型按设计先说一句"确认吗"就会被判 ungrounded → 直接拒答（实测踩过）。
        graph = AgentGraph(llm, registry,
                           config=AgentConfig(max_steps=4, evidence_gate=False,
                                              refuse_on_insufficient=False,
                                              enable_reflection=False),
                           memory=memory, agent_name="ab")
        state = graph.run(scenario.query)
        payload = state.to_dict()
        payload["status"] = state.status
        answer_text = str(state.answer or "")
        asked = bool(ASK_PATTERN.search(answer_text))
        redundant = bool(scenario.redundant_pattern.search(answer_text)) \
            if scenario.redundant_pattern is not None else False
        return {
            "state": payload,
            "success": bool(scenario.check(payload, orders, order_id)) if scenario.check else False,
            "asked_user": asked,
            "asked_redundant": redundant,
            "tool_calls": len(state.tool_calls),
            "tokens": state.tokens_total,
            "answer": answer_text[:200],
        }
    finally:
        orders.close()
        if store is not None:
            store.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="真模型的记忆答案层 AB")
    add_llm_args(ap)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个场景（省钱验链路用）")
    ap.add_argument("--repeat", type=int, default=2,
                    help="每个条件重复跑几次（模型有随机性，单次不足以判断）")
    ap.add_argument("--json-out", default=os.path.join(ROOT, "eval", "memory_ab_real.json"))
    args = ap.parse_args()

    kb = KnowledgeBase.load(FIXTURE_KB if os.path.exists(FIXTURE_KB) else None)
    llm = make_llm(args, known_tools=ToolRegistry(kb).names())
    scenarios = SCENARIOS[:args.limit] if args.limit else SCENARIOS
    tmp = tempfile.mkdtemp(prefix="mem-ab-real-")

    print("=" * 78)
    print(f"记忆答案层 AB（{llm_label(args)}）")
    print("=" * 78)

    rows: List[Dict[str, Any]] = []
    repeats = max(1, args.repeat)
    for scenario in scenarios:
        seed_mem = os.path.join(tmp, f"{scenario.sid}_seed_mem.db")
        seed_orders = os.path.join(tmp, f"{scenario.sid}_seed_orders.db")
        order_id = seed(seed_mem, seed_orders, scenario)
        if scenario.need_order and not order_id:
            print(f"  [!!] {scenario.sid} 预置工单失败，跳过")
            continue

        results: Dict[str, List[Dict[str, Any]]] = {"on": [], "off": []}
        for condition in ("on", "off"):
            for rep in range(repeats):
                # 每跑一次都从**同一份预置状态**复制，保证 ON/OFF 起跑线完全一致
                mem_path = os.path.join(tmp, f"{scenario.sid}_{condition}_{rep}_mem.db")
                orders_path = os.path.join(tmp, f"{scenario.sid}_{condition}_{rep}_orders.db")
                shutil.copy(seed_mem, mem_path)
                shutil.copy(seed_orders, orders_path)
                results[condition].append(
                    run_case(scenario, condition == "on", llm, kb,
                             mem_path, orders_path, order_id))

        def rate(items: List[Dict[str, Any]], key: str) -> float:
            return sum(1 for item in items if item[key]) / max(1, len(items))

        agg = {
            "on": {"success_rate": rate(results["on"], "success"),
                   "redundant_rate": rate(results["on"], "asked_redundant"),
                   "avg_calls": round(sum(i["tool_calls"] for i in results["on"]) / repeats, 2),
                   "avg_tokens": round(sum(i["tokens"] for i in results["on"]) / repeats)},
            "off": {"success_rate": rate(results["off"], "success"),
                    "redundant_rate": rate(results["off"], "asked_redundant"),
                    "avg_calls": round(sum(i["tool_calls"] for i in results["off"]) / repeats, 2),
                    "avg_tokens": round(sum(i["tokens"] for i in results["off"]) / repeats)},
        }
        rows.append({"scenario": scenario.sid, "note": scenario.note, "order_id": order_id,
                     "repeats": repeats, **agg,
                     "samples": {"on": [i["answer"][:160] for i in results["on"]],
                                 "off": [i["answer"][:160] for i in results["off"]]}})
        print(f"\n  [{scenario.sid}] {scenario.note}（{repeats} 次）")
        for name, data in (("ON ", agg["on"]), ("OFF", agg["off"])):
            print(f"    {name} 一次办成 {data['success_rate']:.0%} ｜ "
                  f"重复追问 {data['redundant_rate']:.0%} ｜ "
                  f"平均工具调用 {data['avg_calls']:.2f} ｜ 平均 tokens {data['avg_tokens']}")

    total = max(1, len(rows) * repeats)
    on_success = sum(round(r["on"]["success_rate"] * repeats) for r in rows)
    off_success = sum(round(r["off"]["success_rate"] * repeats) for r in rows)
    on_redundant = sum(round(r["on"]["redundant_rate"] * repeats) for r in rows)
    off_redundant = sum(round(r["off"]["redundant_rate"] * repeats) for r in rows)
    on_calls = sum(r["on"]["avg_calls"] for r in rows) / max(1, len(rows))
    off_calls = sum(r["off"]["avg_calls"] for r in rows) / max(1, len(rows))
    fields = backend_fields(args, llm)

    print("\n" + "-" * 78)
    print(f"一次办成：记忆 ON {on_success}/{total} ｜ OFF {off_success}/{total}")
    print(f"**重复追问已说过的信息**：ON {on_redundant}/{total} ｜ "
          f"OFF {off_redundant}/{total}")
    print(f"平均工具调用：ON {on_calls:.2f} ｜ OFF {off_calls:.2f}")
    print(cost_line(fields))

    summary = {
        "backend": fields["backend"], "model": fields["model"],
        "scenarios": len(rows), "repeats": repeats, "runs_total": total,
        "success_on": on_success, "success_off": off_success,
        "redundant_on": on_redundant, "redundant_off": off_redundant,
        "avg_tool_calls_on": round(on_calls, 2), "avg_tool_calls_off": round(off_calls, 2),
        **fields,
    }
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, ensure_ascii=False, indent=2)
    print(f"报告：{args.json_out}")

    if fields["llm_calls"] == 0:
        print("[!] 没有发生模型调用（provider=planner？）——这个脚本要用真实模型才有意义")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
