# -*- coding: utf-8 -*-
"""真模型的记忆答案层 AB：同一个跨会话任务，记忆 ON/OFF，比"一次办成"和"要不要追问"。

    python eval/memory_ab_real.py --provider deepseek
    python eval/memory_ab_real.py --provider deepseek --limit 1     # 先花一分钱验链路

和 `eval/memory_ab.py` 的区别
-----------------------------
`memory_ab.py` 测的是**上下文**：事实有没有进 system prompt（用 `SpyPlanner` 抓实发 messages）。
那个指标可复现，但只证明"模型看得到"。这里补**答案层**：用真模型跑同一个跨会话任务，
看它能不能**一次办成**、会不会回头问用户要已经说过的信息。

12 个场景 × 记忆 ON/OFF（初始数据库完全相同，只有"有没有长期记忆"不同）
------------------------------------------------------------------------
覆盖画像四层 × 记忆收益的三种形态（完整定义见 `SCENARIOS`）：
1. 门店偏好 ×3：昨天说过"我一般去 X 店"，今天只说"帮我约保养"——
   ON 的门店应该填对；OFF 只能问，或者落到默认门店（错的）。
2. 偏好漂移 ×3：昨天先说 A 店、又改口 B 店——ON 应按**最新**的 B 店建单。
3. 工单生命周期 ×3：跨会话改期 / 查询 / 取消——ON 直接从记忆拿单号，OFF 先查列表。
4. 双工单定位 ×3：库里同时有保养单和维修单——记忆要定位"对的那张"；
   实测结论：**v1 词面召回对口语/歧义指代失效（虚词 bigram 稀释覆盖率），
   本类场景 ON 无收益，是 v2 嵌入召回的首要目标**，保留作为基线对照。

如实说明
--------
- 门店场景里的"默认门店"是**桩数据**（`OrderStore.create` 的兜底门店），
  所以 OFF 侧不是一定失败，而是**门店会填错**——这恰好是记忆该解决的错；
- 写操作场景按真实产品流程跑**确认循环（最多 4 轮）**：尝试 → 被拦
  （needs_confirmation）→ 按实际参数补确认键 → 用户"确认" → 重试办成；
  单轮评测曾把"模型违反系统提示词直接下单"误奖为成功（早期 pref_store 的 3/3 即此类），
  确认循环修正了这一点；
- 汇总指标带 **bootstrap 95% 置信区间**（样本池化重采样，零依赖），
  单场景 n=3 的方差不再影响结论的可信度；
- 诚实边界（为什么不是 30 个场景）：否定修正（"不是那个店"）、季节性偏好、隐私遗忘
  等挑战**无法用 v1 记忆表达**（正则抽取 + 词面召回不支持）；车型档案场景实测
  ON/OFF 无区分度（模型能从工具 schema 示例猜到演示车型），已被砍掉。
  它们都随画像管线（阶段 3）加入。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
from agent.profile.pipeline import (  # noqa: E402
    ProfilePipeline,
)
from agent.profile.pipeline import (
    order_event_recorder as v2_order_event_recorder,
)
from agent.profile.retrieve import LocalEmbedder  # noqa: E402
from agent.profile.store import ProfileStore  # noqa: E402
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
# 用相对"今天"的真实未来日期，而不是 2099 年的固定日期：deepseek-flash 会把 2099 年
# 判成"距今 70 多年的笔误"并拒绝继续（pref_store2 ON 0/3 的第三层根因）。
# 日期不真实时，评测测的是模型的日期洁癖，不是长期记忆。
ORDER_DATE = (date.today() + timedelta(days=3)).isoformat()
ORDER_DATE2 = (date.today() + timedelta(days=5)).isoformat()
NEW_DATE = (date.today() + timedelta(days=7)).isoformat()
# 两家门店都故意与 OrderStore 的兜底门店（浦东金桥店）不同，这样"门店对不对"可判
CENTER = "领克中心(徐汇店)"
CENTER2 = "领克中心(杭州西湖店)"
SHENZHEN = "领克中心(深圳南山店)"      # 第三家店：门店偏好/漂移场景的变体
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
    seed_text2: str = ""              # 昨天的第二句话（漂移/改口场景：后说覆盖先说）
    need_order: bool = False          # 昨天有没有建过单
    query: str = ""                   # 今天说的话
    note: str = ""
    order_date: str = ORDER_DATE      # 本场景"今天"要预约的日期（预置确认键必须与 query 一致）
    # 昨天的额外工单（多工单场景：库里不止一张单，记忆要定位"对的那张"）
    extra_orders: List[Dict[str, str]] = field(default_factory=list)
    # check/确认键针对哪张单：0 = 主工单，1..n = extra_orders 的第 n 张
    target_order: int = 0
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


def _check_contains(substr: str) -> Callable[[Dict[str, Any], OrderStore, str], bool]:
    """答案里必须出现该事实（车型参数场景：记忆给出车型 → 工具返回对应参数）。"""
    def check(state: Dict[str, Any], orders: OrderStore, order_id: str) -> bool:
        return substr in str(state.get("answer") or "")
    return check


# ── v1/v2 记忆后端切换（基线矩阵的接口层）────────────────────────────────

class MemoryBackend:
    """v1（longterm）与 v2（profile）的统一接口：AB 脚本只认这个接口。

    基线矩阵（阶段 4）就是：同 12 场景、同 harness，只换 `--profile`。
    """

    def extract_and_remember(self, text: str) -> None:
        raise NotImplementedError

    def defaults(self) -> Dict[str, str]:
        raise NotImplementedError

    def context_text(self, query: str) -> str:
        raise NotImplementedError

    def order_recorder(self, session_id: str, query: str) -> Optional[Callable]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class V1Backend(MemoryBackend):
    """v1：LongTermMemory（正则抽取 + 词面召回，冻结为基线）。"""

    def __init__(self, mem_path: str, embedder: Any = None):
        self.store = LongTermMemory(mem_path)

    def extract_and_remember(self, text: str) -> None:
        self.store.extract_and_remember(text, user_id=USER)

    def defaults(self) -> Dict[str, str]:
        return defaults_from_preferences(self.store, USER)

    def context_text(self, query: str) -> str:
        return self.store.context_text(query, user_id=USER)

    def order_recorder(self, session_id: str, query: str):
        return order_event_recorder(self.store, USER, session_id, query)

    def close(self) -> None:
        self.store.close()


class V2Backend(MemoryBackend):
    """v2：ProfilePipeline（五层画像 + 冲突消解 + 嵌入/词面双路召回）。"""

    def __init__(self, mem_path: str, embedder: Any = None):
        self.pipeline = ProfilePipeline(ProfileStore(mem_path), embedder=embedder, user_id=USER)

    def extract_and_remember(self, text: str) -> None:
        self.pipeline.extract_and_remember(text)

    def defaults(self) -> Dict[str, str]:
        return self.pipeline.defaults_from_preferences()

    def context_text(self, query: str) -> str:
        return self.pipeline.context_text(query)

    def order_recorder(self, session_id: str, query: str):
        return v2_order_event_recorder(self.pipeline, USER, session_id, query)

    def close(self) -> None:
        self.pipeline.store.close()


def make_backend(profile: str, mem_path: str, embedder: Any = None) -> MemoryBackend:
    return V2Backend(mem_path, embedder) if profile == "v2" else V1Backend(mem_path)


BOOK = "帮我约 {date} 到店保养，换机油机滤。我已经确认过了，直接办。"
# 双工单场景的"另一张单"：维修类，日期与门店都与主工单（保养）不同
REPAIR_EXTRA = {"item": "更换刹车片", "preferred_date": ORDER_DATE2,
                "center": CENTER2, "query": "维修"}

SCENARIOS: List[Scenario] = [
    # ── 门店偏好 ×3（画像·偏好层：惯用门店自动填参）──
    Scenario("pref_store", seed_text=f"我一般去{CENTER}，以后保养都去这家",
             query=BOOK.format(date=ORDER_DATE), note="门店偏好（徐汇）",
             confirm_template="create_service_order:{item}:{date}",
             redundant_pattern=ASK_FOR_STORE, check=_pref_store_check(CENTER)),
    Scenario("pref_store2", seed_text=f"我一般去{CENTER2}，以后都去这家",
             query=BOOK.format(date=ORDER_DATE2), note="门店偏好（杭州）",
             order_date=ORDER_DATE2, confirm_template="create_service_order:{item}:{date}",
             redundant_pattern=ASK_FOR_STORE, check=_pref_store_check(CENTER2)),
    Scenario("pref_store3", seed_text=f"我一般去{SHENZHEN}，以后保养都去这家",
             query=BOOK.format(date=ORDER_DATE2), note="门店偏好（深圳）",
             order_date=ORDER_DATE2, confirm_template="create_service_order:{item}:{date}",
             redundant_pattern=ASK_FOR_STORE, check=_pref_store_check(SHENZHEN)),
    # ── 偏好漂移 ×3（画像·冲突消解层：后说的覆盖先说的）──
    Scenario("pref_drift1", seed_text=f"我一般去{CENTER}，以后都去这家",
             seed_text2=f"不对，我一般去{CENTER2}，以后都去这家",
             query=BOOK.format(date=ORDER_DATE), note="漂移：徐汇→杭州",
             confirm_template="create_service_order:{item}:{date}",
             redundant_pattern=ASK_FOR_STORE, check=_pref_store_check(CENTER2)),
    Scenario("pref_drift2", seed_text=f"我一般去{CENTER2}，以后都去这家",
             seed_text2=f"不对，我一般去{SHENZHEN}，以后都去这家",
             query=BOOK.format(date=ORDER_DATE), note="漂移：杭州→深圳",
             confirm_template="create_service_order:{item}:{date}",
             redundant_pattern=ASK_FOR_STORE, check=_pref_store_check(SHENZHEN)),
    Scenario("pref_drift3", seed_text=f"我一般去{SHENZHEN}，以后都去这家",
             seed_text2=f"不对，我一般去{CENTER}，以后都去这家",
             query=BOOK.format(date=ORDER_DATE), note="漂移：深圳→徐汇",
             confirm_template="create_service_order:{item}:{date}",
             redundant_pattern=ASK_FOR_STORE, check=_pref_store_check(CENTER)),
    # ── 工单生命周期 ×3（情景记忆：跨会话定位自己的工单）──
    Scenario("order_reschedule", need_order=True,
             query=f"上次那个保养我想改到 {NEW_DATE}。我已经确认过了，直接改。",
             note="跨会话改期", confirm_template="reschedule_order:{order_id}:{new_date}",
             redundant_pattern=ASK_FOR_ORDER, check=_check_reschedule),
    Scenario("order_query", need_order=True, query="我那个保养约的是哪天？",
             note="跨会话查询", check=_check_query),
    Scenario("order_cancel", need_order=True, query="上次那个保养取消掉吧，我已经确认过了。",
             note="跨会话取消", confirm_template="cancel_order:{order_id}",
             redundant_pattern=ASK_FOR_ORDER, check=_check_cancel),
    # ── 多工单情景定位 ×3（库里两张单，记忆要定位"对的那张"）──
    Scenario("multi_reschedule", need_order=True, extra_orders=[REPAIR_EXTRA],
             query=f"上次那个保养我想改到 {NEW_DATE}。我已经确认过了，直接改。",
             note="双工单·定位保养改期", confirm_template="reschedule_order:{order_id}:{new_date}",
             redundant_pattern=ASK_FOR_ORDER, check=_check_reschedule),
    Scenario("multi_cancel_repair", need_order=True, extra_orders=[REPAIR_EXTRA],
             query="上次那个维修取消掉吧，我已经确认过了。",
             note="双工单·定位维修取消", target_order=1,
             confirm_template="cancel_order:{order_id}",
             redundant_pattern=ASK_FOR_ORDER, check=_check_cancel),
    Scenario("multi_reschedule_repair", need_order=True, extra_orders=[REPAIR_EXTRA],
             query=f"上次那个维修我想改到 {NEW_DATE}。我已经确认过了，直接改。",
             note="双工单·定位维修改期", target_order=1,
             confirm_template="reschedule_order:{order_id}:{new_date}",
             redundant_pattern=ASK_FOR_ORDER, check=_check_reschedule),
]

# 被砍掉的场景类型（诚实记录，别偷偷塞回）：
# 1. 车型档案（"我的车是领克08" → 问轮胎规格）：OFF 侧模型从工具 schema 的示例
#    里就能猜到演示车型并调 lookup_vehicle_spec，ON/OFF 无区分度（实测 OFF 100%）。
#    这类场景归入阶段 4 的画像基准（v1 vs v2），比较的是冗余反问率与参数正确率。
# 2. 更多多工单变体：v1 词面召回对口语/歧义指代失效（见 multi_* 场景注释），
#    扩大变体只会放大同一个已知弱点，等 v2 嵌入召回落地后再扩。


def seed(backend: MemoryBackend, orders_path: str, scenario: Scenario) -> Dict[str, Any]:
    """把"昨天发生过的事"写进两个库；返回 {"primary": 主工单号, "extras": [额外工单号…]}。

    情景记忆的写入带**按工单区分的 query 前缀**（"用户提到「保养」；已建工单…"）：
    多工单场景下如果两张单的摘要共享同一个前缀，召回会把两张都捞出来，
    "上次那个维修"就定位不到正确的那张（第一版就是这么设计失败的）。
    """
    orders = OrderStore(orders_path, enforce_future_date=True)
    oids: Dict[str, Any] = {"primary": "", "extras": []}
    try:
        if scenario.seed_text:
            backend.extract_and_remember(scenario.seed_text)
        if scenario.seed_text2:
            # 漂移/改口：第二句按写入顺序覆盖第一句（overwrite 语义）
            backend.extract_and_remember(scenario.seed_text2)
        defaults = backend.defaults() if scenario.seed_text else {}
        registry = ToolRegistry(KnowledgeBase.load(FIXTURE_KB), orders=orders,
                                user_id=USER, defaults=defaults, on_event=None)

        def create_one(item: str, pdate: str, center: str, prefix: str) -> str:
            """建一张单（预置确认），并把情景记忆按业务前缀手动写入。

            多工单场景的坑：如果两张单的摘要共享同一个 query 前缀，词面召回会把
            两张都捞出来，"上次那个维修"就定位不到正确的那张（第一版设计如此）。
            """
            args = {"item": item, "preferred_date": pdate, "center": center}
            pending = registry.call("create_service_order", args)
            if pending.meta.get("needs_confirmation"):
                registry.confirm(pending.meta["action_key"])
            result = registry.call("create_service_order", args)
            if result.ok and result.data:
                rec = backend.order_recorder("seed", prefix)
                if rec:
                    rec("order_created", result.data)
                return str(result.data.get("预约单号") or "")
            return ""

        if scenario.need_order:
            oids["primary"] = create_one(ITEM, ORDER_DATE,
                                         defaults.get("center", "") or CENTER, "保养")
        for i, extra in enumerate(scenario.extra_orders):
            oid = create_one(extra.get("item", ""), extra.get("preferred_date", ""),
                             extra.get("center", "") or CENTER,
                             extra.get("query", f"工单{i + 1}"))
            oids["extras"].append(oid)
    finally:
        orders.close()
    return oids


def run_case(scenario: Scenario, use_memory: bool, llm: Any, kb: Any,
             profile: str, embedder: Any, mem_path: str, orders_path: str,
             order_id: str = "") -> Dict[str, Any]:
    backend = make_backend(profile, mem_path, embedder) if use_memory else None
    orders = OrderStore(orders_path, enforce_future_date=False)
    try:
        registry = ToolRegistry(
            kb, orders=orders, user_id=USER,
            defaults=backend.defaults() if backend else {},
            on_event=backend.order_recorder("ab", scenario.query) if backend else None,
        )
        # 写操作需要车主确认：评测里把"用户已确认"预置好（两种条件完全一致），
        # 否则会把"确认流程"误判成"任务失败"。
        # 注意：确认键里的日期必须用**本场景的 order_date**，与 query 的日期一致；
        # 早前版本统一用 ORDER_DATE，导致 pref_store2（约 06-25）预确认的是 06-18，
        # 工具端按实际日期生成的 action_key 匹配不上 → 永远落在"需二次确认"→ ON 假性 0/3。
        if scenario.confirm_template:
            action_key = scenario.confirm_template.format(
                item=ITEM, date=scenario.order_date or ORDER_DATE,
                new_date=NEW_DATE, order_id=order_id)
            registry.confirm(action_key)
        memory = ConversationMemory()
        if backend is not None:
            context = backend.context_text(scenario.query)
            memory.attach_longterm(context)
        # 这是"任务完成"评测，不是 RAG 接地评测：关掉证据门控与接地拒答，
        # 否则模型按设计先说一句"确认吗"就会被判 ungrounded → 直接拒答（实测踩过）。
        graph = AgentGraph(llm, registry,
                           config=AgentConfig(max_steps=4, evidence_gate=False,
                                              refuse_on_insufficient=False,
                                              enable_reflection=False),
                           memory=memory, agent_name="ab")
        states = [graph.run(scenario.query)]
        # ── 写操作场景的多轮确认：对齐真实产品流程（尝试 → 拦下 → 确认 → 重试）──
        # 产品路径：agent 调写工具 → 未确认被拦（needs_confirmation）→ 客户端拿
        # action_key 调 /v1/writes/confirm → agent 重跑办成。单轮评测只能奖励
        # "模型违反系统提示词直接下单"的运气行为（早期 pref_store 的 3/3 即此类，
        # 不可复现、也不该鼓励），所以这里改为最多 3 轮的确认循环：
        #   1) 每次跑完，把被拦下的写调用按**模型实际参数**补进 confirmed_actions——
        #      模型会把"换机油机滤"扩展成保养全项列表，确认键必须跟着实际参数走，
        #      否则 exact-match 的键永远对不上（pref_store2 0/3 的第四层根因）；
        #   2) 已办成（check 通过）就停；否则用户说"确认"再跑一轮；
        #   3) 记忆的收益体现在：任何一轮都不该再追问门店/单号（redundant），
        #      而不是"省掉确认轮"——确认轮是产品规则，不是记忆的失职。
        if scenario.confirm_template:
            for _ in range(3):
                for call in states[-1].tool_calls:
                    name = call.get("tool") or ""
                    if name not in ("create_service_order", "reschedule_order", "cancel_order"):
                        continue
                    args = call.get("arguments") or {}
                    if name == "create_service_order":
                        key = f"create_service_order:{args.get('item')}:{args.get('preferred_date')}"
                    else:
                        key = f"{name}:{args.get('order_id')}:{args.get('new_date')}"
                    registry.confirm(key)
                ok_now = bool(scenario.check(states[-1].to_dict(), orders, order_id)) \
                    if scenario.check else True
                if ok_now:
                    break
                states.append(graph.run("确认，就按刚才说的办"))
        final = states[-1]
        payload = final.to_dict()
        payload["status"] = final.status
        answers = [str(s.answer or "") for s in states]
        asked = any(bool(ASK_PATTERN.search(a)) for a in answers)
        redundant = any(bool(scenario.redundant_pattern.search(a)) for a in answers) \
            if scenario.redundant_pattern is not None else False
        return {
            "state": payload,
            "success": bool(scenario.check(payload, orders, order_id)) if scenario.check else False,
            "asked_user": asked,
            "asked_redundant": redundant,
            "tool_calls": sum(len(s.tool_calls) for s in states),
            "tokens": sum(s.tokens_total for s in states),
            "answer": "\n\n".join(answers)[:320],
            # 逐轮明细：失败归因要靠它——"最后一轮到底调没调工具、参数是什么"。
            "turns": [{"status": s.status,
                       "calls": [{"tool": c.get("tool"), "ok": c.get("ok"),
                                  "args": c.get("arguments"),
                                  "error": str(c.get("error"))[:80]} for c in s.tool_calls],
                       "answer": str(s.answer or "")[:200]} for s in states],
        }
    finally:
        orders.close()
        if backend is not None:
            backend.close()


def bootstrap_ci(vals: Sequence[int], n_resample: int = 2000,
                 seed_rng: int = 7) -> Tuple[float, float, float]:
    """二值样本均值的 bootstrap 95% 置信区间（零依赖）。

    为什么不用正态近似：单场景 3 次重复的 n 太小，正态区间的假设不成立；
    这里把所有场景×重复的样本池化后重采样，报"均值的区间"而不是点估计。
    """
    bits = [1.0 if v else 0.0 for v in vals]
    if not bits:
        return (0.0, 0.0, 0.0)
    mean = sum(bits) / len(bits)
    if len(bits) == 1:
        return (mean, mean, mean)
    rng = random.Random(seed_rng)
    means: List[float] = []
    for _ in range(n_resample):
        sample = [bits[rng.randrange(len(bits))] for _ in range(len(bits))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * (n_resample - 1))]
    hi = means[int(0.975 * (n_resample - 1))]
    return (mean, lo, hi)


def main() -> int:
    ap = argparse.ArgumentParser(description="真模型的记忆答案层 AB")
    add_llm_args(ap)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个场景（省钱验链路用）")
    ap.add_argument("--sid", default="", help="只跑指定场景 id（如 pref_store2），留空跑全部")
    ap.add_argument("--repeat", type=int, default=2,
                    help="每个条件重复跑几次（模型有随机性，单次不足以判断）")
    ap.add_argument("--profile", choices=["v1", "v2"], default="v1",
                    help="记忆后端：v1=longterm 基线（默认），v2=profile 画像管线")
    ap.add_argument("--json-out", default=os.path.join(ROOT, "eval", "memory_ab_real.json"))
    args = ap.parse_args()

    kb = KnowledgeBase.load(FIXTURE_KB if os.path.exists(FIXTURE_KB) else None)
    llm = make_llm(args, known_tools=ToolRegistry(kb).names())
    embedder = LocalEmbedder() if args.profile == "v2" else None
    scenarios = SCENARIOS[:args.limit] if args.limit else SCENARIOS
    if args.sid:
        scenarios = [s for s in scenarios if s.sid == args.sid]
        if not scenarios:
            print(f"[!] 没有场景 {args.sid}；可用：{[s.sid for s in SCENARIOS]}")
            return 2
    tmp = tempfile.mkdtemp(prefix="mem-ab-real-")

    print("=" * 78)
    print(f"记忆答案层 AB · 后端 {args.profile}（{llm_label(args)}）")
    print("=" * 78)

    rows: List[Dict[str, Any]] = []
    repeats = max(1, args.repeat)
    bits: Dict[str, List[int]] = {"on_success": [], "off_success": [],
                                  "on_redundant": [], "off_redundant": []}
    for scenario in scenarios:
        seed_mem = os.path.join(tmp, f"{scenario.sid}_seed_mem.db")
        seed_orders = os.path.join(tmp, f"{scenario.sid}_seed_orders.db")
        seed_backend = make_backend(args.profile, seed_mem, embedder)
        oids = seed(seed_backend, seed_orders, scenario)
        seed_backend.close()
        all_ids = [oids["primary"]] + list(oids["extras"])
        target_id = all_ids[scenario.target_order] if scenario.target_order < len(all_ids) \
            else oids["primary"]
        if scenario.need_order and not target_id:
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
                item = run_case(scenario, condition == "on", llm, kb,
                                args.profile, embedder, mem_path, orders_path, target_id)
                results[condition].append(item)
                bits[f"{condition}_success"].append(1 if item["success"] else 0)
                bits[f"{condition}_redundant"].append(1 if item["asked_redundant"] else 0)

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
        rows.append({"scenario": scenario.sid, "note": scenario.note,
                     "order_id": target_id, "target_order": scenario.target_order,
                     "repeats": repeats, **agg,
                     "samples": {"on": [i["answer"][:160] for i in results["on"]],
                                 "off": [i["answer"][:160] for i in results["off"]]},
                     "details": {"on": [i["turns"] for i in results["on"]],
                                 "off": [i["turns"] for i in results["off"]]}})
        print(f"\n  [{scenario.sid}] {scenario.note}（{repeats} 次）")
        for name, data in (("ON ", agg["on"]), ("OFF", agg["off"])):
            print(f"    {name} 一次办成 {data['success_rate']:.0%} ｜ "
                  f"重复追问 {data['redundant_rate']:.0%} ｜ "
                  f"平均工具调用 {data['avg_calls']:.2f} ｜ 平均 tokens {data['avg_tokens']}")

    total = max(1, len(bits["on_success"]))
    on_success = sum(bits["on_success"])
    off_success = sum(bits["off_success"])
    on_redundant = sum(bits["on_redundant"])
    off_redundant = sum(bits["off_redundant"])
    on_calls = sum(r["on"]["avg_calls"] for r in rows) / max(1, len(rows))
    off_calls = sum(r["off"]["avg_calls"] for r in rows) / max(1, len(rows))
    fields = backend_fields(args, llm)

    def ci_line(name: str, vals: List[int]) -> str:
        mean, lo, hi = bootstrap_ci(vals)
        return f"{name} {mean:.1%}  [95% CI {lo:.1%}–{hi:.1%}]  n={len(vals)}"

    print("\n" + "-" * 78)
    print(f"一次办成：记忆 ON {on_success}/{total} ｜ OFF {off_success}/{total}")
    print("  " + ci_line("ON 办成率 ", bits["on_success"]))
    print("  " + ci_line("OFF 办成率", bits["off_success"]))
    print(f"**重复追问已说过的信息**：ON {on_redundant}/{total} ｜ "
          f"OFF {off_redundant}/{total}")
    print("  " + ci_line("ON 重复追问 ", bits["on_redundant"]))
    print("  " + ci_line("OFF 重复追问", bits["off_redundant"]))
    print(f"平均工具调用：ON {on_calls:.2f} ｜ OFF {off_calls:.2f}")
    print(cost_line(fields))

    ci = {name: list(bootstrap_ci(vals)) for name, vals in bits.items()}
    summary = {
        "backend": fields["backend"], "model": fields["model"],
        "profile": args.profile,
        "scenarios": len(rows), "repeats": repeats, "runs_total": total,
        "success_on": on_success, "success_off": off_success,
        "redundant_on": on_redundant, "redundant_off": off_redundant,
        "avg_tool_calls_on": round(on_calls, 2), "avg_tool_calls_off": round(off_calls, 2),
        "ci": {name: {"mean": round(v[0], 4), "lo": round(v[1], 4), "hi": round(v[2], 4)}
               for name, v in ci.items()},
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
