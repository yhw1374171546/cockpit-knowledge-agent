# -*- coding: utf-8 -*-
"""服务顾问：保养计划、到店预约与工单跟进（改期 / 取消 / 查询）。

唯一持有**写操作**工具的 Agent（创建、改期、取消），因此它也是安全策略的重点：
写操作必须车主确认，且注入风险升高时该工具会被策略引擎直接禁用。
工单是真实有状态的对象（见 `agent/orders.py`），也是"跨会话记忆"与"MCP 复用"的消费者。
"""

from __future__ import annotations

from agent.agents.base import SubAgent


class ServiceAdvisor(SubAgent):
    name = "service_advisor"
    title = "服务顾问"
    kind = "maintenance"
    description = ("按里程给出保养建议；在车主确认后创建、改期或取消到店服务工单（写操作）；"
                   "也能查询车主已有的工单")
    allowed_tools = ("get_maintenance_plan", "search_manual", "create_service_order",
                     "get_my_orders", "reschedule_order", "cancel_order")
    planner_kinds = ("booking", "maintenance")
