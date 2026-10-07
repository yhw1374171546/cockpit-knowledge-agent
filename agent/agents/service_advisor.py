# -*- coding: utf-8 -*-
"""服务顾问：保养计划与到店预约。

唯一持有**写操作**工具的 Agent（create_service_order），因此它也是安全策略的重点：
写操作必须车主确认，且注入风险升高时该工具会被策略引擎直接禁用。
"""

from __future__ import annotations

from agent.agents.base import SubAgent


class ServiceAdvisor(SubAgent):
    name = "service_advisor"
    title = "服务顾问"
    kind = "maintenance"
    description = "按里程给出保养项目建议，并在车主确认后创建到店服务预约（写操作）"
    allowed_tools = ("get_maintenance_plan", "search_manual", "create_service_order")
    planner_kinds = ("booking", "maintenance")
