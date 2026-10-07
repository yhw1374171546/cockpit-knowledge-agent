# -*- coding: utf-8 -*-
"""手册专家：只读手册知识库，负责"说明书上怎么写的"，必须给出页码出处。"""

from __future__ import annotations

from agent.agents.base import SubAgent


class ManualExpert(SubAgent):
    name = "manual_expert"
    title = "手册专家"
    kind = "manual_qa"
    description = "检索车主手册与车型参数，回答功能操作、部件说明、故障处置步骤类问题，答案必须带页码出处"
    allowed_tools = ("search_manual", "lookup_vehicle_spec")
    planner_kinds = ("spec", "manual_qa")
