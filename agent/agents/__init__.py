# -*- coding: utf-8 -*-
"""专家子 Agent 集合：手册 / 车控 / 服务 / 安全评审。"""

from agent.agents.base import ScopedPlanner, SubAgent
from agent.agents.manual_expert import ManualExpert
from agent.agents.safety_critic import SafetyCritic
from agent.agents.service_advisor import ServiceAdvisor
from agent.agents.vehicle_control import VehicleControlAgent, assess_telemetry

SPECIALIST_CLASSES = (ManualExpert, VehicleControlAgent, ServiceAdvisor)


def build_specialists(kb, llm=None, parent_registry=None, config=None, tracer=None, policy=None):
    """构建四个专家子 Agent（安全评审不是 SubAgent，而是聚合后的评审器）。"""
    return {
        cls.name: cls(kb, llm=llm, parent_registry=parent_registry, config=config,
                      tracer=tracer, policy=policy)
        for cls in SPECIALIST_CLASSES
    }


__all__ = ["ScopedPlanner", "SubAgent", "ManualExpert", "VehicleControlAgent",
           "ServiceAdvisor", "SafetyCritic", "assess_telemetry",
           "SPECIALIST_CLASSES", "build_specialists"]
