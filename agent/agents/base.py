# -*- coding: utf-8 -*-
"""子 Agent 基类：工具白名单 + 独立预算 + 结构化产出。

与"单 Agent 挂全部工具"的本质区别：
1. **工具白名单**——子 Agent 看不到不该用的工具，从物理上杜绝误调用；
2. **独立失败域**——一个子 Agent 报错/超预算不会拖垮整轮；
3. **结构化产出**——统一返回 AgentResult（含风险等级、证据、置信度），供 Supervisor 仲裁。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

from agent.guardrails.budget import Budget, BudgetTracker
from agent.guardrails.policy import PolicyEngine
from agent.graph import AgentConfig, AgentGraph
from agent.kb import KnowledgeBase
from agent.llm import LLMBackend, RuleBasedPlannerLLM
from agent.memory import ConversationMemory, VehicleProfile
from agent.obs.tracer import Tracer
from agent.protocols import (AGENT_PRIORITY, RISK_OK, AgentResult, Handoff, SubTask)
from agent.reflection import NO_ANSWER, Reflector
from agent.tools import build_default_registry


class ScopedPlanner(RuleBasedPlannerLLM):
    """把"意图"限定在子 Agent 的职责范围内（Supervisor 已经路由过了）。

    复用 RuleBasedPlannerLLM 的规划与答案组装逻辑，只覆写 route()：
    在 allowed_kinds 里找一个匹配的意图，找不到就用主意图，
    这样离线环境下子 Agent 的行为依然完全确定、可复现。
    """

    def __init__(self, kind: str, allowed_kinds: Optional[Sequence[str]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.kind = kind
        self.allowed_kinds = tuple(allowed_kinds or (kind,))

    def route(self, question: str) -> str:      # noqa: D102
        for intent, keys in self.INTENT_RULES:
            if intent in self.allowed_kinds and any(k in question for k in keys):
                return intent
        return self.kind


class SubAgent:
    """专家子 Agent：一个受限的 AgentGraph + 结构化结果封装。"""

    name: str = "sub_agent"
    title: str = "子 Agent"
    kind: str = "manual_qa"
    description: str = ""
    allowed_tools: Sequence[str] = ("search_manual",)
    planner_kinds: Sequence[str] = ()

    def __init__(self, kb: KnowledgeBase, llm: Optional[LLMBackend] = None,
                 parent_registry=None, config: Optional[AgentConfig] = None,
                 tracer: Optional[Tracer] = None, policy: Optional[PolicyEngine] = None):
        self.kb = kb
        self.tracer = tracer or Tracer()
        self.policy = policy or PolicyEngine()
        base_registry = parent_registry or build_default_registry()
        self.registry = base_registry.subset(self.allowed_tools, self.name)
        # 注意：必须**复制**配置再改，不能直接改传入的 config ——
        # Supervisor 的 baseline 图与所有子 Agent 共享同一个 config 对象，
        # 直接改会把 baseline 的安全门控也一起关掉（别名 bug，实测导致
        # 单 Agent 基线的安全指令合规率错误地停在 0%）。
        self.config = replace(config) if config is not None else AgentConfig()
        # 多 Agent 模式：安全指令由 safety_critic 独立评审并负责注入，
        # 子 Agent 只做风险判定（见 AgentConfig.enable_safety_gate 的说明）。
        self.config.enable_safety_gate = False
        self.llm = llm or ScopedPlanner(self.kind, self.planner_kinds or (self.kind,))
        self.reflector = Reflector(evidence_overlap_threshold=self.config.evidence_overlap_threshold)
        self.graph = AgentGraph(self.llm, self.registry, reflector=self.reflector,
                                memory=ConversationMemory(profile=VehicleProfile()),
                                config=self.config, tracer=self.tracer, policy=self.policy,
                                agent_name=self.name)

    # ── 对外入口 ──
    def run(self, handoff: Handoff, memory: Optional[ConversationMemory] = None) -> AgentResult:
        task = handoff.task
        budget = BudgetTracker(Budget(label=self.name))
        self.graph.budget = budget
        self.graph.memory = memory or ConversationMemory(profile=VehicleProfile())
        with self.tracer.span(f"subagent.{self.name}", self.name, task=task.query[:40]) as span:
            state = self.graph.run(task.query)
            span.attrs["status"] = state.status
            result = self._to_result(task, state, budget)
        return result

    def _to_result(self, task: SubTask, state, budget: BudgetTracker) -> AgentResult:
        risk, findings = self.assess(state)
        grounded = (state.reflection or {}).get("grounded_ratio")
        return AgentResult(
            agent=self.name, subtask_id=task.id, kind=self.kind,
            answer=(state.answer or "").strip(),
            evidence=state.evidence, citations=state.citations,
            status=state.status,
            risk_level=risk, findings=findings,
            confidence=float(grounded) if grounded is not None else (1.0 if state.answer else 0.0),
            priority=AGENT_PRIORITY.get(self.name, task.priority),
            policy_flags=[b["reason"] for b in state.policy_blocks],
            needs_confirmation=state.status == "needs_confirmation",
            injection_flagged=state.injection_flagged,
            injection_risk=state.injection_risk,
            latency_ms=state.latency_ms.get("total", 0.0),
            tokens_total=state.tokens_total, steps=state.steps,
        )

    def assess(self, state) -> tuple:
        """子 Agent 的风险评估钩子；默认继承 AgentState 的注入风险。"""
        if state.injection_flagged:
            return "warning", ["检索内容疑似包含注入指令"]
        return RISK_OK, []

    def tools(self) -> List[str]:
        return list(self.registry.names())
