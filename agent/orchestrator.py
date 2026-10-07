# -*- coding: utf-8 -*-
"""Supervisor 编排器：拆解 → 委派（并行）→ 仲裁 → 聚合 → 安全评审。

与"单 Agent 挂全部工具"的对比是这个模块存在的意义：
- **拆解**：多意图请求拆成子任务，各自找对专家；
- **委派**：结构化 handoff + 子预算，工具白名单收窄，物理上减少误调用；
- **仲裁**：冲突有明确优先级规则（安全 > 实时车况 > 服务政策 > 手册通用说明），不靠模型自由发挥；
- **隔离**：子 Agent 并行执行，单个失败/超预算不影响其它子任务；
- **把关**：安全评审 Agent 对最终答案做接地 + 安全合规检查。
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

from agent.agents.base import SubAgent
from agent.agents.manual_expert import ManualExpert
from agent.agents.safety_critic import SafetyCritic
from agent.agents.service_advisor import ServiceAdvisor
from agent.agents.vehicle_control import VehicleControlAgent
from agent.guardrails.budget import Budget
from agent.guardrails.policy import PolicyEngine
from agent.graph import AgentConfig, AgentGraph
from agent.llm import LLMBackend, RuleBasedPlannerLLM
from agent.memory import ConversationMemory, VehicleProfile
from agent.obs.tracer import Tracer
from agent.protocols import (AGENT_PRIORITY, PERMISSIVE_PATTERNS, RISK_CRITICAL,
                            AgentResult, ConflictReport, Handoff, Plan, SubTask,
                            SupervisorState, extract_numbers, risk_at_least)
from agent.reflection import NO_ANSWER, Reflector
from agent.tools import ToolRegistry

SUPERVISOR_PROMPT = """你是智能座舱的调度中枢（Supervisor）。你的职责不是自己回答问题，而是：
1. 判断车主请求需要哪些专家：手册专家（功能/故障/说明书）、车控专家（实时车况/告警）、服务顾问（保养/预约）；
2. 把请求拆成互不依赖的子任务并委派，每个子任务只交给最合适的专家；
3. 汇总各专家结论时遵守仲裁优先级：安全 > 实时车况 > 服务政策 > 手册通用说明；
4. 任何写操作（预约到店）必须获得车主确认；实时告警为严重级别时必须给出停驶与联系中心的指令。
"""

# 强拆分标记：出现即说明车主在提**新的一件事情**
STRONG_SPLIT = ("顺便", "另外", "同时", "以及", "还有", "并且", "接着", "然后", "再帮我", "；", ";")
# 弱拆分标记：逗号后面可能只是补充说明（"胎压报警了，还能继续开吗"是同一件事），
# 因此只有该片段自身命中意图关键词时才独立成一个子任务，否则并入前一个片段。
WEAK_SPLIT = ("，",)
# 引用手册的说法 → 说明车主在拿"手册的通用说明"和"当下的实际情况"做对比，
# 这类问题必须同时派出「手册专家」和「实时车况专家」，冲突仲裁才有意义。
REFERENCE_CUES = ("手册", "说明书", "听谁的", "哪个对", "书上说", "不是说")

KIND_TO_AGENT = {
    "manual_qa": "manual_expert",
    "spec": "manual_expert",
    "status": "vehicle_control",
    "maintenance": "service_advisor",
    "booking": "service_advisor",
}


class Supervisor:
    def __init__(self, kb, llm: Optional[LLMBackend] = None,
                 config: Optional[AgentConfig] = None,
                 tracer: Optional[Tracer] = None,
                 policy: Optional[PolicyEngine] = None,
                 telemetry: Optional[Dict] = None,
                 budget: Optional[Budget] = None,
                 parallel: bool = True):
        self.kb = kb
        self.tracer = tracer or Tracer()
        self.policy = policy or PolicyEngine()
        self.config = config or AgentConfig()
        self.parallel = parallel
        self.budget = budget or Budget(label="supervisor")
        self.registry = ToolRegistry(kb, telemetry=telemetry)
        self.critic = SafetyCritic(Reflector(evidence_overlap_threshold=self.config.evidence_overlap_threshold))
        # 单 Agent 基线：挂全部工具，用于对比（同一个 LLM 后端）
        self.baseline = AgentGraph(llm or RuleBasedPlannerLLM(), self.registry,
                                   reflector=Reflector(evidence_overlap_threshold=self.config.evidence_overlap_threshold),
                                   memory=ConversationMemory(profile=VehicleProfile()),
                                   config=self.config, tracer=self.tracer, policy=self.policy,
                                   agent_name="single_agent")
        self.specialists: Dict[str, SubAgent] = {}
        for cls in (ManualExpert, VehicleControlAgent, ServiceAdvisor):
            agent = cls(kb, llm=llm, parent_registry=self.registry, config=self.config,
                        tracer=self.tracer, policy=self.policy)
            self.specialists[agent.name] = agent
        self.last_plan: Optional[Plan] = None

    # ── 拆解 ──
    def decompose(self, question: str) -> Plan:
        fragments = self._split(question)
        subtasks: List[SubTask] = []
        for idx, frag in enumerate(fragments):
            kind = self._classify(frag)
            agent = KIND_TO_AGENT.get(kind, "manual_expert")
            subtasks.append(SubTask(id=f"t{idx+1}", agent=agent, kind=kind, query=frag,
                                    goal=f"{agent} 处理：{frag[:24]}",
                                    priority=AGENT_PRIORITY.get(agent, 50)))
            # 片段里引用了"手册的说法" → 额外派一个手册专家去核实原文
            if any(cue in frag for cue in REFERENCE_CUES) and agent != "manual_expert":
                subtasks.append(SubTask(id=f"t{idx+1}m", agent="manual_expert", kind="manual_qa",
                                        query=frag, goal=f"核实手册原文：{frag[:20]}",
                                        priority=AGENT_PRIORITY["manual_expert"]))
        # 同一专家的多个子任务合并（避免重复检索同样内容）
        merged: Dict[str, SubTask] = {}
        for st in subtasks:
            if st.agent in merged:
                merged[st.agent].query += " " + st.query
                merged[st.agent].goal += "；" + st.goal
            else:
                merged[st.agent] = st
        final = sorted(merged.values(), key=lambda s: -s.priority)
        route = "multi" if len(final) > 1 else "single"
        reason = (f"识别到 {len(final)} 个专家分工：{', '.join(s.agent for s in final)}"
                  if route == "multi" else f"单意图，交由 {final[0].agent}")
        self.last_plan = Plan(route=route, subtasks=final, reason=reason)
        return self.last_plan

    def _split(self, question: str) -> List[str]:
        """两阶段拆分：强标记直接分段；弱标记（逗号）只在片段自带意图信号时才独立。

        这样"胎压报警了，还能继续开到维修站吗"不会被误拆成两个专家任务，
        而"胎压报警了怎么办，另外座椅加热怎么关闭"仍能正确拆成两件事。
        """
        strong_parts = [question.strip()]
        for marker in STRONG_SPLIT:
            expanded: List[str] = []
            for part in strong_parts:
                expanded.extend(p.strip() for p in part.split(marker) if p.strip())
            strong_parts = expanded

        merged: List[str] = []
        for part in strong_parts:
            weak = [p.strip() for p in part.split(WEAK_SPLIT[0]) if p.strip()] or [part]
            current = weak[0]
            for fragment in weak[1:]:
                if self._has_intent_signal(fragment):
                    merged.append(current)
                    current = fragment
                else:
                    current = f"{current}，{fragment}"      # 视为补充说明，并入前一片段
            merged.append(current)
        return [p for p in merged if len(p) >= 3] or [question.strip()]

    def _has_intent_signal(self, text: str) -> bool:
        if any(any(k in text for k in keys)
               for _intent, keys in RuleBasedPlannerLLM.INTENT_RULES):
            return True
        # 引用手册的说法本身就是"要查手册"的信号（"手册说可以继续开，我该听谁的"）
        return any(cue in text for cue in REFERENCE_CUES)

    def _classify(self, text: str) -> str:
        for intent, keys in RuleBasedPlannerLLM.INTENT_RULES:
            if any(k in text for k in keys):
                return intent
        return "manual_qa"

    # ── 执行 ──
    def run(self, question: str, memory: Optional[ConversationMemory] = None) -> SupervisorState:
        start = time.perf_counter()
        mem = memory or ConversationMemory(profile=VehicleProfile())
        with self.tracer.span("supervisor.run", "supervisor", question=question[:40]) as span:
            state = SupervisorState(question=question)
            plan = self.decompose(question)
            state.plan = plan

            t0 = time.perf_counter()
            state.results = self._dispatch(plan, mem, state)
            state.latency_ms["dispatch"] = (time.perf_counter() - t0) * 1000

            state.conflicts = detect_conflicts(state.results)
            state.answer = self._aggregate(state)
            state.tokens_total = sum(r.tokens_total for r in state.results)
            state.injection_risk = sum(r.injection_risk for r in state.results)

            # 安全评审（最后一道关）
            t1 = time.perf_counter()
            evidence = [e for r in state.results for e in r.evidence]
            verdict = self.critic.review(question, state.answer, state.results, evidence)
            state.verdict = verdict
            state.latency_ms["review"] = (time.perf_counter() - t1) * 1000
            if verdict.verdict in ("revise", "block") and verdict.revised_answer is not None:
                state.answer = verdict.revised_answer
                if verdict.verdict == "block":
                    state.status = "refused"
            if not state.status or state.status == "answered":
                state.status = self._status_from_results(state.results)
            state.citations = [c for r in sorted(state.results, key=lambda x: -x.priority)
                               for c in r.citations][:6]
            state.latency_ms["total"] = (time.perf_counter() - start) * 1000
            state.failure = self.tracer.classify_failure({
                "injection_flagged": any(r.injection_flagged for r in state.results),
                "policy_blocked": any(r.policy_flags for r in state.results),
                "refused": state.answer.strip() == NO_ANSWER,
                "under_call": not state.results,
            })
            span.attrs.update({"route": plan.route, "status": state.status,
                               "n_subtasks": len(plan.subtasks)})
            return state

    def _dispatch(self, plan: Plan, mem: ConversationMemory,
                  state: SupervisorState) -> List[AgentResult]:
        results: List[AgentResult] = []
        handoffs: List[Handoff] = []
        for task in plan.subtasks:
            child = self.budget.slice(task.agent, step_ratio=0.7, tool_ratio=0.7,
                                     token_ratio=0.7, cost_ratio=0.7)
            handoffs.append(Handoff(from_agent="supervisor", to_agent=task.agent, task=task,
                                    context_digest=mem.system_context(),
                                    injected_constraints=["工具白名单受限",
                                                          "写操作需车主确认"]))
        state.handoffs = handoffs

        def _run(handoff: Handoff) -> AgentResult:
            agent = self.specialists.get(handoff.to_agent)
            if agent is None:
                return AgentResult(agent=handoff.to_agent, subtask_id=handoff.task.id,
                                   status="blocked", answer=NO_ANSWER)
            return agent.run(handoff, memory=ConversationMemory(profile=mem.profile))

        if self.parallel and len(handoffs) > 1:
            with ThreadPoolExecutor(max_workers=min(3, len(handoffs))) as pool:
                results = list(pool.map(_run, handoffs))
        else:
            results = [_run(h) for h in handoffs]
        return results

    def _aggregate(self, state: SupervisorState) -> str:
        ordered = sorted(state.results, key=lambda r: -r.priority)
        pending = [r for r in ordered if r.needs_confirmation]
        if pending:
            return pending[0].answer or "该操作需要您确认后再执行。"
        answered = [r for r in ordered if r.answered]
        if not answered:
            return NO_ANSWER
        # 有安全冲突时，实时车况的结论必须排在第一位
        if any(c.kind == "safety_override" for c in state.conflicts):
            answered.sort(key=lambda r: (0 if risk_at_least(r.risk_level, RISK_CRITICAL) else 1,
                                         -r.priority))
        parts, seen = [], set()
        for r in answered:
            text = r.answer.strip()
            if text and text not in seen:
                parts.append(text)
                seen.add(text)
        return "；".join(parts) if len(parts) > 1 else parts[0]

    def _status_from_results(self, results: Sequence[AgentResult]) -> str:
        if any(r.needs_confirmation for r in results):
            return "needs_confirmation"
        if any(r.answered for r in results):
            return "answered"
        return "refused"

    # ── 单 Agent 基线（同一套护栏与 trace，保证对比公平）──
    def run_baseline(self, question: str, memory: Optional[ConversationMemory] = None):
        return self.baseline.run(question, memory=memory)

    def stats(self) -> Dict[str, Any]:
        return {"tools_total": len(self.registry.names()),
                "specialists": {n: a.tools() for n, a in self.specialists.items()},
                "critic": self.critic.stats()}


# ── 冲突检测与仲裁 ───────────────────────────────────────────────────


def detect_conflicts(results: Sequence[AgentResult]) -> List[ConflictReport]:
    """检测并仲裁子 Agent 之间的结论冲突。

    规则（可扩展）：
    1. **safety_override**：某专家判定实时风险为 critical，而另一个专家（通常是手册类）
       给出了"可继续行驶"这类许可性建议 → 安全优先，实时车况覆盖通用说明；
    2. **numeric_mismatch**：两个专家对同一可比较事实（白名单：续航/胎压）给出不同数值
       → 取优先级更高来源，并标记 escalated（需人工复核）。
    """
    conflicts: List[ConflictReport] = []

    critical = [r for r in results if risk_at_least(r.risk_level, RISK_CRITICAL)]
    critical_agents = {r.agent for r in critical}
    # 必须是**不同专家之间**的分歧才算冲突：
    # 同一个专家的答案里既报风险又引了手册的许可性原文，那是"自问自答"，不是冲突。
    permissive = [r for r in results
                  if PERMISSIVE_PATTERNS.search(r.answer or "") and r.agent not in critical_agents]
    if critical and permissive:
        conflicts.append(ConflictReport(
            kind="safety_override", topic="当前是否可以继续行驶",
            participants=[critical[0].agent, permissive[0].agent],
            values={critical[0].agent: f"{critical[0].risk_level}：" + "；".join(critical[0].findings[:2]),
                    permissive[0].agent: PERMISSIVE_PATTERNS.search(permissive[0].answer).group(0)},
            winner=critical[0].agent,
            resolution="安全优先：实时车况覆盖手册通用说明，输出停驶与联系中心的安全指令",
            escalated=False))

    # 可比较事实白名单：只有这些上下文里的数值才参与冲突判定，避免误报
    COMPARABLE = {"续航": ("km",), "胎压": ("kpa",)}
    facts: Dict[str, Dict[str, Any]] = {}
    for r in results:
        for context, units in COMPARABLE.items():
            if context not in (r.answer or ""):
                continue
            numbers = extract_numbers(r.answer)
            for unit in units:
                if unit in numbers:
                    key = f"{context}/{unit}"
                    prev = facts.get(key)
                    if prev and abs(prev["value"] - numbers[unit]) > 1e-6:
                        winner = prev["agent"] if AGENT_PRIORITY.get(prev["agent"], 0) >= \
                            AGENT_PRIORITY.get(r.agent, 0) else r.agent
                        conflicts.append(ConflictReport(
                            kind="numeric_mismatch", topic=f"{context}（{unit}）",
                            participants=[prev["agent"], r.agent],
                            values={prev["agent"]: str(prev["value"]), r.agent: str(numbers[unit])},
                            winner=winner,
                            resolution=f"取优先级更高的来源（{winner}），并标记需人工复核",
                            escalated=True))
                    else:
                        facts.setdefault(key, {"agent": r.agent, "value": numbers[unit]})
    return conflicts
