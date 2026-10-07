# -*- coding: utf-8 -*-
"""多 Agent 协作协议：结构化 handoff、子任务、结果、冲突报告、仲裁规则。

设计原则：**禁止自由文本传话**。
Agent 之间如果靠自然语言互相对话，就会出现三个问题：不可观测、不可测、预算无法归属。
因此这里把所有交互都定义成结构化对象：
    SubTask  →  Handoff（带预算与约束）  →  AgentResult（带证据、风险、置信度）  →  ConflictReport
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── 风险等级（仲裁的核心依据）────────────────────────────────────────
RISK_OK = "ok"
RISK_NOTICE = "notice"
RISK_WARNING = "warning"
RISK_CRITICAL = "critical"
RISK_ORDER = {RISK_OK: 0, RISK_NOTICE: 1, RISK_WARNING: 2, RISK_CRITICAL: 3}

# ── 仲裁优先级：安全 > 实时车况 > 服务政策 > 手册通用说明 ─────────────
AGENT_PRIORITY: Dict[str, int] = {
    "safety_critic": 100,
    "vehicle_control": 80,
    "service_advisor": 70,
    "manual_expert": 60,
    "chitchat": 10,
}

# 安全处置话术（风险为 critical 时强制出现在答案里）
SAFETY_DIRECTIVES = (
    "请立即在安全位置靠边停车，不要继续行驶",
    "请联系 Lynk&Co 领克中心或道路救援",
)

# 与"安全优先"冲突的通用建议（手册里常见的可继续行驶表述）
PERMISSIVE_PATTERNS = re.compile(
    r"(可继续|可以继续|正常行驶|继续低速行驶|无大碍|不影响使用|放心使用|低速行驶|"
    r"行驶几分钟|以\s*\d+\s*(?:km/h|公里/小时|码)[^。]{0,10}?行驶)")

NUM_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(km|公里|kPa|千帕|kWh|度电|%)", re.I)


@dataclass
class SubTask:
    """Supervisor 拆解出的一个子任务。"""
    id: str
    agent: str
    kind: str
    query: str
    goal: str = ""
    constraints: List[str] = field(default_factory=list)
    priority: int = 50
    depends_on: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "agent": self.agent, "kind": self.kind, "query": self.query,
                "goal": self.goal, "constraints": self.constraints, "priority": self.priority,
                "depends_on": self.depends_on}


@dataclass
class Plan:
    route: str                       # single | multi
    subtasks: List[SubTask] = field(default_factory=list)
    reason: str = ""

    @property
    def is_multi(self) -> bool:
        return self.route == "multi" and len(self.subtasks) > 1

    def to_dict(self) -> Dict[str, Any]:
        return {"route": self.route, "reason": self.reason,
                "subtasks": [s.to_dict() for s in self.subtasks]}


@dataclass
class Handoff:
    """Supervisor → 子 Agent 的结构化委派（含预算与上下文摘要）。"""
    from_agent: str
    to_agent: str
    task: SubTask
    context_digest: str = ""
    injected_constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"from": self.from_agent, "to": self.to_agent, "task": self.task.to_dict(),
                "context_digest": self.context_digest[:120],
                "constraints": self.injected_constraints}


@dataclass
class AgentResult:
    """子 Agent 的结构化产出。"""
    agent: str
    subtask_id: str
    kind: str = "manual_qa"
    answer: str = ""
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    status: str = "answered"            # answered | refused | needs_confirmation | blocked | budget_exhausted
    risk_level: str = RISK_OK
    findings: List[str] = field(default_factory=list)
    confidence: float = 0.0
    priority: int = 50
    policy_flags: List[str] = field(default_factory=list)
    needs_confirmation: bool = False
    injection_flagged: bool = False
    injection_risk: int = 0
    latency_ms: float = 0.0
    tokens_total: int = 0
    steps: int = 0

    @property
    def answered(self) -> bool:
        return self.status == "answered" and bool(self.answer.strip())

    def to_dict(self) -> Dict[str, Any]:
        return {"agent": self.agent, "subtask_id": self.subtask_id, "kind": self.kind,
                "answer": self.answer, "citations": self.citations[:3],
                "status": self.status, "risk_level": self.risk_level, "findings": self.findings,
                "confidence": round(self.confidence, 3), "priority": self.priority,
                "policy_flags": self.policy_flags, "needs_confirmation": self.needs_confirmation,
                "injection_flagged": self.injection_flagged,
                "injection_risk": self.injection_risk,
                "latency_ms": round(self.latency_ms, 2), "tokens_total": self.tokens_total,
                "steps": self.steps, "n_evidence": len(self.evidence)}


@dataclass
class ConflictReport:
    """两个子 Agent 的结论冲突及仲裁结果。"""
    kind: str                     # safety_override | numeric_mismatch | tie_escalation
    topic: str
    participants: List[str] = field(default_factory=list)
    values: Dict[str, str] = field(default_factory=dict)
    winner: Optional[str] = None
    resolution: str = ""
    escalated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "topic": self.topic, "participants": self.participants,
                "values": self.values, "winner": self.winner, "resolution": self.resolution,
                "escalated": self.escalated}


@dataclass
class CriticVerdict:
    """安全评审结论。"""
    verdict: str                  # approve | revise | block
    reasons: List[str] = field(default_factory=list)
    revised_answer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "reasons": self.reasons,
                "revised": (self.revised_answer or "")[:120]}


@dataclass
class SupervisorState:
    question: str
    plan: Optional[Plan] = None
    results: List[AgentResult] = field(default_factory=list)
    conflicts: List[ConflictReport] = field(default_factory=list)
    verdict: Optional[CriticVerdict] = None
    answer: str = ""
    citations: List[str] = field(default_factory=list)
    status: str = "answered"
    handoffs: List[Handoff] = field(default_factory=list)
    latency_ms: Dict[str, float] = field(default_factory=dict)
    tokens_total: int = 0
    injection_risk: int = 0
    failure: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {"question": self.question,
                "plan": self.plan.to_dict() if self.plan else None,
                "results": [r.to_dict() for r in self.results],
                "conflicts": [c.to_dict() for c in self.conflicts],
                "verdict": self.verdict.to_dict() if self.verdict else None,
                "answer": self.answer, "citations": self.citations[:6], "status": self.status,
                "handoffs": [h.to_dict() for h in self.handoffs],
                "latency_ms": self.latency_ms, "tokens_total": self.tokens_total,
                "injection_risk": self.injection_risk, "failure": self.failure}


def risk_at_least(level: str, threshold: str = RISK_CRITICAL) -> bool:
    return RISK_ORDER.get(level, 0) >= RISK_ORDER.get(threshold, 3)


def extract_numbers(text: str) -> Dict[str, float]:
    """抽取"数值+单位"，用于发现两个 Agent 对同一事实给出的数值不一致。"""
    out: Dict[str, float] = {}
    for value, unit in NUM_UNIT_RE.findall(text or ""):
        key = unit.lower().replace("千帕", "kpa").replace("公里", "km").replace("度电", "kwh")
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out
