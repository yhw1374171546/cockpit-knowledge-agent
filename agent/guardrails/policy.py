# -*- coding: utf-8 -*-
"""工具策略引擎：权限、写操作确认、参数校验、敏感信息脱敏、审计。

设计目标：把「工具该不该被调用」从模型的自由裁量，变成**可审计的策略判定**。
模型可以请求任何工具，但最终执行由策略引擎裁决。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

# ── 敏感信息脱敏 ─────────────────────────────────────────────────────
SENSITIVE_PATTERNS: List[tuple] = [
    ("vin", re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")),
    ("phone", re.compile(r"\b1[3-9]\d{9}\b")),
    ("id_card", re.compile(r"\b\d{17}[\dXx]\b")),
]


def mask_sensitive(text: str) -> str:
    """对 VIN / 手机号 / 身份证做脱敏（审计与对外输出前调用）。"""
    text = text or ""
    for name, pattern in SENSITIVE_PATTERNS:
        text = pattern.sub(lambda m: m.group(0)[:3] + "*" * (len(m.group(0)) - 6)
                           + m.group(0)[-3:], text)
    return text


@dataclass
class ToolPolicy:
    name: str
    allow: bool = True
    requires_confirmation: bool = False      # 写操作：必须车主确认
    max_calls_per_turn: int = 3              # 单轮最多调用次数（防循环/钱包攻击）
    arg_validators: Dict[str, Callable[[Any], bool]] = field(default_factory=dict)
    require_injection_clear: bool = False    # 注入风险高时是否禁用该工具
    roles: Sequence[str] = ("driver", "guest")

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        for key, check in self.arg_validators.items():
            if key in args and not check(args[key]):
                return f"参数 {key} 不合法"
        return None


@dataclass
class PolicyContext:
    user_role: str = "driver"
    injection_risk: int = 0
    injection_suspicious: bool = False
    agent: str = "agent"
    turn: int = 0

    def to_dict(self) -> Dict:
        return {"user_role": self.user_role, "injection_risk": self.injection_risk,
                "injection_suspicious": self.injection_suspicious,
                "agent": self.agent, "turn": self.turn}


@dataclass
class Decision:
    allow: bool
    needs_confirmation: bool = False
    reason: str = ""
    masked_args: Dict[str, Any] = field(default_factory=dict)
    audit: List[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return not self.allow

    def to_dict(self) -> Dict:
        return {"allow": self.allow, "needs_confirmation": self.needs_confirmation,
                "reason": self.reason, "masked_args": self.masked_args, "audit": self.audit}


DEFAULT_POLICIES: Dict[str, ToolPolicy] = {
    "search_manual": ToolPolicy("search_manual", max_calls_per_turn=4),
    "get_vehicle_status": ToolPolicy("get_vehicle_status", max_calls_per_turn=2),
    "lookup_vehicle_spec": ToolPolicy("lookup_vehicle_spec", max_calls_per_turn=3),
    "get_maintenance_plan": ToolPolicy("get_maintenance_plan", max_calls_per_turn=2),
    # 写操作：必须确认，且一旦检测到注入风险直接禁用
    "create_service_order": ToolPolicy("create_service_order", requires_confirmation=True,
                                       max_calls_per_turn=1, require_injection_clear=True),
}


class PolicyEngine:
    """工具策略裁决 + 审计日志。"""

    def __init__(self, policies: Optional[Dict[str, ToolPolicy]] = None,
                 unknown_tool_deny: bool = False):
        self.policies = dict(policies or DEFAULT_POLICIES)
        self.unknown_tool_deny = unknown_tool_deny
        self.audit_log: List[Dict] = []
        self._turn_calls: Dict[str, int] = {}

    # ── 注册 ──
    def register(self, policy: ToolPolicy) -> None:
        self.policies[policy.name] = policy

    def new_turn(self) -> None:
        self._turn_calls.clear()

    # ── 裁决 ──
    def check(self, tool: str, args: Dict[str, Any], context: Optional[PolicyContext] = None,
              confirmed: bool = False) -> Decision:
        ctx = context or PolicyContext()
        policy = self.policies.get(tool)
        if policy is None:
            if self.unknown_tool_deny:
                return self._log(Decision(False, reason=f"未注册的工具 {tool} 被策略拒绝（默认拒绝）"), tool, args, ctx)
            return self._log(Decision(True, reason="未注册工具，按默认放行"), tool, args, ctx)

        if not policy.allow:
            return self._log(Decision(False, reason=f"工具 {tool} 已被策略禁用"), tool, args, ctx)

        if ctx.user_role not in policy.roles:
            return self._log(Decision(False, reason=f"角色 {ctx.user_role} 无权调用 {tool}"), tool, args, ctx)

        if policy.require_injection_clear and ctx.injection_suspicious:
            return self._log(Decision(False, reason=f"检测到疑似提示注入（风险分 {ctx.injection_risk}），"
                                                    f"已禁用高风险工具 {tool}"), tool, args, ctx)

        bad = policy.validate_args(args)
        if bad:
            return self._log(Decision(False, reason=bad), tool, args, ctx)

        used = self._turn_calls.get(tool, 0)
        if used >= policy.max_calls_per_turn:
            return self._log(Decision(False, reason=f"{tool} 单轮调用已达上限 {policy.max_calls_per_turn}，"
                                                    f"疑似循环调用"), tool, args, ctx)

        masked = {k: (mask_sensitive(str(v)) if isinstance(v, str) else v) for k, v in (args or {}).items()}
        if policy.requires_confirmation and not confirmed:
            return self._log(Decision(False, needs_confirmation=True,
                                      reason=f"写操作 {tool} 需要车主确认", masked_args=masked),
                             tool, args, ctx)

        return self._log(Decision(True, masked_args=masked), tool, args, ctx)

    def note_executed(self, tool: str) -> None:
        self._turn_calls[tool] = self._turn_calls.get(tool, 0) + 1

    # ── 审计 ──
    def _log(self, decision: Decision, tool: str, args: Dict[str, Any],
             ctx: PolicyContext) -> Decision:
        entry = {"tool": tool, "args": decision.masked_args or args, "allow": decision.allow,
                 "needs_confirmation": decision.needs_confirmation, "reason": decision.reason,
                 "context": ctx.to_dict()}
        self.audit_log.append(entry)
        if decision.reason:
            decision.audit.append(decision.reason)
        return decision

    def audit_summary(self) -> Dict[str, int]:
        return {"total": len(self.audit_log),
                "allowed": sum(1 for e in self.audit_log if e["allow"]),
                "blocked": sum(1 for e in self.audit_log if not e["allow"]),
                "needs_confirmation": sum(1 for e in self.audit_log if e["needs_confirmation"])}
