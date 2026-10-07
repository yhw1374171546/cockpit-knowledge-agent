# -*- coding: utf-8 -*-
"""预算与熔断：token / 步数 / 工具调用 / 金额四维预算，支持向子 Agent 传播。

为什么需要：Agent 的循环 + 工具调用会让成本不可预测（denial of wallet）。
多 Agent 场景下更危险——一个子 Agent 可能吃掉全部预算，导致其它子任务饿死。
因此预算是**可传播、可回收**的：Supervisor 分配子预算，子 Agent 用完即止。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


class BudgetExceeded(Exception):
    def __init__(self, reason: str, scope: str = ""):
        super().__init__(f"预算耗尽[{scope}]：{reason}")
        self.reason = reason
        self.scope = scope


@dataclass
class CostModel:
    """按 1K token 计价（美元），默认取常见开源模型自建推理的近似值。"""
    price_per_1k_input: float = 0.0002
    price_per_1k_output: float = 0.0004

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in / 1000.0) * self.price_per_1k_input + \
               (tokens_out / 1000.0) * self.price_per_1k_output


@dataclass
class Budget:
    max_steps: int = 8
    max_tool_calls: int = 8
    max_tokens: int = 12000
    max_cost_usd: float = 0.02
    label: str = "root"

    def slice(self, label: str, step_ratio: float = 0.5, tool_ratio: float = 0.5,
              token_ratio: float = 0.5, cost_ratio: float = 0.5) -> "Budget":
        """按比例切出一份子预算（Supervisor → 子 Agent）。"""
        return Budget(
            max_steps=max(1, int(self.max_steps * step_ratio)),
            max_tool_calls=max(1, int(self.max_tool_calls * tool_ratio)),
            max_tokens=max(64, int(self.max_tokens * token_ratio)),
            max_cost_usd=self.max_cost_usd * cost_ratio,
            label=label,
        )


@dataclass
class BudgetTracker:
    budget: Budget
    steps: int = 0
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    tool_calls_by_name: Dict[str, int] = field(default_factory=dict)
    exhausted_reason: Optional[str] = None
    stop_events: List[str] = field(default_factory=list)
    cost_model: CostModel = field(default_factory=CostModel)

    # ── 记账 ──
    def step(self) -> None:
        self.steps += 1
        if self.steps > self.budget.max_steps:
            self._stop(f"步数超限 {self.budget.max_steps}")

    def tool_call(self, name: str) -> None:
        self.tool_calls += 1
        self.tool_calls_by_name[name] = self.tool_calls_by_name.get(name, 0) + 1
        if self.tool_calls > self.budget.max_tool_calls:
            self._stop(f"工具调用数超限 {self.budget.max_tool_calls}")

    def charge_tokens(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        self.tokens_in += max(0, tokens_in)
        self.tokens_out += max(0, tokens_out)
        self.cost_usd = self.cost_model.cost(self.tokens_in, self.tokens_out)
        if self.tokens_in + self.tokens_out > self.budget.max_tokens:
            self._stop(f"token 超限 {self.budget.max_tokens}")
        if self.cost_usd > self.budget.max_cost_usd:
            self._stop(f"成本超限 ${self.budget.max_cost_usd}")

    def _stop(self, reason: str) -> None:
        # 只保留**首次**触发原因（后续超限都是它的连锁反应），并去重事件，便于归因
        if self.exhausted_reason is None:
            self.exhausted_reason = reason
        if reason not in self.stop_events:
            self.stop_events.append(reason)

    @property
    def exhausted(self) -> bool:
        return self.exhausted_reason is not None

    def ensure(self) -> None:
        """熔断检查：预算耗尽即抛异常，由上层转换为"预算耗尽"话术。"""
        if self.exhausted:
            raise BudgetExceeded(self.exhausted_reason or "unknown", self.budget.label)

    # ── 报表 ──
    def to_dict(self) -> Dict:
        return {"label": self.budget.label, "steps": self.steps,
                "tool_calls": self.tool_calls, "tool_calls_by_name": self.tool_calls_by_name,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "tokens_total": self.tokens_in + self.tokens_out,
                "cost_usd": round(self.cost_usd, 6),
                "limits": {"steps": self.budget.max_steps, "tool_calls": self.budget.max_tool_calls,
                           "tokens": self.budget.max_tokens,
                           "cost_usd": self.budget.max_cost_usd},
                "exhausted": self.exhausted, "exhausted_reason": self.exhausted_reason,
                "stop_events": self.stop_events}

    def merge(self, other: "BudgetTracker") -> None:
        """把子 Agent 的消耗合并回父预算（预算回收）。"""
        self.steps += other.steps
        self.tool_calls += other.tool_calls
        for k, v in other.tool_calls_by_name.items():
            self.tool_calls_by_name[k] = self.tool_calls_by_name.get(k, 0) + v
        self.charge_tokens(other.tokens_in, other.tokens_out)
