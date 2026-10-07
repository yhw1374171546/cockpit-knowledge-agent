# -*- coding: utf-8 -*-
"""工具执行器：策略裁决 → 幂等去重 → 并行执行 → 观察回填 → 记账。

把「执行工具」这一步从 Agent 主循环里抽出来，是因为它承担了四件与决策无关的工程职责：
1. **策略裁决**：模型可以"请求"任何工具，但能不能执行由 PolicyEngine 说了算；
2. **幂等性**：写操作（预约到店）重试不能重复下单——按 idempotency key 复用首次结果；
3. **并行执行**：一次响应里的多个 tool_calls 并发执行（真实场景工具是网络 I/O）；
4. **可观测**：每次调用都进 trace 与预算账本，失败可归因。

并行收益说明：离线 BM25 是 CPU 密集型（受 GIL 限制），并行收益有限；
座舱线上工具是**网络 I/O**（车况接口、门店查询、车控 API），并行才有意义。
因此评测里用 `mock_io_latency_ms` 显式注入模拟网络时延来说明这一点（报告中已标注为模拟）。
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from agent.guardrails.budget import BudgetTracker
from agent.guardrails.policy import PolicyContext, PolicyEngine
from agent.llm import ToolCall
from agent.obs.tracer import Tracer
from agent.tools import ToolRegistry, ToolResult

WRITE_TOOLS = {"create_service_order"}


@dataclass
class ExecutedCall:
    call: ToolCall
    result: ToolResult
    blocked: bool = False
    reason: str = ""
    idempotent_reuse: bool = False
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.call.name, "arguments": self.call.arguments,
                "ok": self.result.ok, "blocked": self.blocked, "reason": self.reason,
                "repeated": self.result.repeated, "idempotent_reuse": self.idempotent_reuse,
                "duration_ms": round(self.duration_ms, 3),
                "repaired": self.call.repaired}


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, policy: Optional[PolicyEngine] = None,
                 tracer: Optional[Tracer] = None, parallel: bool = True,
                 max_workers: int = 4, mock_io_latency_ms: float = 0.0,
                 agent: str = "agent"):
        self.registry = registry
        self.policy = policy or PolicyEngine()
        self.tracer = tracer or Tracer(enabled=False)
        self.parallel = parallel
        self.max_workers = max_workers
        self.mock_io_latency_ms = mock_io_latency_ms
        self.agent = agent
        self._idempotency: Dict[str, ToolResult] = {}
        self.history: List[ExecutedCall] = []
        self.wall_clock_ms = 0.0
        self.serial_estimate_ms = 0.0

    # ── 幂等键 ──
    @staticmethod
    def idempotency_key(call: ToolCall) -> str:
        payload = json.dumps(call.arguments or {}, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(f"{call.name}:{payload}".encode("utf-8")).hexdigest()[:16]

    # ── 执行入口 ──
    def execute(self, calls: Sequence[ToolCall], context: Optional[PolicyContext] = None,
                budget: Optional[BudgetTracker] = None) -> List[ExecutedCall]:
        if not calls:
            return []
        prepared: List[ExecutedCall] = []
        runnable: List[tuple] = []      # (index, call, key)

        for idx, call in enumerate(calls):
            call.idempotency_key = call.idempotency_key or self.idempotency_key(call)

            # ① 幂等优先于配额：同一个写请求的**重试**不是新调用，
            #    既不重复下单，也不该被"单轮调用上限"拦掉。顺序反了会变成"重试即报错"。
            if call.name in WRITE_TOOLS and call.idempotency_key in self._idempotency:
                res = self._idempotency[call.idempotency_key]
                prepared.append(ExecutedCall(call, res, idempotent_reuse=True,
                                             reason="命中幂等键，复用首次执行结果"))
                continue

            # ② 策略裁决：模型可以"请求"工具，能不能执行由策略引擎决定
            ctx = context or PolicyContext(agent=self.agent)
            decision = self.policy.check(call.name, call.arguments or {}, ctx,
                                         confirmed=self._is_confirmed(call))
            if decision.blocked or decision.needs_confirmation:
                reason = decision.reason
                res = ToolResult(False, error=reason,
                                 hint=("请向车主确认后再执行" if decision.needs_confirmation
                                       else "该工具调用被安全策略拦截，请改用其它方式或拒答"),
                                 meta={"needs_confirmation": decision.needs_confirmation,
                                       "policy_blocked": decision.blocked})
                self.tracer.record_guard("confirm" if decision.needs_confirmation else "block",
                                         reason, agent=self.agent)
                prepared.append(ExecutedCall(call, res, blocked=True, reason=reason))
                continue
            prepared.append(ExecutedCall(call, ToolResult(False, error="pending")))
            runnable.append((idx, call, call.idempotency_key))

        if runnable:
            t0 = time.perf_counter()
            if self.parallel and len(runnable) > 1:
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(runnable))) as pool:
                    futures = {pool.submit(self._run_one, call, key, budget): idx
                               for idx, call, key in runnable}
                    for fut in futures:
                        idx = futures[fut]
                        prepared[idx] = fut.result()
            else:
                for idx, call, key in runnable:
                    prepared[idx] = self._run_one(call, key, budget)
            self.wall_clock_ms = (time.perf_counter() - t0) * 1000
            self.serial_estimate_ms = sum(p.duration_ms for p in prepared)
            for p in prepared:
                if not p.blocked:
                    self.policy.note_executed(p.call.name)
        else:
            self.wall_clock_ms = self.serial_estimate_ms = 0.0

        self.history.extend(prepared)
        return prepared

    def _run_one(self, call: ToolCall, key: str, budget: Optional[BudgetTracker]) -> ExecutedCall:
        start = time.perf_counter()
        if self.mock_io_latency_ms > 0:
            time.sleep(self.mock_io_latency_ms / 1000.0)     # 模拟真实网络 I/O
        if budget is not None:
            budget.tool_call(call.name)
        result = self.registry.call(call.name, call.arguments or {})
        duration = (time.perf_counter() - start) * 1000
        call.duration_ms = duration
        if call.name in WRITE_TOOLS and result.ok:
            self._idempotency[key] = result
        self.tracer.record_tool(call.name, duration, ok=result.ok, repeated=result.repeated,
                                blocked=False, agent=self.agent,
                                error=result.error or "")
        return ExecutedCall(call, result, duration_ms=duration)

    def _is_confirmed(self, call: ToolCall) -> bool:
        key = f"{call.name}:{call.arguments.get('item')}:{call.arguments.get('preferred_date')}"
        return key in getattr(self.registry, "confirmed_actions", set())

    # ── 报表 ──
    def stats(self) -> Dict[str, Any]:
        return {
            "calls": len(self.history),
            "blocked": sum(1 for c in self.history if c.blocked),
            "idempotent_reuse": sum(1 for c in self.history if c.idempotent_reuse),
            "errors": sum(1 for c in self.history if not c.result.ok and not c.blocked),
            "serial_estimate_ms": round(self.serial_estimate_ms, 3),
            "wall_clock_ms": round(self.wall_clock_ms, 3),
            "parallel_speedup": (round(self.serial_estimate_ms / self.wall_clock_ms, 2)
                                 if self.wall_clock_ms > 0 else None),
            "policy": self.policy.audit_summary(),
        }
