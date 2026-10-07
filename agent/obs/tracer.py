# -*- coding: utf-8 -*-
"""可观测层：trace/span、token 成本核算、失败归因。

目标：Agent 的每一步都可解释、可计费、可归因。座舱线上问题定位靠的就是这套东西：
"这次为什么答错了 / 为什么慢了 / 为什么贵了"，必须能从 trace 里读出来。
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from agent.guardrails.budget import CostModel

# 失败归因分类（写进 trace，用于自动统计与回归对比）
FAILURE_CATEGORIES = (
    "ok",
    "retrieval_miss",      # 检索没召回正确内容
    "under_call",          # 该调工具却没调（直接编）
    "over_call",           # 不该调工具却调了
    "wrong_tool",          # 调了错误的工具
    "bad_args",            # 参数错误/解析失败
    "hallucination",       # 答案无证据支撑
    "false_refusal",       # 可答题被拒答
    "policy_block",        # 被护栏拦截
    "budget_exhausted",    # 预算耗尽
    "tool_error",          # 工具执行报错
    "injection_flagged",   # 检测到注入风险
)


@dataclass
class Span:
    id: str
    name: str
    agent: str = "agent"
    parent_id: Optional[str] = None
    start_ts: float = field(default_factory=time.time)
    end_ts: Optional[float] = None
    duration_ms: float = 0.0
    status: str = "ok"
    attrs: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "agent": self.agent,
                "parent_id": self.parent_id, "duration_ms": round(self.duration_ms, 3),
                "status": self.status, "attrs": self.attrs, "events": self.events}


class Tracer:
    """轻量 trace 收集器：线程安全（多 Agent 并行时会并发写入）。"""

    def __init__(self, cost_model: Optional[CostModel] = None, enabled: bool = True):
        self.cost_model = cost_model or CostModel()
        self.enabled = enabled
        self.spans: List[Span] = []
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost_usd = 0.0
        self.ttft_ms: Optional[float] = None
        self.failure: str = "ok"
        self.notes: List[str] = []
        self._lock = threading.Lock()
        self._stack: List[str] = []

    # ── span ──
    @contextmanager
    def span(self, name: str, agent: str = "agent", **attrs: Any) -> Iterator[Span]:
        if not self.enabled:
            yield Span(id="-", name=name, agent=agent)
            return
        sp = Span(id=uuid.uuid4().hex[:8], name=name, agent=agent,
                  parent_id=self._stack[-1] if self._stack else None, attrs=dict(attrs))
        with self._lock:
            self.spans.append(sp)
            self._stack.append(sp.id)
        start = time.perf_counter()
        try:
            yield sp
        except Exception as exc:                      # noqa: BLE001
            sp.status = "error"
            sp.attrs["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            sp.duration_ms = (time.perf_counter() - start) * 1000
            sp.end_ts = time.time()
            with self._lock:
                if self._stack and self._stack[-1] == sp.id:
                    self._stack.pop()

    # ── 记录 ──
    def record_llm(self, tokens_in: int = 0, tokens_out: int = 0, model: str = "",
                   agent: str = "", latency_ms: float = 0.0) -> Dict[str, Any]:
        cost = self.cost_model.cost(tokens_in, tokens_out)
        with self._lock:
            self.tokens_in += tokens_in
            self.tokens_out += tokens_out
            self.cost_usd += cost
            self.spans.append(Span(id=uuid.uuid4().hex[:8], name="llm.call", agent=agent or "agent",
                                   duration_ms=latency_ms, attrs={
                                       "model": model, "tokens_in": tokens_in,
                                       "tokens_out": tokens_out, "cost_usd": round(cost, 6)}))
        return {"tokens_in": tokens_in, "tokens_out": tokens_out, "cost_usd": cost}

    def record_tool(self, name: str, latency_ms: float, ok: bool = True,
                    repeated: bool = False, blocked: bool = False,
                    agent: str = "", error: str = "") -> None:
        if not self.enabled:
            return
        status = "blocked" if blocked else ("ok" if ok else "error")
        with self._lock:
            self.spans.append(Span(id=uuid.uuid4().hex[:8], name=f"tool.{name}", agent=agent or "agent",
                                   duration_ms=latency_ms, status=status,
                                   attrs={"repeated": repeated, "error": error}))

    def record_guard(self, action: str, reason: str = "", agent: str = "") -> None:
        if not self.enabled:
            return
        with self._lock:
            self.spans.append(Span(id=uuid.uuid4().hex[:8], name=f"guard.{action}",
                                   agent=agent or "guard", status="blocked" if action == "block" else "ok",
                                   attrs={"reason": reason}))

    def mark_first_token(self, elapsed_ms: float) -> None:
        with self._lock:
            if self.ttft_ms is None or elapsed_ms < self.ttft_ms:
                self.ttft_ms = elapsed_ms

    def note(self, text: str) -> None:
        with self._lock:
            self.notes.append(text)

    # ── 汇总 ──
    def totals(self) -> Dict[str, Any]:
        blocked = [s for s in self.spans if s.status == "blocked"]
        tools = [s for s in self.spans if s.name.startswith("tool.")]
        return {
            "n_spans": len(self.spans),
            "duration_ms": round(sum(s.duration_ms for s in self.spans
                                     if s.name in ("agent.run", "supervisor.run")), 3),
            "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
            "tokens_total": self.tokens_in + self.tokens_out,
            "cost_usd": round(self.cost_usd, 6),
            "tool_calls": len(tools),
            "blocked_spans": len(blocked),
            "ttft_ms": round(self.ttft_ms, 3) if self.ttft_ms is not None else None,
            "failure": self.failure,
        }

    def stage_breakdown(self) -> Dict[str, float]:
        """按阶段聚合耗时（ms），用于定位"时间花在哪"。"""
        out: Dict[str, float] = {}
        for s in self.spans:
            key = s.name.split(".")[0]
            out[key] = round(out.get(key, 0.0) + s.duration_ms, 3)
        return out

    def waterfall(self, limit: int = 40) -> str:
        rows = ["{:<28} {:<10} {:>10}  {}".format("span", "agent", "ms", "status")]
        for s in self.spans[:limit]:
            rows.append("{:<28} {:<10} {:>10.2f}  {}".format(s.name, s.agent, s.duration_ms, s.status))
        t = self.totals()
        rows.append("-" * 62)
        rows.append(f"tokens={t['tokens_total']} (in {t['tokens_in']} / out {t['tokens_out']})  "
                    f"cost=${t['cost_usd']}  ttft={t['ttft_ms']}ms  failure={t['failure']}")
        return "\n".join(rows)

    def to_jsonl(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for s in self.spans:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        return path

    # ── 失败归因 ──
    def classify_failure(self, signals: Optional[Dict[str, Any]] = None) -> str:
        """按优先级归因：护栏 > 预算 > 工具 > 检索 > 生成 > 拒答。"""
        sig = signals or {}
        if sig.get("injection_flagged"):
            self.failure = "injection_flagged"
        elif sig.get("policy_blocked"):
            self.failure = "policy_block"
        elif sig.get("budget_exhausted"):
            self.failure = "budget_exhausted"
        elif sig.get("tool_error"):
            self.failure = "tool_error"
        elif sig.get("parse_failed"):
            self.failure = "bad_args"
        elif sig.get("expected_tools") and not sig.get("called_tools"):
            self.failure = "under_call"
        elif sig.get("expected_tools") and sig.get("called_tools"):
            exp, got = set(sig["expected_tools"]), set(sig["called_tools"])
            if not exp & got:
                self.failure = "wrong_tool"
            elif sig.get("gold_answerable") is False and got:
                self.failure = "over_call"
        if self.failure == "ok" and sig.get("grounded_ratio") is not None \
                and sig["grounded_ratio"] < 0.34:
            self.failure = "hallucination"
        if self.failure == "ok" and sig.get("refused") and sig.get("gold_answerable"):
            self.failure = "false_refusal"
        if self.failure == "ok" and sig.get("evidence_recall") is not None \
                and sig["evidence_recall"] == 0.0:
            self.failure = "retrieval_miss"
        return self.failure


def new_tracer(enabled: bool = True, **kwargs: Any) -> Tracer:
    return Tracer(enabled=enabled, **kwargs)
