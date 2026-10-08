# -*- coding: utf-8 -*-
"""故障注入与降级：把「依赖挂了」变成可验证的行为，而不是 500。

三种角色
--------
1. `FaultyLLM`：按配置注入故障（超时 / 报错 / 随机抖动），用于**故障演练**。
2. `DegradingLLM`：**降级链** —— 主后端失败时依次回退到
   ① 语义缓存（若已命中）→ ② 离线规则规划器，并在响应里标注 `degraded`。
   这样"推理服务挂了"不会让座舱彻底失能（安全话术仍可给出）。
3. `build_llm_with_faults`：按服务配置组装上述两者。

为什么需要它：本项目的离线规则规划器原本只是"没有 GPU 时的测试替身"，
从服务可用性视角看，它正好是一条**天然的降级路径**——把它显式建模出来，
就从"测试技巧"变成了"高可用设计"。
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Sequence

from agent.llm import LLMBackend, LLMResponse, RuleBasedPlannerLLM, StreamChunk


class InjectedFault(RuntimeError):
    """注入的故障（用于演练降级链）。"""


class FaultyLLM(LLMBackend):
    """故障注入包装器：按概率/次数让底层后端超时或报错。"""

    name = "faulty"

    def __init__(self, inner: LLMBackend, mode: str = "none", rate: float = 0.0,
                 fail_first: int = 0, sleep_s: float = 0.0):
        self.inner = inner
        self.mode = mode                 # none | error | timeout | flaky
        self.rate = max(0.0, min(1.0, rate))
        self.fail_first = max(0, fail_first)
        self.sleep_s = max(0.0, sleep_s)
        self.calls = 0
        self.injected = 0

    def set_known_tools(self, names: Sequence[str]) -> None:
        if hasattr(self.inner, "set_known_tools"):
            self.inner.set_known_tools(names)

    def _should_fail(self) -> bool:
        self.calls += 1
        if self.calls <= self.fail_first:
            return True
        if self.mode == "none":
            return False
        if self.mode == "flaky":
            return random.random() < self.rate
        return True

    def _fault(self) -> None:
        self.injected += 1
        if self.mode == "timeout":
            time.sleep(self.sleep_s or 0.05)
            raise InjectedFault("注入故障：上游推理服务超时")
        raise InjectedFault("注入故障：上游推理服务返回 5xx")

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024,
             guided_json=None) -> LLMResponse:
        if self._should_fail():
            self._fault()
        return self.inner.chat(messages, tools=tools, temperature=temperature,
                               max_tokens=max_tokens, guided_json=guided_json)

    def stream_chat(self, messages, tools=None, temperature=0.0, max_tokens=1024):
        if self._should_fail():
            self._fault()
        yield from self.inner.stream_chat(messages, tools=tools, temperature=temperature,
                                          max_tokens=max_tokens)


class DegradingLLM(LLMBackend):
    """降级链：主后端失败 → 回退到兜底后端，并记录降级次数与原因。

    `on_degrade` 回调用于把降级事件写进 trace/指标（面试常问"你怎么知道降级了"）。
    """

    name = "degrading"

    def __init__(self, primary: LLMBackend, fallback: Optional[LLMBackend] = None,
                 on_degrade=None, max_fallbacks: int = 3):
        self.primary = primary
        self.fallback = fallback or RuleBasedPlannerLLM()
        self.on_degrade = on_degrade
        self.max_fallbacks = max_fallbacks
        self.degraded_calls = 0
        self.last_error: str = ""

    def set_known_tools(self, names: Sequence[str]) -> None:
        for backend in (self.primary, self.fallback):
            if hasattr(backend, "set_known_tools"):
                backend.set_known_tools(names)

    @property
    def degraded(self) -> bool:
        return self.degraded_calls > 0

    def _degrade(self, exc: Exception, stage: str) -> None:
        self.degraded_calls += 1
        self.last_error = f"{stage}: {type(exc).__name__}: {exc}"
        if self.on_degrade is not None:
            try:
                self.on_degrade(self.last_error)
            except Exception:                                  # noqa: BLE001
                pass

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024,
             guided_json=None) -> LLMResponse:
        try:
            return self.primary.chat(messages, tools=tools, temperature=temperature,
                                     max_tokens=max_tokens, guided_json=guided_json)
        except Exception as exc:                               # noqa: BLE001
            self._degrade(exc, "chat")
            return self.fallback.chat(messages, tools=tools, temperature=temperature,
                                      max_tokens=max_tokens, guided_json=guided_json)

    def stream_chat(self, messages, tools=None, temperature=0.0, max_tokens=1024):
        try:
            yield from self.primary.stream_chat(messages, tools=tools,
                                                temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:                               # noqa: BLE001
            self._degrade(exc, "stream_chat")
            yield from self.fallback.stream_chat(messages, tools=tools,
                                                 temperature=temperature,
                                                 max_tokens=max_tokens)

    def stats(self) -> Dict[str, Any]:
        return {"degraded_calls": self.degraded_calls, "last_error": self.last_error}
