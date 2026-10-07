# -*- coding: utf-8 -*-
"""安全护栏层：提示注入防御、工具策略、预算熔断。

座舱场景的特殊性：检索到的"手册内容"会被塞进大模型上下文，而工具里存在**写操作**
（预约到店）。因此本层针对三类真实风险：
1. **间接提示注入**——被污染的手册块/工具返回值里藏指令，劫持模型行为；
2. **工具滥用 / 越权写操作**——模型被诱导执行副作用操作；
3. **钱包攻击（denial of wallet）**——循环调用把 token 预算烧光。
"""

from agent.guardrails.budget import Budget, BudgetExceeded, BudgetTracker, CostModel
from agent.guardrails.injection import InjectionDetector, InjectionReport, isolate_evidence
from agent.guardrails.output_guard import LeakReport, OutputGuard
from agent.guardrails.policy import Decision, PolicyContext, PolicyEngine, ToolPolicy

__all__ = [
    "Budget", "BudgetExceeded", "BudgetTracker", "CostModel",
    "InjectionDetector", "InjectionReport", "isolate_evidence",
    "LeakReport", "OutputGuard",
    "Decision", "PolicyContext", "PolicyEngine", "ToolPolicy",
]
