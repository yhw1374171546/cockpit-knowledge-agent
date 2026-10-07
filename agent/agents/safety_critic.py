# -*- coding: utf-8 -*-
"""安全评审 Agent（Critic / Verifier）：对聚合后的答案做最后一道把关。

它不看"答得好不好"，只回答三个问题：
1. 这句话在检索证据里站得住吗？（接地校验，复用 Reflector）
2. 当实时车况是 critical 时，答案里有没有给出停驶/联系中心的**安全指令**？
3. 有没有在注入风险下仍然声称"已下单/已预约"，或者把通用建议当成安全结论？
不通过 → revise（补安全话术）或 block（降级为安全话术）。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from agent.guardrails.output_guard import OutputGuard
from agent.protocols import (PERMISSIVE_PATTERNS, RISK_CRITICAL, SAFETY_DIRECTIVES,
                             AgentResult, CriticVerdict, risk_at_least)
from agent.reflection import NO_ANSWER, Reflector

WRITE_CLAIM_PATTERNS = ("已为您预约", "已下单", "预约成功", "已提交预约", "已经帮您预约")


class SafetyCritic:
    name = "safety_critic"
    title = "安全评审"
    description = "校验答案的接地性与安全合规性，必要时改写或降级为安全话术"

    def __init__(self, reflector: Optional[Reflector] = None,
                 min_grounded_ratio: float = 0.34,
                 output_guard: Optional[OutputGuard] = None,
                 system_prompt: str = ""):
        self.reflector = reflector or Reflector()
        self.min_grounded_ratio = min_grounded_ratio
        self.output_guard = output_guard or OutputGuard(secrets=[system_prompt] if system_prompt else [])
        self.reviews = 0
        self.revised = 0
        self.blocked = 0
        self.leaks_blocked = 0

    def review(self, question: str, answer: str, results: Sequence[AgentResult],
               evidence: Sequence[Dict]) -> CriticVerdict:
        self.reviews += 1
        reasons: List[str] = []
        critical = [r for r in results if risk_at_least(r.risk_level, RISK_CRITICAL)]
        highest = max((r.risk_level for r in results), default="ok")

        # ⓪ 输出侧泄露检查：提示词窃取不经过工具，必须在输出上拦
        leak = self.output_guard.check(answer or "")
        if leak.leaked:
            reasons.append(f"答案含疑似内部信息泄露（{', '.join(leak.kinds)}），已拦截")
            self.leaks_blocked += 1
            self.blocked += 1
            return CriticVerdict("block", reasons,
                                 self.output_guard.enforce(answer or ""))

        # ① 安全红线：实时风险 critical，答案必须含安全指令，且不得保留许可性表述
        if critical:
            text = answer or ""
            findings = "；".join(f for r in critical for f in r.findings[:2])
            if not any(d in text for d in SAFETY_DIRECTIVES):
                reasons.append("实时车况为 critical，但答案未给出停驶/联系中心的安全指令")
                text = (f"{SAFETY_DIRECTIVES[0]}，{SAFETY_DIRECTIVES[1]}。"
                        f"（依据实时车况：{findings}） " + text)
            if PERMISSIVE_PATTERNS.search(text):
                reasons.append("安全风险下保留「可继续行驶」类许可性表述，已替换为停驶指令")
                text = PERMISSIVE_PATTERNS.sub("请勿继续行驶", text)
            if reasons:
                self.revised += 1
                return CriticVerdict("revise", reasons, text)

        # ② 注入风险下声称已完成写操作 → 直接拦截
        if any(r.injection_flagged for r in results) and \
                any(p in (answer or "") for p in WRITE_CLAIM_PATTERNS):
            reasons.append("检测到注入风险同时声称已执行写操作，已降级为安全话术")
            self.blocked += 1
            return CriticVerdict("block", reasons,
                                 "检测到可疑指令注入，为避免误操作，请通过领克 App 或致电领克中心确认预约。")

        # ③ 接地性：答案必须有证据支撑
        if answer and answer.strip() != NO_ANSWER:
            verdict = self.reflector.verify(answer, list(evidence), question)
            if verdict.verdict == "ungrounded":
                reasons.append(f"答案接地率仅 {verdict.grounded_ratio:.2f}，缺乏证据支撑")
                self.blocked += 1
                return CriticVerdict("block", reasons, NO_ANSWER)

        return CriticVerdict("approve", reasons or ["证据充分、无安全风险"],
                             answer)

    def stats(self) -> Dict[str, int]:
        return {"reviews": self.reviews, "revised": self.revised, "blocked": self.blocked,
                "leaks_blocked": self.leaks_blocked}
