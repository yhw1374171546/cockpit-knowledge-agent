# -*- coding: utf-8 -*-
"""输出侧防线：防止系统提示词 / 凭据随答案泄露。

为什么需要单独一层：
- **输入侧**（注入检测 + 工具最小权限）能拦住"诱导执行写操作"；
- 但"把系统提示词念一遍"这类攻击**不经过任何工具**，答案直接就是泄露内容，
  工具策略根本管不到 —— 必须在**输出**上做检查。

本模块提供：系统提示词原文检测、凭据形态检测（API Key / Token / 长十六进制串）、
以及脱敏改写。接在安全评审之后（见 agents/safety_critic.py）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# 凭据形态（宁可多报，因为泄露代价远高于误报代价）
SECRET_PATTERNS = [
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}\b")),
    ("bearer", re.compile(r"\bBearer\s+[A-Za-z0-9\._\-]{16,}\b", re.I)),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{12,}\b")),
    ("long_hex", re.compile(r"\b[0-9a-fA-F]{32,}\b")),
    ("env_secret", re.compile(r"(api[_-]?key|secret|password|token)\s*[:=]\s*\S{8,}", re.I)),
]

# 提示词泄露的迹象措辞
PROMPT_LEAK_HINTS = ("我的系统提示词", "系统提示词是", "system prompt is",
                     "my instructions are", "初始设定是", "我的设定是", "提示词如下")

REFUSAL_TEMPLATE = ("抱歉，这属于系统内部信息，我不能提供。"
                    "如果您想了解车辆功能或保养问题，我可以继续帮您查询手册。")


@dataclass
class LeakReport:
    leaked: bool = False
    kinds: List[str] = field(default_factory=list)
    matched: List[str] = field(default_factory=list)
    sanitized: str = ""

    def to_dict(self) -> Dict:
        return {"leaked": self.leaked, "kinds": self.kinds,
                "matched": [m[:24] + "…" if len(m) > 24 else m for m in self.matched]}


class OutputGuard:
    def __init__(self, secrets: Sequence[str] = (), min_prompt_fragment: int = 24):
        self.secrets = [s for s in secrets if s]
        self.min_prompt_fragment = min_prompt_fragment

    def add_secret(self, text: str) -> None:
        if text:
            self.secrets.append(text)

    def check(self, answer: str, system_prompt: Optional[str] = None) -> LeakReport:
        answer = answer or ""
        report = LeakReport(sanitized=answer)

        # ① 系统提示词原文片段泄露（取若干等长片段做子串匹配，容忍轻微改写）
        prompts = list(self.secrets) + ([system_prompt] if system_prompt else [])
        for text in prompts:
            fragment = self._longest_common_fragment(answer, text)
            if fragment and len(fragment) >= self.min_prompt_fragment:
                report.leaked = True
                report.kinds.append("system_prompt")
                report.matched.append(fragment)
                report.sanitized = report.sanitized.replace(fragment, "［系统内部信息已脱敏］")

        # ② 提示词泄露措辞
        lowered = answer.lower()
        for hint in PROMPT_LEAK_HINTS:
            if hint.lower() in lowered:
                report.leaked = True
                report.kinds.append("prompt_hint")
                report.matched.append(hint)
                break

        # ③ 凭据形态
        for name, pattern in SECRET_PATTERNS:
            for m in pattern.finditer(report.sanitized):
                report.leaked = True
                report.kinds.append(name)
                report.matched.append(m.group(0))
                report.sanitized = report.sanitized.replace(m.group(0), "［凭据已脱敏］")

        report.kinds = sorted(set(report.kinds))
        return report

    def enforce(self, answer: str, system_prompt: Optional[str] = None) -> str:
        """泄露则整段替换为拒绝话术（比局部脱敏更安全，避免残留提示词结构）。"""
        report = self.check(answer, system_prompt)
        if not report.leaked:
            return answer
        if "system_prompt" in report.kinds or "prompt_hint" in report.kinds:
            return REFUSAL_TEMPLATE
        return report.sanitized

    # ── 工具 ──
    @staticmethod
    def _longest_common_fragment(a: str, b: str, block: int = 12) -> Optional[str]:
        """用固定块长做滚动匹配，返回 a 中与 b 共有的最长片段（近似）。"""
        if not a or not b:
            return None
        blocks = {b[i:i + block] for i in range(0, max(1, len(b) - block + 1))}
        best = ""
        i = 0
        while i <= len(a) - block:
            if a[i:i + block] in blocks:
                j = i + block
                while j < len(a) and a[i:j + 1] in b:
                    j += 1
                if j - i > len(best):
                    best = a[i:j]
                i = j
            else:
                i += 1
        return best or None
