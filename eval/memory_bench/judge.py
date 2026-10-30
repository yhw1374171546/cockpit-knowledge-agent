# -*- coding: utf-8 -*-
"""judge：题型化的 LLM 判定（答案层指标用）。

口径对齐调研结论（LongMemEval 的按题型定制 judge + RAGAS/FairEval 的
rationale-before-verdict + 固定输出格式）：

- 每题型有自己的判定规则（KU 含旧值但新值正确判对；ABS 必须拒答、编造判错）；
- 输出强制 `理由：…\nVERDICT: yes|no`，解析器只认 VERDICT 行；
- **启用前置条件**：meta-eval 与人工标注一致率 ≥90% 才启用 judge（30 题/题型），
  否则回退人工——这是调研里的标定锚点，写死在 docstring 里。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from eval.memory_bench.schema import TYPE_NOTES

JUDGE_PROMPT = """你是座舱记忆问答的裁判。判断助手回答是否正确回答了问题。
规则：
1. 先给判断理由，再给结论；结论必须是单独一行 `VERDICT: yes` 或 `VERDICT: no`；
2. 回答只需**包含**正确信息即可，不需要逐字一致；
3. {type_note}
4. 回答无关、含糊、或明显缺失关键信息 → no。

问题：{question}
期望答案：{gold}
助手回答：{answer}

理由：
"""

VERDICT_RE = re.compile(r"VERDICT:\s*(yes|no)", re.IGNORECASE)


def judge(llm: Any, qtype: str, question: str, gold: str, answer: str) -> Dict[str, Any]:
    """用 LLM 判定一道题；返回 {verdict, reason, raw}。

    `llm` 是任何实现 `chat(messages, temperature=..., max_tokens=...)` 的后端
    （DeepSeek/vLLM/ScriptedLLM 均可——测试用 ScriptedLLM 确定性回放）。
    """
    note = TYPE_NOTES.get(qtype, TYPE_NOTES["IE"])
    prompt = JUDGE_PROMPT.format(type_note=note, question=question, gold=gold, answer=answer)
    resp = llm.chat([{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=256)
    raw = (resp.content or "").strip()
    m = VERDICT_RE.search(raw)
    verdict = m.group(1).lower() if m else "no"
    return {"verdict": verdict, "reason": raw, "raw": raw}


def parse_verdict(text: str) -> Optional[str]:
    """从 judge 输出里取 VERDICT（测试与手工复核用）。"""
    m = VERDICT_RE.search(text or "")
    return m.group(1).lower() if m else None


def meta_eval_report(pairs: list[Tuple[str, bool]]) -> Dict[str, Any]:
    """meta-eval：judge 判定 vs 人工标注的一致性（启用 judge 的准入依据）。

    pairs: [(judge_verdict, human_label), ...]；human_label 用 "yes"/"no" 或 bool。
    口径：一致率 ≥90% 才允许用 judge 替代人工（LongMemEval App. A.4 锚点）。
    """
    n = len(pairs)
    if not n:
        return {"n": 0, "agreement": None, "usable": False}
    agree = 0
    for jv, hl in pairs:
        hv = "yes" if hl in (True, "yes", "y", 1) else "no"
        agree += 1 if jv == hv else 0
    agreement = agree / n
    return {"n": n, "agreement": round(agreement, 4),
            "usable": agreement >= 0.9}
