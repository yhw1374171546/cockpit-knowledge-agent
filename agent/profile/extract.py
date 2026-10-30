# -*- coding: utf-8 -*-
"""extract：从车主一句话里抽取画像候选（双后端，与全项目一致的"离线可复现"原则）。

两条路径
--------
1. `extract_rules`（离线规则抽取，CI 基线）：v1 正则的超集，新增两类业务难点——
   **漂移改口**（"不对，我一般去浦东店"）与**否定修正**（"不是那个店，是浦东店"）；
   确定性、零成本、可复现。它衡量"规则能覆盖多少明示句式"，不衡量模型能力。
2. `extract_with_llm`（LLM 结构化抽取）：guided_json 约束到 extraction_schema()，
   并对每条候选做**依据定位校验**——evidence_span 必须逐字出现在原文里，
   否则视为幻觉写入直接丢弃（这是"只记明示 + 防幻觉"在 v2 的落地）。
"""

from __future__ import annotations

import json
import re
from typing import Any, List

from agent.profile.consolidate import MemoryCandidate
from agent.profile.schema import extraction_schema, spec_for

# ── 规则模式（v1 超集）────────────────────────────────────────────────

CITY_WORDS = ("北京", "上海", "深圳", "广州", "杭州", "成都", "合肥", "武汉", "南京", "苏州")
CITY_CUE = re.compile(r"(?:我住|我常住|我家在|我定居|我们住)(?:在)?\s*(" + "|".join(CITY_WORDS) + ")")
MODEL_CUE = re.compile(r"(?:我的车是|我开的是|我买的是|我这台是)\s*(领克\s*0?[1-9])")
CENTER_CUE = re.compile(
    r"(?:我(?:一般|通常|都|常|习惯)去|我就在)\s*([^\s，。！？、；]{2,30})")
# 漂移改口："不对/说错了/改一下，我一般去 X"——与 CENTER_CUE 共用主体，但需要
# 显式标记"这是改口"，consolidate 层才能把它当 overwrite 而不是 insert。
DRIFT_CUE = re.compile(r"(?:不对|说错了|不是这样|改一下|更正)[^。！？\n]{0,12}?"
                       r"我(?:一般|通常|都|常|习惯)去\s*([^\s，。！？、；]{2,30})")
# 否定修正："不是那个店，是浦东店"——v1 完全抽不到（CENTER_CUE 需要"我一般去"开头）
NEGATE_CUE = re.compile(r"不是(?:那个|这家|那家)?(?:店|的)?[，,、]?\s*(?:是|去|应该是)?\s*"
                        r"([^\s，。！？、；]{2,30})(?:店|中心)?(?=$|[，。！？、；])")
PREF_CUE = re.compile(
    r"(?:以后|今后|默认|每次|一直|记住)[^。！？\n]{0,12}?"
    r"(座椅加热|座椅通风|空调|氛围灯|方向盘加热|后视镜加热|天窗)(?:都|就|要)?"
    r"(打开|关闭|开|关|调低|调高|开启)")
PLATE_CUE = re.compile(
    r"(?:车牌|牌照)(?:是|号是|上是)?\s*([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Za-z])")
DO_NOT_CUE = re.compile(
    r"(?:以后|记住|千万别|不要再|不许|永远别)(?:给我|帮我|再)?[^。！？\n]{0,10}?"
    r"(不要|别|禁止|关闭|取消|停止)([^。！？\n]{0,10})")
ACTION_NORMALIZE = {"开": "开启", "关": "关闭", "打开": "开启", "关闭": "关闭",
                    "调低": "调低", "调高": "调高", "开启": "开启"}


def _candidate(key: str, value: str, span: str, source: str = "user_explicit") -> MemoryCandidate:
    spec = spec_for(key)
    return MemoryCandidate(key=key, value=value.strip(), source=source,
                           confidence=spec.default_confidence,
                           evidence_span=span, layer=spec.layer)


def extract_rules(text: str) -> List[MemoryCandidate]:
    """离线规则抽取：只抽**明示**信息（与 v1 的"只记明示"策略一致）。"""
    out: List[MemoryCandidate] = []
    text = text or ""
    for m in CITY_CUE.finditer(text):
        out.append(_candidate("city", m.group(1), m.group(0)))
    for m in MODEL_CUE.finditer(text):
        name = re.sub(r"\s+", "", m.group(1))
        digits = re.search(r"(\d{1,2})$", name)
        if digits:
            name = f"领克{int(digits.group(1)):02d}"
        out.append(_candidate("vehicle_model", name, m.group(0)))
    for m in PLATE_CUE.finditer(text):
        out.append(_candidate("plate_region", m.group(1).upper(), m.group(0)))
    for m in DRIFT_CUE.finditer(text):
        value = m.group(1).strip()
        if "店" in value or "中心" in value:
            out.append(_candidate("usual_center", value, m.group(0)))
    for m in CENTER_CUE.finditer(text):
        value = m.group(1).strip().rstrip("，。")
        if ("店" in value or "中心" in value) and not any(
                c.value == value for c in out if c.key == "usual_center"):
            out.append(_candidate("usual_center", value, m.group(0)))
    if not any(c.key == "usual_center" for c in out):
        for m in NEGATE_CUE.finditer(text):
            value = m.group(1).strip()
            if "店" in value or "中心" in value:
                out.append(_candidate("usual_center", value, m.group(0)))
    for m in PREF_CUE.finditer(text):
        feature, action = m.group(1), ACTION_NORMALIZE.get(m.group(2), m.group(2))
        out.append(_candidate(f"pref:{feature}", action, m.group(0)))
    for m in DO_NOT_CUE.finditer(text):
        value = (m.group(1) + (m.group(2) or "")).strip()
        if value:
            out.append(_candidate("do_not", value, m.group(0)))
    return out


# ── LLM 结构化抽取 ────────────────────────────────────────────────────

EXTRACT_PROMPT = """你是座舱车主画像的抽取器。从车主的**这句话**里抽取可以长期记住的事实，并以 JSON 格式输出（{"facts": [...]}）。
只抽用户**明说**的（source=user_explicit）；从上下文明显能推断的才标 inferred（极少用）。
规则：
1. 只抽能改善后续服务的事实：车型、牌照区域、常住城市、惯用门店、功能偏好、车主红线；
2. evidence_span 必须**逐字摘录**原文片段，禁止改写或拼接；
3. 抽不到就返回空 facts 数组，绝不编造。"""


# 模型常把 schema 里的 "key" 写成 "type" 并用中文标签（实测 deepseek-flash 即如此），
# 解析层做别名映射——这是真实模型对接的典型坑：约束 schema 是提示，不是保证。
KEY_ALIASES = {
    "车型": "vehicle_model", "车辆型号": "vehicle_model", "model": "vehicle_model",
    "牌照": "plate_region", "牌照区域": "plate_region", "车牌": "plate_region",
    "常住城市": "city", "城市": "city", "居住地": "city",
    "惯用门店": "usual_center", "常用门店": "usual_center", "门店": "usual_center",
    "preferred_service_center": "usual_center", "preferred_center": "usual_center",
    "下次保养": "next_due", "保养轨迹": "maintenance_history",
    "车主红线": "do_not", "红线": "do_not", "禁忌": "do_not",
}


def _canonical_key(raw_key: str) -> str:
    key = (raw_key or "").strip()
    if not key:
        return ""
    if key.startswith("pref:"):
        return key
    return KEY_ALIASES.get(key, key)


def extract_with_llm(text: str, llm: Any) -> List[MemoryCandidate]:
    """LLM 结构化抽取 + 依据定位硬校验（防幻觉写入）。

    `llm` 需支持 guided_json（vLLM/DeepSeek 均已适配，见 agent/llm.py）；
    离线替身（RuleBasedPlannerLLM）不支持此能力，调用方应先用 `extract_rules` 兜底。
    """
    resp = llm.chat(
        [{"role": "system", "content": EXTRACT_PROMPT},
         {"role": "user", "content": (text or "").strip()}],
        guided_json=extraction_schema(), temperature=0.0, max_tokens=1024)
    raw = (resp.content or "").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    facts = payload.get("facts") or []
    out: List[MemoryCandidate] = []
    for item in facts:
        if not isinstance(item, dict):
            continue
        key = _canonical_key(str(item.get("key") or item.get("type") or ""))
        value = str(item.get("value") or item.get("content") or "").strip()
        span = str(item.get("evidence_span") or item.get("evidence") or "").strip()
        source = "inferred" if item.get("source") == "inferred" else "user_explicit"
        # 依据定位硬校验：候选必须能在原文找到出处，否则是幻觉，直接丢弃
        if not key or not value or span not in text:
            continue
        spec = spec_for(key)
        out.append(MemoryCandidate(key=key, value=value, source=source,
                                   confidence=spec.default_confidence,
                                   evidence_span=span, layer=spec.layer))
    return out
