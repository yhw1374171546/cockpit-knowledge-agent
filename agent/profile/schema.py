# -*- coding: utf-8 -*-
"""画像 schema：座舱五层画像 + 每个字段的四要素（来源优先级 / 置信度 / 衰减 / 冲突消解）。

为什么是"五层"而不是通用 assistant 的三类记忆（semantic/episodic/procedural）
----------------------------------------------------------------------------
通用方案（LangMem 等）的记忆类型是"语义/情景/程序性"，那是**存储形态**的分类；
座舱画像要回答的是**业务问题**：这辆车的生命周期到哪了、车主习惯去哪家店、
什么偏好会随季节翻转、哪些是安全红线。这些字段通用方案里根本不存在
（调研结论：Mem0 无强类型 schema、LangMem 的 Pydantic 是通用文档模型），
所以这里是"针对性建模"而不是"套现成方案"——对标出处见 docs/业界长期记忆方案调研.md。

设计对齐（调研结论的直接落地）
------------------------------
- 来源优先级 + 冲突消解：借 Mem0 的 update 冲突观（新信息矛盾时旧记忆失效）
  与 Zep 的 Edge Invalidation（旧事实保留失效、新信息优先），落地为 SOURCE_RANK；
- append_only（保养轨迹）：借 Zep episode 层"非有损"思想——历史事实只追加不覆盖；
- seasonal（季节偏好）：调研未找到现成方案，属业务自定义（已按承诺标注）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# ── 五层 ──
STATIC_PROFILE = "static_profile"   # 静态档案：车型 / 年款 / 牌照区域（几乎不衰减）
LIFECYCLE = "lifecycle"             # 生命周期：保养轨迹 / 下次保养（工单系统为权威来源）
USAGE_PATTERN = "usage_pattern"     # 使用模式：高频功能（仅推断来源，置信度低、衰减快）
PREFERENCE = "preference"           # 偏好：门店 / 座椅加热 / 空调（含季节性）
CONSTRAINT = "constraint"           # 约束红线：仅用户明示，不可被任何推断覆盖

LAYERS: Tuple[str, ...] = (STATIC_PROFILE, LIFECYCLE, USAGE_PATTERN, PREFERENCE, CONSTRAINT)

# 来源优先级：数值越大越权威。冲突消解规则：新来源优先级 < 旧来源 → 拒绝覆盖；
# >= → 旧事实标 t_invalid（保留演化史，借 Zep Edge Invalidation）并写入新事实。
SOURCE_RANK = {
    "vin_parse": 3,        # 车联网 VIN 解析（车辆出厂事实）
    "order_system": 3,     # 工单系统（保养轨迹的权威来源）
    "telematics": 3,       # 远程车况
    "user_explicit": 2,    # 用户明示（"我一般去徐汇店"）
    "inferred": 1,         # 行为/上下文推断（置信度低，不可覆盖明示）
}


@dataclass(frozen=True)
class FieldSpec:
    """画像字段的四要素：来源 / 置信度 / 衰减 / 冲突消解。

    - `sources`：允许的来源，按优先级从高到低排列（第一个是权威来源）；
    - `ttl_days=None`：不衰减（历史事实与红线永远为真，直到被更高优先级覆盖）；
    - `seasonal`：季节偏好——到期不是删除而是"休眠"，来年同期复活（v2 retrieve 层实现）；
    - `append_only`：生命周期轨迹——只追加不覆盖（历史不可改写）。
    """
    key: str
    label: str
    layer: str
    sources: Tuple[str, ...]
    default_confidence: float = 0.9
    ttl_days: Optional[int] = None
    seasonal: bool = False
    append_only: bool = False


FIELDS: Dict[str, FieldSpec] = {
    "vehicle_model": FieldSpec(
        "vehicle_model", "车型", STATIC_PROFILE,
        ("vin_parse", "user_explicit"), 0.95),
    "plate_region": FieldSpec(
        "plate_region", "牌照区域", STATIC_PROFILE,
        ("telematics", "user_explicit"), 0.9),
    "maintenance_history": FieldSpec(
        "maintenance_history", "保养轨迹", LIFECYCLE,
        ("order_system", "user_explicit", "inferred"), 1.0,
        append_only=True),
    "next_due": FieldSpec(
        "next_due", "下次保养", LIFECYCLE,
        ("order_system", "user_explicit"), 0.9),
    "usual_center": FieldSpec(
        "usual_center", "惯用门店", PREFERENCE,
        ("user_explicit", "order_system", "inferred"), 0.85, ttl_days=90),
    "city": FieldSpec(
        "city", "常住城市", PREFERENCE,
        ("user_explicit", "order_system"), 0.9, ttl_days=90),
    "pref:seat_heat": FieldSpec(
        "pref:seat_heat", "座椅加热偏好", PREFERENCE,
        ("user_explicit",), 0.9, ttl_days=180, seasonal=True),
    "pref:ac": FieldSpec(
        "pref:ac", "空调偏好", PREFERENCE,
        ("user_explicit",), 0.9, ttl_days=180, seasonal=True),
    "do_not": FieldSpec(
        "do_not", "车主红线", CONSTRAINT,
        ("user_explicit",), 1.0),
}

# 通用功能偏好模板：pref:{功能名} 动态字段（"以后氛围灯都关掉" → pref:氛围灯）
PREF_FEATURES = ("座椅加热", "座椅通风", "空调", "氛围灯", "方向盘加热", "后视镜加热", "天窗")


def pref_field(feature: str) -> FieldSpec:
    """动态偏好字段：功能名不在 FIELDS 里时按模板合成（季节性与座椅/空调一致）。"""
    return FieldSpec(f"pref:{feature}", f"功能偏好·{feature}", PREFERENCE,
                     ("user_explicit",), 0.9, ttl_days=180, seasonal=True)


def spec_for(key: str) -> FieldSpec:
    if key in FIELDS:
        return FIELDS[key]
    if key.startswith("pref:"):
        feature = key.split(":", 1)[1]
        if feature:
            return pref_field(feature)
    # 未知字段：保守默认——只有用户明示可写、90 天衰减、可覆盖
    return FieldSpec(key, key, PREFERENCE, ("user_explicit",), 0.8, ttl_days=90)


def source_rank(source: str) -> int:
    return SOURCE_RANK.get(source, 0)


def extraction_schema() -> Dict:
    """LLM 结构化抽取的 JSON Schema（guided_json 用）。

    对齐 LangMem 的 Pydantic-schema 抽取思路，但用项目自有的手写 JSON Schema
    （agent/tools.py 同款风格，零依赖）。evidence_span 是**依据定位**：
    候选必须能在原文里逐字找到出处，否则视为幻觉写入（extract 层硬校验）。
    """
    return {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "description": "从车主这句话里抽取的画像事实；没有可写事实就返回空数组",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": ("字段名：vehicle_model / plate_region / usual_center / "
                                            "city / next_due / do_not，或 pref:功能名（如 pref:座椅加热）"),
                        },
                        "value": {"type": "string", "description": "归一化后的值（如 领克08 / 浦东金桥店）"},
                        "source": {
                            "type": "string",
                            "enum": ["user_explicit", "inferred"],
                            "description": "user_explicit=用户明说；inferred=从上下文推断（会被低优先级对待）",
                        },
                        "evidence_span": {
                            "type": "string",
                            "description": "支撑该事实的原文片段，必须逐字摘录，禁止改写",
                        },
                    },
                    "required": ["key", "value", "source", "evidence_span"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["facts"],
        "additionalProperties": False,
    }
