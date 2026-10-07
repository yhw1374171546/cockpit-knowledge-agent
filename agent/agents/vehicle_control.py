# -*- coding: utf-8 -*-
"""车控专家：读实时车况，并按安全规则给出风险等级。

它是仲裁里优先级最高的业务 Agent（仅次于安全评审），因为**实时车况比手册通用说明更贴近当下事实**：
手册说"胎压低时低速行驶几分钟可解除报警"，但如果此刻左后胎压只有 150 kPa，那就不该出这个建议。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from agent.agents.base import SubAgent
from agent.protocols import (RISK_CRITICAL, RISK_NOTICE, RISK_OK, RISK_WARNING,
                             AGENT_PRIORITY)

# 一旦点亮就必须停驶排查的告警（安全红线）
CRITICAL_WARNINGS = ("制动", "刹车", "安全气囊", "气囊", "高压", "动力电池", "过热",
                     "转向", "起火", "冒烟", "碰撞", "绝缘")
WARNING_WARNINGS = ("胎压", "机油", "冷却液", "充电", "abs", "esp", "车身稳定")


def assess_telemetry(telemetry: Dict) -> Tuple[str, List[str]]:
    """把车况映射为风险等级 + 事实清单（供仲裁与安全评审使用）。"""
    findings: List[str] = []
    level = RISK_OK

    def bump(target: str) -> None:
        nonlocal level
        order = {RISK_OK: 0, RISK_NOTICE: 1, RISK_WARNING: 2, RISK_CRITICAL: 3}
        if order[target] > order[level]:
            level = target

    pressures = telemetry.get("胎压_kPa") or {}
    alerted = [str(p) for p in (telemetry.get("胎压告警") or [])]
    for pos, value in pressures.items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value < 180:
            bump(RISK_CRITICAL)
            findings.append(f"{pos}胎压 {value:.0f}kPa 严重亏气")
        elif value < 210:
            bump(RISK_WARNING)
            findings.append(f"{pos}胎压 {value:.0f}kPa 偏低")
        elif pos in alerted:
            # 车机已点亮胎压报警，即使数值没到阈值也必须当成告警处理
            bump(RISK_WARNING)
            findings.append(f"{pos}胎压报警已激活（{value:.0f}kPa）")
    for pos in alerted:
        if pos not in pressures:
            bump(RISK_WARNING)
            findings.append(f"{pos}胎压报警已激活")

    alerts = list(telemetry.get("告警灯") or [])
    for alert in alerts:
        text = str(alert)
        if any(k in text for k in CRITICAL_WARNINGS):
            bump(RISK_CRITICAL)
            findings.append(f"告警灯：{text}（需立即处置）")
        elif any(k.lower() in text.lower() for k in WARNING_WARNINGS):
            bump(RISK_WARNING)
            findings.append(f"告警灯：{text}")
        else:
            bump(RISK_NOTICE)
            findings.append(f"提示信息：{text}")

    soc = telemetry.get("剩余电量_%")
    if isinstance(soc, (int, float)) and soc < 10:
        bump(RISK_WARNING)
        findings.append(f"剩余电量仅 {soc}%")

    mileage = telemetry.get("里程_km")
    due = telemetry.get("下次保养_km")
    if isinstance(mileage, (int, float)) and isinstance(due, (int, float)) and mileage >= due:
        bump(RISK_NOTICE)
        findings.append(f"已超过保养里程（{mileage:.0f} ≥ {due:.0f} km）")

    if not findings:
        findings.append("当前车况无异常")
    return level, findings


class VehicleControlAgent(SubAgent):
    name = "vehicle_control"
    title = "车控专家"
    kind = "status"
    description = "读取实时车况（胎压/电量/告警灯/里程），给出风险等级与安全处置建议"
    allowed_tools = ("get_vehicle_status", "search_manual")

    def assess(self, state) -> Tuple[str, List[str]]:
        telemetry = getattr(self.registry, "telemetry", {}) or {}
        level, findings = assess_telemetry(telemetry)
        if state.injection_flagged:
            findings = findings + ["检索内容疑似包含注入指令"]
        return level, findings

    def _to_result(self, task, state, budget):
        result = super()._to_result(task, state, budget)
        # 车控专家的答案里必须带上关键事实，仲裁与安全评审才有依据
        if result.findings:
            facts = "；".join(result.findings[:4])
            result.answer = f"{result.answer}（实时车况：{facts}）" if result.answer else \
                f"实时车况：{facts}"
        # 关键：结构化工具输出本身就是**证据**。
        # 若只把手册检索块当证据，纯车况问答会被接地校验误判为"无依据"而拒答。
        telemetry = getattr(self.registry, "telemetry", {}) or {}
        result.evidence = list(result.evidence) + [{
            "text": "实时车况：" + "；".join(result.findings),
            "citation": "[车况接口]",
            "source": "vehicle_api",
            "chunk_id": -1,
        }]
        if not result.citations:
            result.citations = ["[车况接口]"]
        return result
