# -*- coding: utf-8 -*-
"""工具层：把 RAG 检索与车辆数据封装成 LLM 可调用的工具（Function Calling）。

设计要点
--------
1. **标准 JSON Schema**：`ToolRegistry.specs()` 直接输出 OpenAI / vLLM 的 `tools` 参数格式，
   可以直接丢给兼容接口，不需要任何适配代码。
2. **结构化错误**：工具失败返回 `{"ok": false, "error": ..., "hint": ...}`，
   让模型看到失败原因后自行决定「换参数重试 / 换工具 / 降级 / 拒答」，
   而不是抛异常打断整个 Agent 循环。
3. **写操作治理**：所有写操作（如预约到店）标记 `write=True / requires_confirmation=True`，
   未获确认时一律拒绝执行，返回待确认状态。
4. **重复调用抑制**：同一工具 + 同样参数第二次调用直接命中缓存并标记 `repeated`，
   配合上层「无进展检测」防止 Agent 空转。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agent.kb import Evidence, KnowledgeBase

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: Optional[str] = None
    hint: Optional[str] = None
    repeated: bool = False
    latency_ms: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_observation(self) -> str:
        if not self.ok:
            return json.dumps({"ok": False, "error": self.error, "hint": self.hint},
                              ensure_ascii=False)
        return json.dumps({"ok": True, "data": self.data}, ensure_ascii=False)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., ToolResult]
    write: bool = False
    requires_confirmation: bool = False

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ── 车辆静态数据（真实项目中来自车云接口，这里内置演示数据） ──────────

VEHICLE_SPECS: Dict[str, Dict[str, str]] = {
    "领克08": {"车型": "领克08 EM-P", "能源类型": "插电式混合动力", "纯电续航": "245 km(CLTC)",
             "综合续航": "1400 km", "电池容量": "39.6 kWh", "快充时间": "30 分钟(30%-80%)",
             "整备质量": "2035 kg", "轮胎规格": "235/50 R19"},
    "领克09": {"车型": "领克09 EM-P", "能源类型": "插电式混合动力", "纯电续航": "190 km(CLTC)",
             "综合续航": "1100 km", "电池容量": "40.1 kWh", "快充时间": "28 分钟(30%-80%)",
             "整备质量": "2320 kg", "轮胎规格": "275/45 R20"},
}

MAINTENANCE_PLAN = [
    (5000, ["首次保养：更换机油机滤", "检查制动系统", "检查轮胎气压与磨损"]),
    (10000, ["更换机油机滤", "更换空调滤芯", "检查高压系统绝缘"]),
    (20000, ["更换机油机滤", "更换空气滤芯", "更换空调滤芯", "检查制动液含水率"]),
    (40000, ["更换机油机滤", "更换火花塞", "更换制动液", "检查动力电池健康度"]),
    (60000, ["更换机油机滤", "更换变速箱油", "更换冷却液", "检查悬挂与转向系统"]),
]

VEHICLE_TELEMETRY = {
    "vin": "L6T**********1234",
    "车型": "领克08 EM-P",
    "里程_km": 23860,
    "胎压_kPa": {"左前": 236, "右前": 241, "左后": 228, "右后": 239},
    "胎压告警": ["左后"],
    "剩余电量_%": 62,
    "续航_km": 148,
    "告警灯": [],
    "下次保养_km": 24000,
    "车门状态": "已锁止",
    "车窗状态": "全部关闭",
}


class ToolRegistry:
    """工具注册表：schema 声明、调用、治理（确认/去重/计数/超时）。"""

    def __init__(self, kb: KnowledgeBase, telemetry: Optional[Dict] = None,
                 specs_data: Optional[Dict] = None, maintenance: Optional[List] = None):
        self.kb = kb
        self.telemetry = telemetry if telemetry is not None else dict(VEHICLE_TELEMETRY)
        self.specs_data = specs_data if specs_data is not None else VEHICLE_SPECS
        self.maintenance = maintenance if maintenance is not None else MAINTENANCE_PLAN
        self.confirmed_actions: set = set()
        self._cache: Dict[str, ToolResult] = {}
        self.call_counts: Dict[str, int] = {}
        self.trace: List[Dict[str, Any]] = []
        self.tools: Dict[str, ToolSpec] = {}
        self._register_all()

    # ── 注册 ──
    def _register_all(self) -> None:
        self.register(ToolSpec(
            name="search_manual",
            description=("检索车主用户手册知识库，返回与问题最相关的原文片段及其出处。"
                         "凡是涉及车辆功能操作、故障处置、保养要求、指示灯含义等问题都必须先调用本工具；"
                         "如果检索结果不足以回答，应当如实说明而不是编造。"),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词或自然语言问题，建议使用手册中的术语"},
                    "top_k": {"type": "integer", "description": "返回片段数量，默认 6，最大 10",
                              "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
            handler=self._search_manual,
        ))
        self.register(ToolSpec(
            name="get_vehicle_status",
            description="读取当前车辆的实时状态，包括里程、胎压、电量、告警灯等。适合与手册知识联合回答「我这台车现在该怎么办」。",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=self._get_vehicle_status,
        ))
        self.register(ToolSpec(
            name="lookup_vehicle_spec",
            description="查询车型参数配置（续航、电池容量、充电时间、轮胎规格等静态参数）。",
            parameters={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "车型名称，例如 领克08 / 领克09"},
                    "field": {"type": "string", "description": "可选，只取某个字段，如 纯电续航"},
                },
                "required": ["model"],
            },
            handler=self._lookup_vehicle_spec,
        ))
        self.register(ToolSpec(
            name="get_maintenance_plan",
            description="根据当前里程给出保养项目建议，说明本次与下次保养该做什么。",
            parameters={
                "type": "object",
                "properties": {"mileage_km": {"type": "integer", "description": "当前里程，缺省则读取车辆实时里程"}},
                "required": [],
            },
            handler=self._get_maintenance_plan,
        ))
        self.register(ToolSpec(
            name="create_service_order",
            description="为车主创建到店服务预约（写操作，需车主二次确认后方可执行）。",
            parameters={
                "type": "object",
                "properties": {
                    "item": {"type": "string", "description": "服务项目，例如 更换机油机滤"},
                    "preferred_date": {"type": "string", "description": "期望到店日期，格式 YYYY-MM-DD"},
                    "center": {"type": "string", "description": "期望到店的门店名称"},
                },
                "required": ["item", "preferred_date"],
            },
            handler=self._create_service_order,
            write=True,
            requires_confirmation=True,
        ))

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def subset(self, names, name: str = "sub-agent") -> "ToolRegistry":
        """派生一个只暴露指定工具的注册表（用于子 Agent 的工具白名单）。

        白名单是最小权限原则的落地：子 Agent 看不到不该用的工具，就不可能误调用——
        这也是多 Agent 相对"单 Agent 挂全部工具"的可量化收益之一。
        """
        scoped = ToolRegistry.__new__(ToolRegistry)
        scoped.kb = self.kb
        scoped.telemetry = self.telemetry
        scoped.specs_data = self.specs_data
        scoped.maintenance = self.maintenance
        scoped.confirmed_actions = self.confirmed_actions      # 确认状态在父子间共享
        scoped._cache = self._cache                            # 调用缓存共享，跨 Agent 去重
        scoped.call_counts = self.call_counts
        scoped.trace = self.trace
        scoped.tools = {n: self.tools[n] for n in names if n in self.tools}
        scoped.scope_name = name
        return scoped

    def names_of_scope(self) -> str:
        return getattr(self, "scope_name", "all")

    def specs(self) -> List[Dict[str, Any]]:
        return [t.to_openai_schema() for t in self.tools.values()]

    def names(self) -> List[str]:
        return list(self.tools.keys())

    def confirm(self, action_key: str) -> None:
        """车主确认某个写操作后调用。"""
        self.confirmed_actions.add(action_key)

    def reset_cache(self) -> None:
        """清空调用缓存（评测单题冷启动耗时前调用，避免命中上一题的缓存）。"""
        self._cache.clear()
        self.call_counts.clear()
        self.trace.clear()

    def cache_stats(self) -> Dict[str, int]:
        repeated = sum(1 for t in self.trace if t.get("repeated"))
        return {"total_calls": len(self.trace), "repeated_calls": repeated,
                "unique_calls": len(self._cache)}

    # ── 调用入口 ──
    def call(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        spec = self.tools.get(name)
        if spec is None:
            return ToolResult(False, error=f"未知工具 {name}",
                              hint=f"可用工具：{', '.join(self.names())}")
        key = f"{name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"
        if key in self._cache:
            cached = self._cache[key]
            repeated = ToolResult(cached.ok, cached.data, cached.error, cached.hint,
                                  repeated=True, latency_ms=0.0, meta=cached.meta)
            self._record(name, arguments, repeated)
            return repeated

        if spec.requires_confirmation:
            action_key = f"{name}:{arguments.get('item')}:{arguments.get('preferred_date')}"
            if action_key not in self.confirmed_actions:
                res = ToolResult(False, error="写操作需要车主确认",
                                 hint=f"请先向车主说明将执行「{arguments.get('item')}」并获取确认，"
                                      f"确认后再调用（action_key={action_key}）",
                                 meta={"needs_confirmation": True, "action_key": action_key})
                self._record(name, arguments, res)
                return res

        start = time.perf_counter()
        try:
            res = spec.handler(**arguments)
        except TypeError as exc:
            res = ToolResult(False, error=f"参数不合法：{exc}",
                             hint="请检查参数名与类型是否与工具 schema 一致")
        except Exception as exc:  # 工具内部异常不打断 Agent 循环
            res = ToolResult(False, error=f"{type(exc).__name__}: {exc}",
                             hint="可尝试调整参数重试，或改用其他工具")
        res.latency_ms = (time.perf_counter() - start) * 1000
        self._cache[key] = res
        self._record(name, arguments, res)
        return res

    def _record(self, name: str, arguments: Dict[str, Any], res: ToolResult) -> None:
        self.call_counts[name] = self.call_counts.get(name, 0) + 1
        self.trace.append({"tool": name, "arguments": arguments, "ok": res.ok,
                           "repeated": res.repeated, "latency_ms": round(res.latency_ms, 2),
                           "error": res.error})

    # ── 具体工具实现 ──
    def _search_manual(self, query: str, top_k: int = 6) -> ToolResult:
        top_k = max(1, min(int(top_k or 6), 10))
        hits: List[Evidence] = self.kb.search(query, top_k=top_k)
        if not hits:
            return ToolResult(False, error="知识库中没有检索到相关内容",
                              hint="可以换用手册中的术语重试，例如把口语「靠背发烫」换成「座椅加热」")
        return ToolResult(True, data=[h.to_dict() for h in hits],
                          meta={"n_hits": len(hits), "retriever": hits[0].retriever})

    def _get_vehicle_status(self) -> ToolResult:
        return ToolResult(True, data=self.telemetry)

    def _lookup_vehicle_spec(self, model: str, field: Optional[str] = None) -> ToolResult:
        entry = None
        for key, value in self.specs_data.items():
            if key in model or value.get("车型", "").startswith(model):
                entry = value
                break
        if entry is None:
            return ToolResult(False, error=f"未收录车型：{model}",
                              hint=f"已收录车型：{', '.join(self.specs_data.keys())}")
        if field:
            for k, v in entry.items():
                if field in k:
                    return ToolResult(True, data={k: v})
            return ToolResult(False, error=f"该车型没有字段「{field}」",
                              hint=f"可用字段：{', '.join(entry.keys())}")
        return ToolResult(True, data=entry)

    def _get_maintenance_plan(self, mileage_km: Optional[int] = None) -> ToolResult:
        mileage = int(mileage_km) if mileage_km is not None else int(self.telemetry.get("里程_km", 0))
        due = [m for m in self.maintenance if m[0] <= mileage]
        upcoming = [m for m in self.maintenance if m[0] > mileage]
        current = due[-1] if due else None
        nxt = upcoming[0] if upcoming else None
        data = {
            "当前里程_km": mileage,
            "本次建议项目": current[1] if current else ["尚未到首保里程，建议按 5000 km 或 6 个月先到为准"],
            "本次对应里程": current[0] if current else None,
            "下次保养": {"里程_km": nxt[0], "项目": nxt[1]} if nxt else None,
            "剩余里程_km": (nxt[0] - mileage) if nxt else None,
        }
        return ToolResult(True, data=data)

    def _create_service_order(self, item: str, preferred_date: str,
                              center: str = "领克中心(浦东金桥店)") -> ToolResult:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", preferred_date or ""):
            return ToolResult(False, error="日期格式应为 YYYY-MM-DD",
                              hint="例如 2025-06-18")
        return ToolResult(True, data={
            "预约单号": f"SO{int(time.time()) % 10**8:08d}",
            "项目": item, "到店日期": preferred_date, "门店": center,
            "状态": "已提交（待门店确认）",
        })


def build_default_registry(kb_path: Optional[str] = None) -> ToolRegistry:
    kb = KnowledgeBase.load(kb_path)
    return ToolRegistry(kb)


if __name__ == "__main__":
    registry = build_default_registry()
    print("工具清单：", registry.names())
    print(json.dumps(registry.specs()[0], ensure_ascii=False, indent=2)[:600])
    print("\n检索工具调用：")
    res = registry.call("search_manual", {"query": "座椅加热怎么关闭", "top_k": 2})
    print(res.to_observation()[:400])
    print("\n写操作未确认：")
    print(registry.call("create_service_order",
                        {"item": "更换机油机滤", "preferred_date": "2025-06-18"}).to_observation())
    print("\n重复调用检测：")
    registry.call("get_vehicle_status", {})
    print(registry.call("get_vehicle_status", {}).repeated)
