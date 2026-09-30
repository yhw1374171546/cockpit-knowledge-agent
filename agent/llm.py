# -*- coding: utf-8 -*-
"""LLM 后端抽象：生产走 OpenAI 兼容接口（vLLM），离线走可复现的规划器。

三种后端
--------
1. `OpenAICompatibleLLM`：直接 POST `/chat/completions`（只用标准库 urllib，
   不依赖 openai 包）。vLLM 起 `vllm.entrypoints.openai.api_server` 后即可对接，
   支持 `tools` 参数做原生 Function Calling。
2. `ScriptedLLM`：按脚本回放固定响应，用于单元测试与回归复现（确定性）。
3. `RuleBasedPlannerLLM`：**离线规划器替身**。用一个确定性的规则策略模拟「模型决策」，
   使整条 Agent 编排链路（路由→规划→工具→反思→回答）在没有 GPU、没有大模型的情况下
   也能真实跑通并被度量。它衡量的是**编排与检索**，不衡量生成质量——生成质量由
   真实大模型跑批的 4 路消融实验给出。
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

NO_ANSWER = "无答案"


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    id: str = "call_0"

    def to_openai(self) -> Dict[str, Any]:
        return {"id": self.id, "type": "function",
                "function": {"name": self.name,
                             "arguments": json.dumps(self.arguments, ensure_ascii=False)}}


@dataclass
class LLMResponse:
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


class LLMBackend:
    name = "base"

    def chat(self, messages: Sequence[Dict[str, Any]], tools: Optional[List[Dict]] = None,
             temperature: float = 0.0, max_tokens: int = 1024) -> LLMResponse:
        raise NotImplementedError


# ── 生产：OpenAI 兼容（vLLM） ────────────────────────────────────────


class OpenAICompatibleLLM(LLMBackend):
    name = "openai-compatible"

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", model: str = "Qwen2_7B",
                 api_key: str = "EMPTY", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"调用 LLM 服务失败：{exc}") from exc

        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = []
        for i, tc in enumerate(message.get("tool_calls") or []):
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(fn.get("name", ""), args, tc.get("id", f"call_{i}")))
        return LLMResponse(content=message.get("content"), tool_calls=calls,
                           finish_reason=choice.get("finish_reason", "stop"),
                           usage=body.get("usage") or {})


# ── 测试：脚本回放 ────────────────────────────────────────────────────


class ScriptedLLM(LLMBackend):
    """按顺序回放预设响应，用于单元测试（确定性、无网络）。"""

    name = "scripted"

    def __init__(self, responses: Sequence[LLMResponse]):
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024) -> LLMResponse:
        self.calls.append({"n_messages": len(messages), "tools": [t["function"]["name"] for t in (tools or [])]})
        if not self.responses:
            return LLMResponse(content="（脚本已耗尽）")
        return self.responses.pop(0)


# ── 离线：规则规划器 ──────────────────────────────────────────────────


class RuleBasedPlannerLLM(LLMBackend):
    """确定性规划器替身：模拟模型的「路由 + 选工具 + 组织答案」三步决策。

    它让没有 GPU 的环境也能真实跑完整条 Agent 链路并度量编排行为。
    """

    name = "rule-based-planner"

    INTENT_RULES = [
        ("booking", ("预约", "到店", "保养预约", "帮我约", "下单", "安排到")),
        ("spec", ("参数", "配置", "电池容量", "几度电", "充电时间", "轮胎规格", "轴距",
                  "纯电续航", "续航里程", "电机功率", "百公里加速")),
        ("maintenance", ("保养", "什么时候保养", "该做什么", "保养项目", "多少公里保养", "首保")),
        ("status", ("胎压", "电量", "剩余续航", "我现在", "我的车", "告警", "故障灯", "剩下")),
    ]

    def __init__(self, overlap_threshold: float = 0.12, rewrite_enabled: bool = True):
        self.overlap_threshold = overlap_threshold
        self.rewrite_enabled = rewrite_enabled
        self.turn = 0
        self.route_history: List[str] = []

    # -- 路由 --
    def route(self, question: str) -> str:
        for intent, keys in self.INTENT_RULES:
            if any(k in question for k in keys):
                self.route_history.append(intent)
                return intent
        self.route_history.append("manual_qa")
        return "manual_qa"

    # -- 决策 --
    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024) -> LLMResponse:
        self.turn += 1
        question = self._last_user(messages)
        observations = self._observations(messages)
        tool_names = {t["function"]["name"] for t in (tools or [])}

        # 第一步：还没调用过任何工具 → 规划工具调用
        if not observations:
            return LLMResponse(tool_calls=self._plan(question, tool_names))

        # 第二步：已有观察结果 → 反思（证据是否足够）
        if not self._has_reflected(messages):
            if self._evidence_sufficient(question, observations):
                return LLMResponse(tool_calls=[ToolCall("__reflect__", {"verdict": "sufficient"})])
            return LLMResponse(tool_calls=[ToolCall("__reflect__", {"verdict": "insufficient"})])

        # 第三步：产出答案（抽取式，模拟「依据上下文生成」）
        return LLMResponse(content=self._compose(question, observations))

    # -- 计划 --
    def _plan(self, question: str, available: set) -> List[ToolCall]:
        intent = self.route(question)
        calls: List[ToolCall] = []
        if intent == "manual_qa":
            if "search_manual" in available:
                calls.append(ToolCall("search_manual", {"query": self._normalize_query(question), "top_k": 6}))
        elif intent == "maintenance":
            if "get_maintenance_plan" in available:
                calls.append(ToolCall("get_maintenance_plan", {}))
            if "search_manual" in available:
                calls.append(ToolCall("search_manual", {"query": "保养项目 保养周期", "top_k": 4}))
        elif intent == "status":
            if "get_vehicle_status" in available:
                calls.append(ToolCall("get_vehicle_status", {}))
            if "search_manual" in available:
                calls.append(ToolCall("search_manual", {"query": self._normalize_query(question), "top_k": 4}))
        elif intent == "spec":
            if "lookup_vehicle_spec" in available:
                calls.append(ToolCall("lookup_vehicle_spec", {"model": "领克08"}))
            if "search_manual" in available:
                calls.append(ToolCall("search_manual", {"query": self._normalize_query(question), "top_k": 4}))
        elif intent == "booking":
            # 写操作：先只取信息，执行留给确认环节
            if "get_maintenance_plan" in available:
                calls.append(ToolCall("get_maintenance_plan", {}))
        return calls

    # -- 证据充分性（离线替身用词面重叠 + 命中数判断） --
    def _evidence_sufficient(self, question: str, observations: List[Dict]) -> bool:
        best = 0.0
        for obs in observations:
            data = obs.get("data")
            if isinstance(data, list):
                for item in data:
                    text = item.get("text", "") if isinstance(item, dict) else str(item)
                    if not text:
                        continue
                    if "".join(sorted(set(re.findall(r"[\u4e00-\u9fff]", question)))) == "":
                        continue
                    overlap = self._char_overlap(question, text)
                    best = max(best, overlap)
        for obs in observations:
            if obs.get("tool") in ("get_vehicle_status", "get_maintenance_plan", "lookup_vehicle_spec") \
                    and obs.get("ok"):
                return True
        return best >= self.overlap_threshold

    @staticmethod
    def _char_overlap(a: str, b: str) -> float:
        """问题关键字符被证据覆盖的比例（去掉常见停用字，只算实词字符）。"""
        stop = set("的了吗呢么怎如何是有什么可以请问一下我你他这那个哪些为对能会要怎样")
        chars = {c for c in a if "\u4e00" <= c <= "\u9fff" and c not in stop}
        if not chars:
            chars = {c for c in a if "\u4e00" <= c <= "\u9fff"}
        if not chars:
            return 1.0
        hit = sum(1 for c in chars if c in b)
        return hit / len(chars)

    def _normalize_query(self, question: str) -> str:
        """口语 → 手册术语的轻量改写（真实项目这一步交给 LLM 做 query rewrite）。"""
        if not self.rewrite_enabled:
            return question
        mapping = {
            "靠背太热": "座椅加热", "靠背发烫": "座椅加热", "屁股热": "座椅加热",
            "怎么关": "关闭", "打不开": "无法开启", "亮黄灯": "警告灯",
            "空调不凉": "空调制冷", "没电了": "电量低", "刹车": "制动",
        }
        query = question
        for k, v in mapping.items():
            if k in query:
                query = query.replace(k, v)
        return query

    def _compose(self, question: str, observations: List[Dict]) -> str:
        evidence_texts: List[str] = []
        extra: List[str] = []
        for obs in observations:
            data = obs.get("data")
            if obs.get("tool") not in ("search_manual", None) and isinstance(data, dict):
                extra.append(self._render_structured(data))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("text"):
                        evidence_texts.append(item["text"])
        if not evidence_texts and not extra:
            return "无答案"
        if not evidence_texts:
            return "；".join(extra)[:200]
        # 抽取式答案：选出与问题重叠最高、且信息量足够的一句
        best_sent, best_score = "", -1.0
        for text in evidence_texts:
            for sent in re.split(r"[。！\n]", text):
                sent = sent.strip()
                if len(sent) < 6:
                    continue
                score = self._char_overlap(question, sent)
                if score > best_score:
                    best_sent, best_score = sent, score
        if not best_sent or best_score < self.overlap_threshold:
            return "无答案"
        prefix = "；".join(extra[:2])
        return f"{prefix}。{best_sent}。" if prefix else f"{best_sent}。"

    @staticmethod
    def _render_structured(data: Dict[str, Any]) -> str:
        """把工具返回的结构化数据渲染成自然语言，避免把 JSON 原样塞进答案。"""
        parts = []
        for key, value in data.items():
            if value is None:
                continue
            if isinstance(value, list):
                value = "、".join(str(v) for v in value)
            elif isinstance(value, dict):
                value = "、".join(f"{k}{v}" for k, v in value.items())
            parts.append(f"{key}：{value}")
        return "；".join(parts)

    # -- 消息解析 --
    @staticmethod
    def _last_user(messages: Sequence[Dict[str, Any]]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return m.get("content") or ""
        return ""

    @staticmethod
    def _observations(messages: Sequence[Dict[str, Any]]) -> List[Dict]:
        out = []
        for m in messages:
            if m.get("role") == "tool":
                try:
                    payload = json.loads(m.get("content") or "{}")
                except json.JSONDecodeError:
                    continue
                payload.setdefault("tool", m.get("name"))
                out.append(payload)
        return out

    @staticmethod
    def _has_reflected(messages: Sequence[Dict[str, Any]]) -> bool:
        for m in messages:
            if m.get("role") == "assistant" and "__reflect__" in json.dumps(m, ensure_ascii=False):
                return True
        return False
