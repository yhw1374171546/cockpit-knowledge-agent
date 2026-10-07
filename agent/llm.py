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
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

NO_ANSWER = "无答案"


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    id: str = "call_0"
    repaired: bool = False          # 参数是否经过「解析修复」
    raw_arguments: str = ""         # 模型原始输出，用于失败归因与回归样本
    idempotency_key: str = ""       # 写操作幂等键（重试不会重复下单）
    duration_ms: float = 0.0        # 并行执行时用于统计

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

    @property
    def repaired_calls(self) -> int:
        return sum(1 for c in self.tool_calls if c.repaired)


@dataclass
class StreamChunk:
    """流式增量：用于测量首字延迟（TTFT）与边生成边展示。"""
    delta: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    done: bool = False
    elapsed_ms: float = 0.0


def estimate_tokens(text: str) -> int:
    """无 tokenizer 时的保守估算：中文≈1.6 字/token，英文≈4 字符/token。"""
    text = text or ""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, int(cjk / 1.6 + other / 4.0))


class LLMBackend:
    name = "base"

    def chat(self, messages: Sequence[Dict[str, Any]], tools: Optional[List[Dict]] = None,
             temperature: float = 0.0, max_tokens: int = 1024,
             guided_json: Optional[Dict] = None) -> LLMResponse:
        raise NotImplementedError

    def stream_chat(self, messages: Sequence[Dict[str, Any]], tools: Optional[List[Dict]] = None,
                    temperature: float = 0.0, max_tokens: int = 1024
                    ) -> Iterator[StreamChunk]:
        """默认实现：非流式后端退化为"一次性吐完"，仍然可用（TTFT = 总耗时）。"""
        resp = self.chat(messages, tools=tools, temperature=temperature, max_tokens=max_tokens)
        yield StreamChunk(delta=resp.content or "", tool_calls=resp.tool_calls, done=True)


# ── 工具调用参数解析与修复 ────────────────────────────────────────────


class ToolCallRepair:
    """把模型输出的"几乎合法"的参数修成合法 JSON。

    真实模型在 Function Calling 上最常见的四类脏输出：
    1. 用代码块包起来：```json {...} ```
    2. 单引号 / Python 字面量：{'query': '座椅加热'}
    3. 尾随逗号：{"query": "座椅加热",}
    4. 括号不闭合 / 截断
    另有工具名幻觉（search_manuals / searchManual），用编辑距离就近纠正。
    """

    def __init__(self, known_tools: Optional[Sequence[str]] = None):
        self.known_tools = list(known_tools or [])
        self.parse_failures = 0
        self.repairs = 0
        self.name_fixes = 0

    # -- 工具名纠正 --
    def fix_name(self, name: str) -> str:
        if not name or name in self.known_tools:
            return name
        best, best_score = name, 0.0
        for cand in self.known_tools:
            score = _similarity(name.lower(), cand.lower())
            if score > best_score:
                best, best_score = cand, score
        if best_score >= 0.6:
            self.name_fixes += 1
            return best
        return name

    # -- 参数解析 --
    def parse(self, raw: Any) -> Tuple[Dict[str, Any], bool]:
        """返回 (参数字典, 是否经过修复)。解析彻底失败则返回 ({}, True) 由上层重试。"""
        if isinstance(raw, dict):
            return raw, False
        if raw is None:
            return {}, False
        text = str(raw).strip()
        if not text:
            return {}, False
        try:
            parsed = json.loads(text)
            return (parsed if isinstance(parsed, dict) else {"value": parsed}), False
        except json.JSONDecodeError:
            pass

        fixed = text
        fixed = re.sub(r"^```(?:json)?|```$", "", fixed, flags=re.M).strip()
        fixed = fixed.replace("'", '"').replace("True", "true").replace("False", "false") \
                     .replace("None", "null")
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)          # 去尾随逗号
        fixed = re.sub(r"(\w+)\s*:", r'"\1":', fixed) if not fixed.lstrip().startswith("{") else fixed
        if not fixed.lstrip().startswith("{"):
            fixed = "{" + fixed
        fixed = fixed.rstrip().rstrip(",")
        opens, closes = fixed.count("{"), fixed.count("}")
        fixed += "}" * max(0, opens - closes)
        try:
            parsed = json.loads(fixed)
            if isinstance(parsed, dict):
                self.repairs += 1
                return parsed, True
        except json.JSONDecodeError:
            pass

        # 最后兜底：容忍 `query:座椅加热, top_k:6` 这类非 JSON 的 k:v 写法
        lenient = self._parse_lenient_pairs(text)
        if lenient is not None:
            self.repairs += 1
            return lenient, True

        self.parse_failures += 1
        return {}, True

    @staticmethod
    def _parse_lenient_pairs(text: str) -> Optional[Dict[str, Any]]:
        """把 `k: v, k2: v2` / `k=v` 形式解析成字典；解析不出内容则返回 None。"""
        cleaned = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M)
        cleaned = cleaned.strip().strip("{}").strip()
        if not cleaned:
            return None
        pairs: Dict[str, Any] = {}
        for chunk in re.split(r"[,;\n]", cleaned):
            if not chunk.strip():
                continue
            m = re.match(r"\s*[\"']?([\w\u4e00-\u9fff]+)[\"']?\s*[:=]\s*(.*?)\s*$", chunk)
            if not m:
                return None
            key, value = m.group(1), m.group(2).strip().strip("\"'")
            if re.fullmatch(r"-?\d+", value):
                pairs[key] = int(value)
            elif re.fullmatch(r"-?\d+\.\d+", value):
                pairs[key] = float(value)
            elif value.lower() in ("true", "false"):
                pairs[key] = value.lower() == "true"
            else:
                pairs[key] = value
        return pairs or None

    def stats(self) -> Dict[str, int]:
        return {"repairs": self.repairs, "parse_failures": self.parse_failures,
                "name_fixes": self.name_fixes}


def _similarity(a: str, b: str) -> float:
    """归一化编辑距离相似度（轻量，无需依赖）。"""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


# ── 生产：OpenAI 兼容（vLLM） ────────────────────────────────────────


class OpenAICompatibleLLM(LLMBackend):
    name = "openai-compatible"

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", model: str = "Qwen2_7B",
                 api_key: str = "EMPTY", timeout: int = 120,
                 known_tools: Optional[Sequence[str]] = None, max_repair_retries: int = 1):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.repair = ToolCallRepair(known_tools)
        self.max_repair_retries = max_repair_retries

    def set_known_tools(self, names: Sequence[str]) -> None:
        """把工具清单告诉修复器，用于纠正模型幻觉出来的工具名。"""
        self.repair.known_tools = list(names)

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024,
             guided_json=None) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if guided_json is not None:
            # vLLM 原生约束解码：让模型输出严格符合 JSON Schema，从源头减少解析失败
            payload["guided_json"] = guided_json
            payload.setdefault("guided_decoding_backend", "xgrammar")
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

        return self._parse_body(body)

    # ── 流式：用于首字延迟（TTFT）优化 ──
    def stream_chat(self, messages, tools=None, temperature=0.0, max_tokens=1024,
                    ) -> Iterator[StreamChunk]:
        payload = {"model": self.model, "messages": list(messages),
                   "temperature": temperature, "max_tokens": max_tokens, "stream": True}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = ((chunk.get("choices") or [{}])[0].get("delta") or {})
                    yield StreamChunk(delta=delta.get("content") or "",
                                      done=False,
                                      elapsed_ms=(time.perf_counter() - start) * 1000)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"调用 LLM 流式服务失败：{exc}") from exc
        yield StreamChunk(done=True, elapsed_ms=(time.perf_counter() - start) * 1000)

    def _parse_body(self, body: Dict[str, Any]) -> LLMResponse:
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls: List[ToolCall] = []
        for i, tc in enumerate(message.get("tool_calls") or []):
            fn = tc.get("function") or {}
            raw = fn.get("arguments")
            args, repaired = self.repair.parse(raw)
            name = self.repair.fix_name(fn.get("name", ""))
            calls.append(ToolCall(name, args, tc.get("id", f"call_{i}"),
                                  repaired=repaired, raw_arguments=str(raw)[:200]))
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

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024,
             guided_json=None) -> LLMResponse:
        self.calls.append({"n_messages": len(messages), "tools": [t["function"]["name"] for t in (tools or [])]})
        if not self.responses:
            return LLMResponse(content="（脚本已耗尽）")
        return self.responses.pop(0)


# ── 离线：Function Calling 管线替身 ──────────────────────────────────


class SimulatedToolCallLLM(LLMBackend):
    """模拟"真实模型输出 Function Calling"的后端，用于**离线验证 FC 管线**。

    它按 OpenAI 的原始格式吐出 `tool_calls`（可注入脏数据：代码块包裹、单引号、尾随逗号、
    截断、工具名幻觉），用来验证：
        Guided/修复解析 → 工具名纠正 → 策略裁决 → 并行执行 → 观察回填
    这条链路本身是否健壮。

    ⚠️ 它衡量的是**管线健壮性**，不衡量模型能力；模型能力必须用真实 LLM 跑（--backend openai）。
    """

    name = "simulated-tool-call"

    def __init__(self, script: Sequence[Dict[str, Any]], dirty_ratio: float = 0.0,
                 seed: int = 7):
        """script: [{"tool": "search_manual", "arguments": {...} | "raw_string", "content": "..."}]
        dirty_ratio: 以一定比例把参数故意写脏，测试修复链路。
        """
        self.script = list(script)
        self.dirty_ratio = dirty_ratio
        self.random = random.Random(seed)
        self.turn = 0
        self.repair = ToolCallRepair()

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024,
             guided_json=None) -> LLMResponse:
        self.turn += 1
        if tools and self.repair.known_tools == []:
            self.repair.known_tools = [t["function"]["name"] for t in tools]
        if not self.script:
            return LLMResponse(content=NO_ANSWER)
        item = self.script.pop(0)
        if "content" in item and "tool" not in item:
            return LLMResponse(content=item["content"])

        raw = item.get("raw_string")
        if raw is None:
            args = item.get("arguments") or {}
            raw = json.dumps(args, ensure_ascii=False)
            if self.random.random() < self.dirty_ratio:
                raw = "```json\n" + raw.replace('"', "'") + ",\n```"   # 制造脏输出
        parsed, repaired = self.repair.parse(raw)
        name = self.repair.fix_name(item.get("tool", ""))
        return LLMResponse(tool_calls=[ToolCall(name, parsed, f"call_{self.turn}",
                                                repaired=repaired, raw_arguments=str(raw)[:200])])

    def stream_chat(self, messages, tools=None, temperature=0.0, max_tokens=1024):
        resp = self.chat(messages, tools=tools)
        for piece in (resp.content or "").split("，"):
            yield StreamChunk(delta=piece + "，", done=False)
        yield StreamChunk(done=True, tool_calls=resp.tool_calls)


class SimulatedStreamingLLM(LLMBackend):
    """模拟流式生成：用可配置的 per-token 时延制造真实的"首字延迟 vs 总时长"差异。

    用途：在没有 GPU 的环境里量化**编排层对流式首字延迟的贡献**
    （检索耗时 + 缓存命中 + 首 token 时间），而不是凭空宣称一个 TTFT 数字。
    """

    name = "simulated-streaming"

    def __init__(self, answer: str = "", ms_per_token: float = 4.0, tokens: int = 150):
        self.answer = answer
        self.ms_per_token = ms_per_token
        self.tokens = tokens

    def _sleep(self, ms: float) -> None:
        time.sleep(max(0.0, ms) / 1000.0)

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024,
             guided_json=None) -> LLMResponse:
        self._sleep(self.ms_per_token * self.tokens)          # 非流式：等全部生成完
        return LLMResponse(content=self.answer or "（模拟答案）",
                           usage={"prompt_tokens": 800, "completion_tokens": self.tokens})

    def stream_chat(self, messages, tools=None, temperature=0.0, max_tokens=1024):
        start = time.perf_counter()
        for i in range(self.tokens):
            self._sleep(self.ms_per_token)
            piece = (self.answer or "（模拟答案）")[i % max(1, len(self.answer or "答"))]
            yield StreamChunk(delta=piece, done=False,
                              elapsed_ms=(time.perf_counter() - start) * 1000)
        yield StreamChunk(done=True, elapsed_ms=(time.perf_counter() - start) * 1000)


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
        ("status", ("胎压", "电量", "剩余续航", "还剩", "多少电", "我现在", "我的车",
                    "告警", "故障灯", "剩下", "车况", "还能开", "还能跑")),
    ]

    # 「不该调用工具」的判定：闲聊/元问题/域外问题不该去查手册。
    # 规则基线最初没有这一层，导致过度调用率 100%（见 eval/tool_call_report.md 的对比）。
    NO_TOOL_RULES = [
        ("greeting", ("你好", "您好", "谢谢", "多谢", "辛苦了", "再见", "嗯嗯", "好的", "算了")),
        ("meta", ("你是谁", "你叫什么", "你能做什么", "你会做什么", "你是什么")),
        ("chitchat", ("讲个笑话", "无聊", "陪我聊", "唱首歌", "聊天")),
        # 域外问题：明确不属于车主手册范围 → 按"无答案"处理（不调用工具）
        ("out_of_domain", ("天气", "机票", "股票", "美股", "足球", "新冠", "电影", "新闻",
                           "彩票", "股票", "汇率")),
    ]
    NO_TOOL_REPLY = {
        "out_of_domain": NO_ANSWER,
        "greeting": "您好，我是您的车辆助手，可以帮您查询车主手册、保养计划、车型参数和实时车况。",
        "meta": "我是智能座舱的车辆助手，能检索车主手册、读取车况、给出保养建议并在您确认后预约到店。",
        "chitchat": "我这方面不太擅长，不过车辆使用、保养和故障处置的问题我都能帮您查手册。",
    }

    def __init__(self, overlap_threshold: float = 0.12, rewrite_enabled: bool = True):
        self.overlap_threshold = overlap_threshold
        self.rewrite_enabled = rewrite_enabled
        self.turn = 0
        self.route_history: List[str] = []

    # -- 路由 --
    def route(self, question: str) -> str:
        no_tool = self.no_tool_intent(question)
        if no_tool:
            self.route_history.append(no_tool)
            return no_tool
        for intent, keys in self.INTENT_RULES:
            if any(k in question for k in keys):
                self.route_history.append(intent)
                return intent
        self.route_history.append("manual_qa")
        return "manual_qa"

    def no_tool_intent(self, question: str) -> Optional[str]:
        """判断是否属于「不该调用工具」的请求（闲聊/元问题/域外）。"""
        q = (question or "").strip()
        if not q:
            return "greeting"
        for intent, keys in self.NO_TOOL_RULES:
            if any(k in q for k in keys):
                return intent
        return None

    # -- 决策 --
    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024,
             guided_json=None) -> LLMResponse:
        self.turn += 1
        question = self._last_user(messages)
        observations = self._observations(messages)
        tool_names = {t["function"]["name"] for t in (tools or [])}

        # 第零步：闲聊/元问题/域外问题 → 不调用工具（避免无谓的检索开销与误答）
        if not observations:
            no_tool = self.no_tool_intent(question)
            if no_tool:
                self.route_history.append(no_tool)
                return LLMResponse(content=self.NO_TOOL_REPLY.get(no_tool, NO_ANSWER))

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

    # 结构化数据的展示优先级：告警类信息优先，避免把一大串状态原样塞进答案
    RENDER_PRIORITY = ("胎压告警", "告警灯", "剩余电量_%", "续航_km", "里程_km", "下次保养_km",
                       "本次建议项目", "下次保养", "剩余里程_km", "车型", "纯电续航",
                       "电池容量", "快充时间", "轮胎规格", "胎压_kPa", "车门状态", "车窗状态")
    RENDER_MAX_CHARS = 140

    @classmethod
    def _render_structured(cls, data: Dict[str, Any]) -> str:
        """把工具返回的结构化数据渲染成自然语言（告警优先 + 截断），避免 JSON 原样入答案。"""
        ordered = [k for k in cls.RENDER_PRIORITY if k in data]
        ordered += [k for k in data if k not in ordered and k != "vin"]
        parts: List[str] = []
        used = 0
        for key in ordered:
            value = data.get(key)
            if value is None or value == "" or value == []:
                continue
            if isinstance(value, list):
                value = "、".join(str(v) for v in value)
            elif isinstance(value, dict):
                value = "、".join(f"{k}{v}" for k, v in value.items())
            piece = f"{key}：{value}"
            if used + len(piece) > cls.RENDER_MAX_CHARS:
                break
            parts.append(piece)
            used += len(piece) + 1
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
