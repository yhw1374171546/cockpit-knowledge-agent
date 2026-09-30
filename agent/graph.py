# -*- coding: utf-8 -*-
"""Agent 编排层：Router → Planner → ToolExecutor → Reflector → Answer。

原生实现（不依赖 LangGraph，零额外依赖即可运行），节点命名与状态字段与 LangGraph
的 StateGraph 一一对应；若环境安装了 `langgraph`，`build_langgraph_app()` 会用真正的
StateGraph 重建同一套节点，便于接入 LangGraph 生态（检查点、持久化、可视化）。

治理策略（防死循环 / 防乱调用）
------------------------------
1. `max_steps`：单轮问题最多交互步数；
2. **重复调用抑制**：相同工具 + 相同参数第二次调用直接命中缓存并标记 repeated（工具层实现）；
3. **无进展检测**：若某一步既没有新证据、也没有新工具，则立即终止（输出已知信息或拒答）；
4. **重检索预算**：证据不足时允许一次 query 改写重试，用尽后拒答；
5. **写操作确认**：写操作未获车主确认一律不执行，返回 needs_confirmation 状态。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agent.llm import LLMBackend, RuleBasedPlannerLLM, ToolCall
from agent.memory import ConversationMemory
from agent.reflection import NO_ANSWER, Reflector
from agent.tools import ToolRegistry

SYSTEM_PROMPT = """你是智能座舱的车辆助手，服务于车主。请遵守以下规则：
1. 涉及车辆功能、故障处置、保养要求的问题，必须先调用 search_manual 检索车主手册，只能依据检索到的原文回答。
2. 如果检索结果不足以回答，必须回答「无答案」，严禁编造。
3. 涉及「我的车现在怎么了」这类问题，请调用 get_vehicle_status 获取实时车况，并与手册知识结合回答。
4. 保养建议调用 get_maintenance_plan；车型参数调用 lookup_vehicle_spec。
5. 任何写操作（预约到店等）必须先向车主口头确认，获得同意后才调用工具。
6. 回答要简洁、专业、中文，并给出处。"""

SYNONYMS = {
    "靠背太热": "座椅加热", "靠背发烫": "座椅加热", "屁股热": "座椅加热",
    "怎么关": "关闭", "怎么打开": "开启", "打不开": "无法开启",
    "空调不凉": "空调制冷", "没电了": "电量低", "刹车": "制动",
    "亮黄灯": "警告灯", "胎压灯": "胎压报警",
}


@dataclass
class AgentConfig:
    max_steps: int = 6
    max_retrieval_retries: int = 1
    top_k: int = 6
    refuse_on_insufficient: bool = True
    enable_reflection: bool = True
    # 生成前的硬门控：问题实词被最佳证据覆盖的比例低于阈值 → 直接拒答（可标定）
    evidence_gate: bool = True
    evidence_overlap_threshold: float = 0.45


@dataclass
class AgentState:
    question: str
    resolved_question: str = ""
    route: str = "unknown"
    status: str = "running"          # answered | refused | needs_confirmation | max_steps | error
    answer: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    retries: int = 0
    reflection: Optional[Dict[str, Any]] = None
    gate_coverage: Optional[float] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question, "resolved_question": self.resolved_question,
            "route": self.route, "status": self.status, "answer": self.answer,
            "citations": self.citations,
            "evidence": [{"chunk_id": e.get("chunk_id"), "citation": e.get("citation"),
                          "score": e.get("score"), "page": e.get("page")} for e in self.evidence],
            "tool_calls": self.tool_calls, "steps": self.steps, "retries": self.retries,
            "gate_coverage": self.gate_coverage,
            "reflection": self.reflection, "latency_ms": self.latency_ms, "trace": self.trace,
        }


class AgentGraph:
    # 与 LangGraph 节点同名的阶段标识，便于对照
    NODES = ("router", "planner", "tool_executor", "reflector", "answer", "refuse")

    def __init__(self, llm: LLMBackend, registry: ToolRegistry,
                 reflector: Optional[Reflector] = None,
                 memory: Optional[ConversationMemory] = None,
                 config: Optional[AgentConfig] = None):
        self.llm = llm
        self.registry = registry
        self.reflector = reflector or Reflector()
        self.memory = memory or ConversationMemory()
        self.config = config or AgentConfig()

    # ── 对外入口 ──
    def run(self, question: str, memory: Optional[ConversationMemory] = None) -> AgentState:
        mem = memory or self.memory
        start = time.perf_counter()
        resolved = mem.resolve(question)
        state = AgentState(question=question, resolved_question=resolved)

        mem.add_user(question)
        state.route = self._route(resolved)
        state.trace.append({"node": "router", "route": state.route})

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + mem.system_context()},
            {"role": "user", "content": resolved},
        ]

        while state.steps < self.config.max_steps:
            state.steps += 1
            t0 = time.perf_counter()
            response = self.llm.chat(messages, tools=self.registry.specs(),
                                     temperature=0.0, max_tokens=1024)
            state.latency_ms.setdefault("llm", 0.0)
            state.latency_ms["llm"] += (time.perf_counter() - t0) * 1000

            # ── 无工具调用：先过硬门控，再产出答案 ──
            if not response.wants_tool:
                answer = (response.content or "").strip()
                if self.config.evidence_gate and answer and answer != NO_ANSWER:
                    texts = [e.get("text", "") for e in state.evidence]
                    # 门控同时看「原始问法」与「术语改写后的问法」，取较大覆盖率：
                    # 车主说「靠背太热」而手册写「座椅加热」，只按原始问法判会误杀。
                    gate_queries = [state.resolved_question, self._reformulate(state.resolved_question)]
                    coverage = max((self.reflector.question_coverage(q, t)
                                    for q in gate_queries for t in texts), default=0.0)
                    if coverage < self.config.evidence_overlap_threshold:
                        state.gate_coverage = round(coverage, 3)
                        state.trace.append({"node": "refuse", "reason": "evidence_gate",
                                            "coverage": round(coverage, 3),
                                            "threshold": self.config.evidence_overlap_threshold})
                        return self._finalize(state, NO_ANSWER, mem, start)
                    state.gate_coverage = round(coverage, 3)
                state.trace.append({"node": "answer", "step": state.steps,
                                    "preview": answer[:40]})
                return self._finalize(state, answer, mem, start)

            # ── 反思伪调用（由规划器发出的自检信号） ──
            reflect_call = next((c for c in response.tool_calls if c.name == "__reflect__"), None)
            if reflect_call is not None:
                messages.append({"role": "assistant", "content": None,
                                 "tool_calls": [reflect_call.to_openai()]})
                verdict = reflect_call.arguments.get("verdict", "sufficient")
                messages.append({"role": "tool", "name": "__reflect__",
                                 "content": json.dumps({"ok": True, "verdict": verdict},
                                                       ensure_ascii=False)})
                state.trace.append({"node": "reflector", "step": state.steps, "verdict": verdict})
                if verdict == "sufficient":
                    continue
                if state.retries >= self.config.max_retrieval_retries:
                    return self._finalize(state, NO_ANSWER, mem, start)
                state.retries += 1
                rewritten = self._reformulate(resolved)
                state.trace.append({"node": "tool_executor", "step": state.steps,
                                    "retry_query": rewritten})
                messages.append({"role": "user",
                                 "content": f"检索证据不足，请用改写后的查询重新检索：{rewritten}"})
                continue

            # ── 正常工具调用 ──
            messages.append({"role": "assistant", "content": None,
                             "tool_calls": [c.to_openai() for c in response.tool_calls]})
            new_evidence = 0
            progressed = False
            for call in response.tool_calls:
                t1 = time.perf_counter()
                result = self.registry.call(call.name, call.arguments or {})
                state.latency_ms.setdefault("tools", 0.0)
                state.latency_ms["tools"] += (time.perf_counter() - t1) * 1000
                state.tool_calls.append({"step": state.steps, "tool": call.name,
                                         "arguments": call.arguments, "ok": result.ok,
                                         "repeated": result.repeated, "error": result.error})
                state.trace.append({"node": "tool_executor", "step": state.steps,
                                    "tool": call.name, "ok": result.ok,
                                    "repeated": result.repeated,
                                    "latency_ms": round(result.latency_ms, 2)})

                if result.meta.get("needs_confirmation"):
                    state.status = "needs_confirmation"
                    state.answer = result.hint
                    mem.add_assistant(state.answer, [call.name])
                    state.latency_ms["total"] = (time.perf_counter() - start) * 1000
                    return state

                messages.append({"role": "tool", "name": call.name,
                                 "content": result.to_observation()})

                if result.ok and not result.repeated:
                    progressed = True
                    payload = result.data if isinstance(result.data, list) else []
                    for item in payload:
                        if isinstance(item, dict) and item.get("text"):
                            if not any(e.get("chunk_id") == item.get("chunk_id")
                                       for e in state.evidence):
                                state.evidence.append(item)
                                new_evidence += 1

            # ── 无进展检测：没有新证据且没有新工具成功执行 → 终止 ──
            if not progressed and state.steps > 1:
                state.trace.append({"node": "planner", "step": state.steps,
                                    "stop_reason": "no_progress"})
                if state.evidence:
                    messages.append({"role": "user",
                                     "content": "没有更多可用信息了，请基于已有证据作答，证据不足就回答「无答案」。"})
                    response2 = self.llm.chat(messages, tools=None, temperature=0.0,
                                              max_tokens=512)
                    answer = (response2.content or NO_ANSWER).strip()
                else:
                    answer = NO_ANSWER
                return self._finalize(state, answer, mem, start)

            if new_evidence == 0 and state.steps >= self.config.max_steps:
                break

        state.status = "max_steps"
        return self._finalize(state, self._best_effort(state), mem, start)

    # ── 节点实现 ──
    def _route(self, question: str) -> str:
        if isinstance(self.llm, RuleBasedPlannerLLM):
            return self.llm.route(question)
        for intent, keys in RuleBasedPlannerLLM.INTENT_RULES:
            if any(k in question for k in keys):
                return intent
        return "manual_qa"

    def _reformulate(self, query: str) -> str:
        """证据不足时的查询改写策略（去疑问词 + 口语→手册术语）。"""
        q = query
        for k, v in SYNONYMS.items():
            q = q.replace(k, v)
        q = re.sub(r"(请问|帮我|怎么|如何|是什么|为什么|呢|吗|？|\?)", " ", q).strip()
        q = re.sub(r"\s+", " ", q)
        return q or query

    def _best_effort(self, state: AgentState) -> str:
        if not state.evidence:
            return NO_ANSWER
        return "；".join(e.get("text", "")[:40] for e in state.evidence[:2]) or NO_ANSWER

    def _finalize(self, state: AgentState, answer: str, mem: ConversationMemory,
                  start: float) -> AgentState:
        answer = (answer or "").strip()

        if self.config.enable_reflection:
            t0 = time.perf_counter()
            reflection = self.reflector.verify(answer, state.evidence, state.resolved_question)
            state.latency_ms.setdefault("reflection", 0.0)
            state.latency_ms["reflection"] += (time.perf_counter() - t0) * 1000
            state.reflection = reflection.to_dict()
            if self.config.refuse_on_insufficient and reflection.verdict == "ungrounded":
                state.trace.append({"node": "refuse", "reason": "ungrounded",
                                    "ratio": round(reflection.grounded_ratio, 3)})
                state.answer = NO_ANSWER
                state.status = "refused"
                state.citations = []
                mem.add_assistant(state.answer)
                state.latency_ms["total"] = (time.perf_counter() - start) * 1000
                return state

        if not state.evidence and answer == NO_ANSWER:
            state.status = "refused"
        elif state.status not in ("needs_confirmation",):
            state.status = "answered"

        state.citations = [e.get("citation", "") for e in state.evidence[:6]]
        state.answer = answer
        if answer and answer != NO_ANSWER:
            topic = state.resolved_question[:20]
            mem.set_topic(topic)
        mem.add_assistant(answer, [t["tool"] for t in state.tool_calls])
        state.latency_ms["total"] = (time.perf_counter() - start) * 1000
        return state


# ── LangGraph 适配器 ─────────────────────────────────────────────────


def build_langgraph_app(agent: AgentGraph):
    """用真正的 LangGraph StateGraph 重建同一套节点（需 pip install langgraph）。

    便于接入 LangGraph 生态：检查点（checkpointer）、人工介入（interrupt）、
    以及图结构可视化。节点逻辑复用 AgentGraph 的同名方法，保持行为一致。
    """
    try:
        from langgraph.graph import END, StateGraph  # type: ignore
    except Exception:
        return None

    from typing import TypedDict  # noqa: WPS433

    class GraphState(TypedDict, total=False):
        question: str
        resolved: str
        route: str
        evidence: List[Dict[str, Any]]
        steps: int
        answer: str
        status: str

    def router(state: GraphState) -> GraphState:
        return {"route": agent._route(state.get("resolved") or state["question"]), "steps": 0}

    def tools(state: GraphState) -> GraphState:
        hits = agent.registry.call("search_manual",
                                   {"query": state.get("resolved") or state["question"],
                                    "top_k": agent.config.top_k})
        evidence = hits.data if hits.ok and isinstance(hits.data, list) else []
        return {"evidence": evidence, "steps": state.get("steps", 0) + 1}

    def answer(state: GraphState) -> GraphState:
        evidence = state.get("evidence") or []
        if not evidence:
            return {"answer": NO_ANSWER, "status": "refused"}
        return {"answer": evidence[0].get("text", "")[:200], "status": "answered"}

    def reflect(state: GraphState) -> GraphState:
        result = agent.reflector.verify(state.get("answer", ""), state.get("evidence") or [],
                                        state.get("resolved") or state.get("question", ""))
        if result.verdict == "ungrounded":
            return {"answer": NO_ANSWER, "status": "refused"}
        return {"status": state.get("status", "answered")}

    graph = StateGraph(GraphState)
    graph.add_node("router", router)
    graph.add_node("tool_executor", tools)
    graph.add_node("answer", answer)
    graph.add_node("reflector", reflect)
    graph.set_entry_point("router")
    graph.add_edge("router", "tool_executor")
    graph.add_edge("tool_executor", "answer")
    graph.add_edge("answer", "reflector")
    graph.add_edge("reflector", END)
    return graph.compile()


if __name__ == "__main__":
    from agent.tools import build_default_registry

    reg = build_default_registry()
    graph = AgentGraph(RuleBasedPlannerLLM(), reg)
    for q in ["靠背太热怎么办", "我这台车该保养了吗", "中国足球的队长是谁", "胎压报警了怎么办"]:
        st = graph.run(q)
        print(f"\nQ: {q}\n  status={st.status} route={st.route} steps={st.steps} "
              f"retries={st.retries} total={st.latency_ms.get('total', 0):.1f}ms")
        print(f"  A: {st.answer}")
        print(f"  cites: {st.citations[:2]}")
