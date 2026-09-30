# -*- coding: utf-8 -*-
"""智能座舱 Agent：把 RAG 检索升级为可规划、可调用工具、可自我校验的智能体。

模块导航
--------
- `agent.kb`          知识库与检索（BM25 自实现 + 可选 m3e/FAISS 向量路，RRF 融合）
- `agent.tools`       工具注册表（OpenAI Function Calling schema / 写操作确认 / 去重）
- `agent.llm`         LLM 后端（OpenAI 兼容 vLLM / 脚本回放 / 离线规则规划器）
- `agent.memory`      多轮记忆、车辆档案、指代消解
- `agent.reflection`  引用校验、证据充分性、拒答决策
- `agent.graph`       编排状态机（原生实现 + LangGraph 适配器）
- `agent.eval_agent`  在 103 题测试集上评测 Agent 的编排与检索行为
"""

from agent.graph import AgentConfig, AgentGraph, build_langgraph_app
from agent.kb import Chunk, Evidence, KnowledgeBase
from agent.llm import LLMBackend, OpenAICompatibleLLM, RuleBasedPlannerLLM, ScriptedLLM
from agent.memory import ConversationMemory, VehicleProfile
from agent.reflection import Reflector
from agent.tools import ToolRegistry, build_default_registry

__all__ = [
    "AgentConfig", "AgentGraph", "build_langgraph_app",
    "Chunk", "Evidence", "KnowledgeBase",
    "LLMBackend", "OpenAICompatibleLLM", "RuleBasedPlannerLLM", "ScriptedLLM",
    "ConversationMemory", "VehicleProfile",
    "Reflector",
    "ToolRegistry", "build_default_registry",
]
