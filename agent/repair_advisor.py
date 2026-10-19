# -*- coding: utf-8 -*-
"""维修顾问：MCP 的**第二个消费者**（只读）。

为什么单独写一个 Agent
----------------------
第二十节里 MCP 只有一个客户端（座舱助手），所以"供多个 Agent 复用"只是潜力。
这个模块是那个"第二方"：

- 连的是**同一个** MCP Server 进程外接口，但用只读 scope token 接入；
- `tools/list` 只返回被授权的工具，`create_service_order` / `reschedule_order` /
  `cancel_order` 根本不在列表里；
- 即使它硬编码去调写操作，也会**在服务端**被拒（`-32001`），而不是靠客户端自觉。

于是 MCP 的价值变成可验证的：**同一份工具定义，两种权限，两个 Agent**。

它复用座舱助手同一套 `AgentGraph`（离线规则规划器），所以差别只在"工具从哪来、能调什么"。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

from agent.graph import AgentConfig, AgentGraph
from agent.llm import RuleBasedPlannerLLM
from agent.mcp.client import McpClient
from agent.mcp.registry import McpToolRegistry
from agent.mcp.scope import issue_scope_token
from agent.tools import ToolRegistry

# 只读授权：能查手册、能看保养计划、能查自己的工单、能读车况；**没有任何写操作**。
READONLY_TOOLS: Tuple[str, ...] = (
    "search_manual", "get_maintenance_plan", "get_my_orders", "get_vehicle_status",
    "get_telemetry",
)
SUBJECT = "repair-advisor"
TOKEN_TTL_S = 3600


class RepairAdvisor:
    """只读售后 Agent：同一个编排层，工具来自 MCP（由 scope 限定）。"""

    def __init__(self, client: McpClient, user_id: str = "repair-user"):
        self.client = client
        self.registry = McpToolRegistry(client, name="repair_advisor", user_id=user_id)
        self.tools: List[str] = self.registry.names()

    def ask(self, question: str) -> Dict[str, Any]:
        # `cast` 只是给 mypy 看的：McpToolRegistry 与 ToolRegistry **同形**（names/specs/call/
        # subset/telemetry），编排层不需要知道工具在哪个进程——这正是 MCP 适配层成立的证据。
        graph = AgentGraph(RuleBasedPlannerLLM(), cast(ToolRegistry, self.registry),
                           config=AgentConfig(), agent_name="repair_advisor")
        state = graph.run(question)
        return {
            "question": question,
            "status": state.status,
            "answer": state.answer,
            "citations": list(state.citations),
            "tools_used": [t["tool"] for t in state.tool_calls],
        }


def open_readonly_advisor(secret: str, *, kb_path: Optional[str] = None,
                          timeout: float = 30.0,
                          ttl_s: int = TOKEN_TTL_S) -> Tuple[McpClient, RepairAdvisor]:
    """签发只读 token → 起一个带 scope 的 MCP 客户端 → 包成维修顾问。"""
    token = issue_scope_token(SUBJECT, READONLY_TOOLS, secret, ttl_s=ttl_s)
    args: List[str] = ["--scope-secret", secret]
    if kb_path:
        args += ["--kb", kb_path]
    client = McpClient(args=args, timeout=timeout, scope_token=token).start()
    return client, RepairAdvisor(client)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.repair_advisor",
                                     description="维修顾问（只读 MCP 消费者）")
    parser.add_argument("--secret", default=os.environ.get("MCP_SCOPE_SECRET", ""),
                        help="与 MCP Server 共享的 scope 密钥（也可用 MCP_SCOPE_SECRET）")
    parser.add_argument("--kb", default=None, help="知识库路径（缺省与 server 一致）")
    parser.add_argument("question", nargs="*", help="要问的问题")
    args = parser.parse_args(argv)

    if not args.secret:
        print("需要 --secret（或 MCP_SCOPE_SECRET）：MCP Server 必须用同一个密钥启动",
              file=sys.stderr)
        return 2

    client, advisor = open_readonly_advisor(args.secret, kb_path=args.kb)
    try:
        print(f"授权工具（服务端过滤后）：{advisor.tools}")
        question = " ".join(args.question) or "座椅加热怎么关闭"
        result = advisor.ask(question)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        if result["citations"]:
            print(f"出处: {' '.join(result['citations'][:4])}")
        print(f"用到工具: {result['tools_used']}")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
