# -*- coding: utf-8 -*-
"""MCP 跨进程冒烟：一条命令演示「协议 → 工具发现 → 白名单 → 写操作确认 → trace → 编排层」。

    python eval/mcp_smoke.py                            # 内置迷你语料（秒级）
    python eval/mcp_smoke.py --kb kb/chunks.jsonl       # 完整语料（首次调用才加载，约 15 s）

为什么这个脚本要单独存在
------------------------
单测证明"协议没写错"，这个脚本证明"**把工具搬到另一个进程之后，业务行为没变**"。
面试时一条命令能演示四件事，且不需要 GPU、不需要真模型：

1. 握手与工具发现是毫秒级（知识库在第一次检索时才加载，不阻塞 tools/list）；
2. 车控专家的白名单让写操作**在客户端就被拦住**，请求根本没到服务端；
3. 写操作确认是**跨进程两步**：先拿到 action_key，确认后再执行；
4. trace_id 会透传到服务端并随结果回来，客户端耗时与服务端 handler 耗时分开记。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.console import harden_stdout  # noqa: E402

harden_stdout()   # 中文 Windows 的 GBK 控制台编不出 [OK]，否则会崩在最后一行

from agent.graph import AgentConfig, AgentGraph  # noqa: E402
from agent.kb import FIXTURE_KB, KnowledgeBase  # noqa: E402
from agent.llm import RuleBasedPlannerLLM  # noqa: E402
from agent.mcp.client import McpClient  # noqa: E402
from agent.mcp.registry import McpToolRegistry  # noqa: E402
from agent.tools import ToolRegistry  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRITE_ARGS = {"item": "更换机油机滤", "preferred_date": "2025-06-18"}


def main() -> int:
    parser = argparse.ArgumentParser(description="MCP 跨进程冒烟")
    parser.add_argument("--kb", default=None, help="知识库路径，缺省用内置迷你语料")
    args = parser.parse_args()
    kb_path = args.kb or (FIXTURE_KB if os.path.exists(FIXTURE_KB) else None)

    checks: List[Dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        flag = "[OK]" if ok else "[FAIL]"
        print(f"  {flag} {name}" + (f" —— {detail}" if detail else ""))
        return ok

    print("=" * 74)
    print("MCP 跨进程冒烟（手写 JSON-RPC over stdio）")
    print("=" * 74)
    print(f"知识库：{kb_path or '<按环境解析>'}")

    started = time.perf_counter()
    client = McpClient(args=["--kb", kb_path] if kb_path else [])
    client.start()
    handshake_ms = (time.perf_counter() - started) * 1000

    try:
        # 1. 握手
        record("initialize 握手", bool(client.server_info.get("name")),
               f"{client.server_info.get('name')} v{client.server_info.get('version')} "
               f"protocol={client.protocol_version}，{handshake_ms:.0f} ms")

        # 2. 工具发现 + schema 一致性
        tools = client.list_tools()
        names = [t["name"] for t in tools]
        local = ToolRegistry(KnowledgeBase.load(kb_path))
        local_specs = {s["function"]["name"]: s["function"] for s in local.specs()}
        mismatched = [t["name"] for t in tools
                      if t["inputSchema"] != local_specs.get(t["name"], {}).get("parameters")]
        record("tools/list 与本地 Function Calling schema 一致",
               not mismatched and len(tools) == len(local_specs),
               f"{len(tools)} 个工具：{', '.join(names)}")

        # 3. 一次真实检索（首次调用才加载知识库）
        registry = McpToolRegistry(client)
        t0 = time.perf_counter()
        hit = registry.call("search_manual", {"query": "座椅加热怎么关闭", "top_k": 3})
        call_ms = (time.perf_counter() - t0) * 1000
        citation = (hit.data or [{}])[0].get("citation") if hit.ok else None
        record("search_manual 跨进程检索", bool(hit.ok and citation),
               f"{call_ms:.0f} ms，首条出处 {citation}，服务端 handler "
               f"{(hit.meta or {}).get('server_ms')} ms")

        # 4. 白名单：本地拦截，服务端收不到
        scoped = registry.subset(["search_manual", "get_vehicle_status"], "vehicle_control")
        before = client.request_count
        blocked = scoped.call("create_service_order", dict(WRITE_ARGS))
        leaked = client.request_count != before
        record("车控专家看不到写工具，且调用被客户端拦下", (not blocked.ok) and not leaked,
               f"specs={[s['function']['name'] for s in scoped.specs()]}；"
               f"服务端请求数 {before} → {client.request_count}")

        # 5. 写操作两步确认
        pending = registry.call("create_service_order", dict(WRITE_ARGS))
        action_key = (pending.meta or {}).get("action_key")
        confirmed_ok = False
        done: Dict[str, Any] = {}
        if action_key:
            registry.confirm(action_key)
            confirmed_ok = action_key in registry.confirmed_actions
            result = registry.call("create_service_order", dict(WRITE_ARGS))
            done = result.data if result.ok and isinstance(result.data, dict) else {}
        record("写操作：未确认 → 拿 action_key → 确认 → 才执行",
               bool(action_key) and confirmed_ok and "预约单号" in done,
               f"action_key={action_key}，预约单号={done.get('预约单号', '-')}")

        # 6. trace 透传
        payload = client.call_tool("get_vehicle_status", {}, trace_id="demo-trace-001")
        meta = payload.get("_meta") or {}
        record("trace_id 透传到服务端并随结果返回",
               meta.get("trace_id") == "demo-trace-001",
               f"trace_id={meta.get('trace_id')}，server_ms={meta.get('server_ms')}")

        # 7. 编排层端到端（AgentGraph 不知道工具在另一个进程）
        graph = AgentGraph(RuleBasedPlannerLLM(), registry, config=AgentConfig())
        state = graph.run("座椅加热怎么关闭")
        record("AgentGraph 用远端工具答题",
               state.status == "answered" and bool(state.citations),
               f"status={state.status}，出处 {len(state.citations)} 条，"
               f"工具调用 {len([t for t in registry.trace if t.get('tool') == 'search_manual'])} 次")
    finally:
        client.close()

    passed = sum(1 for c in checks if c["ok"])
    print("-" * 74)
    print(f"合计 {passed}/{len(checks)} 项通过")
    report = os.path.join(ROOT, "eval", "mcp_smoke_report.json")
    with open(report, "w", encoding="utf-8") as fh:
        json.dump({"checks": checks, "passed": passed, "total": len(checks)},
                  fh, ensure_ascii=False, indent=2)
    print(f"报告：{report}")
    if passed != len(checks):
        print("[FAIL] 存在未通过项")
        return 1
    print("[OK] 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
