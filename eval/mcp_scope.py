# -*- coding: utf-8 -*-
"""MCP 第二消费者 + 服务端 scope：证明"复用"和"权限"都不是口号。

    python eval/mcp_scope.py

演示三件事
----------
1. **服务端鉴权**：Server 用 `--scope-secret` 起来后，不带 token 的连接直接握手失败；
2. **工具级授权**：同一份工具定义，全量 token 看到 8 个，只读 token 只看到 4 个；
   只读客户端**硬调**写操作也会被服务端拒（`-32001`），不是靠客户端自觉；
3. **第二个消费者**：维修顾问（`agent/repair_advisor.py`）用只读 token 接同一套工具，
   能答出带页出处的手册问题——它和座舱助手共用同一个 `AgentGraph` 与同一份 schema。

如实说明：MCP 的 stdio 传输是**单客户端**的，所以这里两个 Agent 各自拉起一个 server 进程，
共用的是**同一份工具实现与定义**（同一入口命令、同一 `ToolRegistry`），
而不是"一个进程同时服务两个客户端"（那需要 HTTP/SSE 传输，未实现）。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.console import harden_stdout  # noqa: E402

harden_stdout()

from agent.kb import FIXTURE_KB, KnowledgeBase  # noqa: E402
from agent.mcp.client import McpClient  # noqa: E402
from agent.mcp.protocol import FORBIDDEN, McpError  # noqa: E402
from agent.mcp.registry import McpToolRegistry  # noqa: E402
from agent.mcp.scope import issue_scope_token  # noqa: E402
from agent.repair_advisor import (  # noqa: E402
    READONLY_TOOLS,
    RepairAdvisor,
    open_readonly_advisor,
)
from agent.tools import ToolRegistry  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = FIXTURE_KB if os.path.exists(FIXTURE_KB) else None
SECRET = "demo-scope-secret"
ALL_TOOLS = ["search_manual", "get_vehicle_status", "lookup_vehicle_spec",
             "get_maintenance_plan", "create_service_order",
             "get_my_orders", "reschedule_order", "cancel_order"]


def client_with(token: str = "", secret: str = SECRET, timeout: float = 20.0) -> McpClient:
    args = ["--scope-secret", secret]
    if KB_PATH:
        args += ["--kb", KB_PATH]
    return McpClient(args=args, scope_token=token, timeout=timeout)


def main() -> int:
    checks: List[Dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"  {'[OK]' if ok else '[FAIL]'} {name}" + (f" —— {detail}" if detail else ""))
        return ok

    print("=" * 78)
    print("MCP 第二消费者 + 服务端 scope（只读维修顾问 vs 座舱助手）")
    print("=" * 78)

    # 1. 不带 token：服务端拒绝握手
    rejected = False
    rejected_code = 0
    try:
        client_with("").start()
    except McpError as exc:
        rejected, rejected_code = True, exc.code
    record("没有 scope token 的连接被服务端拒绝", rejected and rejected_code == FORBIDDEN,
           f"code={rejected_code}（期望 {FORBIDDEN}）")

    # 2. 只读 token：工具列表被服务端裁剪
    read_token = issue_scope_token("repair-advisor", READONLY_TOOLS, SECRET)
    readonly = client_with(read_token).start()
    readonly_names: List[str] = []
    try:
        readonly_names = [t["name"] for t in readonly.list_tools()]
        expected = {"search_manual", "get_maintenance_plan", "get_my_orders", "get_vehicle_status"}
        wrote = {"create_service_order", "reschedule_order", "cancel_order"}
        record("只读 token 只看到 4 个工具（且不含任何写操作）",
               set(readonly_names) == expected and not (set(readonly_names) & wrote),
               f"tools={readonly_names}")

        # 3. 硬调写操作 → 服务端拒绝（客户端没有白名单也拦不住这条路）
        write_denied = False
        try:
            readonly.call_tool("create_service_order",
                               {"item": "更换机油机滤", "preferred_date": "2099-06-18"})
        except McpError as exc:
            write_denied = exc.code == FORBIDDEN
        record("只读客户端硬调写操作被**服务端**拒绝", write_denied,
               f"服务端 scope 拦截（{FORBIDDEN}），不是客户端白名单挡的")
    finally:
        readonly.close()

    # 4. 全量 token：同一份工具定义，8 个工具都在
    full_token = issue_scope_token("cockpit-assistant", ALL_TOOLS, SECRET)
    full = client_with(full_token).start()
    try:
        full_names = [t["name"] for t in full.list_tools()]
        record("全量 token 看到 8 个工具（同一 server、不同授权）",
               full_names == ALL_TOOLS, f"tools={len(full_names)} 个")
        local = ToolRegistry(KnowledgeBase.load(KB_PATH))
        local_schema = {s["function"]["name"]: s["function"]["parameters"]
                        for s in local.specs()}
        remote = {t["name"]: t["inputSchema"] for t in full.list_tools()}
        mismatch = [n for n, schema in remote.items() if schema != local_schema.get(n)]
        record("两处 schema 同源（MCP 的 inputSchema 与本地 Function Calling 一致）",
               not mismatch, f"不一致：{mismatch or '无'}")
    finally:
        full.close()

    # 5. 第二消费者：维修顾问用只读 token 答一题
    client, advisor = open_readonly_advisor(SECRET, kb_path=KB_PATH)
    try:
        result = advisor.ask("座椅加热怎么关闭")
        record("维修顾问（第二消费者）用 MCP 工具答出带出处的手册问题",
               result["status"] == "answered" and bool(result["citations"]),
               f"工具={advisor.tools}，用到={result['tools_used']}，"
               f"出处 {len(result['citations'])} 条")
    finally:
        client.close()

    passed = sum(1 for c in checks if c["ok"])
    summary = {
        "checks": len(checks), "passed": passed,
        "readonly_tools": len(READONLY_TOOLS) - 1,      # get_telemetry 是控制面，不进 tools/list
        "full_tools": len(ALL_TOOLS),
        "server_rejections": 1,
    }
    print("-" * 78)
    print(f"合计 {passed}/{len(checks)} 项通过｜只读工具 {summary['readonly_tools']} 个 / "
          f"全量 {summary['full_tools']} 个｜服务端越权拦截 {summary['server_rejections']} 次")
    report = os.path.join(ROOT, "eval", "mcp_scope_report.json")
    with open(report, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "checks": checks}, fh, ensure_ascii=False, indent=2)
    print(f"报告：{report}")
    if passed != len(checks):
        print("[FAIL] 存在未通过项")
        return 1
    print("[OK] 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
