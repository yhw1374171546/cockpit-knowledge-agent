# -*- coding: utf-8 -*-
"""MCP 工具注册表：把远端 MCP Server 的工具包成"和本地 ToolRegistry 同形"的对象。

为什么必须同形
--------------
编排层（`AgentGraph` / `ToolExecutor` / `Supervisor` 的子 Agent）只认这几个方法：
`names()` / `specs()` / `call()` / `subset()` / `confirm()` / `telemetry`。
只要形状一致，**"工具在本地函数里"和"工具在另一个进程里"对上层完全透明**——
这也是唯一能证明"MCP 是接入层替换、不是重写业务"的方式（有端到端测试覆盖：
`test_agent_graph_runs_over_mcp`）。

跨进程带来的三个真实差异（本文件逐个处理）
------------------------------------------
1. **白名单只能在客户端做**：MCP Server 不知道调用方是哪个子 Agent，
   所以 `subset()` 过滤的是**客户端**的 `specs()`；服务端仍会接受任何已知工具名。
   这是 stdio 单客户端场景的边界，跨团队共享时应改成服务端 scope token。
2. **确认状态是两处**：`ToolExecutor._is_confirmed()` 读客户端的 `confirmed_actions`，
   而真正执行时的校验在服务端。所以 `confirm()` 要**先让服务端确认、再写本地镜像**，
   两边都以服务端为准（镜像只用于前置判断）。
3. **耗时口径要分开**：客户端记端到端，服务端记 handler 用时，两个数都进 meta——
   否则"工具变慢了"到底是网络慢还是检索慢，说不清。
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Set

from agent.mcp.client import McpClient
from agent.mcp.protocol import McpError, McpTimeout, McpTransportError
from agent.tools import ToolResult


class McpToolRegistry:
    """与 `ToolRegistry` 同形的远端工具注册表（只读工具发现结果）。"""

    def __init__(self, client: McpClient, name: str = "mcp",
                 allowed: Optional[Sequence[str]] = None,
                 timeout: Optional[float] = None,
                 user_id: Optional[str] = None):
        self.client = client
        self.scope_name = name
        self.timeout = timeout
        self.transport = "mcp"
        # 调用者身份：服务端没有"当前用户"的概念，必须每次调用随 `_meta` 带过去。
        # 工单类工具（查/改/取消）全靠它做用户隔离。
        self.user_id = user_id
        self._tools: Dict[str, Dict[str, Any]] = {}
        for tool in client.list_tools():
            tool_name = tool.get("name")
            if isinstance(tool_name, str) and tool_name:
                self._tools[tool_name] = tool
        if not self._tools:
            raise RuntimeError("MCP Server 没有暴露任何工具（tools/list 为空）")
        unknown = [n for n in (allowed or []) if n not in self._tools]
        if unknown:
            raise ValueError(f"白名单里有服务端未提供的工具：{unknown}；可用：{list(self._tools)}")
        self._allowed: List[str] = list(allowed) if allowed is not None else list(self._tools)
        # 与 ToolRegistry 同名同义的状态（子 Agent 共享同一份，见 subset）
        self.confirmed_actions: Set[str] = set()
        self.trace: List[Dict[str, Any]] = []
        self.call_counts: Dict[str, int] = {}
        self._telemetry: Optional[Dict[str, Any]] = None

    # ── 与 ToolRegistry 对齐的只读接口 ──
    def names(self) -> List[str]:
        return list(self._allowed)

    def names_of_scope(self) -> str:
        return self.scope_name

    def specs(self) -> List[Dict[str, Any]]:
        """输出 OpenAI / vLLM 的 `tools` 格式（参数直接来自服务端的 inputSchema）。"""
        out: List[Dict[str, Any]] = []
        for name in self._allowed:
            tool = self._tools[name]
            schema = tool.get("inputSchema")
            out.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description") or "",
                    "parameters": schema if isinstance(schema, dict) else {"type": "object",
                                                                          "properties": {}},
                },
            })
        return out

    def subset(self, names: Sequence[str], name: str = "sub-agent") -> "McpToolRegistry":
        """派生一个只暴露指定工具的注册表（白名单在**客户端**生效，见模块说明第 1 条）。"""
        unknown = [n for n in names if n not in self._tools]
        if unknown:
            raise ValueError(f"白名单里有服务端未提供的工具：{unknown}；可用：{list(self._tools)}")
        scoped = McpToolRegistry.__new__(McpToolRegistry)
        scoped.client = self.client
        scoped.scope_name = name
        scoped.timeout = self.timeout
        scoped.transport = self.transport
        scoped.user_id = self.user_id
        scoped._tools = self._tools
        scoped._allowed = [n for n in names if n in self._allowed]
        scoped.confirmed_actions = self.confirmed_actions      # 确认状态父子共享
        scoped.trace = self.trace                              # 观测面共享
        scoped.call_counts = self.call_counts
        scoped._telemetry = self._telemetry
        return scoped

    # ── 车况：编排层做安全判定要读它（对应 ToolRegistry.telemetry）──
    @property
    def telemetry(self) -> Dict[str, Any]:
        if self._telemetry is None:
            try:
                payload = self.client.call_tool("get_telemetry", {},
                                                trace_id=self._new_trace_id(),
                                                timeout=self.timeout)
                content = payload.get("structuredContent") or {}
                data = content.get("data")
                self._telemetry = dict(data) if isinstance(data, dict) else {}
            except (McpError, McpTransportError) as exc:       # 读不到就退化为空，安全门控按"无告警"处理
                self._telemetry = {}
                self.trace.append({"tool": "get_telemetry", "ok": False, "error": str(exc),
                                   "transport": "mcp"})
        return dict(self._telemetry)

    # ── 调用 ──
    def call(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> ToolResult:
        arguments = arguments or {}
        if name not in self._allowed:
            result = ToolResult(
                False,
                error=f"工具 {name} 不在当前白名单",
                hint=f"{self.scope_name} 可用工具：{', '.join(self._allowed) or '（无）'}",
                meta={"blocked_by": "whitelist", "transport": "mcp"},
            )
            self._record(name, arguments, result)
            return result

        trace_id = self._new_trace_id()
        started = time.perf_counter()
        try:
            payload = self.client.call_tool(name, arguments, trace_id=trace_id,
                                            user_id=self.user_id, timeout=self.timeout)
        except McpTimeout as exc:
            return self._fail(name, arguments, trace_id, "MCP 调用超时",
                              f"{exc}；写操作不会被自动重试，避免重复下单")
        except McpError as exc:
            return self._fail(name, arguments, trace_id, f"MCP 协议错误 {exc.code}", exc.message)
        except McpTransportError as exc:
            return self._fail(name, arguments, trace_id, "MCP 传输失败", str(exc))

        content = payload.get("structuredContent")
        if not isinstance(content, dict):
            content = {"ok": not payload.get("isError")}
        raw_meta = payload.get("_meta")
        meta: Dict[str, Any] = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        result = ToolResult(
            ok=bool(content.get("ok")),
            data=content.get("data"),
            error=content.get("error"),
            hint=content.get("hint"),
            repeated=bool(content.get("repeated")),
            latency_ms=(time.perf_counter() - started) * 1000,
            meta={**meta, "transport": "mcp", "trace_id": meta.get("trace_id") or trace_id,
                  "client_ms": round((time.perf_counter() - started) * 1000, 3)},
        )
        if name == "get_vehicle_status" and result.ok and isinstance(result.data, dict):
            self._telemetry = dict(result.data)                 # 车况顺手更新，省一次控制面往返
        self._record(name, arguments, result)
        return result

    def _fail(self, name: str, arguments: Dict[str, Any], trace_id: str,
              error: str, hint: str) -> ToolResult:
        result = ToolResult(False, error=error, hint=hint,
                            meta={"transport": "mcp", "trace_id": trace_id})
        self._record(name, arguments, result)
        return result

    def _record(self, name: str, arguments: Dict[str, Any], result: ToolResult) -> None:
        self.call_counts[name] = self.call_counts.get(name, 0) + 1
        self.trace.append({
            "tool": name, "arguments": arguments, "ok": result.ok,
            "repeated": result.repeated, "latency_ms": round(result.latency_ms, 2),
            "error": result.error, "transport": "mcp",
            "trace_id": (result.meta or {}).get("trace_id"),
            "server_ms": (result.meta or {}).get("server_ms"),
        })

    # ── 写操作确认（两步跨进程）──
    def confirm(self, action_key: str) -> None:
        if not action_key:
            raise ValueError("action_key 不能为空")
        payload = self.client.call_tool("confirm_action", {"action_key": action_key},
                                        trace_id=self._new_trace_id(), timeout=self.timeout)
        content = payload.get("structuredContent") or {}
        if not content.get("ok"):
            raise McpError(-32603, f"服务端确认失败：{content.get('error')}")
        self.confirmed_actions.add(action_key)                  # 本地镜像，供执行器前置判断

    def reset_cache(self) -> None:
        self.client.call_tool("reset_cache", {}, trace_id=self._new_trace_id(),
                              timeout=self.timeout)
        self.trace.clear()
        self.call_counts.clear()
        self._telemetry = None

    def cache_stats(self) -> Dict[str, int]:
        repeated = sum(1 for item in self.trace if item.get("repeated"))
        unique = len({f"{item.get('tool')}:{sorted((item.get('arguments') or {}).items())}"
                      for item in self.trace if item.get("transport") == "mcp"})
        return {"total_calls": len(self.trace), "repeated_calls": repeated, "unique_calls": unique}

    @staticmethod
    def _new_trace_id() -> str:
        return f"mcp-{uuid.uuid4().hex[:12]}"
