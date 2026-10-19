# -*- coding: utf-8 -*-
"""MCP 服务端：把座舱工具暴露成标准 MCP Server（stdio 传输，按行分隔的 JSON-RPC 2.0）。

    python -m agent.mcp.server                     # 默认按 AGENT_KB_PATH → kb/chunks.jsonl → 内置语料
    python -m agent.mcp.server --kb eval/fixtures/mini_corpus.jsonl --telemetry severe

三个刻意的设计
--------------
1. **工具发现与语料解耦**：`initialize` / `tools/list` 只读工具 schema，不碰知识库；
   语料在**第一次真正调用检索**时才加载（`_LazyKnowledgeBase`）。完整语料下
   `KnowledgeBase.load` ≈ 14.6 s，如果握手时加载，客户端会以为服务端挂了。
2. **控制面不污染观测面**：`confirm_action` / `reset_cache` / `get_telemetry` 走 `tools/call`
   但**不登记进 trace / 缓存**，也不出现在 `tools/list` 里——它们是"客户端才能调用"的运维接口。
3. **结果双通道**：`content[].text` 给模型看（与现有 `ToolResult.to_observation()` 一致），
   `structuredContent` 给客户端解析（避免客户端再解析一遍 JSON 文本）；`isError` 与 `ok` 同步。

诚实边界：车况/车型/保养数据是内置演示数据（真实项目来自车云），
因此 MCP 这层能证明的是**协议与治理**，不是"接上了真实车控 API"。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from agent.kb import KnowledgeBase
from agent.mcp.protocol import (
    CONTROL_TOOLS,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    PROTOCOL_VERSION,
    SERVER_NAME,
    SERVER_VERSION,
    McpError,
    encode,
    make_error,
    make_result,
)
from agent.tools import TELEMETRY_PROFILES, VEHICLE_TELEMETRY, ToolRegistry, ToolResult


def log(message: str) -> None:
    """日志一律走 stderr：stdout 是协议通道，混一点非 JSON 就会把客户端打挂。"""
    sys.stderr.write(f"[mcp-server] {message}\n")
    sys.stderr.flush()


class _LazyKnowledgeBase(KnowledgeBase):
    """延迟加载的知识库：构造是 O(1)，第一次检索才真正读盘 + 建 BM25 索引。

    为什么需要它：`ToolRegistry.__init__` 拿到 kb 就注册工具，而工具 schema 与语料无关。
    直接 `KnowledgeBase.load()` 会让 `tools/list` 付 14.6 s（完整语料）——客户端在
    `initialize` 阶段就超时了。这里把加载推迟到"第一次真的要检索"。

    实现方式：先构造空库，加载后把 `chunks / tokenizer / _bm25 / path` 拷进来，
    因此后续 `super().search()` 走的是完全相同的检索实现，没有第二套逻辑。
    """

    def __init__(self, path: Optional[str] = None):
        super().__init__()
        self._pending_path = path
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        started = time.perf_counter()
        real = KnowledgeBase.load(self._pending_path)
        self.chunks = real.chunks
        self.tokenizer = real.tokenizer
        self._bm25 = real._bm25
        self.path = real.path
        self._loaded = True
        log(f"知识库加载完成：{len(self.chunks)} 块，耗时 {(time.perf_counter() - started) * 1000:.0f} ms")

    def search(self, query: str, *args: Any, **kwargs: Any) -> List[Any]:     # noqa: D102
        self._ensure_loaded()
        return super().search(query, *args, **kwargs)


def tool_payload(result: ToolResult, trace_id: Optional[str] = None,
                 server_ms: float = 0.0, control: bool = False) -> Dict[str, Any]:
    """把内部 ToolResult 翻译成 MCP 的 tools/call 返回体。"""
    meta = dict(result.meta or {})
    meta.update({
        "trace_id": trace_id,
        "server_ms": round(server_ms, 3),
        "repeated": result.repeated,
        "control": control,
    })
    return {
        "content": [{"type": "text", "text": result.to_observation()}],
        "structuredContent": {
            "ok": result.ok,
            "data": result.data,
            "error": result.error,
            "hint": result.hint,
            "repeated": result.repeated,
            "handler_ms": round(result.latency_ms, 3),
        },
        "isError": not result.ok,
        "_meta": meta,
    }


class McpServer:
    """一个 stdio MCP Server：单客户端、单线程、按行收发。"""

    def __init__(self, kb_path: Optional[str] = None, telemetry_profile: str = "normal",
                 registry: Optional[ToolRegistry] = None):
        self.kb_path = kb_path
        self.telemetry_profile = telemetry_profile
        self._registry = registry
        self.initialized = False
        self.calls_served = 0
        self.control_calls = 0

    # ── 延迟构造 ──
    @property
    def registry(self) -> ToolRegistry:
        if self._registry is None:
            telemetry = TELEMETRY_PROFILES.get(self.telemetry_profile, VEHICLE_TELEMETRY)
            self._registry = ToolRegistry(_LazyKnowledgeBase(self.kb_path), telemetry=dict(telemetry))
        return self._registry

    # ── 协议处理 ──
    def handle_line(self, line: str) -> Optional[Dict[str, Any]]:
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            return make_error(None, PARSE_ERROR, f"无效的 JSON：{exc.msg}")
        if not isinstance(message, dict):
            return make_error(None, INVALID_REQUEST, "请求必须是 JSON 对象")
        return self.handle(message)

    def handle(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if message.get("jsonrpc") != "2.0":
            return make_error(message.get("id"), INVALID_REQUEST, "jsonrpc 必须为 \"2.0\"")
        method = message.get("method")
        if not isinstance(method, str) or not method:
            return make_error(message.get("id"), INVALID_REQUEST, "缺少 method")
        is_notification = "id" not in message
        params = message.get("params") or {}
        try:
            result = self._dispatch(method, params if isinstance(params, dict) else {})
        except McpError as exc:
            if is_notification:
                return None
            return make_error(message.get("id"), exc.code, exc.message, exc.data)
        except Exception as exc:                                    # noqa: BLE001
            log(f"[error] {method}: {type(exc).__name__}: {exc}")
            if is_notification:
                return None
            return make_error(message.get("id"), INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
        if is_notification:
            return None
        return make_result(message.get("id"), result)

    def _dispatch(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == "initialize":
            self.initialized = True
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "instructions": ("座舱知识大脑工具服务：手册检索 / 实时车况 / 车型参数 / 保养计划 / 到店预约。"
                                 "写操作（create_service_order）必须先经车主确认，"
                                 "客户端用控制面工具 confirm_action 传入 action_key 后才能执行。"),
            }
        if method in ("notifications/initialized", "initialized"):
            self.initialized = True
            return {}
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": self.tool_descriptors()}
        if method == "tools/call":
            return self._tools_call(params)
        raise McpError(METHOD_NOT_FOUND, f"未知方法 {method}")

    # ── 工具清单 ──
    def tool_descriptors(self) -> List[Dict[str, Any]]:
        """把 ToolRegistry 的 OpenAI schema 翻译成 MCP 的 `{name, description, inputSchema}`。

        复用同一份 `parameters` 很关键：MCP 与 Function Calling 描述的是**同一组工具**，
        两份 schema 各写一次，迟早会漂移。
        """
        out: List[Dict[str, Any]] = []
        for spec in self.registry.tools.values():
            if spec.name in CONTROL_TOOLS:
                continue
            function = spec.to_openai_schema()["function"]
            out.append({
                "name": function["name"],
                "description": function["description"],
                "inputSchema": function["parameters"],
                "annotations": {
                    "title": spec.name,
                    "readOnlyHint": not spec.write,
                    "destructiveHint": spec.write,
                    "idempotentHint": not spec.write,
                    "requiresConfirmation": spec.requires_confirmation,
                },
            })
        return out

    # ── 工具调用 ──
    def _tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not name:
            raise McpError(INVALID_PARAMS, "tools/call 需要字符串 name")
        if not isinstance(arguments, dict):
            raise McpError(INVALID_PARAMS, "tools/call 的 arguments 必须是对象")

        meta = params.get("_meta") or {}
        trace_id = meta.get("trace_id") if isinstance(meta, dict) else None
        # 身份随调用走：服务端不持有"当前用户"，每次调用由客户端在 _meta 里带 user_id。
        # 这是 stdio 单客户端的简化做法；多客户端场景应换成服务端签发的 scope token（见 README）。
        user_id = meta.get("user_id") if isinstance(meta, dict) else None
        previous_user = self.registry.user_id
        if isinstance(user_id, str) and user_id:
            self.registry.user_id = user_id
        started = time.perf_counter()
        try:
            if name in CONTROL_TOOLS:
                self.control_calls += 1
                result = self._control(name, arguments)
                return tool_payload(result, trace_id=trace_id, control=True,
                                    server_ms=(time.perf_counter() - started) * 1000)

            if name not in self.registry.names():
                raise McpError(INVALID_PARAMS, f"未知工具 {name}", {"tools": self.registry.names()})

            result = self.registry.call(name, arguments)
            self.calls_served += 1
            return tool_payload(result, trace_id=trace_id,
                                server_ms=(time.perf_counter() - started) * 1000)
        finally:
            self.registry.user_id = previous_user

    def _control(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        if name == "confirm_action":
            action_key = arguments.get("action_key")
            if not isinstance(action_key, str) or not action_key:
                return ToolResult(False, error="confirm_action 需要 action_key",
                                  hint="action_key 来自写操作返回的 _meta.action_key")
            self.registry.confirm(action_key)
            return ToolResult(True, data={"confirmed": action_key})
        if name == "reset_cache":
            self.registry.reset_cache()
            return ToolResult(True, data={"reset": True})
        if name == "get_telemetry":
            return ToolResult(True, data=dict(self.registry.telemetry))
        return ToolResult(False, error=f"未知控制方法 {name}")

    # ── 主循环 ──
    def serve(self, stdin: Any = None, stdout: Any = None) -> None:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        while True:
            line = stdin.readline()
            if not line:
                break                                # EOF：客户端关闭了 stdin
            line = line.strip()
            if not line:
                continue
            response = self.handle_line(line)
            if response is not None:
                stdout.write(encode(response) + "\n")
                stdout.flush()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.mcp.server",
                                     description="座舱知识大脑 MCP Server（stdio）")
    parser.add_argument("--kb", default=None, help="知识库路径；缺省按 AGENT_KB_PATH 解析")
    parser.add_argument("--telemetry", default="normal", choices=sorted(TELEMETRY_PROFILES),
                        help="车况场景（演示数据）")
    args = parser.parse_args(argv)

    for stream in (sys.stdout,):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", newline="\n")
            except Exception:                                    # noqa: BLE001
                pass

    log(f"启动 pid={os.getpid()} kb={args.kb or '<按环境解析>'} telemetry={args.telemetry}")
    server = McpServer(kb_path=args.kb, telemetry_profile=args.telemetry)
    try:
        server.serve()
    except KeyboardInterrupt:
        log("收到中断，退出")
    log(f"退出：业务调用 {server.calls_served} 次，控制面调用 {server.control_calls} 次")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
