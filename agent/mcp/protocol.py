# -*- coding: utf-8 -*-
"""MCP 协议层：JSON-RPC 2.0 的编解码、错误码与错误类型。

为什么手写协议，而不是装官方 SDK
--------------------------------
1. 仓库的定位是**零依赖、克隆即跑**（CI 只装 jieba）。引入 `mcp` SDK 会让"能不能演示"
   取决于能不能联网装包；而 MCP 的 stdio 传输本身就是**按行分隔的 JSON-RPC 2.0**，
   手写一层覆盖得住。
2. 手写之后每个字段为什么这么填都说得清：`tools/list` 返回 `inputSchema`、
   `tools/call` 返回 `content` + `structuredContent` —— 面试问"协议长什么样"能直接对着代码讲，
   而不是"我调了库"。

实现边界（不夸大）
------------------
- 只实现 **tools** 能力：`initialize` / `notifications/initialized` / `ping` /
  `tools/list` / `tools/call`；
- 没有 resources / prompts / sampling / roots；传输只有 **stdio**，没有 streamable HTTP；
- 不支持 JSON-RPC 批量（batch）——stdio 单客户端场景用不上。

版本说明：`PROTOCOL_VERSION` 取我们**真正实现**的 2024-11-05。服务端不做版本协商，
只是如实回报自己支持的那一版（MCP 规定：服务端不支持客户端请求的版本时，返回自己支持的版本）。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "cockpit-knowledge-mcp"
SERVER_VERSION = "1.0.0"

# 控制面工具：**不出现在 tools/list 里**，模型看不到也无法选择，只允许客户端按名字调用。
# 为什么不让模型看见：
#   - `confirm_action` 一旦被模型自己调用，写操作的"人工二次确认"就形同虚设；
#   - `reset_cache` / `get_telemetry` 是运维与观测接口，不是业务能力。
# 为什么还走 tools/call 而不是自定义 method：MCP 的 method 集合是固定的，
# 服务端私自加 method 会破坏"标准客户端可直接接入"这一点。
CONTROL_TOOLS = ("confirm_action", "reset_cache", "get_telemetry")

# JSON-RPC 2.0 标准错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

JsonRpcId = Union[int, str, None]


class McpError(Exception):
    """协议层错误：会被翻译成 JSON-RPC 的 `error` 字段返回给客户端。"""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class McpTransportError(RuntimeError):
    """传输层错误：进程死了、管道断了、协议行不是 JSON。"""


class McpTimeout(McpTransportError):
    """等响应超时（服务端卡住，或工具执行超过客户端给的 deadline）。"""


def encode(message: Dict[str, Any]) -> str:
    """序列化成**单行** JSON（stdio 传输按行分帧，不能带裸换行）。"""
    return json.dumps(message, ensure_ascii=False, separators=(",", ":"))


def make_result(request_id: JsonRpcId, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def make_error(request_id: JsonRpcId, code: int, message: str,
               data: Optional[Any] = None) -> Dict[str, Any]:
    error: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}
