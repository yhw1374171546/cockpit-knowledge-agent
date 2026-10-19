# -*- coding: utf-8 -*-
"""MCP 工具接入层的包入口。

四个模块各管一件事：
- `protocol.py`：JSON-RPC 2.0 编解码与错误码（手写，零依赖）；
- `server.py`：把 5 个座舱工具暴露成 MCP Server（stdio 传输）；
- `client.py`：拉起 server 子进程，按行收发 JSON-RPC；
- `registry.py`：把远端工具包成与本地 `ToolRegistry` **同形**的注册表。

    python -m agent.mcp.server          # 单独起一个 MCP Server
    python eval/mcp_smoke.py            # 跨进程冒烟（握手 / 白名单 / 确认 / trace / 端到端）

`server` 故意不在这里导入：`python -m agent.mcp.server` 会先导入本包，
如果本包再导入 server，会让该模块被当作 `__main__` 与 `agent.mcp.server` 各执行一遍。
"""

from agent.mcp.client import McpClient
from agent.mcp.protocol import CONTROL_TOOLS, PROTOCOL_VERSION
from agent.mcp.registry import McpToolRegistry

__all__ = ["CONTROL_TOOLS", "PROTOCOL_VERSION", "McpClient", "McpToolRegistry"]
