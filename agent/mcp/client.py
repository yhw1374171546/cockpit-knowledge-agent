# -*- coding: utf-8 -*-
"""MCP 客户端：拉起一个 stdio MCP Server 子进程，按 JSON-RPC 收发。

    with McpClient(args=["--kb", "eval/fixtures/mini_corpus.jsonl"]) as client:
        registry = McpToolRegistry(client)

为什么用「子进程 + 线程读管道」而不是同步 readline：
- 工具执行可能超过调用方的 deadline（完整语料下首次检索要建 BM25），
  Windows 上没有可用的 `select()` 去给管道设读超时，只能让**读线程**把行投进队列，
  主线程用 `Queue.get(timeout=)` 实现真正的超时；
- 超时后进程不杀，但该请求的响应会被标记成"迟到响应"直接丢弃（按 id 匹配），
  这样一次超时不会污染后续调用。

如实说明的边界：一个客户端对应一个 server 进程，没有连接池/重连；
多副本部署时每个副本各自拉起一个 server（这也正是 MCP stdio 的常规用法）。
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from agent.mcp.protocol import (
    PROTOCOL_VERSION,
    McpError,
    McpTimeout,
    McpTransportError,
    encode,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_COMMAND: Tuple[str, ...] = (sys.executable, "-m", "agent.mcp.server")


class McpClient:
    """最小 MCP 客户端：initialize 握手 + tools/list + tools/call。"""

    def __init__(self, command: Optional[Sequence[str]] = None,
                 args: Optional[Sequence[str]] = None,
                 timeout: float = 30.0,
                 cwd: Optional[str] = None,
                 env: Optional[Dict[str, str]] = None,
                 scope_token: str = ""):
        base = list(command) if command else list(DEFAULT_COMMAND)
        self.command: List[str] = base + list(args or [])
        self.timeout = timeout
        self.cwd = cwd or ROOT
        self._env_overrides = dict(env or {})
        # 服务端开启鉴权时，必须在 initialize 里带上它；不带的连接会被直接拒绝
        self.scope_token = scope_token
        self._proc: Optional[subprocess.Popen] = None
        self._responses: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self._stderr: Deque[str] = deque(maxlen=50)
        self._lock = threading.Lock()
        self._next_id = 0
        self._reader: Optional[threading.Thread] = None
        self._err_reader: Optional[threading.Thread] = None
        self.request_count = 0
        self.timed_out = 0
        self.server_info: Dict[str, Any] = {}
        self.scope_info: Dict[str, Any] = {}
        self.protocol_version = ""

    # ── 生命周期 ──
    def start(self) -> "McpClient":
        env = dict(os.environ)
        env.update(self._env_overrides)
        env["PYTHONPATH"] = os.pathsep.join(
            [self.cwd, env["PYTHONPATH"]] if env.get("PYTHONPATH") else [self.cwd])
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        kwargs: Dict[str, Any] = dict(
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1, cwd=self.cwd, env=env,
        )
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(self.command, **kwargs)      # noqa: S603
        self._reader = threading.Thread(target=self._read_loop, name="mcp-reader", daemon=True)
        self._err_reader = threading.Thread(target=self._stderr_loop, name="mcp-stderr", daemon=True)
        self._reader.start()
        self._err_reader.start()
        try:
            self._handshake()
        except Exception:
            # 鉴权失败/服务端拒绝握手：别把子进程与线程留在那里
            self.close()
            raise
        return self

    def _handshake(self) -> None:
        params: Dict[str, Any] = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "cockpit-knowledge-agent", "version": "1.0.0"},
        }
        if self.scope_token:
            params["_meta"] = {"scope_token": self.scope_token}
        result = self.request("initialize", params)
        self.protocol_version = str(result.get("protocolVersion") or "")
        self.server_info = dict(result.get("serverInfo") or {})
        # 服务端回报的授权范围（subject + 工具白名单），用于日志与断言
        self.scope_info = dict(result.get("_meta") or {})
        self.notify("notifications/initialized")

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:                                          # noqa: BLE001
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
        # 先等读线程自然退出（EOF），再关管道——否则在线程里读已关闭的文件会抛 ValueError
        for thread in (self._reader, self._err_reader):
            if thread is not None:
                thread.join(timeout=1.0)
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except Exception:                                      # noqa: BLE001
                pass
        self._responses.put(None)

    def __enter__(self) -> "McpClient":
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ── 收发 ──
    def _read_loop(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    self._stderr.append(f"<- 非 JSON 输出：{line[:160]}")
                    continue
                if isinstance(message, dict) and "id" in message:
                    self._responses.put(message)
                # 服务端主动发的通知（目前没有）直接忽略
        finally:
            self._responses.put(None)                                # EOF 哨兵

    def _stderr_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            self._stderr.append(line.rstrip())

    def _write(self, message: Dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise McpTransportError("MCP 服务端未启动或已退出")
        try:
            proc.stdin.write(encode(message) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise McpTransportError(f"写入失败：{exc}") from exc

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        message: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        with self._lock:
            self._write(message)

    def request(self, method: str, params: Optional[Dict[str, Any]] = None,
                timeout: Optional[float] = None) -> Dict[str, Any]:
        deadline = time.monotonic() + (timeout if timeout is not None else self.timeout)
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            message: Dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
            if params is not None:
                message["params"] = params
            self._write(message)
            self.request_count += 1
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.timed_out += 1
                    raise McpTimeout(f"{method} 超时（{(timeout or self.timeout):.1f}s）")
                try:
                    response = self._responses.get(timeout=remaining)
                except queue.Empty:
                    self.timed_out += 1
                    raise McpTimeout(f"{method} 超时（{(timeout or self.timeout):.1f}s）")
                if response is None:
                    raise McpTransportError(f"MCP 服务端已退出：{self.stderr_tail()}")
                if response.get("id") != request_id:
                    continue                    # 迟到/串号的响应：丢弃，不污染本次结果
                if "error" in response:
                    error = response.get("error") or {}
                    raise McpError(int(error.get("code", -32603)),
                                   str(error.get("message", "unknown")), error.get("data"))
                result = response.get("result")
                return dict(result) if isinstance(result, dict) else {}

    # ── 协议方法 ──
    def list_tools(self, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        result = self.request("tools/list", {}, timeout=timeout)
        tools = result.get("tools") or []
        return [t for t in tools if isinstance(t, dict)]

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None,
                  trace_id: Optional[str] = None, user_id: Optional[str] = None,
                  timeout: Optional[float] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"name": name, "arguments": arguments or {}}
        meta: Dict[str, Any] = {}
        if trace_id:
            meta["trace_id"] = trace_id
        if user_id:
            meta["user_id"] = user_id
        if meta:
            params["_meta"] = meta
        return self.request("tools/call", params, timeout=timeout)

    def ping(self) -> bool:
        return self.request("ping", {}) == {}

    # ── 诊断 ──
    def stderr_tail(self, lines: int = 5) -> str:
        return " | ".join(list(self._stderr)[-lines:])
