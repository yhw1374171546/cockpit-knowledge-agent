# -*- coding: utf-8 -*-
"""SSE 事件重放缓冲（BE7：断线重连）。

为什么需要
----------
SSE 是单向流，网络抖动、切后台、隧道断网都会让连接断掉。浏览器会自动重连，
但重连后是**一个新的请求**——原来那条流的 token 已经吐出去一半，客户端手里只有半句话。

这就是要解决的两件事：
1. **重连能拿到已产出的内容**（不必重头再问，也不用再付一次钱）；
2. **断开时立刻取消上游**（见 `AgentConfig.cancel_event`）——否则用户切走了，
   后台还在跑检索和生成，**token 照扣**，这笔浪费在线上是无声的。

设计取舍（重要，别被问住）
--------------------------
**"断开即取消" 与 "重连续跑" 是冲突的**：想让重连无缝续上，就得在客户端可能永远不回来时
继续付费；想省钱就得取消。我选择**取消优先**，因为座舱里"用户切走"比"刚刚断了一下"更常见。

因此本模块提供的是**"部分重放"**：
- 缓冲里已产出的事件按 `Last-Event-ID` 补发；
- 如果原流没有正常结束（没有 `done`），补发完再发一个 `event: interrupted`，
  明确告诉客户端"这条流是断的，要完整答案请重新发起"——**不假装成功**。

流标识用什么？用 `trace_id`。客户端重连时带上相同的 `X-Trace-Id` 请求头
（服务端本来就支持这个头）即可找回同一条流；`Last-Event-ID` 则指明"从哪个事件之后补发"。
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


class SSEReplayBuffer:
    """按 stream_id 保存最近若干事件，支持按 Last-Event-ID 补发（有容量与 TTL 上限）。"""

    def __init__(self, max_streams: int = 256, max_events: int = 512, ttl_s: float = 300.0):
        self.max_streams = max(1, max_streams)
        self.max_events = max(1, max_events)
        self.ttl_s = ttl_s
        self._streams: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
        self._lock = threading.RLock()
        self.stats = {"streams_started": 0, "replays": 0, "replayed_events": 0,
                      "interrupted": 0, "evicted": 0, "expired": 0}

    # ── 生命周期 ──
    def open(self, stream_id: str) -> None:
        with self._lock:
            self._streams[stream_id] = {"events": [], "done": False, "ts": time.time()}
            self._streams.move_to_end(stream_id)
            self.stats["streams_started"] += 1
            self._evict()

    def append(self, stream_id: str, event_id: int, chunk: str) -> None:
        """记录一个已下发的事件（chunk 是完整的 SSE 文本，含 id/event/data）。"""
        with self._lock:
            record = self._streams.get(stream_id)
            if record is None:
                self.open(stream_id)
                record = self._streams[stream_id]
            events: List[Tuple[int, str]] = record["events"]           # type: ignore[assignment]
            events.append((int(event_id), chunk))
            if len(events) > self.max_events:
                del events[: len(events) - self.max_events]
            record["ts"] = time.time()

    def close(self, stream_id: str, completed: bool = True) -> None:
        """标记流已结束。`completed=False` 表示中断（断开/取消）。"""
        with self._lock:
            record = self._streams.get(stream_id)
            if record is None:
                return
            record["done"] = bool(completed)
            record["ts"] = time.time()
            if not completed:
                self.stats["interrupted"] += 1

    # ── 补发 ──
    def replay(self, stream_id: str, last_event_id: Optional[int]) -> Tuple[List[str], bool]:
        """返回 (要补发的 chunk 列表, 原流是否已完成)。

        - 找不到流（未开始/已过期/换了实例）→ 返回 ([], False)，调用方按"新请求"处理；
        - `last_event_id is None` → 从头补发（客户端没给 id 但给了同一个 trace_id）。
        """
        with self._lock:
            record = self._streams.get(stream_id)
            if record is None:
                return [], False
            if self.ttl_s and (time.time() - float(record["ts"])) > self.ttl_s:   # type: ignore[arg-type]
                self._streams.pop(stream_id, None)
                self.stats["expired"] += 1
                return [], False
            events: List[Tuple[int, str]] = record["events"]           # type: ignore[assignment]
            if last_event_id is None:
                picked = [chunk for _, chunk in events]
            else:
                picked = [chunk for eid, chunk in events if eid > int(last_event_id)]
            self.stats["replays"] += 1
            self.stats["replayed_events"] += len(picked)
            return picked, bool(record["done"])

    def has(self, stream_id: str) -> bool:
        with self._lock:
            return stream_id in self._streams

    # ── 容量控制 ──
    def _evict(self) -> None:
        now = time.time()
        if self.ttl_s:
            for key in [k for k, v in self._streams.items()
                        if now - float(v["ts"]) > self.ttl_s]:        # type: ignore[arg-type]
                self._streams.pop(key, None)
                self.stats["expired"] += 1
        while len(self._streams) > self.max_streams:
            self._streams.popitem(last=False)
            self.stats["evicted"] += 1

    def stats_snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {**self.stats, "streams": len(self._streams)}
