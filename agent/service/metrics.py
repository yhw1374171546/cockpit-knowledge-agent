# -*- coding: utf-8 -*-
"""服务指标（Prometheus 文本格式，零依赖手写）。

设计说明
--------
- **不引入 `prometheus_client`**：本仓库的离线测试与 CI 都不装它，而 exposition 格式本身
  就是纯文本，手写 60 行即可；这样 `/metrics` 在任何环境都能用。
- **多进程语义**：计数器是**进程内**的。`uvicorn --workers 4` 时每个 worker 各有一份，
  由 Prometheus 分别 scrape 各实例后在服务端聚合——这是标准做法，不要试图在进程内合并。
  为此每个指标都带 `instance` 标签（取自 PID/端口），便于区分。
- 需要跨实例的全局值（消息数、token 总量、成本）时，用 `/v1/admin/metrics`（走数据库聚合）。
"""

from __future__ import annotations

import os
import threading
import time
from typing import Dict, List, Tuple

# 延迟直方图分桶（秒）：覆盖「毫秒级编排」到「秒级生成」的全区间
LATENCY_BUCKETS: Tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
TTFT_BUCKETS: Tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


class _Histogram:
    def __init__(self, buckets: Tuple[float, ...]):
        self.buckets = buckets
        self.counts: List[int] = [0] * (len(buckets) + 1)
        self.total = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        for i, edge in enumerate(self.buckets):
            if value <= edge:
                self.counts[i] += 1
                return
        self.counts[-1] += 1


class Metrics:
    """进程内指标注册表（线程安全）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.instance = f"{os.getpid()}"
        self.started_at = time.time()
        self.counters: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
        self.latency = _Histogram(LATENCY_BUCKETS)
        self.ttft = _Histogram(TTFT_BUCKETS)

    # ── 写入 ──
    def inc(self, name: str, labels: Dict[str, str] = None, value: float = 1.0) -> None:
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            self.counters[key] = self.counters.get(key, 0.0) + value

    def observe_latency(self, seconds: float) -> None:
        with self._lock:
            self.latency.observe(max(0.0, seconds))

    def observe_ttft(self, seconds: float) -> None:
        with self._lock:
            self.ttft.observe(max(0.0, seconds))

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return {f"{name}{_fmt_labels(labels)}": v
                    for (name, labels), v in sorted(self.counters.items())}

    # ── 渲染 ──
    def render(self) -> str:
        lines: List[str] = []
        inst = f'{{instance="{self.instance}"}}'

        lines.append("# HELP agent_up 服务是否存活（恒为 1）")
        lines.append("# TYPE agent_up gauge")
        lines.append(f"agent_up{inst} 1")
        lines.append("# HELP agent_uptime_seconds 进程运行时长")
        lines.append("# TYPE agent_uptime_seconds gauge")
        lines.append(f"agent_uptime_seconds{inst} {time.time() - self.started_at:.1f}")

        with self._lock:
            groups: Dict[str, List[Tuple[str, float]]] = {}
            for (name, labels), value in sorted(self.counters.items()):
                groups.setdefault(name, []).append((_fmt_labels(labels), value))

        for name in sorted(groups):
            lines.append(f"# TYPE {name} counter")
            for label_str, value in groups[name]:
                merged = _merge_instance(label_str, self.instance)
                lines.append(f"{name}{merged} {_num(value)}")

        lines.extend(_render_hist("agent_request_duration_seconds",
                                 "请求耗时分布", self.latency, self.instance))
        lines.extend(_render_hist("agent_ttft_seconds",
                                 "首字延迟分布（SSE）", self.ttft, self.instance))
        return "\n".join(lines) + "\n"


def _num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.6f}"


def _fmt_labels(labels: Tuple[Tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in labels)
    return "{" + inner + "}"


def _merge_instance(label_str: str, instance: str) -> str:
    if not label_str:
        return f'{{instance="{instance}"}}'
    return label_str[:-1] + f',instance="{instance}"}}'


def _render_hist(name: str, help_text: str, hist: _Histogram, instance: str) -> List[str]:
    out = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
    cumulative = 0
    for edge, count in zip(hist.buckets, hist.counts):
        cumulative += count
        out.append(f'{name}_bucket{{le="{edge}",instance="{instance}"}} {cumulative}')
    cumulative += hist.counts[-1]
    out.append(f'{name}_bucket{{le="+Inf",instance="{instance}"}} {cumulative}')
    out.append(f'{name}_sum{{instance="{instance}"}} {hist.total:.6f}')
    out.append(f'{name}_count{{instance="{instance}"}} {hist.count}')
    return out


# 全局单例：中间件与各端点共用
METRICS = Metrics()
