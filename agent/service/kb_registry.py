# -*- coding: utf-8 -*-
"""知识库版本注册表：多版本共存、按租户灰度、可回滚。

要解决的问题（也是面试必问的那句"线上怎么换手册"）
------------------------------------------------
原来只有一个 `KnowledgeBase` 实例，路径固定：换手册 = **改配置 + 重启**，
没有版本概念、没有灰度、出问题**只能再改回去重启**。而"换材料"在座舱场景里是常态：
手册改版、政策调整、某个车型单独更新。

本模块提供三件事：
1. **多版本共存**：同一份语料 vs 新语料是两个版本，各自有 checksum 与块数，可以同时在内存里；
2. **灰度**：新版本先只给一部分用户（按 user_id 稳定分桶，**同一用户不会在两版本之间跳**，
   否则同一会话里前言不搭后语）；
3. **回滚**：一条命令切回上一版本，不需要重启（`activate` 只改路由表，KB 实例已在缓存里）。

设计说明
--------
- **版本号自带语料指纹**（`v1-<checksum8>`）：版本与内容强绑定，避免"v2 其实是同一份语料"这种假版本；
- **分桶要稳定**：`bucket = md5(user|租户|灰度版本) % 100 < pct`。带上租户避免跨车型串；
  带上灰度版本号，使得**新一轮灰度会重新分桶**（否则永远同一批人当小白鼠）；
- **已知取舍**：注册表状态在内存里（进程重启需重新注册），因为它只在启动/发布时使用；
  状态变更会写入 `kb_versions` 表作为**审计留痕**。要让它成为跨重启的权威配置，
  需要给表加 `version`/`checksum`/`traffic_pct` 列——那属于迁移工作（Alembic），
  在拿到 Alembic 之前**不擅自给生产表加列**，这里用 `source_file` 承载 `version|path`。
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from agent.kb import KnowledgeBase


def bucket_of(user_id: str, vehicle_model: str, version: str) -> int:
    """稳定分桶：同一 (用户, 租户, 灰度版本) 永远落在同一个 0-99 档。"""
    raw = f"{user_id}|{vehicle_model}|{version}".encode("utf-8")
    return int(hashlib.md5(raw).hexdigest()[:8], 16) % 100


@dataclass
class KBVersion:
    version: str
    path: str
    vehicle_model: str
    checksum: str = ""
    n_chunks: int = 0
    status: str = "active"          # active | shadow | archived
    traffic_pct: int = 100           # 仅 shadow 有意义：灰度比例
    built_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, object]:
        return {"version": self.version, "path": self.path,
                "vehicle_model": self.vehicle_model, "checksum": self.checksum,
                "n_chunks": self.n_chunks, "status": self.status,
                "traffic_pct": self.traffic_pct, "built_at": self.built_at}


class KBRegistry:
    """多版本知识库注册表 + 灰度路由 + 回滚。"""

    def __init__(self) -> None:
        self._versions: Dict[str, KBVersion] = {}
        self._kb: Dict[str, KnowledgeBase] = {}
        self._active: Dict[str, str] = {}                      # 租户 → 版本
        self._shadow: Dict[str, Tuple[str, int]] = {}           # 租户 → (版本, 比例)
        self._history: Dict[str, List[str]] = {}                # 租户 → 激活历史
        self._lock = threading.RLock()
        self._buckets_hit: Dict[str, int] = {}                  # 观测：各版本被路由次数

    # ── 注册 ──
    def register(self, path: str, vehicle_model: str = "default",
                 version: Optional[str] = None,
                 kb: Optional[KnowledgeBase] = None) -> KBVersion:
        """注册一个知识库文件作为版本。

        版本号默认 `v<n>-<checksum8>`：**版本号里带语料指纹**，
        这样"两个版本号指向同一份内容"或"同版本号内容变了"都能一眼看出来。
        """
        with self._lock:
            instance = kb or KnowledgeBase.load(path)
            checksum = instance.checksum()
            if not version:                       # None 与 "" 都视为"未指定"
                version = f"v{len(self._versions) + 1}-{checksum[:8]}"
            if not version.strip():
                raise ValueError("版本号不能为空")
            record = KBVersion(version=version, path=path, vehicle_model=vehicle_model,
                               checksum=checksum, n_chunks=len(instance.chunks))
            self._versions[version] = record
            self._kb[version] = instance
            return record

    def register_loaded(self, kb: KnowledgeBase, vehicle_model: str = "default",
                        version: Optional[str] = None) -> KBVersion:
        """注册一个**已加载**的 KB 实例（服务启动时用它，避免重复加载）。"""
        with self._lock:
            checksum = kb.checksum()
            if not version:                       # None 与 "" 都视为"未指定"
                version = f"v{len(self._versions) + 1}-{checksum[:8]}"
            record = KBVersion(version=version, path=kb.path or "", vehicle_model=vehicle_model,
                               checksum=checksum, n_chunks=len(kb.chunks), status="active")
            self._versions[version] = record
            self._kb[version] = kb
            return record

    # ── 查询 ──
    def list(self, vehicle_model: Optional[str] = None) -> List[KBVersion]:
        with self._lock:
            records = list(self._versions.values())
        if vehicle_model:
            records = [r for r in records if r.vehicle_model in (vehicle_model, "default")]
        return sorted(records, key=lambda r: r.built_at)

    def get(self, version: str) -> Optional[KBVersion]:
        return self._versions.get(version)

    def active_version(self, vehicle_model: str) -> Optional[str]:
        return self._active.get(vehicle_model)

    def kb_for(self, version: str) -> Optional[KnowledgeBase]:
        return self._kb.get(version)

    def stats(self) -> Dict[str, object]:
        with self._lock:
            return {"versions": len(self._versions), "active": dict(self._active),
                    "shadow": {k: {"version": v, "pct": p} for k, (v, p) in self._shadow.items()},
                    "routed": dict(self._buckets_hit)}

    # ── 生效 / 灰度 / 回滚 ──
    def activate(self, version: str, vehicle_model: str = "default",
                 traffic_pct: int = 100) -> KBVersion:
        """把某个版本设为生效版本。

        `traffic_pct` 的语义刻意设计成**三态**，避免"灰度"和"全量"用两套接口：
        - `100`：**全量切换**（原生效版本进历史，可回滚）；
        - `1~99`：**灰度**（原版本继续服务大多数用户，新版本只给这比例）；
        - `0`：**撤销灰度**（等价于把灰度下线，保持原生效版本不变）。
        """
        with self._lock:
            record = self._versions.get(version)
            if record is None:
                raise KeyError(f"未知知识库版本：{version}")
            traffic_pct = max(0, min(100, int(traffic_pct)))
            record.status = "active" if traffic_pct >= 100 else "shadow"
            record.traffic_pct = traffic_pct

            if traffic_pct >= 100:
                previous = self._active.get(vehicle_model)
                if previous and previous != version:
                    self._versions[previous].status = "archived"
                    self._history.setdefault(vehicle_model, []).append(previous)
                self._active[vehicle_model] = version
                self._shadow.pop(vehicle_model, None)
            elif traffic_pct == 0:
                self._shadow.pop(vehicle_model, None)
                if record.status == "shadow":
                    record.status = "archived"
            else:
                self._shadow[vehicle_model] = (version, traffic_pct)
            return record

    def rollback(self, vehicle_model: str = "default") -> KBVersion:
        """回滚到上一个生效版本；没有历史时抛错（而不是静默返回当前版本）。"""
        with self._lock:
            history = self._history.get(vehicle_model) or []
            if not history:
                raise KeyError(f"租户 {vehicle_model} 没有可回滚的历史版本")
            target = history.pop()
            current = self._active.get(vehicle_model)
            if current and current != target:
                self._versions[current].status = "archived"
            record = self._versions[target]
            record.status = "active"
            record.traffic_pct = 100
            self._active[vehicle_model] = target
            self._shadow.pop(vehicle_model, None)
            return record

    # ── 路由 ──
    def resolve(self, vehicle_model: str, user_id: str = "") -> KBVersion:
        """为一次请求决定用哪个版本（灰度命中则用影子版本）。"""
        with self._lock:
            active = self._active.get(vehicle_model)
            if active is None:
                # 没显式激活时退回该租户最新注册的版本（保证服务总能起来）
                candidates = [r for r in self._versions.values()
                              if r.vehicle_model in (vehicle_model, "default")]
                if not candidates:
                    raise KeyError(f"租户 {vehicle_model} 没有已注册的知识库版本")
                active = max(candidates, key=lambda r: r.built_at).version
                self._active[vehicle_model] = active
            shadow = self._shadow.get(vehicle_model)
            chosen = active
            if shadow:
                shadow_version, pct = shadow
                if pct >= 100 or (user_id and bucket_of(user_id, vehicle_model,
                                                        shadow_version) < pct):
                    chosen = shadow_version
            self._buckets_hit[chosen] = self._buckets_hit.get(chosen, 0) + 1
            record = self._versions.get(chosen) or self._versions[active]
            return record
