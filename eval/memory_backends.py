# -*- coding: utf-8 -*-
"""v1/v2 记忆后端的统一接口（AB 与 memory_bench 共用）。

- `extract_and_remember`：写入（用户一句话 → 画像候选 → 冲突消解）；
- `remember_episode`：情景记忆（工单轨迹）；
- `context_text`：召回注入文本（进 system prompt 的【跨会话记忆】块）；
- `defaults`：工具默认参数（惯用门店 → center）；
- `history`：某字段的演化史（TR 题型与"当时 vs 现在"查询的基础；
  v1 的 overwrite 会丢历史，如实返回空——这本身就是 v1/v2 的差异点）；
- `close`：释放连接。
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from agent.longterm import (  # noqa: E402
    LongTermMemory,
    defaults_from_preferences,
    order_event_recorder,
)
from agent.profile.extract import extract_rules  # noqa: E402
from agent.profile.pipeline import (  # noqa: E402
    ProfilePipeline,
)
from agent.profile.pipeline import (
    order_event_recorder as v2_order_event_recorder,
)
from agent.profile.retrieve import LocalEmbedder, cosine, overlap_score  # noqa: E402
from agent.profile.store import ProfileStore  # noqa: E402

USER = "u1"


class MemoryBackend:
    """v1（longterm）与 v2（profile）的统一接口。"""

    def extract_and_remember(self, text: str, user_id: str = USER) -> None:
        raise NotImplementedError

    def remember_episode(self, summary: str, user_id: str = USER,
                         session_id: str = "", dedupe_key: Optional[str] = None) -> None:
        raise NotImplementedError

    def defaults(self, user_id: str = USER) -> Dict[str, str]:
        raise NotImplementedError

    def context_text(self, query: str, user_id: str = USER) -> str:
        raise NotImplementedError

    def history(self, key: str, user_id: str = USER) -> List[str]:
        raise NotImplementedError

    def order_recorder(self, session_id: str, query: str,
                       user_id: str = USER) -> Optional[Callable]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class V1Backend(MemoryBackend):
    """v1：LongTermMemory（正则抽取 + 词面召回，冻结为基线）。"""

    def __init__(self, mem_path: str, embedder: Any = None):
        self.store = LongTermMemory(mem_path)

    def extract_and_remember(self, text: str, user_id: str = USER) -> None:
        self.store.extract_and_remember(text, user_id=user_id)

    def remember_episode(self, summary: str, user_id: str = USER,
                         session_id: str = "", dedupe_key: Optional[str] = None) -> None:
        self.store.remember_episode(summary, user_id=user_id, session_id=session_id,
                                    dedupe_key=dedupe_key)

    def defaults(self, user_id: str = USER) -> Dict[str, str]:
        return defaults_from_preferences(self.store, user_id)

    def context_text(self, query: str, user_id: str = USER) -> str:
        return self.store.context_text(query, user_id=user_id)

    def history(self, key: str, user_id: str = USER) -> List[str]:
        # v1 的 overwrite 不保留旧值：演化史无从查起（如实返回空，这是 v1/v2 的差异点）
        return []

    def order_recorder(self, session_id: str, query: str,
                       user_id: str = USER) -> Optional[Callable]:
        return order_event_recorder(self.store, user_id, session_id, query)

    def close(self) -> None:
        self.store.close()


class V2Backend(MemoryBackend):
    """v2：ProfilePipeline（五层画像 + 冲突消解 + 嵌入/词面双路召回 + 双时态历史）。

    可选 `llm`：注入后 extract 走 LLM 结构化抽取（guided_json + 依据定位），
    规则抽取降级为兜底（LLM 返回空时用）——v0.2 润色评测集就是它的用武之地
    （规则抽取对措辞变化脆弱，v0.1 100% → v0.2 48.4% 是实测证据）。
    """

    def __init__(self, mem_path: str, embedder: Any = None, llm: Any = None):
        self.pipeline = ProfilePipeline(ProfileStore(mem_path), embedder=embedder,
                                        user_id=USER)
        self.llm = llm

    def extract_and_remember(self, text: str, user_id: str = USER) -> None:
        if self.llm is not None:
            from agent.profile.consolidate import consolidate
            from agent.profile.extract import extract_rules, extract_with_llm
            candidates = extract_with_llm(text, self.llm) or extract_rules(text)
            for cand in candidates:
                consolidate(cand, self.pipeline.store, user_id=user_id,
                            allow_inferred=self.pipeline.allow_inferred)
            return
        self.pipeline.extract_and_remember(text, user_id=user_id)

    def remember_episode(self, summary: str, user_id: str = USER,
                         session_id: str = "", dedupe_key: Optional[str] = None) -> None:
        self.pipeline.remember_episode(summary, user_id=user_id, session_id=session_id,
                                       dedupe_key=dedupe_key)

    def defaults(self, user_id: str = USER) -> Dict[str, str]:
        return self.pipeline.defaults_from_preferences(user_id)

    def context_text(self, query: str, user_id: str = USER) -> str:
        return self.pipeline.context_text(query, user_id=user_id)

    def history(self, key: str, user_id: str = USER) -> List[str]:
        return [f.value for f in self.pipeline.store.history(user_id, key)]

    def order_recorder(self, session_id: str, query: str,
                       user_id: str = USER) -> Optional[Callable]:
        return v2_order_event_recorder(self.pipeline, user_id, session_id, query)

    def close(self) -> None:
        self.pipeline.store.close()


class Mem0SemanticsBackend(MemoryBackend):
    """Mem0 语义对照档：按 Mem0 **公开语义**的本地复现（非 API 调用）。

    对照目的（诚实声明）：回答"如果开箱用 Mem0 会发生什么"——
    - 写入 ADD-only：每条抽取只增不覆盖、无冲突消解（Mem0 生产版语义，
      漂移靠平台 Dream 机制，OSS 不提供）；新旧值**并存**；
    - 检索：query 语义搜索（嵌入余弦 + threshold=0.1 对齐 Mem0 默认），
      无主题路由、无业务 schema、无红线概念、无事实有效期；
    - 因此预期：KU 新旧值混着返回（答新值的"干净率"低）、TR 无演化史、红线缺失。
    它是本地复现语义，不是 Mem0 产品——README 与报告中如实标注。
    """

    def __init__(self, mem_path: str, embedder: Any = None):
        self.embedder = embedder
        self._conn = sqlite3.connect(mem_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, user_id TEXT,"
            " text TEXT, created_at REAL)")

    def _add(self, text: str, user_id: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._conn.execute("INSERT INTO memories (id, user_id, text, created_at)"
                           " VALUES (?,?,?,?)", (uuid.uuid4().hex[:16], user_id,
                                                 text, time.time()))
        self._conn.commit()

    def extract_and_remember(self, text: str, user_id: str = USER) -> None:
        # ADD-only：无冲突消解、无覆盖——与 Mem0 OSS 的 add 语义一致
        for cand in extract_rules(text):
            self._add(f"{cand.key}={cand.value}", user_id)

    def remember_episode(self, summary: str, user_id: str = USER,
                         session_id: str = "", dedupe_key: Optional[str] = None) -> None:
        self._add(summary, user_id)                      # 应用侧把工单摘要当普通记忆存

    def defaults(self, user_id: str = USER) -> Dict[str, str]:
        return {}                                        # Mem0 无业务默认参数概念

    def context_text(self, query: str, user_id: str = USER) -> str:
        rows = self._conn.execute(
            "SELECT text FROM memories WHERE user_id=?", (user_id,)).fetchall()
        texts = [str(r[0]) for r in rows]
        if not texts:
            return ""
        scored: List[Any] = []
        if self.embedder is not None and self.embedder.available:
            try:
                qv = self.embedder.encode([query])[0]
                vecs = self.embedder.encode(texts) or []
                scored = sorted(((cosine(qv, v), t) for t, v in zip(texts, vecs, strict=True)),
                                key=lambda x: -x[0])
            except Exception:                            # noqa: BLE001
                scored = []
        if not scored:
            scored = sorted(((overlap_score(query, t), t) for t in texts),
                            key=lambda x: -x[0])
        top = [t for s, t in scored if s >= 0.1][:5]     # threshold 对齐 Mem0 默认
        return "【跨会话记忆】\n- " + "；".join(top) if top else ""

    def history(self, key: str, user_id: str = USER) -> List[str]:
        return []                                        # 无事实有效期概念

    def order_recorder(self, session_id: str, query: str,
                       user_id: str = USER) -> Optional[Callable]:
        return None

    def close(self) -> None:
        self._conn.close()


def make_backend(profile: str, mem_path: str, embedder: Any = None,
                 llm: Any = None) -> MemoryBackend:
    if profile == "v2":
        return V2Backend(mem_path, embedder, llm=llm)
    if profile == "mem0":
        return Mem0SemanticsBackend(mem_path, embedder)
    return V1Backend(mem_path)
