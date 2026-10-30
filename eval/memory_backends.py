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

from typing import Any, Callable, Dict, List, Optional

from agent.longterm import (  # noqa: E402
    LongTermMemory,
    defaults_from_preferences,
    order_event_recorder,
)
from agent.profile.pipeline import (  # noqa: E402
    ProfilePipeline,
)
from agent.profile.pipeline import (
    order_event_recorder as v2_order_event_recorder,
)
from agent.profile.retrieve import LocalEmbedder  # noqa: E402
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
    """v2：ProfilePipeline（五层画像 + 冲突消解 + 嵌入/词面双路召回 + 双时态历史）。"""

    def __init__(self, mem_path: str, embedder: Any = None):
        self.pipeline = ProfilePipeline(ProfileStore(mem_path), embedder=embedder,
                                        user_id=USER)

    def extract_and_remember(self, text: str, user_id: str = USER) -> None:
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


def make_backend(profile: str, mem_path: str, embedder: Any = None) -> MemoryBackend:
    return V2Backend(mem_path, embedder) if profile == "v2" else V1Backend(mem_path)
