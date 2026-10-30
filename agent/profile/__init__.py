# -*- coding: utf-8 -*-
"""v2 画像管线：extract → consolidate → retrieve → forget（阶段 3）。

已完成：schema / store / extract（规则+LLM）/ consolidate / retrieve（嵌入+词面双路）
/ forget（TTL/容量/显式/合规）/ pipeline（v1 同款接口门面）。
v1（agent/longterm.py）冻结为基线，本包与其并存、不互相依赖。
"""

from agent.profile.consolidate import (  # noqa: F401
    ConsolidateResult,
    MemoryCandidate,
    consolidate,
)
from agent.profile.extract import extract_rules, extract_with_llm  # noqa: F401
from agent.profile.forget import (  # noqa: F401
    cap_active,
    erase_user,
    expire,
    explicit_forget,
    maintain,
)
from agent.profile.pipeline import ProfilePipeline  # noqa: F401
from agent.profile.retrieve import (  # noqa: F401
    LocalEmbedder,
    MemoryContext,
    MemoryItem,
    cosine,
    overlap_score,
    retrieve,
)
from agent.profile.schema import FIELDS, LAYERS, FieldSpec, pref_field, spec_for  # noqa: F401
from agent.profile.store import ProfileFact, ProfileStore  # noqa: F401

__all__ = [
    "ProfileStore", "ProfileFact", "FieldSpec", "FIELDS", "LAYERS",
    "spec_for", "pref_field", "MemoryCandidate", "ConsolidateResult",
    "consolidate", "extract_rules", "extract_with_llm",
    "LocalEmbedder", "MemoryContext", "MemoryItem", "retrieve",
    "cosine", "overlap_score", "ProfilePipeline",
    "expire", "cap_active", "explicit_forget", "erase_user", "maintain",
]
