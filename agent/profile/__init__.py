# -*- coding: utf-8 -*-
"""v2 画像管线：extract → consolidate → retrieve → forget（阶段 3）。

当前进度：schema / store / extract（规则+LLM）/ consolidate 已落地；
retrieve（嵌入召回 + 时间衰减 + 季节复活）与 forget（TTL/容量/合规删除）随阶段 3 后续推进。
v1（agent/longterm.py）冻结为基线，本包与其并存、不互相依赖。
"""

from agent.profile.consolidate import (  # noqa: F401
    ConsolidateResult,
    MemoryCandidate,
    consolidate,
)
from agent.profile.extract import extract_rules, extract_with_llm  # noqa: F401
from agent.profile.schema import FIELDS, LAYERS, FieldSpec, pref_field, spec_for  # noqa: F401
from agent.profile.store import ProfileFact, ProfileStore  # noqa: F401

__all__ = [
    "ProfileStore", "ProfileFact", "FieldSpec", "FIELDS", "LAYERS",
    "spec_for", "pref_field", "MemoryCandidate", "ConsolidateResult",
    "consolidate", "extract_rules", "extract_with_llm",
]
