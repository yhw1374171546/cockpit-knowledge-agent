# -*- coding: utf-8 -*-
"""consolidate：候选事实入库的冲突消解（对齐 Mem0 的 update 算子语义 + Zep 的失效模型）。

四类动作（决策矩阵见 `consolidate()` docstring）：
- insert：无既有事实 → 直接写入；
- reinforce：值与现状相同 → 只提置信度（复述是强信号），不重复插行；
- overwrite：值不同且新来源优先级 >= 旧来源 → 旧事实标 t_invalid（保留演化史）+ 写入新事实；
- reject：值不同且新来源优先级更低（如推断想覆盖明示）→ 拒绝，审计留痕。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from agent.profile.schema import source_rank, spec_for
from agent.profile.store import ProfileFact, ProfileStore, _append_value


@dataclass
class MemoryCandidate:
    """extract 层的产出：一条待入库的画像事实候选。"""
    key: str
    value: str
    source: str                          # user_explicit | inferred | order_system ...
    confidence: float
    evidence_span: str = ""              # 原文依据（extract 层已校验）
    layer: str = ""


@dataclass
class ConsolidateResult:
    action: str                          # insert | reinforce | overwrite | append | reject
    reason: str
    fact: Optional[ProfileFact] = None   # 写入/更新后的有效事实（reject 时为 None）

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "reason": self.reason,
                "fact": self.fact.to_dict() if self.fact else None}


def consolidate(candidate: MemoryCandidate, store: ProfileStore,
                user_id: str = "default", now: Optional[float] = None,
                allow_inferred: bool = False) -> ConsolidateResult:
    """把一条候选事实按冲突矩阵写入画像库。

    决策矩阵：
    | 现状 | 候选来源优先级 | 结果 |
    | --- | --- | --- |
    | 无 | 任意（推断需 allow_inferred） | insert |
    | 值相同 | 任意 | reinforce（置信度取 max） |
    | 值不同 | >= 现状 | overwrite（旧标 t_invalid + 新事实，借 Zep） |
    | 值不同 | < 现状 | reject（如 inferred 试图覆盖 user_explicit） |
    | append_only 字段 | 任意 | append（去重追加，历史不可改写） |
    """
    now = store._now() if now is None else now
    spec = spec_for(candidate.key)

    if candidate.source not in spec.sources:
        return ConsolidateResult("reject",
                                 f"来源 {candidate.source} 不在字段 {candidate.key} 的允许来源内")
    if candidate.source == "inferred" and not allow_inferred:
        return ConsolidateResult("reject", "推断来源默认不落库（allow_inferred=False）")
    if not candidate.value.strip():
        return ConsolidateResult("reject", "值为空")

    current = store.current(user_id, candidate.key)

    if current is None:
        from agent.profile.store import make_fact
        fact = make_fact(user_id, candidate.key, candidate.value,
                         candidate.source, candidate.confidence, now)
        store.insert(fact)
        store._event(user_id, spec.layer, candidate.key, "insert",
                     None, candidate.value, candidate.source, now)
        return ConsolidateResult("insert", "新建事实", fact)

    if spec.append_only:
        merged = _append_value(current.value, candidate.value)
        if merged == current.value:
            store.bump(current, current.confidence, "append_duplicate", now)
            return ConsolidateResult("reinforce", "轨迹已存在，仅触碰", current)
        store.bump(current, current.confidence, "append", now)
        # 追加语义：值更新但事实身份不变（t_valid 保持），审计记录旧值
        store._event(user_id, spec.layer, candidate.key, "append",
                     current.value, merged, candidate.source, now)
        with store._lock, store._conn:
            store._conn.execute(
                "UPDATE profile_facts SET value=?, updated_at=? "
                "WHERE user_id=? AND key=? AND t_valid=?",
                (merged, now, user_id, candidate.key, current.t_valid))
        updated = store.current(user_id, candidate.key)
        return ConsolidateResult("append", "追加到轨迹", updated)

    if current.value == candidate.value:
        boosted = max(float(candidate.confidence), float(current.confidence))
        store.bump(current, boosted, candidate.source, now)
        refreshed = store.current(user_id, candidate.key)
        return ConsolidateResult("reinforce", "同一事实复述，置信度回升", refreshed)

    if source_rank(candidate.source) < source_rank(current.source):
        store._event(user_id, spec.layer, candidate.key, "reject_override",
                     current.value, candidate.value,
                     f"{candidate.source}({source_rank(candidate.source)}) "
                     f"< {current.source}({source_rank(current.source)})", now)
        return ConsolidateResult("reject", "新来源优先级低于现状，拒绝覆盖")

    # overwrite：旧事实保留失效（借 Zep Edge Invalidation），新事实生效
    store.invalidate(current, f"被 {candidate.source} 覆盖", now)
    from agent.profile.store import make_fact
    fact = make_fact(user_id, candidate.key, candidate.value,
                     candidate.source, candidate.confidence, now)
    store.insert(fact)
    store._event(user_id, spec.layer, candidate.key, "overwrite",
                 current.value, candidate.value, candidate.source, now)
    return ConsolidateResult("overwrite", "新信息优先，旧事实标失效", fact)
