# -*- coding: utf-8 -*-
"""ProfilePipeline：extract → consolidate → retrieve → forget 的一站式入口。

对外接口刻意镜像 v1 `LongTermMemory`（extract_and_remember / context_text /
preference / defaults_from_preferences / fingerprint / maintain），
让 v1 → v2 的替换是"换一个构造参数"而不是"重写调用方"——基线矩阵（阶段 4）
能跑同一条 AB 脚本对比两代记忆。
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from agent.profile.consolidate import ConsolidateResult, consolidate
from agent.profile.extract import extract_rules, extract_with_llm
from agent.profile.forget import maintain as _maintain
from agent.profile.retrieve import LocalEmbedder, MemoryContext, retrieve
from agent.profile.store import ProfileStore


class ProfilePipeline:
    """v2 画像管线（阶段 3）。规则抽取零成本走 CI；LLM 抽取只在规则 0 命中时兜底。"""

    def __init__(self, store: ProfileStore, embedder: Optional[LocalEmbedder] = None,
                 llm: Any = None, allow_inferred: bool = False,
                 user_id: str = "default"):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.allow_inferred = allow_inferred
        self.user_id = user_id

    # ── 写入（extract + consolidate）──
    def extract_and_remember(self, text: str, user_id: Optional[str] = None) -> List[ConsolidateResult]:
        """从一句话里抽取候选并入库。返回每条候选的冲突消解结果（日志/测试断言用）。"""
        uid = user_id or self.user_id
        candidates = extract_rules(text)
        if not candidates and self.llm is not None:
            try:
                candidates = extract_with_llm(text, self.llm)
            except Exception:                              # noqa: BLE001 LLM 不可用不阻断
                candidates = []
        results: List[ConsolidateResult] = []
        for cand in candidates:
            results.append(consolidate(cand, self.store, user_id=uid,
                                       allow_inferred=self.allow_inferred))
        return results

    # ── 召回 ──
    def recall(self, query: str, user_id: Optional[str] = None,
               now: Optional[float] = None) -> MemoryContext:
        return retrieve(query, self.store, user_id=user_id or self.user_id,
                        embedder=self.embedder, now=now)

    def context_text(self, query: str, user_id: Optional[str] = None,
                     now: Optional[float] = None) -> str:
        return self.recall(query, user_id=user_id, now=now).to_prompt()

    def preference(self, user_id: str, key: str) -> Optional[str]:
        """当前有效值（工具默认参数用，与 v1 preference() 同语义）。"""
        fact = self.store.current(user_id, key)
        return fact.value if fact else None

    def defaults_from_preferences(self, user_id: Optional[str] = None) -> Dict[str, str]:
        """把画像翻译成工具默认参数（对齐 v1：目前映射惯用门店）。"""
        uid = user_id or self.user_id
        out: Dict[str, str] = {}
        center = self.preference(uid, "usual_center")
        if center:
            out["center"] = center
        return out

    # ── 情景记忆（episodes）──
    def remember_episode(self, summary: str, user_id: Optional[str] = None,
                         session_id: str = "", dedupe_key: Optional[str] = None,
                         ttl_days: int = 30) -> Optional[str]:
        return self.store.remember_episode(summary, user_id=user_id or self.user_id,
                                           session_id=session_id,
                                           dedupe_key=dedupe_key, ttl_days=ttl_days)

    # ── 观测与维护 ──
    def fingerprint(self, user_id: Optional[str] = None) -> str:
        """记忆指纹：进语义缓存的 scope（记忆一变，缓存自然失效——与 v1 同语义）。"""
        uid = user_id or self.user_id
        with self.store._lock:
            row = self.store._conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(MAX(updated_at), 0) AS m FROM profile_facts"
                " WHERE user_id=? AND t_invalid IS NULL", (uid,)).fetchone()
        raw = f"{row['n']}:{float(row['m']):.0f}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]

    def maintain(self, user_id: Optional[str] = None, max_facts: int = 100) -> Dict[str, int]:
        return _maintain(self.store, user_id=user_id or self.user_id, max_facts=max_facts)


def order_event_recorder(pipeline: ProfilePipeline, user_id: str,
                         session_id: str = "orders",
                         query: str = "") -> Any:
    """工单写操作 → 情景记忆（与 v1 的 order_event_recorder 同签名同格式，
    便于 AB 脚本无缝切换后端）。

    格式完全对齐 v1（"用户提到「…」；已建工单 …"）：带 query 前缀是为了
    让词面/嵌入召回都找得到——用户说"上次那个维修"，摘要里要有"维修"。
    """
    prefix = f"用户提到「{query[:30]}」；" if query else ""

    def record(action: str, payload: Dict[str, Any]) -> None:
        order_id = str(payload.get("预约单号") or "")
        item = str(payload.get("项目") or "")
        date = str(payload.get("到店日期") or "")
        center = str(payload.get("门店") or "")
        if action == "order_created":
            text = f"{prefix}已建工单 {order_id}：{item}，{date} 到 {center}"
        elif action == "order_reused":
            text = f"{prefix}工单 {order_id} 已存在：{item}，{date} 到 {center}"
        elif action == "order_rescheduled":
            text = f"{prefix}工单 {order_id} 改期到 {date}" + (f"（{center}）" if center else "")
        elif action == "order_cancelled":
            text = f"{prefix}工单 {order_id} 已取消"
        else:
            return
        pipeline.remember_episode(text, user_id=user_id, session_id=session_id,
                                  dedupe_key=f"{session_id}:{action}:{order_id}:{date}")

    return record
