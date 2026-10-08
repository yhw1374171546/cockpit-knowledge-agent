# -*- coding: utf-8 -*-
"""服务级安全与可靠性组件：限流、配额、幂等、确认、审计。

这里解决的是**单机脚本没有的问题**：
- 限流：防止单个用户/车型把服务打满（按 用户+车型 维度固定窗口计数，KV 原子 incr）；
- 配额：按自然日聚合 token，超额降级（防 denial-of-wallet 跨用户维度）；
- 幂等：写操作按 (user_id, Idempotency-Key) **落库唯一约束**，重放直接返回首次响应，
  且**同一 key 换 body 会被拒绝**（否则幂等键就成了绕过校验的后门）；
- 确认：车主对写操作的确认落库，Agent 每次运行前加载 —— 重启不丢；
- 审计：写操作 before/after + trace_id 落库，可回溯。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from agent.service.config import Settings
from agent.service.db import (AuditLog, Confirmation, Database, IdempotencyKey, Quota)
from agent.service.kv import KVStore


# ── 限流（固定窗口计数；生产可用 Redis + Lua 做令牌桶更平滑）────────────

@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    limit: int
    retry_after: int = 0
    bucket: str = ""


class RateLimiter:
    def __init__(self, kv: KVStore, settings: Settings):
        self.kv = kv
        self.settings = settings
        # 窗口长度 = 容量 / 补充速率（例如 20 容量、2/s → 10 秒窗口）
        self.window = max(1, int(settings.rate_limit_capacity /
                                 max(0.001, settings.rate_limit_refill_per_sec)))

    def hit(self, principal_id: str, vehicle_model: str) -> RateLimitResult:
        bucket = f"rl:{principal_id}:{vehicle_model}"
        window_start = int(time.time()) // self.window
        key = f"{bucket}:{window_start}"
        count = self.kv.incr(key, 1, ttl=self.window + 1)
        limit = self.settings.rate_limit_capacity
        if count > limit:
            retry = self.window - (int(time.time()) % self.window)
            return RateLimitResult(False, 0, limit, retry_after=retry, bucket=bucket)
        return RateLimitResult(True, limit - count, limit, bucket=bucket)


# ── 配额（按自然日聚合 token，落库）──────────────────────────────────

@dataclass
class QuotaResult:
    allowed: bool
    used: int
    limit: int
    remaining: int


class QuotaService:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings

    def _row(self, session, user_id: str) -> Quota:
        row = session.execute(
            select(Quota).where(Quota.user_id == user_id, Quota.day == date.today())
        ).scalar_one_or_none()
        if row is None:
            row = Quota(user_id=user_id, day=date.today(), tokens_used=0, cost_usd=0.0, requests=0)
            session.add(row)
            session.flush()
        return row

    def check(self, user_id: str, estimated_tokens: int = 0) -> QuotaResult:
        with self.db.session() as session:
            row = self._row(session, user_id)
            limit = self.settings.quota_tokens_per_day
            used = int(row.tokens_used or 0)
            return QuotaResult(used + estimated_tokens <= limit, used, limit,
                               max(0, limit - used))

    def record(self, user_id: str, tokens: int, cost_usd: float = 0.0) -> QuotaResult:
        with self.db.session() as session:
            row = self._row(session, user_id)
            row.tokens_used = int(row.tokens_used or 0) + max(0, tokens)
            row.cost_usd = float(row.cost_usd or 0.0) + max(0.0, cost_usd)
            row.requests = int(row.requests or 0) + 1
            limit = self.settings.quota_tokens_per_day
            return QuotaResult(True, int(row.tokens_used), limit,
                               max(0, limit - int(row.tokens_used)))


# ── 幂等（落库唯一约束）──────────────────────────────────────────────

@dataclass
class IdempotencyOutcome:
    replay: bool = False
    in_progress: bool = False
    conflict: bool = False                 # 同一 key 但 body 不同
    response: Optional[Dict[str, Any]] = None
    reason: str = ""


class IdempotencyStore:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings

    @staticmethod
    def hash_request(payload: Any) -> str:
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def begin(self, user_id: str, key: str, endpoint: str,
              request_hash: str) -> IdempotencyOutcome:
        with self.db.session() as session:
            existing = session.execute(
                select(IdempotencyKey).where(IdempotencyKey.user_id == user_id,
                                             IdempotencyKey.key == key)
            ).scalar_one_or_none()
            if existing is not None:
                return self._replay_outcome(existing, request_hash)
            session.add(IdempotencyKey(user_id=user_id, key=key, endpoint=endpoint,
                                       request_hash=request_hash, status="in_progress"))
            try:
                session.flush()
            except IntegrityError:
                session.rollback()
                with self.db.session() as s2:
                    row = s2.execute(
                        select(IdempotencyKey).where(IdempotencyKey.user_id == user_id,
                                                     IdempotencyKey.key == key)
                    ).scalar_one_or_none()
                    if row is None:
                        return IdempotencyOutcome(reason="幂等键并发冲突，请重试")
                    return self._replay_outcome(row, request_hash)
        return IdempotencyOutcome()

    @staticmethod
    def _replay_outcome(row: IdempotencyKey, request_hash: str) -> IdempotencyOutcome:
        if row.request_hash and row.request_hash != request_hash:
            return IdempotencyOutcome(conflict=True,
                                      reason="同一 Idempotency-Key 被用于不同的请求体")
        if row.status == "completed" and row.response_body:
            try:
                return IdempotencyOutcome(replay=True, response=json.loads(row.response_body))
            except json.JSONDecodeError:
                return IdempotencyOutcome(replay=True, response=None)
        return IdempotencyOutcome(in_progress=True,
                                  reason="相同 Idempotency-Key 的请求正在处理中")

    def complete(self, user_id: str, key: str, response: Dict[str, Any]) -> None:
        with self.db.session() as session:
            row = session.execute(
                select(IdempotencyKey).where(IdempotencyKey.user_id == user_id,
                                             IdempotencyKey.key == key)
            ).scalar_one_or_none()
            if row is None:
                return
            row.status = "completed"
            row.response_body = json.dumps(response, ensure_ascii=False, default=str)
            row.completed_at = datetime.utcnow()

    def release(self, user_id: str, key: str) -> None:
        """请求失败时释放幂等键，避免用户被永久卡住。"""
        with self.db.session() as session:
            row = session.execute(
                select(IdempotencyKey).where(IdempotencyKey.user_id == user_id,
                                             IdempotencyKey.key == key)
            ).scalar_one_or_none()
            if row is not None and row.status != "completed":
                session.delete(row)


# ── 写操作确认（HITL 落库）───────────────────────────────────────────

class ConfirmationStore:
    def __init__(self, db: Database):
        self.db = db

    def confirm(self, user_id: str, action_key: str, session_id: Optional[str] = None) -> bool:
        with self.db.session() as session:
            exists = session.execute(
                select(Confirmation).where(Confirmation.user_id == user_id,
                                           Confirmation.action_key == action_key)
            ).scalar_one_or_none()
            if exists is not None:
                return False
            session.add(Confirmation(user_id=user_id, action_key=action_key,
                                     session_id=session_id))
            try:
                session.flush()
            except IntegrityError:
                session.rollback()
                return False
        return True

    def load(self, user_id: str) -> Set[str]:
        with self.db.session() as session:
            rows = session.execute(
                select(Confirmation.action_key).where(Confirmation.user_id == user_id)
            ).scalars().all()
            return set(rows)


# ── 审计 ─────────────────────────────────────────────────────────────

class AuditService:
    def __init__(self, db: Database):
        self.db = db

    @staticmethod
    def log_in_session(session, actor: str, action: str, target: str = "", before: Any = "",
                       after: Any = "", trace_id: str = "") -> None:
        """在**已有事务**里写审计（避免嵌套 session 造成锁冲突）。"""
        session.add(AuditLog(
            actor=actor, action=action, target=target,
            before=json.dumps(before, ensure_ascii=False, default=str)[:2000],
            after=json.dumps(after, ensure_ascii=False, default=str)[:2000],
            trace_id=trace_id))

    def log(self, actor: str, action: str, target: str = "", before: Any = "",
            after: Any = "", trace_id: str = "") -> None:
        with self.db.session() as session:
            self.log_in_session(session, actor, action, target, before, after, trace_id)

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self.db.session() as session:
            rows = session.execute(
                select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)
            ).scalars().all()
            return [{"id": r.id, "actor": r.actor, "action": r.action, "target": r.target,
                     "trace_id": r.trace_id,
                     "created_at": r.created_at.isoformat() if r.created_at else None}
                    for r in rows]
