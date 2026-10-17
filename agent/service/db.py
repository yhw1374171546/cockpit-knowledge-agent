# -*- coding: utf-8 -*-
"""数据层：SQLAlchemy 模型与会话管理。

默认 **SQLite**（零依赖、仓库克隆即可跑、CI 可跑）；生产把
`SERVICE_DATABASE_URL` 指向 PostgreSQL 即可，schema 保持兼容（不使用任何 SQLite 专有类型）。

关键设计：
- `tool_calls.idempotency_key` 建**唯一索引** —— 幂等从"进程内字典"变成"数据库约束"；
- `idempotency_keys` 表按 (user_id, key) 唯一 —— 同一请求重放直接返回首次响应；
- `quotas` 表按 (user_id, day) 唯一 —— 配额按自然日聚合；
- `audit_logs` 记录写操作的 before/after 与 trace_id —— 合规可回溯。
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import date, datetime
from typing import Dict, Iterator, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class Vehicle(Base):
    __tablename__ = "vehicles"
    id = Column(Integer, primary_key=True)
    vin = Column(String(32), unique=True, index=True)
    model = Column(String(64), index=True)
    mileage_km = Column(Integer, default=0)
    owner_id = Column(String(64), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class SessionRow(Base):
    __tablename__ = "sessions"
    id = Column(String(36), primary_key=True)
    vehicle_model = Column(String(64), index=True)
    user_id = Column(String(64), index=True)
    title = Column(String(128), default="")
    status = Column(String(16), default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages = relationship("MessageRow", back_populates="session",
                            cascade="all, delete-orphan")


class MessageRow(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    role = Column(String(16))
    content = Column(Text)
    status = Column(String(24), default="answered")
    citations = Column(Text, default="")
    tokens_in = Column(Integer, default=0)
    tokens_out = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    trace_id = Column(String(36), index=True)
    failure = Column(String(32), default="ok")
    latency_ms = Column(Float, default=0.0)
    ttft_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    session = relationship("SessionRow", back_populates="messages")


class ToolCallRow(Base):
    __tablename__ = "tool_calls"
    id = Column(Integer, primary_key=True)
    session_id = Column(String(36), index=True)
    message_id = Column(Integer, index=True)
    tool = Column(String(64), index=True)
    arguments = Column(Text, default="{}")
    ok = Column(Boolean, default=True)
    blocked = Column(Boolean, default=False)
    reason = Column(String(255), default="")
    repeated = Column(Boolean, default=False)
    latency_ms = Column(Float, default=0.0)
    idempotency_key = Column(String(64), nullable=True)
    trace_id = Column(String(36), index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


# 说明：这里**故意不建唯一索引**。
# 幂等的数据库级保证在 `idempotency_keys(user_id, key)` 上（请求级重放 = 同一结果）；
# `tool_calls` 这一列只用于审计与排查。若在此加唯一约束，一个请求触发多次写操作就会插入失败。
Index("ix_tool_calls_idempotency", ToolCallRow.idempotency_key)


class IdempotencyKey(Base):
    __tablename__ = "idempotency_keys"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(64))
    key = Column(String(64))
    endpoint = Column(String(64))
    request_hash = Column(String(64))
    status = Column(String(16), default="in_progress")   # in_progress | completed
    response_body = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


Index("uq_idempotency_user_key", IdempotencyKey.user_id, IdempotencyKey.key, unique=True)


class Confirmation(Base):
    """车主对写操作的确认（HITL 落库；Agent 每次运行前从这里加载）。"""
    __tablename__ = "confirmations"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(64), index=True)
    action_key = Column(String(160))
    session_id = Column(String(36), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


Index("uq_confirmation_user_action", Confirmation.user_id, Confirmation.action_key, unique=True)


class Quota(Base):
    __tablename__ = "quotas"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(64))
    day = Column(Date, default=date.today)
    tokens_used = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    requests = Column(Integer, default=0)


Index("uq_quota_user_day", Quota.user_id, Quota.day, unique=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True)
    actor = Column(String(64), index=True)
    action = Column(String(64))
    target = Column(String(128), default="")
    before = Column(Text, default="")
    after = Column(Text, default="")
    trace_id = Column(String(36), index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, index=True)
    user_id = Column(String(64), index=True)
    rating = Column(Integer)              # 1 有帮助 / -1 没帮助
    comment = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)


class KbVersion(Base):
    __tablename__ = "kb_versions"
    id = Column(Integer, primary_key=True)
    vehicle_model = Column(String(64), index=True)
    source_file = Column(String(255))
    n_chunks = Column(Integer, default=0)
    status = Column(String(16), default="active")     # building | active | shadow | archived
    built_at = Column(DateTime, default=datetime.utcnow)
    # ── 版本化所需的字段（BE5 拿到 Alembic 之后才敢加列）──
    # 之前没有迁移机制，加列会让老库直接报 "no such column"，
    # 所以当时用 source_file 承载 "version|path" 这个折中；现在靠迁移补上正式字段。
    version = Column(String(64), index=True, default="")
    checksum = Column(String(32), default="")
    traffic_pct = Column(Integer, default=100)


def record_kb_version(db: "Database", *, vehicle_model: str, version: str, path: str,
                      checksum: str, n_chunks: int, status: str,
                      traffic_pct: int = 100) -> None:
    """把知识库版本状态写进表（同一 (租户, 版本) 做 upsert）。

    这张表以前**定义了但零使用**——表结构在那儿、语义在那儿，却没人写。
    现在它承担两件事：① 审计留痕（谁在什么时候把哪版设成 active/shadow/archived）；
    ② **进程重启后的状态恢复**（否则灰度做到一半、一重启就回到 v1）。
    """
    with db.session() as session:
        row = (session.query(KbVersion)
               .filter(KbVersion.vehicle_model == vehicle_model,
                       KbVersion.version == version)
               .one_or_none())
        if row is None:
            row = KbVersion(vehicle_model=vehicle_model, version=version)
            row.source_file = path
            session.add(row)
        row.source_file = path
        row.n_chunks = int(n_chunks)
        row.checksum = checksum
        row.status = status
        row.traffic_pct = int(traffic_pct)
        row.built_at = datetime.utcnow()


def load_kb_versions(db: "Database") -> List[Dict[str, object]]:
    """读取全部版本状态（供启动时恢复 active/shadow 路由）。"""
    with db.session() as session:
        rows = session.query(KbVersion).order_by(KbVersion.built_at).all()
        return [{"vehicle_model": r.vehicle_model or "", "version": r.version or "",
                 "path": r.source_file or "", "checksum": r.checksum or "",
                 "n_chunks": int(r.n_chunks or 0), "status": r.status or "",
                 "traffic_pct": int(r.traffic_pct or 100)} for r in rows]


class Database:
    def __init__(self, url: str, echo: bool = False) -> None:
        is_sqlite = url.startswith("sqlite")
        # SQLite 在服务化场景下的两个必要设置：
        # ① check_same_thread=False —— 请求会在工作线程里执行 Agent 并写库；
        # ② WAL + busy_timeout —— 允许"一写多读"并发，避免 database is locked。
        connect_args = {"check_same_thread": False, "timeout": 30} if is_sqlite else {}
        self.engine = create_engine(url, echo=echo, future=True, connect_args=connect_args)
        if is_sqlite:
            from sqlalchemy import event

            @event.listens_for(self.engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _record):        # pragma: no cover - 驱动回调
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        self._maker = sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False)

    def init(self, create_all_tables: bool = True) -> None:
        """初始化表结构。

        `create_all_tables=True` 只用于**开发/测试**：`create_all` 只建不存在的表，
        **永远不会修改已有表**——所以给老库加列时它什么也不做，新代码一查就
        `no such column`。生产必须用 `alembic upgrade head`（BE5），
        这里通过 `SERVICE_AUTO_CREATE_SCHEMA=0` 关掉自动建表。
        """
        if create_all_tables:
            Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator:
        """一个事务一个会话。

        ⚠️ 不要在 `with db.session()` 内部再开一个 session 写库：
        SQLite 下会撞 "database is locked"（PostgreSQL 下则是两个独立事务，语义也可能错）。
        需要复用同一事务时，用 `AuditService.log_in_session(session, ...)` 这类 *_in_session 方法。
        """
        db = self._maker()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def health(self) -> bool:
        try:
            with self.session() as db:
                db.execute(func.now() if self.engine.dialect.name != "sqlite" else func.current_timestamp())
            return True
        except Exception:      # noqa: BLE001
            return False


_db: Optional[Database] = None


def get_database(url: Optional[str] = None, echo: bool = False) -> Database:
    global _db
    if _db is None:
        _db = Database(url or os.environ.get("SERVICE_DATABASE_URL", "sqlite:///./agent_service.db"), echo)
    return _db


def reset_database() -> None:
    """测试用：丢弃单例，下次重新按 URL 建。"""
    global _db
    _db = None
