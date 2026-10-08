# -*- coding: utf-8 -*-
"""Agent 服务层：FastAPI 应用。

这一步把「能跑通的 Agent」变成「能上线的服务」，补齐脚本形态没有的东西：

| 能力 | 实现 |
| --- | --- |
| 接口契约 | OpenAPI 自动生成；统一错误码；`/v1` 版本前缀 |
| 流式 | `POST /v1/chat/stream`（SSE）：stage → delta×N → citations → done，含真实 TTFT |
| 鉴权与租户 | JWT（HS256）+ `X-Vehicle-Model` 车型租户校验（越权 403） |
| 限流 | 按 用户+车型 维度固定窗口计数（KV 原子 incr），429 + `Retry-After` |
| 配额 | 按自然日聚合 token，超限 429 并提示额度 |
| 幂等 | `Idempotency-Key` 落库唯一约束：重放返回首次响应；同 key 换 body 拒绝 |
| 会话持久化 | sessions / messages 落库，支持续聊与历史查询 |
| 审计 | 写操作 before/after + trace_id 落库；`/v1/admin/audit` 可查 |
| 可观测 | 结构化日志（含 trace_id）、每请求 trace/成本、失败归因落库 |
| 健康检查 | `/healthz`（存活）、`/readyz`（依赖就绪） |

本地运行：
    python -m uvicorn agent.service.app:app --port 8077
    SERVICE_LLM_BACKEND=streaming python -m uvicorn agent.service.app:app --port 8077
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from agent.graph import AgentConfig, AgentGraph
from agent.guardrails.policy import PolicyEngine, ToolPolicy
from agent.kb import KnowledgeBase
from agent.llm import (LLMBackend, OpenAICompatibleLLM, RuleBasedPlannerLLM,
                       StreamingRulePlanner)
from agent.memory import ConversationMemory, VehicleProfile
from agent.obs.tracer import Tracer
from agent.reflection import Reflector
from agent.service.auth import (AuthError, Principal, decode_token, issue_token,
                                resolve_tenant)
from agent.service.config import Settings
from agent.service.db import (Database, Feedback, MessageRow, SessionRow, ToolCallRow,
                              get_database, reset_database)
from agent.service.kv import KVStore, build_kv
from agent.service.schemas import (ChatRequest, ChatResponse, ConfirmRequest,
                                   ConfirmResponse, ErrorResponse, FeedbackRequest,
                                   HealthResponse, SessionInfo, TokenRequest,
                                   TokenResponse, ToolCallInfo, ToolInfo)
from agent.service.security import (AuditService, ConfirmationStore, IdempotencyStore,
                                    QuotaService, RateLimiter)
from agent.tools import ToolRegistry, build_default_registry

logger = logging.getLogger("agent.service")
WRITE_TOOLS = {"create_service_order"}


# ── 运行时：按租户缓存知识库，按请求构建 Agent ────────────────────────

class AgentRuntime:
    """知识库按（车型, KB 路径）缓存；Agent 与 Tracer 按请求新建，保证请求间隔离。

    - KB 缓存：BM25 索引构建是秒级开销，绝不能每请求重建；
    - Agent/Tracer 每请求新建：规则规划器带状态（turn），Tracer 要按请求统计成本。
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._kb_cache: Dict[str, KnowledgeBase] = {}
        self._lock = threading.Lock()
        self.kb_load_ms = 0.0

    def kb(self, vehicle_model: str) -> KnowledgeBase:
        key = f"{vehicle_model}|{self.settings.kb_path or 'default'}"
        with self._lock:
            kb = self._kb_cache.get(key)
            if kb is None:
                t0 = time.perf_counter()
                kb = KnowledgeBase.load(self.settings.kb_path or None)
                self.kb_load_ms = (time.perf_counter() - t0) * 1000
                self._kb_cache[key] = kb
                logger.info("已加载知识库 tenant=%s chunks=%d 耗时=%.0fms",
                            vehicle_model, len(kb.chunks), self.kb_load_ms)
            return kb

    def build_llm(self, on_delta: Optional[Callable[[str], None]] = None) -> LLMBackend:
        backend = (self.settings.llm_backend or "planner").lower()
        if backend in ("streaming", "simulated"):
            return StreamingRulePlanner(ms_per_chunk=self.settings.llm_ms_per_token * 6)
        if backend == "openai":
            return OpenAICompatibleLLM(base_url=self.settings.llm_base_url,
                                       model=self.settings.llm_model)
        return RuleBasedPlannerLLM()

    def build_agent(self, vehicle_model: str, confirmed: Optional[set] = None,
                    on_delta: Optional[Callable[[str], None]] = None,
                    idempotency_key: str = ""
                    ) -> Tuple[AgentGraph, Tracer, ToolRegistry]:
        kb = self.kb(vehicle_model)
        registry = build_default_registry(self.settings.kb_path or None)
        registry.kb = kb
        if confirmed:
            registry.confirmed_actions |= set(confirmed)
        tracer = Tracer()
        policy = self.build_policy()
        config = AgentConfig()
        # 服务层收紧：写操作必须带 Idempotency-Key（客户端重试不会重复下单）
        config.idempotency_key = idempotency_key
        agent = AgentGraph(self.build_llm(on_delta), registry,
                          reflector=Reflector(
                              evidence_overlap_threshold=config.evidence_overlap_threshold),
                          memory=ConversationMemory(profile=VehicleProfile(model=vehicle_model)),
                          config=config, tracer=tracer, policy=policy, agent_name="service")
        return agent, tracer, registry

    def build_policy(self) -> PolicyEngine:
        """在默认策略之上收紧写操作：强制 Idempotency-Key。

        这就是"同一套护栏，服务层可按场景收紧"的体现——本地离线测试用默认策略，
        对外服务用严格策略，两者共用同一实现。
        """
        policy = PolicyEngine()
        write = policy.policies["create_service_order"]
        write.require_idempotency_key = self.settings.require_idempotency_for_writes
        return policy


# ── 依赖 ─────────────────────────────────────────────────────────────

def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _runtime(request: Request) -> AgentRuntime:
    return request.app.state.runtime


def _db(request: Request) -> Database:
    return request.app.state.db


def _kv(request: Request) -> KVStore:
    return request.app.state.kv


def principal_of(request: Request,
                 authorization: Optional[str] = Header(None)) -> Principal:
    settings = _settings(request)
    if not settings.auth_required:
        model = request.headers.get("X-Vehicle-Model") or settings.default_vehicle_model
        return Principal(user_id=request.headers.get("X-User-Id", "dev-user"),
                         role="driver", vehicle_model=model)
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="缺少 Bearer 令牌",
                            headers={"WWW-Authenticate": "Bearer"})
    token = authorization.split(" ", 1)[1].strip()
    try:
        return decode_token(token, settings)
    except AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


def _trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", uuid.uuid4().hex[:16])


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


# ── 应用工厂 ─────────────────────────────────────────────────────────

def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        reset_database()
        db = get_database(settings.database_url, settings.db_echo)
        db.init()
        app.state.db = db
        app.state.kv = build_kv(settings.kv_backend, settings.redis_url)
        app.state.runtime = AgentRuntime(settings)
        app.state.started_at = time.time()
        # 预热默认租户知识库，让 /readyz 有真实含义
        app.state.runtime.kb(settings.default_vehicle_model)
        if settings.jwt_secret == "dev-secret-change-me":
            logger.warning("正在使用默认 JWT 密钥，生产环境请通过 SERVICE_JWT_SECRET 注入")
        logger.info("服务启动完成 db=%s kv=%s llm=%s",
                    db.engine.url.get_backend_name(), app.state.kv.backend(),
                    settings.llm_backend or "planner")
        try:
            yield
        finally:
            db.engine.dispose()

    app = FastAPI(title="智能座舱 Agent 服务", version=settings.version,
                  description="RAG + 多 Agent 座舱助手的服务化封装（鉴权/限流/配额/幂等/审计/SSE）",
                  lifespan=lifespan)

    # ── 中间件：trace_id 贯穿 + 结构化访问日志 ──
    @app.middleware("http")
    async def trace_middleware(request: Request, call_next):
        trace_id = request.headers.get("X-Trace-Id") or uuid.uuid4().hex[:16]
        request.state.trace_id = trace_id
        start = time.perf_counter()
        response = await call_next(request)
        cost_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Trace-Id"] = trace_id
        logger.info(json.dumps({
            "trace_id": trace_id, "method": request.method, "path": request.url.path,
            "status": response.status_code, "ms": round(cost_ms, 2),
            "user": getattr(request.state, "user_id", None),
            "vehicle_model": getattr(request.state, "vehicle_model", None),
        }, ensure_ascii=False))
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code,
                            content={"error": exc.detail, "detail": "",
                                     "trace_id": _trace_id(request)},
                            headers=getattr(exc, "headers", None))

    # ── 健康检查 ──
    @app.get("/healthz", tags=["ops"], summary="存活检查")
    async def healthz():
        return {"status": "ok", "uptime_s": round(time.time() - app.state.started_at, 1)}

    @app.get("/readyz", tags=["ops"], response_model=HealthResponse, summary="就绪检查")
    async def readyz(request: Request):
        db = _db(request)
        runtime = _runtime(request)
        kb = runtime.kb(settings.default_vehicle_model)
        ok = db.health()
        return HealthResponse(status="ready" if ok else "degraded", version=settings.version,
                              database=ok, kv_backend=_kv(request).backend(),
                              llm_backend=settings.llm_backend or "planner",
                              kb_chunks=len(kb.chunks))

    # ── 鉴权（开发用签发；生产由认证服务签发）──
    @app.post("/v1/auth/token", response_model=TokenResponse, tags=["auth"],
              summary="签发开发令牌（生产请关闭）")
    async def token(body: TokenRequest):
        if not settings.dev_token_enabled:
            raise HTTPException(status_code=403, detail="开发令牌接口已关闭，请使用统一认证服务")
        return TokenResponse(**issue_token(body.user_id, settings, body.role,
                                           body.vehicle_model or ""))

    # ── 工具清单 ──
    @app.get("/v1/tools", response_model=List[ToolInfo], tags=["agent"],
              summary="列出可用工具（OpenAI Function Calling schema）")
    async def tools(request: Request, principal: Principal = Depends(principal_of)):
        registry = build_default_registry(settings.kb_path or None)
        return [ToolInfo(name=s["function"]["name"], description=s["function"]["description"],
                         parameters=s["function"]["parameters"]) for s in registry.specs()]

    # ── 会话 ──
    @app.post("/v1/sessions", response_model=SessionInfo, tags=["session"],
              summary="创建会话")
    async def create_session(request: Request, principal: Principal = Depends(principal_of),
                             x_vehicle_model: Optional[str] = Header(None)):
        model = _tenant(principal, x_vehicle_model, settings)
        request.state.user_id, request.state.vehicle_model = principal.user_id, model
        sid = str(uuid.uuid4())
        with _db(request).session() as db:
            row = SessionRow(id=sid, vehicle_model=model, user_id=principal.user_id,
                             title="新会话")
            db.add(row)
            db.flush()
            return SessionInfo(session_id=row.id, vehicle_model=row.vehicle_model,
                               title=row.title, status=row.status,
                               created_at=row.created_at.isoformat() if row.created_at else None)

    @app.get("/v1/sessions", response_model=List[SessionInfo], tags=["session"],
             summary="列出我的会话")
    async def list_sessions(request: Request, principal: Principal = Depends(principal_of),
                            limit: int = 20):
        with _db(request).session() as db:
            rows = (db.query(SessionRow)
                    .filter(SessionRow.user_id == principal.user_id)
                    .order_by(SessionRow.updated_at.desc()).limit(min(limit, 100)).all())
            return [SessionInfo(session_id=r.id, vehicle_model=r.vehicle_model, title=r.title,
                                status=r.status,
                                created_at=r.created_at.isoformat() if r.created_at else None,
                                updated_at=r.updated_at.isoformat() if r.updated_at else None)
                    for r in rows]

    # ── 核心：对话（非流式）──
    @app.post("/v1/chat", response_model=ChatResponse, tags=["agent"],
              summary="对话（非流式）",
              responses={429: {"model": ErrorResponse}, 403: {"model": ErrorResponse}})
    async def chat(request: Request, body: ChatRequest,
                   principal: Principal = Depends(principal_of),
                   x_vehicle_model: Optional[str] = Header(None),
                   idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key")):
        payload = await _prepare(request, body, principal, x_vehicle_model, idempotency_key)
        if isinstance(payload, ChatResponse):
            return payload
        ctx = payload
        result = await asyncio.to_thread(_run_and_persist, request, ctx, None)
        return result

    # ── 核心：对话（SSE 流式）──
    @app.post("/v1/chat/stream", tags=["agent"], summary="对话（SSE 流式）")
    async def chat_stream(request: Request, body: ChatRequest,
                          principal: Principal = Depends(principal_of),
                          x_vehicle_model: Optional[str] = Header(None),
                          idempotency_key: Optional[str] = Header(None,
                                                                  alias="Idempotency-Key")):
        payload = await _prepare(request, body, principal, x_vehicle_model, idempotency_key)
        if isinstance(payload, ChatResponse):        # 幂等重放：直接把结果作为一次性流返回
            async def replay():
                yield _sse("done", payload.model_dump())
            return StreamingResponse(replay(), media_type="text/event-stream")
        ctx = payload

        q: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        t0 = time.perf_counter()
        first_delta = {"ms": None}

        def on_delta(text: str) -> None:
            if first_delta["ms"] is None:
                first_delta["ms"] = (time.perf_counter() - t0) * 1000
            q.put(("delta", text))

        holder: Dict[str, Any] = {}

        def work() -> None:
            try:
                holder["result"] = _run_and_persist(request, ctx, on_delta)
            except Exception as exc:                     # noqa: BLE001
                holder["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                q.put(("__end__", None))

        thread = threading.Thread(target=work, daemon=True)
        thread.start()

        async def event_gen():
            yield _sse("meta", {"trace_id": ctx["trace_id"], "session_id": ctx["session_id"],
                                "vehicle_model": ctx["vehicle_model"],
                                "model": settings.llm_backend or "planner"})
            yield _sse("stage", {"name": "retrieval", "status": "start",
                                 "label": "正在查询手册与车况…"})
            while True:
                try:
                    kind, value = q.get_nowait()
                except queue.Empty:
                    if not thread.is_alive() and q.empty():
                        break
                    await asyncio.sleep(0.01)
                    continue
                if kind == "__end__":
                    break
                if kind == "delta":
                    yield _sse("delta", {"text": value,
                                         "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2)})
            thread.join(timeout=5)
            if "error" in holder:
                yield _sse("error", {"error": holder["error"], "trace_id": ctx["trace_id"]})
                return
            result: ChatResponse = holder["result"]
            result.ttft_ms = round(first_delta["ms"], 2) if first_delta["ms"] else result.latency_ms
            yield _sse("citations", {"citations": result.citations,
                                     "risk_level": result.risk_level})
            yield _sse("done", result.model_dump())

        return StreamingResponse(event_gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no",
                                          "X-Trace-Id": ctx["trace_id"]})

    # ── 写操作确认（HITL 落库）──
    @app.post("/v1/writes/confirm", response_model=ConfirmResponse, tags=["agent"],
              summary="确认写操作（预约到店等）")
    async def confirm_write(request: Request, body: ConfirmRequest,
                            principal: Principal = Depends(principal_of)):
        store = ConfirmationStore(_db(request))
        created = store.confirm(principal.user_id, body.action_key, body.session_id)
        AuditService(_db(request)).log(principal.user_id, "confirm_write", body.action_key,
                                       before="unconfirmed", after="confirmed",
                                       trace_id=_trace_id(request))
        return ConfirmResponse(action_key=body.action_key, confirmed=True,
                               already_confirmed=not created)

    # ── 反馈（闭环数据）──
    @app.post("/v1/feedback", status_code=201, tags=["agent"], summary="提交回答反馈")
    async def feedback(request: Request, body: FeedbackRequest,
                       principal: Principal = Depends(principal_of)):
        with _db(request).session() as db:
            db.add(Feedback(message_id=body.message_id, user_id=principal.user_id,
                            rating=body.rating, comment=body.comment[:1000]))
        return {"ok": True}

    # ── 运维/治理接口 ──
    @app.get("/v1/admin/audit", tags=["ops"], summary="最近审计日志（需 service/admin 角色）")
    async def audit(request: Request, limit: int = 20,
                    principal: Principal = Depends(principal_of)):
        if not principal.is_privileged:
            raise HTTPException(status_code=403, detail="需要 service 或 admin 角色")
        return AuditService(_db(request)).recent(min(limit, 200))

    @app.get("/v1/admin/metrics", tags=["ops"], summary="运行指标聚合（需 service/admin 角色）")
    async def metrics(request: Request, principal: Principal = Depends(principal_of)):
        if not principal.is_privileged:
            raise HTTPException(status_code=403, detail="需要 service 或 admin 角色")
        db = _db(request)
        with db.session() as session:
            from sqlalchemy import func
            from agent.service.db import IdempotencyKey, Quota, ToolCallRow as TC
            n_messages = session.query(func.count(MessageRow.id)).scalar() or 0
            tokens = session.query(func.sum(MessageRow.tokens_in + MessageRow.tokens_out)).scalar() or 0
            cost = session.query(func.sum(MessageRow.cost_usd)).scalar() or 0.0
            blocked = session.query(func.count(TC.id)).filter(TC.blocked.is_(True)).scalar() or 0
            tool_calls = session.query(func.count(TC.id)).scalar() or 0
            idem = session.query(func.count(IdempotencyKey.id)).scalar() or 0
            quota_tokens = session.query(func.sum(Quota.tokens_used)).scalar() or 0
            failures = (session.query(MessageRow.failure, func.count(MessageRow.id))
                        .group_by(MessageRow.failure).all())
        return {"messages": n_messages, "tokens_total": int(tokens), "cost_usd": round(float(cost), 6),
                "tool_calls": tool_calls, "tool_blocked": blocked,
                "idempotency_keys": idem, "quota_tokens_today": int(quota_tokens),
                "failure_distribution": {str(k): v for k, v in failures},
                "kv_backend": _kv(request).backend(),
                "kb_load_ms": round(_runtime(request).kb_load_ms, 1)}

    return app


# ── 请求上下文准备（鉴权之后、执行之前的所有校验与幂等）────────────────

def _tenant(principal: Principal, requested: Optional[str], settings: Settings) -> str:
    try:
        return resolve_tenant(principal, requested, settings)
    except AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


async def _prepare(request: Request, body: ChatRequest, principal: Principal,
                   x_vehicle_model: Optional[str], idempotency_key: Optional[str]):
    """返回执行上下文 dict；若命中幂等重放则直接返回 ChatResponse。"""
    settings = _settings(request)
    db = _db(request)
    kv = _kv(request)
    model = _tenant(principal, body.vehicle_model or x_vehicle_model, settings)
    request.state.user_id, request.state.vehicle_model = principal.user_id, model

    # ① 限流
    limit = RateLimiter(kv, settings).hit(principal.user_id, model)
    if not limit.allowed:
        raise HTTPException(status_code=429,
                            detail=f"请求过于频繁，请在 {limit.retry_after}s 后重试",
                            headers={"Retry-After": str(limit.retry_after)})

    # ② 配额
    quota = QuotaService(db, settings).check(principal.user_id)
    if not quota.allowed:
        raise HTTPException(status_code=429,
                            detail=f"今日 token 配额已用完（{quota.used}/{quota.limit}）")

    # ③ 幂等
    idem = IdempotencyStore(db, settings)
    request_hash = idem.hash_request({"message": body.message, "model": model})
    if idempotency_key:
        outcome = idem.begin(principal.user_id, idempotency_key, "/v1/chat", request_hash)
        if outcome.conflict:
            raise HTTPException(status_code=422, detail=outcome.reason)
        if outcome.in_progress:
            raise HTTPException(status_code=409, detail=outcome.reason,
                                headers={"Retry-After": "1"})
        if outcome.replay and outcome.response:
            replay = ChatResponse(**outcome.response)
            replay.idempotent_replay = True
            return replay

    # ④ 会话
    session_id = body.session_id
    with db.session() as session:
        row = session.get(SessionRow, session_id) if session_id else None
        if row is None:
            session_id = session_id or str(uuid.uuid4())
            row = SessionRow(id=session_id, vehicle_model=model, user_id=principal.user_id,
                             title=body.message[:24])
            session.add(row)
        elif row.user_id != principal.user_id:
            raise HTTPException(status_code=403, detail="无权访问该会话")
        session.add(MessageRow(session_id=session_id, role="user", content=body.message))

    return {"trace_id": _trace_id(request), "session_id": session_id, "vehicle_model": model,
            "user_id": principal.user_id, "message": body.message,
            "idempotency_key": idempotency_key, "principal": principal,
            "request_hash": request_hash}


def _run_and_persist(request: Request, ctx: Dict[str, Any],
                     on_delta: Optional[Callable[[str], None]]) -> ChatResponse:
    """在工作线程里执行 Agent，并把消息、工具调用、审计、配额落库。"""
    settings = _settings(request)
    db = _db(request)
    runtime = _runtime(request)
    t0 = time.perf_counter()

    confirmed = ConfirmationStore(db).load(ctx["user_id"])
    agent, tracer, registry = runtime.build_agent(ctx["vehicle_model"], confirmed, on_delta,
                                                  idempotency_key=ctx.get("idempotency_key") or "")
    # on_delta 必须传进 run()：AgentGraph 据此把最终答案改为流式生成（SSE 的 token 来源）
    state = agent.run(ctx["message"], on_delta=on_delta)

    latency_ms = (time.perf_counter() - t0) * 1000
    tokens_in, tokens_out = state.tokens_in, state.tokens_out
    cost = tracer.totals()["cost_usd"]
    failure = tracer.failure
    needs_confirmation = None
    for call in state.tool_calls:
        if call["tool"] in WRITE_TOOLS and not call["ok"]:
            needs_confirmation = call["arguments"].get("item")

    audit = AuditService(db)
    with db.session() as session:
        message = MessageRow(session_id=ctx["session_id"], role="assistant",
                             content=state.answer or "", status=state.status,
                             citations=json.dumps(state.citations, ensure_ascii=False),
                             tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost,
                             trace_id=ctx["trace_id"], failure=failure, latency_ms=latency_ms)
        session.add(message)
        session.flush()
        message_id = message.id
        for call in state.tool_calls:
            ok = bool(call.get("ok"))
            blocked = bool(call.get("blocked"))
            write = call["tool"] in WRITE_TOOLS
            session.add(ToolCallRow(
                session_id=ctx["session_id"], message_id=message_id, tool=call["tool"],
                arguments=json.dumps(call.get("arguments") or {}, ensure_ascii=False),
                ok=ok, blocked=blocked,
                reason=(call.get("error") or ""),
                repeated=bool(call.get("repeated")),
                latency_ms=float(call.get("latency_ms") or 0.0),
                # 幂等键只记录在写操作上（请求级唯一约束在 idempotency_keys 表）
                idempotency_key=(ctx["idempotency_key"] if write else None),
                trace_id=ctx["trace_id"]))
            if write:
                # 同一事务里写审计，避免嵌套 session（SQLite 下会 database is locked）
                audit.log_in_session(session, ctx["user_id"], "write_tool", call["tool"],
                                     before="requested",
                                     after={"ok": ok, "blocked": blocked,
                                            "args": call.get("arguments")},
                                     trace_id=ctx["trace_id"])
        session.query(SessionRow).filter(SessionRow.id == ctx["session_id"]).update(
            {"updated_at": dt.datetime.utcnow()})

    quota = QuotaService(db, settings).record(ctx["user_id"], tokens_in + tokens_out, cost)

    response = ChatResponse(
        trace_id=ctx["trace_id"], session_id=ctx["session_id"], message_id=message_id,
        status=state.status, answer=state.answer or "", citations=state.citations,
        tool_calls=[ToolCallInfo(tool=c["tool"], arguments=c.get("arguments") or {},
                                 ok=bool(c.get("ok")), blocked=bool(c.get("blocked")),
                                 repeated=bool(c.get("repeated")),
                                 repaired=bool(c.get("repaired")),
                                 reason=str(c.get("error") or ""),
                                 idempotency_key=(ctx.get("idempotency_key") or ""
                                                  if c["tool"] in WRITE_TOOLS else ""),
                                 latency_ms=float(c.get("latency_ms") or 0.0))
                    for c in state.tool_calls],
        risk_level="critical" if state.injection_flagged else "ok",
        needs_confirmation=needs_confirmation, injection_flagged=state.injection_flagged,
        tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost, latency_ms=round(latency_ms, 2),
        ttft_ms=round(latency_ms, 2),
        quota={"used": quota.used, "limit": quota.limit, "remaining": quota.remaining})

    if ctx.get("idempotency_key"):
        IdempotencyStore(db, settings).complete(ctx["user_id"], ctx["idempotency_key"],
                                                response.model_dump())
    return response


app = create_app()
