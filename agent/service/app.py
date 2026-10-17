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
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from agent.cache import SemanticCache, cacheable_status
from agent.graph import AgentConfig, AgentGraph, AgentState
from agent.guardrails.policy import PolicyEngine
from agent.kb import KnowledgeBase
from agent.llm import LLMBackend, OpenAICompatibleLLM, RuleBasedPlannerLLM, StreamingRulePlanner
from agent.memory import ConversationMemory, VehicleProfile
from agent.obs.tracer import Tracer
from agent.reflection import Reflector
from agent.service.auth import AuthError, Principal, decode_token, issue_token, resolve_tenant
from agent.service.config import Settings
from agent.service.db import (
    Database,
    Feedback,
    MessageRow,
    SessionRow,
    ToolCallRow,
    get_database,
    reset_database,
)
from agent.service.faults import DegradingLLM, FaultyLLM
from agent.service.kb_registry import KBRegistry
from agent.service.kv import KVStore, build_kv
from agent.service.metrics import METRICS
from agent.service.schemas import (
    ChatRequest,
    ChatResponse,
    ConfirmRequest,
    ConfirmResponse,
    ErrorResponse,
    FeedbackRequest,
    HealthResponse,
    KBActivateRequest,
    KBRegisterRequest,
    KBRollbackRequest,
    SessionInfo,
    ShutdownPrepareRequest,
    TokenRequest,
    TokenResponse,
    ToolCallInfo,
    ToolInfo,
)
from agent.service.security import (
    AuditService,
    ConfirmationStore,
    IdempotencyStore,
    QuotaService,
    RateLimiter,
)
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
            inner: LLMBackend = StreamingRulePlanner(
                ms_per_chunk=self.settings.llm_ms_per_token * 6)
        elif backend == "openai":
            inner = OpenAICompatibleLLM(base_url=self.settings.llm_base_url,
                                        model=self.settings.llm_model,
                                        provider="vllm")
        else:
            inner = RuleBasedPlannerLLM()
        # 故障注入（演练用）：包一层可注入超时/报错
        if (self.settings.fault_mode or "none").lower() != "none":
            inner = FaultyLLM(inner, mode=self.settings.fault_mode,
                              rate=self.settings.fault_rate,
                              sleep_s=self.settings.fault_sleep_s)
        # 降级链：主后端失败 → 回退到离线规则规划器（座舱不至于彻底失能）
        if self.settings.enable_degradation:
            inner = DegradingLLM(
                inner, on_degrade=lambda reason: METRICS.inc("agent_degraded_total",
                                                             {"reason": reason[:40]}))
        return inner

    def build_agent(self, vehicle_model: str, confirmed: Optional[set] = None,
                    on_delta: Optional[Callable[[str], None]] = None,
                    idempotency_key: str = "",
                    history: Optional[List[Tuple[str, str]]] = None,
                    kb: Optional[KnowledgeBase] = None
                    ) -> Tuple[AgentGraph, Tracer, ToolRegistry]:
        # kb 允许外部注入：多版本路由（灰度/回滚）时由 KBRegistry 决定用哪个版本的实例，
        # 不再从这里按车型固定取缓存。
        kb = kb if kb is not None else self.kb(vehicle_model)
        # 把**已缓存的 kb** 传进去：否则这里会从磁盘重新加载并重建 BM25 索引
        # （9301 块实测 ≈16 s/次，而单次检索只要 ≈12 ms），缓存就白做了。
        registry = build_default_registry(self.settings.kb_path or None, kb=kb)
        if confirmed:
            registry.confirmed_actions |= set(confirmed)
        tracer = Tracer()
        policy = self.build_policy()
        config = AgentConfig()
        # 服务层收紧：写操作必须带 Idempotency-Key（客户端重试不会重复下单）
        config.idempotency_key = idempotency_key
        memory = ConversationMemory(profile=VehicleProfile(model=vehicle_model))
        # 会话记忆落库回填：没有这一步，服务层的多轮是**断的**——
        # `sessions`/`messages` 表里存了历史，但每个请求都新建一份空记忆，
        # 于是"它怎么关闭"这类追问无法解析指代。
        for role, content in (history or []):
            if role == "user":
                memory.add_user(content)
            elif role == "assistant" and content:
                memory.add_assistant(content)
        agent = AgentGraph(self.build_llm(on_delta), registry,
                          reflector=Reflector(
                              evidence_overlap_threshold=config.evidence_overlap_threshold),
                          memory=memory, config=config, tracer=tracer, policy=policy,
                          agent_name="service")
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



# ══ 优雅停机（BE8）══════════════════════════════════════════════════
# 为什么需要它：滚动发布/缩容时进程收到 SIGTERM 就退出，**在途请求会被硬切**——
# 用户看到 500 或半个流式答案，而且此前做的降级链在发布瞬间完全没用（它救不了被砍断的连接）。
# 做法是标准的三段式：**先把流量摘掉（/readyz 503）→ 等排空（有上限）→ 再关资源**。
# 顺序不能反：先关资源会让在途请求直接失败。

def _inflight_inc(app: FastAPI) -> int:
    with app.state.inflight_lock:
        app.state.inflight += 1
        return app.state.inflight


def _inflight_dec(app: FastAPI) -> int:
    with app.state.inflight_lock:
        app.state.inflight = max(0, app.state.inflight - 1)
        return app.state.inflight


def prepare_shutdown(app: FastAPI, reason: str = "manual", grace_s: float = 20.0) -> Dict[str, Any]:
    """进入"停机准备"状态：/readyz 立刻返回 503，但**继续服务在途与短时新请求**。

    生产环境应当由 K8s 的 preStop hook 调用（`POST /v1/admin/shutdown/prepare`），
    给负载均衡器留出摘流量的时间——这是"优雅停机"里最容易被漏掉的一环：
    进程自己排空了，但 LB 还在往它上面转发。
    """
    app.state.shutting_down = True
    app.state.shutdown_reason = reason
    app.state.shutdown_deadline = time.time() + max(0.0, grace_s)
    logger.warning("进入停机准备 reason=%s grace=%.1fs inflight=%s",
                   reason, grace_s, app.state.inflight)
    return {"shutting_down": True, "reason": reason, "grace_s": grace_s,
            "inflight": app.state.inflight}


def drain(app: FastAPI, timeout: float = 20.0, poll_s: float = 0.05) -> bool:
    """等待在途请求归零；超时返回 False（**不阻塞退出**，否则停机窗口会无限拉长）。"""
    deadline = time.time() + max(0.0, timeout)
    while time.time() < deadline:
        with app.state.inflight_lock:
            if app.state.inflight <= 0:
                return True
        time.sleep(poll_s)
    return False


# 管理类接口的权限门：知识库版本切换会影响**所有用户**，必须 service/admin。
def _require_service(principal: Principal) -> None:
    if getattr(principal, "role", "") not in ("service", "admin"):
        raise HTTPException(status_code=403, detail="需要 service 或 admin 角色")


def _audit(request: Request, principal: Principal, action: str, target: str,
           after: Dict[str, Any]) -> None:
    """把管理动作写进审计表——换手册/回滚必须留痕，否则"谁在什么时候切了哪版"说不清。"""
    try:
        with _db(request).session() as session:
            AuditService(_db(request)).log_in_session(
                session, principal.user_id, action, target,
                before="", after=after, trace_id=_trace_id(request))
    except Exception as exc:                                       # noqa: BLE001
        logger.warning("审计写入失败 action=%s target=%s err=%s", action, target, exc)


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _runtime(request: Request) -> AgentRuntime:
    return request.app.state.runtime


def _db(request: Request) -> Database:
    return request.app.state.db


def _kv(request: Request) -> KVStore:
    return request.app.state.kv


def _cache(request: Request) -> Optional[SemanticCache]:
    """语义缓存实例（未启用时返回 None）。"""
    return getattr(request.app.state, "semantic_cache", None)


def _kb_registry(request: Request):
    """知识库版本注册表（决定本次请求用哪个版本）。"""
    return getattr(request.app.state, "kb_registry", None)


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
        app.state.kv = build_kv(settings.kv_backend, settings.redis_url,
                               settings.kv_sqlite_path)
        # 语义缓存：座舱重复率高，命中一次就省掉「检索 + 一次大模型调用」。
        # 默认落 KV（多副本共享、进程重启不丢）；`SERVICE_SEMANTIC_CACHE_SHARED=0` 则退回进程内。
        app.state.semantic_cache = None
        if settings.enable_semantic_cache:
            app.state.semantic_cache = SemanticCache(
                threshold=settings.semantic_cache_threshold,
                max_entries=settings.semantic_cache_max_entries,
                ttl_seconds=settings.semantic_cache_ttl_s,
                kv=app.state.kv if settings.semantic_cache_shared else None)
        app.state.runtime = AgentRuntime(settings)
        # ── 知识库版本注册表（多版本共存 + 灰度 + 回滚）──
        # 启动时把当前语料注册为 v1，并设为生效版本；换手册 = 注册新版本 + 灰度/切换，
        # 不需要改配置重启（这是"线上怎么换手册、改错了怎么回滚"的答案）。
        kb_registry = KBRegistry()
        default_kb = app.state.runtime.kb(settings.default_vehicle_model)
        initial = kb_registry.register_loaded(
            default_kb, vehicle_model=settings.default_vehicle_model,
            version=f"v1-{default_kb.checksum()[:8]}")
        for model in settings.allowed_models:
            kb_registry.activate(initial.version, model, 100)
        app.state.kb_registry = kb_registry
        app.state.started_at = time.time()
        # 停机状态（BE8）：inflight 是"正在处理中的请求数"，drain 靠它判断能不能退出
        app.state.inflight = 0
        app.state.inflight_lock = threading.Lock()
        app.state.shutting_down = False
        app.state.shutdown_reason = ""
        app.state.shutdown_deadline = 0.0
        # 预热默认租户知识库，让 /readyz 有真实含义
        app.state.runtime.kb(settings.default_vehicle_model)
        if settings.jwt_secret == "dev-secret-change-me":
            logger.warning("正在使用默认 JWT 密钥，生产环境请通过 SERVICE_JWT_SECRET 注入")
        logger.info("服务启动完成 db=%s kv=%s llm=%s cache=%s",
                    db.engine.url.get_backend_name(), app.state.kv.backend(),
                    settings.llm_backend or "planner",
                    ("shared" if getattr(app.state.semantic_cache, "shared", False)
                     else ("local" if app.state.semantic_cache else "off")))
        try:
            yield
        finally:
            # 优雅停机：先摘流量（/readyz 已因 shutting_down 返回 503），再等排空，最后关资源。
            prepare_shutdown(app, "lifespan_exit", settings.shutdown_grace_s)
            drained = drain(app, settings.shutdown_grace_s)
            with app.state.inflight_lock:
                left = app.state.inflight
            if drained:
                logger.info("优雅停机完成：在途请求已排空")
            else:
                logger.warning("优雅停机超时：仍有 %s 个在途请求，强制关闭（上限 %.0fs）",
                               left, settings.shutdown_grace_s)
            METRICS.inc("agent_shutdown_total", {"drained": str(drained).lower()})
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
        # 在途计数必须用 try/finally：异常路径也要减一，否则排空会永远等下去
        _inflight_inc(request.app)
        try:
            response = await call_next(request)
        finally:
            _inflight_dec(request.app)
        cost_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Trace-Id"] = trace_id
        # 指标：请求量按（路径, 状态码）计数 + 耗时直方图（多进程时各实例各一份，由 Prometheus 聚合）
        METRICS.inc("agent_http_requests_total",
                    {"path": request.url.path, "status": str(response.status_code)})
        METRICS.observe_latency(cost_ms / 1000.0)
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
        # 停机准备中：**立刻**返回 503，让负载均衡器把流量摘走；
        # 此时 /healthz 仍是 200 —— 存活探针不能先挂，否则容器会先被 K8s 杀掉，排空就白做了。
        if getattr(request.app.state, "shutting_down", False):
            raise HTTPException(status_code=503,
                                detail=f"服务正在停机准备（reason={request.app.state.shutdown_reason}，"
                                       f"在途 {request.app.state.inflight}）")
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
    async def tools(request: Request, principal: Principal = Depends(principal_of),
                    x_vehicle_model: Optional[str] = Header(None)):
        # 复用运行时缓存的知识库（不要在这里重新加载：9301 块 ≈16 s）
        model = settings.default_vehicle_model
        kb = _runtime(request).kb(model)
        registry = build_default_registry(settings.kb_path or None, kb=kb)
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
            if first_delta["ms"]:
                METRICS.observe_ttft(first_delta["ms"] / 1000.0)     # SSE 首字延迟直方图
            yield _sse("citations", {"citations": result.citations,
                                     "risk_level": result.risk_level})
            done_payload = result.model_dump()
            # 停机中完成的流：明确告诉客户端"这条流是被停机收尾的"，便于它决定是否重连取剩余内容
            done_payload["shutdown"] = bool(getattr(request.app.state, "shutting_down", False))
            yield _sse("done", done_payload)

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

    # ── 停机准备（供 K8s preStop / 发布脚本调用）──
    @app.post("/v1/admin/shutdown/prepare", tags=["ops"], summary="进入停机准备（摘流量+排空）")
    async def shutdown_prepare(request: Request, body: ShutdownPrepareRequest,
                               principal: Principal = Depends(principal_of)):
        """把"要停机了"提前告诉实例：/readyz 转 503、等待在途排空。

        典型用法（K8s）：
            lifecycle.preStop: curl -X POST /v1/admin/shutdown/prepare -d '{"grace_s":15}'
        这样 LB 有时间摘流量，进程也有时间排空——**两步都做完才算优雅**。
        """
        _require_service(principal)
        info = prepare_shutdown(request.app, body.reason or "prestop",
                               body.grace_s or settings.shutdown_grace_s)
        _audit(request, principal, "shutdown_prepare", f"inflight={info['inflight']}", info)
        return info

    @app.get("/v1/admin/shutdown/status", tags=["ops"], summary="停机状态与在途请求数")
    async def shutdown_status(request: Request,
                              principal: Principal = Depends(principal_of)):
        _require_service(principal)
        return {"shutting_down": bool(getattr(request.app.state, "shutting_down", False)),
                "reason": getattr(request.app.state, "shutdown_reason", ""),
                "inflight": getattr(request.app.state, "inflight", 0),
                "deadline_in_s": max(0.0, round(
                    getattr(request.app.state, "shutdown_deadline", 0.0) - time.time(), 2))}

    # ── 知识库版本管理（多版本 / 灰度 / 回滚）──
    @app.get("/v1/admin/kb/versions", tags=["ops"], summary="列出知识库版本")
    async def list_kb_versions(request: Request, vehicle_model: Optional[str] = None,
                               principal: Principal = Depends(principal_of)):
        _require_service(principal)
        registry = _kb_registry(request)
        records = registry.list(vehicle_model) if registry else []
        return {"versions": [r.to_dict() for r in records],
                "stats": registry.stats() if registry else {}}

    @app.post("/v1/admin/kb/register", tags=["ops"], summary="注册新的知识库版本")
    async def register_kb_version(request: Request, body: KBRegisterRequest,
                                  principal: Principal = Depends(principal_of)):
        """注册一个知识库文件为新版本（算 checksum、加载、入注册表），**不改变线上路由**。"""
        _require_service(principal)
        registry = _kb_registry(request)
        if registry is None:
            raise HTTPException(status_code=503, detail="知识库注册表不可用")
        if not body.path or not os.path.isfile(body.path):
            raise HTTPException(status_code=400, detail=f"文件不存在：{body.path}")
        try:
            record = registry.register(body.path, body.vehicle_model or settings.default_vehicle_model,
                                       body.version)
        except Exception as exc:                                   # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"注册失败：{exc}") from exc
        METRICS.inc("agent_kb_version_registered_total")
        _audit(request, principal, "kb_register", record.version, record.to_dict())
        return record.to_dict()

    @app.post("/v1/admin/kb/activate", tags=["ops"], summary="切换/灰度生效知识库版本")
    async def activate_kb_version(request: Request, body: KBActivateRequest,
                                  principal: Principal = Depends(principal_of)):
        """`traffic_pct` 三态：100=全量切换（可回滚）；1~99=灰度；0=撤销灰度。"""
        _require_service(principal)
        registry = _kb_registry(request)
        if registry is None:
            raise HTTPException(status_code=503, detail="知识库注册表不可用")
        try:
            record = registry.activate(body.version,
                                       body.vehicle_model or settings.default_vehicle_model,
                                       body.traffic_pct)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        METRICS.inc("agent_kb_version_activated_total",
                    {"pct": str(body.traffic_pct if body.traffic_pct in (0, 100) else "gray")})
        _audit(request, principal, "kb_activate", body.version,
               {"traffic_pct": body.traffic_pct, "vehicle_model": body.vehicle_model})
        return record.to_dict()

    @app.post("/v1/admin/kb/rollback", tags=["ops"], summary="回滚到上一个生效版本")
    async def rollback_kb_version(request: Request, body: KBRollbackRequest,
                                  principal: Principal = Depends(principal_of)):
        _require_service(principal)
        registry = _kb_registry(request)
        if registry is None:
            raise HTTPException(status_code=503, detail="知识库注册表不可用")
        try:
            record = registry.rollback(body.vehicle_model or settings.default_vehicle_model)
        except KeyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        METRICS.inc("agent_kb_version_rollback_total")
        _audit(request, principal, "kb_rollback", record.version, record.to_dict())
        return record.to_dict()

    # ── 指标（Prometheus 文本格式，零依赖手写）──
    @app.get("/metrics", tags=["ops"], summary="Prometheus 指标",
             response_class=PlainTextResponse)
    async def metrics_endpoint(request: Request,
                              authorization: Optional[str] = Header(None)):
        """默认仅 service/admin 可访问；`SERVICE_METRICS_PUBLIC=1` 时公开（Prometheus 抓取）。

        注意：计数器是**进程内**的，`--workers 4` 时每个 worker 各一份，
        由 Prometheus 分别 scrape 后在服务端聚合（标准做法）。
        """
        if not settings.metrics_public:
            try:
                principal = principal_of(request, authorization)
            except HTTPException as exc:
                raise exc
            if not principal.is_privileged:
                raise HTTPException(status_code=403, detail="需要 service 或 admin 角色")
        return PlainTextResponse(METRICS.render())

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

            from agent.service.db import IdempotencyKey, Quota
            from agent.service.db import ToolCallRow as TC
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
        METRICS.inc("agent_rate_limited_total", {"vehicle_model": model})
        raise HTTPException(status_code=429,
                            detail=f"请求过于频繁，请在 {limit.retry_after}s 后重试",
                            headers={"Retry-After": str(limit.retry_after)})

    # ② 配额
    quota = QuotaService(db, settings).check(principal.user_id)
    if not quota.allowed:
        METRICS.inc("agent_quota_exceeded_total")
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
            METRICS.inc("agent_idempotent_replays_total")
            replay = ChatResponse(**outcome.response)
            replay.idempotent_replay = True
            return replay

    # ④ 会话：**这里只读、不写**。
    # 写路径优化（针对压测暴露的 SQLite 单写者瓶颈）：把「用户消息 + 助手消息 + 工具调用
    # + 审计 + 配额 + 幂等完成」合并到 _run_and_persist 里的**同一个事务**。
    # 优化前每请求要开 4 次写事务（user 消息 / assistant+工具+审计 / 配额 / 幂等），
    # 而 SQLite 只允许一个写者 —— 事务数直接决定并发下的排队长度。
    session_id = body.session_id
    session_is_new = False
    with db.session() as session:
        row = session.get(SessionRow, session_id) if session_id else None
        if row is not None and row.user_id != principal.user_id:
            raise HTTPException(status_code=403, detail="无权访问该会话")
        if row is None:
            session_id = session_id or str(uuid.uuid4())
            session_is_new = True

    return {"trace_id": _trace_id(request), "session_id": session_id, "vehicle_model": model,
            "user_id": principal.user_id, "message": body.message,
            "idempotency_key": idempotency_key, "principal": principal,
            "request_hash": request_hash, "session_is_new": session_is_new}


def _load_history(db: Database, session_id: str, max_turns: int = 6) -> List[Tuple[str, str]]:
    """读取会话最近的历史消息（按时间正序），用于回填对话记忆。

    只取最近 `max_turns` 轮，避免把整段历史塞进 prompt（成本与上下文都要控）。
    """
    with db.session() as session:
        rows = (session.query(MessageRow)
                .filter(MessageRow.session_id == session_id)
                .order_by(MessageRow.id.desc())
                .limit(max_turns * 2).all())
    history: List[Tuple[str, str]] = [(r.role, r.content or "") for r in reversed(rows)]
    # 去掉末尾刚写入的那条 user 消息（它就是本次提问，run() 内部会再 add_user）
    while history and history[-1][0] == "user":
        history.pop()
    return history


def _run_and_persist(request: Request, ctx: Dict[str, Any],
                     on_delta: Optional[Callable[[str], None]]) -> ChatResponse:
    """在工作线程里执行 Agent，并把消息、工具调用、审计、配额落库。"""
    settings = _settings(request)
    db = _db(request)
    runtime = _runtime(request)
    t0 = time.perf_counter()

    confirmed = ConfirmationStore(db).load(ctx["user_id"])
    history = _load_history(db, ctx["session_id"])

    # ── 知识库版本路由（多版本共存 / 灰度 / 回滚）──
    # 灰度命中时用"影子版本"的 KB 实例；同一用户的分桶是稳定的，不会在一次会话里跳版本。
    kb_registry = _kb_registry(request)
    kb_version, versioned_kb = "", None
    if kb_registry is not None:
        record = kb_registry.resolve(ctx["vehicle_model"], ctx["user_id"])
        kb_version = record.version
        versioned_kb = kb_registry.kb_for(record.version)
        METRICS.inc("agent_kb_version_routed_total", {"version": record.version})

    agent, tracer, registry = runtime.build_agent(
        ctx["vehicle_model"], confirmed, on_delta,
        idempotency_key=ctx.get("idempotency_key") or "", history=history,
        kb=versioned_kb)

    # ── 语义缓存：命中则**完全跳过** agent.run()（省一次检索 + 一次大模型调用）──
    # 键的作用域是「车型租户」，指纹里包含车况与角色：
    #   ① 不同租户即使问同一句，也各自独立（手册版本/政策口径可能不同）；
    #   ② 车况变了（胎压 228→148、告警灯新增）指纹就变，绝不拿旧车况的答案答新状态。
    cache = _cache(request)
    fingerprint = SemanticCache.fingerprint(getattr(registry, "telemetry", {}) or {},
                                            role=getattr(ctx.get("principal"), "role", "driver"))
    # 作用域必须含**知识库版本**：否则切换/灰度版本之后，语义缓存仍会把旧版本的答案返给新版本
    # 的请求——"版本切了但回答还是老的"，这是比缓存不命中严重得多的正确性问题。
    scope = f"{ctx['vehicle_model']}|{kb_version}"
    # 有历史的多轮请求不查缓存：同一个问法在不同上下文里的含义可能不同（指代/话题继承）。
    cacheable_req = (cache is not None and not ctx.get("idempotency_key")
                     and not history and len(ctx["message"]) >= 4)
    lookup = (cache.get(ctx["message"], fingerprint, cacheable=cacheable_req, scope=scope)
              if cache is not None else None)
    cache_hit = bool(lookup and lookup.hit)

    if cache_hit and lookup is not None and lookup.entry is not None:
        METRICS.inc("agent_cache_lookups_total", {"result": lookup.kind})
        entry = lookup.entry
        state = AgentState(question=ctx["message"], resolved_question=ctx["message"],
                           route="cache", status=entry.status or "answered",
                           answer=entry.answer, citations=list(entry.citations))
        state.trace.append({"node": "semantic_cache", "hit": lookup.kind,
                            "similarity": lookup.similarity})
        # 流式接口也要吐 token：直接把缓存答案推给 on_delta，首字延迟≈0
        if on_delta is not None:
            on_delta(entry.answer)
    else:
        if cache is not None:
            METRICS.inc("agent_cache_lookups_total",
                        {"result": lookup.kind if lookup else "off"})
        # on_delta 必须传进 run()：AgentGraph 据此把最终答案改为流式生成（SSE 的 token 来源）
        state = agent.run(ctx["message"], on_delta=on_delta)

    # 降级统计（主推理后端不可用时回退到了离线规则规划器）
    degraded, degrade_reason = False, ""
    degrade_stats = getattr(agent.llm, "stats", None)
    if callable(degrade_stats):
        info = degrade_stats()
        degraded = bool(info.get("degraded_calls"))
        degrade_reason = str(info.get("last_error") or "")[:200]
        if degraded:
            METRICS.inc("agent_degraded_requests_total")

    latency_ms = (time.perf_counter() - t0) * 1000
    tokens_in, tokens_out = state.tokens_in, state.tokens_out
    cost = tracer.totals()["cost_usd"]
    failure = tracer.failure
    needs_confirmation = None
    for call in state.tool_calls:
        if call["tool"] in WRITE_TOOLS and not call["ok"]:
            needs_confirmation = call["arguments"].get("item")

    # ── 语义缓存写入（只在"这一轮的答案值得复用"时才写）──
    # 五条排除规则，每条都对应一次真实的风险：
    #   ① 非 answered（拒答/待确认/出错/步数耗尽）—— 把一次失败固化成长期答案；
    #   ② 降级产物 —— 故障期间的兜底内容，缓存它会把一次故障的影响延长到 TTL 结束；
    #   ③ 调用过写操作 —— 写结果是"这一次的事"，不能被下一轮复用（且已被幂等/审计管理）；
    #   ④ 注入标记 —— 被污染输入产生的答案不可复用；
    #   ⑤ degraded/needs_confirmation 之外的 safety_injected 不排除：安全指令本来就该复用。
    if cache is not None and not cache_hit:
        wrote = any(c["tool"] in WRITE_TOOLS for c in state.tool_calls)
        blocked_reason = ""
        if not cacheable_status(state.status):
            blocked_reason = f"status={state.status}"
        elif degraded:
            blocked_reason = "degraded"
        elif wrote:
            blocked_reason = "write_tool"
        elif state.injection_flagged:
            blocked_reason = "injection_flagged"
        elif history:
            blocked_reason = "multi_turn"
        if blocked_reason:
            METRICS.inc("agent_cache_skipped_total", {"reason": blocked_reason[:40]})
            state.trace.append({"node": "semantic_cache", "stored": False,
                                "reason": blocked_reason})
        elif cache.put(ctx["message"], state.answer or "", state.citations,
                       status=state.status, fingerprint=fingerprint, scope=scope) is not None:
            METRICS.inc("agent_cache_stored_total")
            state.trace.append({"node": "semantic_cache", "stored": True})

    audit = AuditService(db)
    quota_service = QuotaService(db, settings)
    idem_store = IdempotencyStore(db, settings)
    # ── 单事务写路径 ──
    # 优化前：user 消息(1) + assistant/工具/审计(1) + 配额(1) + 幂等完成(1) = 4 次写事务/请求。
    # SQLite 只允许一个写者，事务数直接决定并发排队长度（压测：QPS 随并发下降、P99 爆到 1.6s）。
    # 现在合并为 1 次：会话(如需) + user 消息 + assistant 消息 + 工具调用 + 审计 + 配额 + 幂等完成。
    with db.session() as session:
        if ctx.get("session_is_new"):
            session.add(SessionRow(id=ctx["session_id"], vehicle_model=ctx["vehicle_model"],
                                   user_id=ctx["user_id"], title=ctx["message"][:24]))
        session.add(MessageRow(session_id=ctx["session_id"], role="user",
                               content=ctx["message"], trace_id=ctx["trace_id"]))
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
                audit.log_in_session(session, ctx["user_id"], "write_tool", call["tool"],
                                     before="requested",
                                     after={"ok": ok, "blocked": blocked,
                                            "args": call.get("arguments")},
                                     trace_id=ctx["trace_id"])
        session.query(SessionRow).filter(SessionRow.id == ctx["session_id"]).update(
            {"updated_at": dt.datetime.utcnow()})
        # 配额与幂等完成也进同一事务（读一次写一次，不再各开一个事务）
        quota = quota_service.record_in_session(session, ctx["user_id"],
                                                tokens_in + tokens_out, cost)
        # 响应必须在事务内构造：message_id 来自 flush，quota 来自上面的记账，
        # 而且要立刻把响应写回幂等键（重放才能拿到同一份结果）。
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
            tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost,
            latency_ms=round(latency_ms, 2), ttft_ms=round(latency_ms, 2),
            degraded=degraded, degrade_reason=degrade_reason,
            cached=cache_hit, kb_version=kb_version,
            quota={"used": quota.used, "limit": quota.limit, "remaining": quota.remaining})
        # 领域指标：token/成本/工具调用/护栏拦截
        METRICS.inc("agent_tokens_total", {"kind": "input"}, tokens_in)
        METRICS.inc("agent_tokens_total", {"kind": "output"}, tokens_out)
        METRICS.inc("agent_cost_usd_total", value=cost)
        for call in state.tool_calls:
            METRICS.inc("agent_tool_calls_total",
                        {"tool": call["tool"], "ok": str(bool(call.get("ok"))).lower()})
            if call.get("blocked"):
                METRICS.inc("agent_guardrail_blocks_total", {"tool": call["tool"]})
        if state.status == "refused":
            METRICS.inc("agent_refusals_total")
        if state.injection_flagged:
            METRICS.inc("agent_injection_flagged_total")
        if ctx.get("idempotency_key"):
            idem_store.complete_in_session(session, ctx["user_id"], ctx["idempotency_key"],
                                           response.model_dump())

    return response


app = create_app()
