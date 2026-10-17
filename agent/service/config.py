# -*- coding: utf-8 -*-
"""Agent 服务层配置：全部通过环境变量注入，默认值面向本地开发。

生产部署建议：
- `SERVICE_DATABASE_URL` 指向 PostgreSQL（`postgresql+psycopg://...`）；
- `SERVICE_KV_BACKEND=redis` + `SERVICE_REDIS_URL`（多实例下限流/配额/幂等才成立）；
- `SERVICE_JWT_SECRET` 从密钥管理服务注入，不要写进代码或镜像。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # 应用
    app_name: str = "cockpit-agent-service"
    version: str = "1.0.0"
    debug: bool = False

    # 存储：默认 SQLite（零依赖可跑），生产换 PostgreSQL
    database_url: str = ""
    db_echo: bool = False

    # KV：默认进程内（单实例），生产换 Redis；sqlite 用于单机多 worker 共享
    kv_backend: str = ""              # memory | redis | sqlite
    redis_url: str = ""
    kv_sqlite_path: str = ""

    # 鉴权
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_ttl_seconds: int = 3600
    auth_required: bool = True
    # /v1/auth/token 仅用于本地联调；生产必须置 0，令牌由统一认证服务签发
    dev_token_enabled: bool = True
    # 写操作强制要求 Idempotency-Key（服务层收紧策略，防止客户端重试导致重复下单）
    require_idempotency_for_writes: bool = True

    # 限流（令牌桶，按 用户+车型 维度）
    rate_limit_capacity: int = 20
    rate_limit_refill_per_sec: float = 2.0

    # 配额（按用户+自然日聚合 token）
    quota_tokens_per_day: int = 200_000

    # 幂等
    idempotency_ttl_seconds: int = 86400

    # LLM 后端：planner（离线规则）| simulated（可流式，演示用）| openai（vLLM）
    llm_backend: str = ""
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_model: str = "Qwen2_5_7B_Instruct"
    llm_ms_per_token: float = 4.0

    # 故障注入与降级（用于高可用演练）
    fault_mode: str = "none"          # none | error | timeout | flaky
    fault_rate: float = 0.0           # flaky 模式下的失败概率
    fault_sleep_s: float = 0.05
    enable_degradation: bool = True   # 主后端失败时回退到离线规则规划器

    # ── 语义缓存（座舱重复率高：命中即省一次检索 + 一次大模型调用）──
    enable_semantic_cache: bool = True
    semantic_cache_threshold: float = 0.92   # 宁严不宽：防"近似但答错题"（座舱答错代价高）
    semantic_cache_max_entries: int = 512
    semantic_cache_ttl_s: float = 3600.0
    semantic_cache_shared: bool = True       # True=落 KV（多副本共享）；False=纯进程内

    # 后台任务队列（BE3）：索引构建/批量评测异步化
    enable_task_worker: bool = True
    task_poll_s: float = 0.5

    # 表结构：开发/测试自动建表；**生产应置 0 并先跑 alembic upgrade head**
    auto_create_schema: bool = True

    # SSE 断线重连（BE7）：事件缓冲的容量上限（按流、按事件）与 TTL
    sse_buffer_streams: int = 256
    sse_buffer_events: int = 512
    sse_buffer_ttl_s: float = 300.0

    # 优雅停机（BE8）：等在途请求排空的上限；超过则强制关闭（不无限拉长停机窗口）
    shutdown_grace_s: float = 20.0

    # 指标暴露：默认仅 service/admin 可访问；置 1 则公开（Prometheus 抓取用）
    metrics_public: bool = False

    # 知识库
    kb_path: str = ""

    # 默认租户（未带 X-Vehicle-Model 时使用）
    # ⚠️ 租户标识必须是 **ASCII**：它会出现在 HTTP 头里，而 HTTP 头不允许非 ASCII 字符。
    # 中文车型名请在网关层映射为代号（显示名单独维护），不要直接塞进 header。
    default_vehicle_model: str = "lynk08"

    allowed_models: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "Settings":
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models = os.environ.get("SERVICE_ALLOWED_MODELS", "lynk08,lynk09")
        return cls(
            debug=os.environ.get("SERVICE_DEBUG", "").lower() in ("1", "true", "yes"),
            database_url=os.environ.get(
                "SERVICE_DATABASE_URL", f"sqlite:///{os.path.join(base_dir, 'agent_service.db')}"),
            db_echo=os.environ.get("SERVICE_DB_ECHO", "").lower() in ("1", "true"),
            kv_backend=os.environ.get("SERVICE_KV_BACKEND", ""),
            redis_url=os.environ.get("SERVICE_REDIS_URL", ""),
            kv_sqlite_path=os.environ.get("SERVICE_KV_SQLITE_PATH", ""),
            jwt_secret=os.environ.get("SERVICE_JWT_SECRET", "dev-secret-change-me"),
            jwt_ttl_seconds=_int("SERVICE_JWT_TTL_SECONDS", 3600),
            auth_required=os.environ.get("SERVICE_AUTH_REQUIRED", "1").lower() not in ("0", "false"),
            dev_token_enabled=os.environ.get("SERVICE_DEV_TOKEN", "1").lower() not in ("0", "false"),
            require_idempotency_for_writes=os.environ.get(
                "SERVICE_REQUIRE_IDEMPOTENCY_FOR_WRITES", "1").lower() not in ("0", "false"),
            rate_limit_capacity=_int("SERVICE_RATE_LIMIT_CAPACITY", 20),
            rate_limit_refill_per_sec=_float("SERVICE_RATE_LIMIT_REFILL_PER_SEC", 2.0),
            quota_tokens_per_day=_int("SERVICE_QUOTA_TOKENS_PER_DAY", 200_000),
            idempotency_ttl_seconds=_int("SERVICE_IDEMPOTENCY_TTL_SECONDS", 86400),
            llm_backend=os.environ.get("SERVICE_LLM_BACKEND", ""),
            llm_base_url=os.environ.get("SERVICE_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            llm_model=os.environ.get("SERVICE_LLM_MODEL", "Qwen2_5_7B_Instruct"),
            llm_ms_per_token=_float("SERVICE_LLM_MS_PER_TOKEN", 4.0),
            fault_mode=os.environ.get("SERVICE_FAULT_MODE", "none"),
            fault_rate=_float("SERVICE_FAULT_RATE", 0.0),
            fault_sleep_s=_float("SERVICE_FAULT_SLEEP_S", 0.05),
            enable_degradation=os.environ.get("SERVICE_ENABLE_DEGRADATION", "1").lower()
            not in ("0", "false"),
            enable_semantic_cache=os.environ.get("SERVICE_SEMANTIC_CACHE", "1").lower()
            not in ("0", "false"),
            semantic_cache_threshold=_float("SERVICE_SEMANTIC_CACHE_THRESHOLD", 0.92),
            semantic_cache_max_entries=_int("SERVICE_SEMANTIC_CACHE_MAX_ENTRIES", 512),
            semantic_cache_ttl_s=_float("SERVICE_SEMANTIC_CACHE_TTL_S", 3600.0),
            semantic_cache_shared=os.environ.get("SERVICE_SEMANTIC_CACHE_SHARED", "1").lower()
            not in ("0", "false"),
            enable_task_worker=os.environ.get("SERVICE_TASK_WORKER", "1").lower()
            not in ("0", "false"),
            task_poll_s=_float("SERVICE_TASK_POLL_S", 0.5),
            auto_create_schema=os.environ.get("SERVICE_AUTO_CREATE_SCHEMA", "1").lower()
            not in ("0", "false"),
            sse_buffer_streams=_int("SERVICE_SSE_BUFFER_STREAMS", 256),
            sse_buffer_events=_int("SERVICE_SSE_BUFFER_EVENTS", 512),
            sse_buffer_ttl_s=_float("SERVICE_SSE_BUFFER_TTL_S", 300.0),
            shutdown_grace_s=_float("SERVICE_SHUTDOWN_GRACE_S", 20.0),
            metrics_public=os.environ.get("SERVICE_METRICS_PUBLIC", "0").lower() in ("1", "true"),
            kb_path=os.environ.get("AGENT_KB_PATH", ""),
            default_vehicle_model=os.environ.get("SERVICE_DEFAULT_VEHICLE_MODEL", "lynk08"),
            allowed_models=[m.strip() for m in models.split(",") if m.strip()],
        )
