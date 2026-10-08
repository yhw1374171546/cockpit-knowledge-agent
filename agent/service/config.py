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

    # KV：默认进程内（单实例），生产换 Redis
    kv_backend: str = ""              # memory | redis
    redis_url: str = ""

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
            kb_path=os.environ.get("AGENT_KB_PATH", ""),
            default_vehicle_model=os.environ.get("SERVICE_DEFAULT_VEHICLE_MODEL", "lynk08"),
            allowed_models=[m.strip() for m in models.split(",") if m.strip()],
        )
