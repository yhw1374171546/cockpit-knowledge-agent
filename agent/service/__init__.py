# -*- coding: utf-8 -*-
"""Agent 服务层：把 Agent 能力封装成可上线的 HTTP 服务。

模块导航
--------
- `config`    环境变量配置（DB / KV / JWT / 限流 / 配额 / LLM 后端）
- `db`        SQLAlchemy 模型与会话（默认 SQLite，生产 PostgreSQL）
- `kv`        KV 抽象（进程内实现 / Redis，限流与缓存用）
- `auth`      JWT 鉴权与车型租户解析
- `security`  限流、配额、幂等（落库）、写操作确认、审计
- `schemas`   Pydantic 请求/响应契约（自动生成 OpenAPI）
- `app`       FastAPI 应用（RBAC、SSE 流式、健康检查、治理接口）
"""

from agent.service.config import Settings

__all__ = ["Settings", "create_app"]


def create_app(settings=None):        # 延迟导入，避免 import 时就拉起 FastAPI 依赖
    from agent.service.app import create_app as _create

    return _create(settings)
