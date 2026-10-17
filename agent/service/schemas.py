# -*- coding: utf-8 -*-
"""接口契约（Pydantic 模型）：请求/响应结构即 API 文档，FastAPI 会据此生成 OpenAPI。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64, description="用户标识（生产由认证服务签发）")
    role: str = Field("driver", description="driver | guest | service | admin")
    vehicle_model: Optional[str] = Field(None, description="绑定的车型")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500, description="车主输入")
    session_id: Optional[str] = Field(None, description="会话 ID；不传则自动创建")
    vehicle_model: Optional[str] = Field(None, description="车型租户；默认取令牌绑定值")
    stream: bool = Field(False, description="是否流式（该字段仅用于 /v1/chat 的便捷开关）")
    top_k: Optional[int] = Field(None, ge=1, le=10)


class ToolCallInfo(BaseModel):
    tool: str
    arguments: Dict[str, Any] = {}
    ok: bool = True
    blocked: bool = False
    repeated: bool = False
    repaired: bool = False
    reason: str = ""
    idempotency_key: str = ""
    latency_ms: float = 0.0


class ChatResponse(BaseModel):
    trace_id: str
    session_id: str
    message_id: Optional[int] = None
    status: str = Field(..., description="answered | refused | needs_confirmation | blocked")
    answer: str
    citations: List[str] = []
    tool_calls: List[ToolCallInfo] = []
    risk_level: str = "ok"
    needs_confirmation: Optional[str] = Field(None, description="需要确认时的 action_key")
    injection_flagged: bool = False
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    ttft_ms: Optional[float] = None
    quota: Optional[Dict[str, Any]] = None
    idempotent_replay: bool = False
    cached: bool = Field(False, description="本次回答是否直接命中语义缓存（未调用大模型）")
    kb_version: str = Field("", description="本次回答使用的知识库版本（灰度/回滚可追溯）")
    degraded: bool = Field(False, description="是否走了降级链路（主推理后端不可用）")
    degrade_reason: str = ""


class SessionInfo(BaseModel):
    session_id: str
    vehicle_model: str
    title: str = ""
    status: str = "active"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ConfirmRequest(BaseModel):
    action_key: str = Field(..., description="写操作的 action_key（由上一次回答返回）")
    session_id: Optional[str] = None


class ConfirmResponse(BaseModel):
    action_key: str
    confirmed: bool
    already_confirmed: bool = False


class FeedbackRequest(BaseModel):
    message_id: int
    rating: int = Field(..., ge=-1, le=1, description="1 有帮助 / -1 没帮助")
    comment: str = ""


class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str
    database: bool
    kv_backend: str
    llm_backend: str
    kb_chunks: int


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
    trace_id: str = ""


class KBRegisterRequest(BaseModel):
    """注册一个新的知识库版本。"""
    path: str = Field(..., description="语料文件路径（jsonl/txt）")
    vehicle_model: str = Field("", description="所属车型租户；空则用服务默认租户")
    version: str = Field("", description="版本号；空则自动生成 v<n>-<checksum8>")


class KBActivateRequest(BaseModel):
    """切换/灰度生效版本。traffic_pct：100=全量切换；1~99=灰度；0=撤销灰度。"""
    version: str
    vehicle_model: str = ""
    traffic_pct: int = Field(100, ge=0, le=100)


class KBRollbackRequest(BaseModel):
    """回滚到上一个生效版本。"""
    vehicle_model: str = ""


class ShutdownPrepareRequest(BaseModel):
    """请求实例进入停机准备（摘流量 + 排空在途请求）。"""
    reason: str = Field("prestop", description="停机原因，写进审计")
    grace_s: float = Field(0.0, ge=0.0, le=600.0,
                           description="排空等待上限（秒）；0 表示用服务端默认 SERVICE_SHUTDOWN_GRACE_S")


class TaskCreateRequest(BaseModel):
    """提交后台任务。kind: noop（验证）| kb_build（注册知识库版本）"""
    kind: str = Field(..., description="任务类型：noop / kb_build")
    payload: Dict[str, Any] = Field(default_factory=dict,
                                    description="任务参数，如 kb_build 的 {path, vehicle_model, version}")
