# -*- coding: utf-8 -*-
"""鉴权与租户：JWT（HS256）+ 车型租户校验。

生产建议：JWT 由统一认证服务签发（OIDC/JWKS），本服务只做校验；
`/v1/auth/token` 这个签发接口**仅供本地开发与联调**，生产必须关闭
（`SERVICE_AUTH_REQUIRED=1` 时应从网关获取公钥校验非对称签名）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import jwt

from agent.service.config import Settings


@dataclass
class Principal:
    user_id: str
    role: str = "driver"           # driver | guest | service | admin
    vehicle_model: str = ""
    raw: Optional[dict] = None

    @property
    def is_privileged(self) -> bool:
        return self.role in ("service", "admin")


class AuthError(Exception):
    def __init__(self, message: str, status_code: int = 401):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def issue_token(user_id: str, settings: Settings, role: str = "driver",
                vehicle_model: str = "") -> dict:
    now = int(time.time())
    payload = {
        "sub": user_id,
        "role": role,
        "vehicle_model": vehicle_model or settings.default_vehicle_model,
        "iat": now,
        "exp": now + settings.jwt_ttl_seconds,
        "iss": settings.app_name,
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return {"access_token": token, "token_type": "bearer",
            "expires_in": settings.jwt_ttl_seconds}


def decode_token(token: str, settings: Settings) -> Principal:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm],
                             issuer=settings.app_name)
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("令牌已过期，请重新获取") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError(f"令牌无效：{exc}") from exc
    user_id = payload.get("sub")
    if not user_id:
        raise AuthError("令牌缺少 sub")
    return Principal(user_id=str(user_id), role=str(payload.get("role", "driver")),
                     vehicle_model=str(payload.get("vehicle_model", "")), raw=payload)


def resolve_tenant(principal: Principal, requested_model: Optional[str],
                   settings: Settings) -> str:
    """租户（车型）解析：请求头指定的车型必须在该用户可见范围内。

    这是多车型平台的最小隔离：知识库、缓存、配额都按这个维度切分。
    """
    model = (requested_model or principal.vehicle_model or settings.default_vehicle_model).strip()
    allowed = settings.allowed_models or [settings.default_vehicle_model]
    if model not in allowed:
        raise AuthError(f"车型 {model} 不在许可范围内：{allowed}", status_code=403)
    if principal.vehicle_model and model != principal.vehicle_model and not principal.is_privileged:
        raise AuthError(f"无权访问车型 {model}", status_code=403)
    return model
