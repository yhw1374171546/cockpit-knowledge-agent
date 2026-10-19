# -*- coding: utf-8 -*-
"""MCP 服务端鉴权：HMAC 签名的 scope token（工具级授权）。

为什么需要它
------------
第二十节的实现里，白名单只能做在**客户端**（`subset()`）——因为 stdio 单客户端场景下
服务端并不知道调用方是谁。一旦出现第二个消费者（例如售后/维修侧只想读、不该下单），
"权限在客户端"就不成立了：客户端换一套代码就能调写操作。

这里给服务端补上工具级授权：客户端在 `initialize` 时带一个 HMAC 签名的 token，
服务端校验后**按授权列表过滤 `tools/list`**，并在 `tools/call` 时**再次**校验。
`_meta.user_id` 仍然随每次调用传（那是"谁的车"，不是"哪个应用"）。

边界（不夸大）
--------------
- 这是**自签 HMAC**，不是 OAuth/OIDC：密钥在部署侧共享，适合内部服务之间；
- token 里只放 subject + 工具白名单 + 过期时间，不做撤销列表（短期 token + 轮换密钥即可）；
- 没有传输加密（stdio 是本地管道）；走网络时要套 TLS。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_TTL_S = 3600


class ScopeTokenError(Exception):
    """token 缺失 / 签名不符 / 过期 / 格式错误。"""


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def _sign(body: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).hexdigest()[:32]


def issue_scope_token(subject: str, tools: Sequence[str], secret: str,
                      ttl_s: int = DEFAULT_TTL_S,
                      now: Optional[float] = None) -> str:
    """签发一个只读/受限 token：`subject` 是调用方标识，`tools` 是允许的工具名。"""
    if not secret:
        raise ScopeTokenError("未配置 scope secret")
    if not subject:
        raise ScopeTokenError("subject 不能为空")
    payload: Dict[str, Any] = {
        "sub": subject,
        "tools": list(dict.fromkeys(tools)),     # 去重且保序
        "exp": int((now if now is not None else time.time()) + ttl_s),
    }
    body = _b64e(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return f"{body}.{_sign(body, secret)}"


def verify_scope_token(token: str, secret: str, now: Optional[float] = None) -> Dict[str, Any]:
    """校验签名与有效期，返回 payload；任何异常都转成 `ScopeTokenError`。"""
    if not secret:
        raise ScopeTokenError("服务端未配置 scope secret")
    if not token or "." not in token:
        raise ScopeTokenError("token 格式错误")
    body, signature = token.rsplit(".", 1)
    if not hmac.compare_digest(_sign(body, secret), signature):
        raise ScopeTokenError("签名不匹配")
    try:
        payload = json.loads(_b64d(body))
    except Exception as exc:                                   # noqa: BLE001
        raise ScopeTokenError(f"payload 无法解析：{exc}") from exc
    if not isinstance(payload, dict):
        raise ScopeTokenError("payload 必须是对象")
    tools = payload.get("tools")
    if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
        raise ScopeTokenError("payload.tools 必须是字符串数组")
    expires_at = payload.get("exp")
    if not isinstance(expires_at, (int, float)):
        raise ScopeTokenError("payload.exp 缺失")
    if (now if now is not None else time.time()) > float(expires_at):
        raise ScopeTokenError("token 已过期")
    return payload


def allowed_tools(payload: Dict[str, Any]) -> List[str]:
    tools = payload.get("tools") or []
    return [str(t) for t in tools]
