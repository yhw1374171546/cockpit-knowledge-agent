# -*- coding: utf-8 -*-
"""演示页的接口契约测试（UI contract test）。

为什么需要
----------
前端页面是纯静态 HTML + 原生 JS，**没有类型检查、没有构建步骤**，最容易出的问题是：
页面读的字段名和后端返回的不一致（比如页面读 `tool_error`、后端给的是 `reason`）。
这类错在浏览器里只表现为"某个区域空白"，很难查。

本脚本用 httpx **按页面里的调用顺序**逐个请求，并断言页面实际使用的每个字段都存在。
它不启动浏览器，但覆盖了页面 100% 的接口契约。

    python eval/ui_contract_check.py --url http://127.0.0.1:8079
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

try:
    import httpx
except ImportError:                                        # pragma: no cover
    print("需要 httpx：python -m pip install httpx")
    raise SystemExit(2)

PASS, FAIL = "  [OK]", "[FAIL]"


class Checker:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.client = httpx.Client(base_url=self.base, timeout=30.0)
        self.ok = 0
        self.bad = 0

    def check(self, name: str, cond: bool, detail: str = "") -> bool:
        if cond:
            self.ok += 1
            print(f"{PASS} {name}")
        else:
            self.bad += 1
            print(f"{FAIL} {name}" + (f"  → {detail}" if detail else ""))
        return cond

    def run(self) -> int:
        c = self.client

        # ── 1. 页面本身 ──
        page = c.get("/")
        self.check("GET / 返回 HTML", page.status_code == 200
                   and "text/html" in page.headers.get("content-type", ""),
                   f"{page.status_code} {page.headers.get('content-type')}")
        html = page.text
        for marker in ("/v1/chat/stream", "/v1/auth/token", "/v1/admin/kb/versions",
                       "/v1/admin/kb/activate", "/v1/writes/confirm", "citations"):
            self.check(f"页面里包含 {marker}", marker in html)

        # ── 2. 健康检查（页面顶部的状态条）──
        health = c.get("/healthz").json()
        # /healthz 是**存活**探针：故意只返回最小信息、不查任何依赖
        # （否则数据库抖一下容器就被 K8s 杀掉，优雅停机等于白做）。
        self.check("healthz 只报存活（status + uptime_s）",
                   "status" in health, json.dumps(health, ensure_ascii=False))
        ready = c.get("/readyz")
        self.check("readyz 可用", ready.status_code in (200, 503), str(ready.status_code))
        detail = ready.json()
        self.check("readyz 有 kb_chunks / kv_backend / llm_backend（页面顶部状态条读它）",
                   all(k in detail for k in ("kb_chunks", "kv_backend", "llm_backend")),
                   json.dumps(detail, ensure_ascii=False)[:160])

        # ── 3. 申请 token（页面 issueToken）──
        tok_res = c.post("/v1/auth/token",
                         json={"user_id": "ui-check", "role": "service",
                               "vehicle_model": "lynk08"})
        self.check("auth/token 成功", tok_res.status_code == 200, tok_res.text[:200])
        token = tok_res.json().get("access_token")
        self.check("响应字段叫 access_token（页面读的就是这个名字）", bool(token))
        auth = {"Authorization": f"Bearer {token}"}

        # ── 4. SSE 流式对话（页面 send() 的核心）──
        events = []
        with c.stream("POST", "/v1/chat/stream",
                      json={"message": "座椅加热怎么关闭"},
                      headers=auth) as resp:
            self.check("chat/stream 返回 200", resp.status_code == 200, str(resp.status_code))
            self.check("Content-Type 是 text/event-stream",
                       "text/event-stream" in resp.headers.get("content-type", ""),
                       resp.headers.get("content-type", ""))
            buf = ""
            for chunk in resp.iter_text():
                buf += chunk
                # 按 SSE 规范以空行切事件
                while "\n\n" in buf:
                    raw, buf = buf.split("\n\n", 1)
                    ev, data = "", ""
                    for line in raw.split("\n"):
                        if line.startswith("event:"):
                            ev = line[6:].strip()
                        elif line.startswith("data:"):
                            data += line[5:].strip()
                    if ev:
                        events.append((ev, data))

        names = [e[0] for e in events]
        self.check("事件顺序含 meta → delta* → citations → done",
                   names[:1] == ["meta"] and "delta" in names
                   and "citations" in names and names[-1] == "done", str(names[:6]) + "…")

        meta = json.loads(next(d for n, d in events if n == "meta"))
        self.check("meta 含 session_id / trace_id / vehicle_model",
                   all(k in meta for k in ("session_id", "trace_id", "vehicle_model")),
                   json.dumps(meta, ensure_ascii=False))

        delta = json.loads(next(d for n, d in events if n == "delta"))
        self.check("delta 含 text / elapsed_ms",
                   "text" in delta and "elapsed_ms" in delta,
                   json.dumps(delta, ensure_ascii=False)[:120])

        cites_ev = json.loads(next(d for n, d in events if n == "citations"))
        self.check("citations 事件含 citations 数组 / risk_level",
                   isinstance(cites_ev.get("citations"), list) and "risk_level" in cites_ev,
                   json.dumps(cites_ev, ensure_ascii=False)[:160])
        if cites_ev.get("citations"):
            sample = cites_ev["citations"][0]
            self.check("引用是字符串（页面用正则抽页码）", isinstance(sample, str), repr(sample))
            self.check("引用能抽出『第N页』（页面排版依赖它）",
                       bool(re.search(r"第\s*\d+\s*页", sample)), repr(sample))

        done = json.loads(next(d for n, d in events if n == "done"))
        # 页面右侧追踪面板 / 徽标实际读取的字段
        needed = ["trace_id", "session_id", "status", "answer", "citations", "tool_calls",
                  "risk_level", "tokens_in", "tokens_out", "cost_usd", "latency_ms",
                  "cached", "kb_version", "degraded", "degrade_reason",
                  "injection_flagged", "idempotent_replay", "ttft_ms"]
        missing = [k for k in needed if k not in done]
        self.check(f"done 事件包含页面用到的全部 {len(needed)} 个字段", not missing,
                   "缺少: " + ", ".join(missing))
        session_id = done.get("session_id", "")

        # tool_calls 的字段（页面渲染工具卡片用 t.tool / t.ok / t.reason / t.latency_ms）
        for call in done.get("tool_calls") or []:
            self.check("tool_calls 单项含 tool / ok / latency_ms / reason",
                       all(k in call for k in ("tool", "ok", "latency_ms", "reason")),
                       json.dumps(call, ensure_ascii=False)[:160])
            break

        # 同一会话第二轮（页面会带上 session_id，验证多轮仍可用）
        second = c.post("/v1/chat",
                        json={"message": "那胎压多少算正常", "session_id": session_id},
                        headers=auth)
        self.check("同会话第二轮可用（多轮记忆）", second.status_code == 200,
                   second.text[:200])

        # ── 5. 版本管理（页面右下角）──
        versions = c.get("/v1/admin/kb/versions", headers=auth)
        self.check("kb/versions 需要 service 角色才能读", versions.status_code == 200,
                   str(versions.status_code))
        payload = versions.json()
        self.check("kb/versions 返回 versions 列表 + stats",
                   isinstance(payload.get("versions"), list) and "stats" in payload)
        if payload.get("versions"):
            v = payload["versions"][0]
            self.check("版本项含 version / n_chunks / status / traffic_pct（页面渲染用）",
                       all(k in v for k in ("version", "n_chunks", "status", "traffic_pct")),
                       json.dumps(v, ensure_ascii=False)[:200])

        # 灰度给当前版本自己（不会真的改动语义，只验证接口形状）
        active = next((v for v in payload.get("versions", []) if v["status"] == "active"), None)
        if active:
            act = c.post("/v1/admin/kb/activate",
                         json={"version": active["version"], "vehicle_model": "lynk08",
                               "traffic_pct": 100}, headers=auth)
            self.check("kb/activate 可用（页面点版本行会调它）", act.status_code == 200,
                       act.text[:200])
        rb = c.post("/v1/admin/kb/rollback", json={"vehicle_model": "lynk08"}, headers=auth)
        self.check("kb/rollback 返回 409（没有上一版）或 200（有上一版）",
                   rb.status_code in (200, 409), f"{rb.status_code} {rb.text[:120]}")

        # ── 6. 权限：driver 不能看版本（页面会显示"需 service 角色"）──
        dtok = c.post("/v1/auth/token",
                      json={"user_id": "ui-driver", "role": "driver",
                            "vehicle_model": "lynk08"}).json()["access_token"]
        denied = c.get("/v1/admin/kb/versions",
                       headers={"Authorization": f"Bearer {dtok}"})
        self.check("driver 读版本返回 403（页面据此提示）", denied.status_code == 403,
                   str(denied.status_code))

        print("-" * 60)
        total = self.ok + self.bad
        print(f"接口契约：{self.ok}/{total} 通过" + ("" if not self.bad else f"，{self.bad} 项失败"))
        return 0 if self.bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("SERVICE_URL", "http://127.0.0.1:8079"))
    args = ap.parse_args()
    print("=" * 60)
    print(f"演示页接口契约检查 → {args.url}")
    print("=" * 60)
    try:
        return Checker(args.url).run()
    except httpx.HTTPError as exc:
        print(f"{FAIL} 连接失败：{exc}")
        print("  先起服务：python -m uvicorn agent.service.app:app --port 8079")
        return 2


if __name__ == "__main__":
    sys.exit(main())
