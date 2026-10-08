# -*- coding: utf-8 -*-
"""服务层端到端冒烟：对**运行中的** Agent 服务做真实 HTTP 检查。

为什么需要它：
- `agent/tests/test_service.py` 是进程内测试（快、可进 CI）；
- 本脚本走真实 HTTP + 真实 SSE，验证"起服务之后确实能用"，
  并且**真实测量首字延迟**（TestClient 会缓冲响应，测不了）。

用法：
    # 终端 A：启动服务
    SERVICE_LLM_BACKEND=streaming AGENT_KB_PATH=eval/fixtures/mini_corpus.jsonl \
        python -m uvicorn agent.service.app:app --port 8077
    # 终端 B：跑冒烟
    python eval/service_smoke.py
    python eval/service_smoke.py --url http://127.0.0.1:8077 --verbose

注意：请用 Python 客户端发中文请求体。PowerShell 的 Invoke-RestMethod 会把非 ASCII
body 编坏，导致服务端收到乱码查询而"正确地"拒答（本项目踩过一次）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    flag = "✅" if ok else "❌"
    print(f"  [{flag}] {name}" + (f"　{detail}" if detail else ""))
    return ok


def header_value(headers, name: str) -> str:
    """HTTP 头不区分大小写（uvicorn 会统一转小写），所以这里必须忽略大小写查找。"""
    for key, value in (headers or {}).items():
        if key.lower() == name.lower():
            return value
    return None


def request(url: str, path: str, body=None, headers=None, method=None, raw=False):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url + path, data=data, method=method or ("POST" if data else "GET"),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = resp.read()
            return resp.status, (payload if raw else json.loads(payload.decode("utf-8"))), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except Exception:                                     # noqa: BLE001
            parsed = {"error": payload.decode("utf-8", "replace")}
        return exc.code, parsed, dict(exc.headers)


def sse_stream(url: str, path: str, body, headers, verbose=False):
    """读取 SSE，返回 (事件列表, 首字延迟 ms, 总耗时 ms)。"""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url + path, data=data,
                                 headers={"Content-Type": "application/json",
                                          "Accept": "text/event-stream", **headers})
    t0 = time.perf_counter()
    ttft = None
    events = []
    buf = ""
    with urllib.request.urlopen(req, timeout=60) as resp:
        while True:
            chunk = resp.read(1)
            if not chunk:
                break
            buf += chunk.decode("utf-8", "replace")
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                event, payload = None, None
                for line in block.split("\n"):
                    if line.startswith("event: "):
                        event = line[7:]
                    elif line.startswith("data: "):
                        payload = json.loads(line[6:])
                if event:
                    events.append((event, payload))
                    if event == "delta" and ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    if verbose and event in ("meta", "stage", "citations"):
                        print(f"      ← {event}: {json.dumps(payload, ensure_ascii=False)[:90]}")
    total = (time.perf_counter() - t0) * 1000
    return events, ttft, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8077")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    url = args.url.rstrip("/")

    print("=" * 72)
    print(f"Agent 服务冒烟：{url}")
    print("=" * 72)

    # ── 1. 健康检查 ──
    print("\n[1] 健康检查")
    status, body, _ = request(url, "/healthz")
    check("存活 /healthz", status == 200 and body.get("status") == "ok", f"uptime={body.get('uptime_s')}s")
    status, ready, _ = request(url, "/readyz")
    check("就绪 /readyz（依赖可用）", status == 200 and ready.get("database") is True,
          f"db={ready.get('database')} kv={ready.get('kv_backend')} kb_chunks={ready.get('kb_chunks')}")
    check("知识库已加载", (ready.get("kb_chunks") or 0) > 0, f"{ready.get('kb_chunks')} 块")

    # ── 2. 契约 ──
    print("\n[2] 接口契约")
    status, spec, _ = request(url, "/openapi.json")
    paths = (spec or {}).get("paths", {})
    check("OpenAPI 已生成", status == 200 and "/v1/chat" in paths,
          f"{len(paths)} 个路径")
    check("SSE 端点已注册", "/v1/chat/stream" in paths)

    # ── 3. 鉴权与租户 ──
    print("\n[3] 鉴权与租户")
    status, _, _ = request(url, "/v1/chat", {"message": "座椅加热怎么关闭"})
    check("无令牌 → 401", status == 401, f"HTTP {status}")
    status, tok, _ = request(url, "/v1/auth/token", {"user_id": "smoke-driver",
                                                     "vehicle_model": "lynk08"})
    check("签发令牌", status == 200 and "access_token" in tok)
    auth = {"Authorization": f"Bearer {tok['access_token']}"}
    status, _, _ = request(url, "/v1/chat", {"message": "座椅加热怎么关闭"},
                           {**auth, "X-Vehicle-Model": "lynk09"})
    check("跨租户访问 → 403", status == 403, f"HTTP {status}")
    status, tools, _ = request(url, "/v1/tools", headers=auth)
    names = {t["name"] for t in (tools if isinstance(tools, list) else [])}
    check("工具清单（FC schema）", "search_manual" in names and "create_service_order" in names,
          f"{len(names)} 个工具")

    # ── 4. 问答链路 ──
    print("\n[4] 问答链路")
    status, ans, _ = request(url, "/v1/chat", {"message": "怎么打开危险警告灯"}, auth)
    check("可答问题 → answered",
          status == 200 and ans.get("status") == "answered" and ans.get("citations"),
          f"{ans.get('latency_ms')}ms tokens={ans.get('tokens_in')}+{ans.get('tokens_out')}")
    check("答案带页码出处", bool(ans.get("citations")),
          (ans.get("citations") or [""])[0])
    check("工具调用成功", all(t["ok"] for t in ans.get("tool_calls", [])) is True
          and bool(ans.get("tool_calls")),
          json.dumps(ans.get("tool_calls", [{}])[0], ensure_ascii=False)[:70])
    check("配额已记账", (ans.get("quota") or {}).get("used", 0) > 0,
          f"used={ans['quota']['used']}/{ans['quota']['limit']}")
    status, ref, _ = request(url, "/v1/chat", {"message": "中国足球的队长是谁"}, auth)
    check("域外问题 → 拒答", ref.get("answer", "").strip() == "无答案",
          f"status={ref.get('status')}")

    # ── 5. 会话 ──
    print("\n[5] 会话持久化")
    status, sess, _ = request(url, "/v1/sessions", method="POST", headers=auth)
    sid = (sess or {}).get("session_id")
    check("创建会话", status == 200 and bool(sid), f"session_id={str(sid)[:8]}…")
    status, cont, _ = request(url, "/v1/chat",
                              {"message": "它怎么关闭", "session_id": sid}, auth)
    check("同会话续聊", cont.get("session_id") == sid)
    status, listing, _ = request(url, "/v1/sessions", headers=auth)
    check("会话出现在列表中", sid in [s["session_id"] for s in (listing or [])],
          f"{len(listing or [])} 个会话")

    # ── 6. 幂等 ──
    print("\n[6] 幂等（Idempotency-Key）")
    idem_h = {**auth, "Idempotency-Key": f"smoke-{int(time.time())}"}
    _, first, _ = request(url, "/v1/chat", {"message": "座椅加热怎么关闭"}, idem_h)
    _, second, _ = request(url, "/v1/chat", {"message": "座椅加热怎么关闭"}, idem_h)
    check("首次执行非重放", first.get("idempotent_replay") is False)
    check("重复请求命中重放", second.get("idempotent_replay") is True,
          f"message_id={second.get('message_id')} 与首次一致={first.get('message_id') == second.get('message_id')}")
    check("重放返回同一答案", first.get("answer") == second.get("answer"))
    _, conflict, _ = request(url, "/v1/chat", {"message": "危险警告灯怎么开"}, idem_h)
    check("同 key 换 body → 422", conflict.get("error", "").find("不同的请求体") >= 0
          or "不同的请求体" in str(conflict), str(conflict.get("error"))[:40])

    # ── 7. 写操作确认 ──
    print("\n[7] 写操作确认（HITL）")
    _, conf, _ = request(url, "/v1/writes/confirm",
                         {"action_key": "create_service_order:更换机油机滤:2025-06-18"},
                         auth)
    check("确认落库", conf.get("confirmed") is True,
          f"already_confirmed={conf.get('already_confirmed')}")
    _, conf2, _ = request(url, "/v1/writes/confirm",
                          {"action_key": "create_service_order:更换机油机滤:2025-06-18"}, auth)
    check("重复确认幂等", conf2.get("already_confirmed") is True)

    # ── 8. SSE 流式（真实 TTFT）──
    print("\n[8] SSE 流式与首字延迟")
    events, ttft, total = sse_stream(url, "/v1/chat/stream",
                                     {"message": "座椅加热怎么关闭"}, auth, args.verbose)
    kinds = [e for e, _ in events]
    check("事件顺序 meta→stage→delta…→done",
          kinds[:2] == ["meta", "stage"] and kinds[-1] == "done" and "delta" in kinds,
          " → ".join(dict.fromkeys(kinds)))
    deltas = [d for e, d in events if e == "delta"]
    done = [d for e, d in events if e == "done"][0]
    check("分多块下发", len(deltas) >= 2, f"{len(deltas)} 块")
    check("拼装结果 = 完整答案",
          "".join(d["text"] for d in deltas) == done["answer"])
    if ttft is not None:
        check("首字延迟 << 总耗时", ttft < total * 0.5,
              f"TTFT={ttft:.1f}ms 总耗时={total:.1f}ms（服务端记录 ttft={done.get('ttft_ms')}ms）")
    else:
        check("收到 delta 事件", False, "未收到任何 delta")

    # ── 9. 治理接口 ──
    print("\n[9] 治理与可观测")
    status, _, _ = request(url, "/v1/admin/metrics", headers=auth)
    check("普通用户禁止访问治理接口 → 403", status == 403, f"HTTP {status}")
    _, admin, _ = request(url, "/v1/auth/token", {"user_id": "smoke-admin", "role": "admin",
                                                  "vehicle_model": "lynk08"})
    admin_h = {"Authorization": f"Bearer {admin['access_token']}"}
    status, metrics, _ = request(url, "/v1/admin/metrics", headers=admin_h)
    check("指标聚合可用", status == 200 and metrics.get("messages", 0) > 0,
          f"messages={metrics.get('messages')} tokens={metrics.get('tokens_total')} "
          f"cost=${metrics.get('cost_usd')}")
    check("失败归因已记录", "failure_distribution" in metrics,
          json.dumps(metrics.get("failure_distribution", {}), ensure_ascii=False))
    status, audit, _ = request(url, "/v1/admin/audit", headers=admin_h)
    check("审计日志可查", status == 200 and isinstance(audit, list),
          f"{len(audit or [])} 条")

    # ── 10. 限流（用独立用户，避免影响前面的检查）──
    print("\n[10] 限流")
    _, rl_tok, _ = request(url, "/v1/auth/token", {"user_id": "smoke-flood",
                                                   "vehicle_model": "lynk08"})
    rl_h = {"Authorization": f"Bearer {rl_tok['access_token']}"}
    hit_429, retry_after = False, None
    for _ in range(80):
        code, body, headers = request(url, "/v1/chat", {"message": "座椅加热怎么关闭"}, rl_h)
        if code == 429:
            hit_429, retry_after = True, header_value(headers, "Retry-After")
            break
    check("超限触发 429", hit_429)
    check("返回 Retry-After", retry_after is not None, f"Retry-After={retry_after}s")

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total_checks = len(RESULTS)
    print("\n" + "-" * 72)
    print(f"冒烟结果：{passed}/{total_checks} 通过　→ {'通过' if passed == total_checks else '存在失败'}")
    if passed != total_checks:
        print("失败项：")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  ❌ {name}　{detail}")
    return 0 if passed == total_checks else 1


if __name__ == "__main__":
    sys.exit(main())
