# -*- coding: utf-8 -*-
"""云端真实模型连通性冒烟：花几分钱先验证接口，再决定是否跑全量评测。

为什么必须先做这一步：真实 API 的坑不在"能不能连"，而在**参数不通用**——
- vLLM 专属的 `guided_json` / `guided_decoding_backend` 发给云端会 400；
- DeepSeek 的思考模式**默认开启**，且带 `tools` 时必须回传 `reasoning_content`，否则 400；
- 思考模式下 `temperature` 无效；`top_p` 只有 0.95~1.0 生效。

本脚本只做 4~5 次调用（约 0.01~0.05 元），验证：
    ① 普通对话  ② Function Calling  ③ 参数解析是否干净
    ④ 流式首字延迟  ⑤ 思考模式开关对 token/延迟的影响

用法：
    python eval/deepseek_smoke.py                 # 读 .env / 环境变量
    python eval/deepseek_smoke.py --model deepseek-v4-pro
    python eval/deepseek_smoke.py --thinking enabled   # 对照思考模式
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm import PROVIDER_PRESETS, build_llm, load_dotenv, redact   # noqa: E402
from agent.tools import build_default_registry                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS = [
    "座椅加热怎么关闭？",
    "我的车胎压报警了，还能继续开吗？",
]


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--model", default="")
    ap.add_argument("--thinking", default="", choices=["", "disabled", "enabled"])
    ap.add_argument("--strict", action="store_true", help="启用 DeepSeek strict 模式（/beta）")
    ap.add_argument("--kb", default=os.environ.get("AGENT_KB_PATH")
                    or os.path.join(ROOT, "eval", "fixtures", "mini_corpus.jsonl"))
    args = ap.parse_args()

    loaded = load_dotenv(os.path.join(ROOT, ".env"))
    print(f"[env] 从 .env 载入 {loaded} 个变量")
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("[错误] 未找到 DEEPSEEK_API_KEY：请设置环境变量或写入 .env（已被 .gitignore 排除）")
        return 2
    print(f"[env] DEEPSEEK_API_KEY 已就绪（{key[:3]}***{key[-2:]}，共 {len(key)} 字符，"
          f"全程不打印完整值）")

    model = args.model or os.environ.get("DEEPSEEK_MODEL") or ""
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or ""
    if args.strict:
        base_url = "https://api.deepseek.com/beta"

    try:
        llm = build_llm(provider=args.provider, model=model, base_url=base_url,
                        thinking=args.thinking, strict_tools=args.strict)
    except RuntimeError as exc:
        print(f"[错误] {redact(exc)}")
        return 2

    preset = PROVIDER_PRESETS.get(args.provider, {})
    print(f"[cfg] provider={llm.provider} model={llm.model}")
    print(f"[cfg] base_url={llm.base_url}")
    print(f"[cfg] thinking={llm.thinking or '（不发送，由服务端默认）'} strict_tools={llm.strict_tools}")
    print(f"[cfg] 计价（每 1M tokens）：输入 ${llm.cost_per_1m_in} / 输出 ${llm.cost_per_1m_out}")

    registry = build_default_registry(args.kb)
    llm.set_known_tools(registry.names())
    print(f"[cfg] 知识库 {len(registry.kb.chunks)} 块；工具 {registry.names()}")

    ok = True

    # ① 普通对话
    hr("① 普通对话（无工具）")
    try:
        t0 = time.perf_counter()
        resp = llm.chat([{"role": "user", "content": "用一句话说明你能做什么。"}],
                        max_tokens=128)
        ms = (time.perf_counter() - t0) * 1000
        print(f"  延迟 {ms:.0f} ms | finish={resp.finish_reason}")
        print(f"  回答：{(resp.content or '')[:120]}")
        print(f"  usage：{resp.usage}")
    except Exception as exc:                                     # noqa: BLE001
        ok = False
        print(f"  ❌ 失败：{redact(exc)}")

    # ② Function Calling
    hr("② Function Calling（真实工具 schema）")
    plan = []
    try:
        hits = registry.call("search_manual", {"query": QUESTIONS[1], "top_k": 3})
        plan = (hits.data or [])[:3]
        messages = [
            {"role": "system", "content": "你是座舱助手。需要资料时调用工具，不要凭记忆回答。"},
            {"role": "user", "content": QUESTIONS[1]},
        ]
        t0 = time.perf_counter()
        resp = llm.chat(messages, tools=registry.specs(), max_tokens=256)
        ms = (time.perf_counter() - t0) * 1000
        print(f"  延迟 {ms:.0f} ms | finish={resp.finish_reason} | tool_calls={len(resp.tool_calls)}")
        for call in resp.tool_calls:
            print(f"  → {call.name}({json.dumps(call.arguments, ensure_ascii=False)})"
                  f"  repaired={call.repaired}")
            print(f"    raw_arguments={call.raw_arguments!r}")
        if not resp.tool_calls:
            ok = False
            print(f"  ❌ 模型没有发起工具调用，直接回答：{(resp.content or '')[:100]}")
        else:
            # 回填工具结果，看模型能否基于证据作答（这一步才会真正暴露接口问题）
            messages.append({"role": "assistant", "content": resp.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.name,
                                                          "arguments": json.dumps(c.arguments,
                                                                                  ensure_ascii=False)}}
                                            for c in resp.tool_calls]})
            for call in resp.tool_calls:
                data = plan if call.name == "search_manual" else []
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(data, ensure_ascii=False)[:1500]})
            resp2 = llm.chat(messages, tools=registry.specs(), max_tokens=320)
            print(f"  回填后回答：{(resp2.content or '')[:200]}")
            if not (resp2.content or "").strip():
                ok = False
                print("  ❌ 回填工具结果后模型没有产出答案")
    except Exception as exc:                                     # noqa: BLE001
        ok = False
        print(f"  ❌ 失败：{redact(exc)}")

    # ③ 参数解析整洁度（不刻意诱导，看自然输出）
    hr("③ 参数解析整洁度")
    try:
        resp = llm.chat([{"role": "user", "content": "查一下领克08的电池容量"}],
                        tools=registry.specs(), max_tokens=256)
        for call in resp.tool_calls:
            raw = call.raw_arguments or ""
            print(f"  → {call.name} args={json.dumps(call.arguments, ensure_ascii=False)}")
            print(f"    是否需要修复器介入：{'是' if call.repaired else '否'}")
            print(f"    原始字符串：{raw!r}")
    except Exception as exc:                                     # noqa: BLE001
        ok = False
        print(f"  ❌ 失败：{redact(exc)}")

    # ④ 流式首字延迟
    hr("④ 流式首字延迟（TTFT）")
    try:
        t0 = time.perf_counter()
        ttft, chars, chunks = None, 0, 0
        for chunk in llm.stream_chat([{"role": "user", "content": "介绍一下胎压报警的处理步骤。"}],
                                     max_tokens=256):
            if chunk.delta:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                chars += len(chunk.delta)
                chunks += 1
        total = (time.perf_counter() - t0) * 1000
        print(f"  TTFT {ttft:.0f} ms | 总耗时 {total:.0f} ms | {chunks} 块 / {chars} 字")
        print("  （这是**公网**模型首字延迟，不等于车端/内网部署延迟）")
    except Exception as exc:                                     # noqa: BLE001
        ok = False
        print(f"  ❌ 失败：{redact(exc)}")

    # ⑤ 成本账本
    hr("⑤ 本次冒烟消耗")
    usage = llm.cost_report()
    print(f"  调用 {usage['calls']} 次 | 输入 {usage['prompt_tokens']} tokens | "
          f"输出 {usage['completion_tokens']} tokens")
    print(f"  预估成本 ${usage['cost_usd']:.4f}（约 {usage['cost_usd'] * 7.2:.2f} 元，按官网单价估算）")

    hr("结论")
    print("  " + ("✅ 接口连通，Function Calling 可用，可以跑全量评测"
                  if ok else "❌ 存在失败项，先按上面的报错修好再跑全量"))
    if preset.get("api_key_env"):
        print("  提醒：这个 key 已在聊天中出现，评测跑完请到平台吊销/轮换。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
