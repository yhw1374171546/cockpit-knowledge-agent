# -*- coding: utf-8 -*-
"""评测脚本的共享 LLM 后端选择层：一键在「离线替身」与「真实云端模型」之间切换。

为什么需要它
-----------
4 个评测脚本（tool_call / eval_agent / multiagent / injection）都要能接真实模型，
但它们**默认必须是可复现的离线基线**（这些脚本已进 CI，默认口径不能变）。
把「参数 → 构造 → 标签 → 成本」收敛到一处，避免 4 份复制粘贴各自漂移。

在评测脚本里的用法（只加几行）
------------------------------
    from eval.llm_backend import add_llm_args, backend_fields, llm_label, make_llm

    ap = argparse.ArgumentParser()
    add_llm_args(ap)                      # --provider/--model/--base-url/--thinking/--strict-tools
    args = ap.parse_args()
    ...
    llm = make_llm(args, known_tools=registry.names())   # planner→离线替身；其余→真实模型
    ...
    summary.update(backend_fields(args, llm))            # backend / tokens / cost_usd

安全约定
--------
1. API key 只从环境变量或本地 `.env` 读取（`.env` 已在 `.gitignore` 中），
   绝不写进代码、日志、指标文件或评测报告；
2. 所有异常信息统一过 `agent.llm.redact()`，避免一次 401 把 key 打进 CI 日志；
3. 报告里只出现 `provider / model` 名称与 token/成本数字。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm import (PROVIDER_PRESETS, RuleBasedPlannerLLM,  # noqa: E402
                       build_llm, load_dotenv, redact)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: `--provider` 的可选值：planner = 离线规则替身（默认，CI 基线）；其余 = 真实模型
PROVIDERS = ["planner", "deepseek", "deepseek-pro", "openai", "vllm"]

#: 成本换算用的汇率（仅用于报告里的"约合人民币"，与 deepseek_smoke.py 保持一致）
USD_TO_CNY = 7.2


def add_llm_args(ap) -> None:
    """给评测脚本挂上统一的 LLM 后端参数（默认 `planner`，即行为完全不变）。"""
    ap.add_argument("--provider", choices=PROVIDERS, default="planner",
                    help="LLM 后端：planner=离线规则替身（默认，CI 基线）；"
                         "deepseek/deepseek-pro/openai/vllm=真实模型")
    ap.add_argument("--model", default="",
                    help="覆盖模型名（默认取 PROVIDER_PRESETS 里的值）")
    ap.add_argument("--base-url", default="",
                    help="覆盖 base_url（默认取 PROVIDER_PRESETS 里的值）")
    ap.add_argument("--thinking", choices=["", "disabled", "enabled"], default="",
                    help="思考模式开关（DeepSeek 专用；空=用 preset 默认值）")
    ap.add_argument("--strict-tools", action="store_true",
                    help="启用厂商 strict Function Calling（DeepSeek 需 /beta）")


def provider_of(args) -> str:
    """从 args 里取 provider（容错：老脚本可能只有 `--backend`）。"""
    return (getattr(args, "provider", "") or "planner").strip().lower()


def model_of(args) -> str:
    """解析本次实际使用的模型名（显式 --model 优先，否则取 preset）。"""
    provider = provider_of(args)
    return getattr(args, "model", "") or PROVIDER_PRESETS.get(provider, {}).get("model", "")


def make_llm(args, known_tools: Optional[Sequence[str]] = None):
    """按 args 构造 LLM 后端。

    - `provider == "planner"` → `RuleBasedPlannerLLM()`（离线、确定、零成本，默认行为）；
    - 其余 → 先 `load_dotenv()` 再 `build_llm()`，并把工具清单交给参数修复器
      （`set_known_tools` 用于纠正模型幻觉出来的工具名）。

    失败时抛 `RuntimeError`，消息已过 `redact()`（不会带出 key）。
    """
    provider = provider_of(args)
    if provider == "planner":
        return RuleBasedPlannerLLM()

    # 云端/本地服务：key 优先取环境变量，其次取仓库根目录的 .env（零依赖读取）
    load_dotenv(os.path.join(ROOT, ".env")) or load_dotenv(".env")
    try:
        llm = build_llm(provider=provider,
                        model=getattr(args, "model", "") or "",
                        base_url=getattr(args, "base_url", "") or "",
                        thinking=getattr(args, "thinking", "") or "",
                        strict_tools=bool(getattr(args, "strict_tools", False)))
    except RuntimeError as exc:
        raise RuntimeError(redact(exc)) from None
    if known_tools and hasattr(llm, "set_known_tools"):
        llm.set_known_tools(list(known_tools))
    return llm


def llm_id(args) -> str:
    """机器可读的后端标识，例如 `planner` / `deepseek/deepseek-flash`（写进指标 JSON）。"""
    provider = provider_of(args)
    if provider == "planner":
        return "planner"
    return f"{provider}/{model_of(args)}"


def llm_label(args) -> str:
    """人读的后端标签，例如 `planner（离线规则替身）` / `deepseek / deepseek-flash`。"""
    provider = provider_of(args)
    if provider == "planner":
        return "planner（离线规则替身）"
    return f"{provider} / {model_of(args)}"


def cost_report(llm) -> Dict[str, Any]:
    """取后端的成本账本；离线替身没有账本 → 返回 `{}`（调用方按 0 处理）。"""
    getter = getattr(llm, "cost_report", None)
    if not callable(getter):
        return {}
    try:
        return dict(getter())
    except Exception as exc:                                   # noqa: BLE001
        return {"error": redact(exc)}


def backend_fields(args, llm=None) -> Dict[str, Any]:
    """指标 JSON 里统一追加的后端/成本字段（含后端标识、token 与美元/人民币成本）。"""
    report = cost_report(llm) if llm is not None else {}
    usd = float(report.get("cost_usd") or 0.0)
    return {
        "backend": llm_id(args),
        "backend_label": llm_label(args),
        "provider": provider_of(args),
        "model": model_of(args),
        "llm_calls": int(report.get("calls") or 0),
        "prompt_tokens": int(report.get("prompt_tokens") or 0),
        "completion_tokens": int(report.get("completion_tokens") or 0),
        "cost_usd": round(usd, 6),
        "cost_cny": round(usd * USD_TO_CNY, 4),
    }


def cost_line(fields: Dict[str, Any]) -> str:
    """把 `backend_fields` 的结果渲染成报告里的一行（离线时为「无 API 调用」）。"""
    if not fields.get("llm_calls"):
        return (f"- 成本：本次为 `{fields.get('backend')}`（离线替身），"
                f"无 API 调用、成本 $0.00")
    return (f"- 成本：调用 **{fields['llm_calls']}** 次 ｜ 输入 {fields['prompt_tokens']} tokens ｜ "
            f"输出 {fields['completion_tokens']} tokens ｜ "
            f"**${fields['cost_usd']:.4f}**（约 {fields['cost_cny']:.2f} 元，按官网单价估算）")
