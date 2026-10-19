# -*- coding: utf-8 -*-
"""长期记忆 AB 对照：关掉记忆 vs 打开记忆，跑同一批跨会话脚本。

    python eval/memory_ab.py

为什么必须做 AB
---------------
只证明"能写进去、能读出来"没有意义——**写进去了没人读**才是这类功能最常见的失败方式
（本仓库已经踩过一次：语义缓存有基准、却从没被服务链路引用）。所以这里量的是
**决策路径的输入**：

1. 事实有没有真的进入 system prompt——用 `SpyPlanner` 抓 `AgentGraph` 实际发出的 messages，
   而不是"我自己拼的字符串里有没有"；
2. 关掉记忆时事实是否**必然缺席**（否则说明它来自别处，AB 无效）；
3. 误注入：与本次问题无关的记忆有没有被塞进去（每个场景都放了背景噪音记忆）；
4. 代价：上下文多了多少字 / 多少 token。

如实说明：本机无 GPU、离线规划器不读 prompt 里的记忆文本，所以**答案层指标
（例如"少追问一次"）测不出来**；这里测的是"模型能不能看到这条记忆"。
真实收益要接真模型才能量化——这条边界与仓库里 TTFT、注入评测的写法一致，不假装。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.console import harden_stdout  # noqa: E402

harden_stdout()

from agent.graph import AgentConfig, AgentGraph  # noqa: E402
from agent.kb import FIXTURE_KB, KnowledgeBase  # noqa: E402
from agent.llm import RuleBasedPlannerLLM  # noqa: E402
from agent.longterm import LongTermMemory  # noqa: E402
from agent.memory import ConversationMemory  # noqa: E402
from agent.tools import ToolRegistry  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAY = 86400.0

# 背景噪音：写进每个场景的记忆库，但不该被任何目标问题召回
DECOY_EPISODE = "问过蓝牙怎么连接，已告知在设置里配对"
DECOY_PREF = ("pref:氛围灯", "关闭")


@dataclass
class Scenario:
    sid: str
    first_turn: str = ""                     # 第一次会话里用户说的话
    summary: str = ""                        # 第一次会话留下的情景摘要
    query: str = ""                          # 第二次会话的问题
    expect: Sequence[str] = ()               # 应当出现在记忆块里的事实片段
    allow_keys: Sequence[str] = ()            # 允许被召回的 key（其余算误注入）
    note: str = ""
    age_days: float = 0.0                    # >0：写完之后隔这么多天再问（测遗忘）


SCENARIOS: List[Scenario] = [
    Scenario("city", "我住在上海", query="帮我约个保养", expect=("上海",),
             allow_keys=("city",), note="常住城市 → 预约类问题"),
    Scenario("center", "我一般去领克中心(浦东金桥店)", query="约下周三到店保养",
             expect=("浦东金桥",), allow_keys=("usual_center",), note="惯用门店"),
    Scenario("seat", "以后座椅加热都关掉", query="座椅加热怎么关", expect=("关闭",),
             allow_keys=("pref:座椅加热",), note="功能偏好"),
    Scenario("model", "我的车是领克08", query="这车续航多少", expect=("领克08",),
             allow_keys=("vehicle_model",), note="车型档案"),
    Scenario("tire", summary="胎压报警，建议到店检查左后轮", query="还是那个胎压问题",
             expect=("胎压",), allow_keys=("episodic",), note="情景记忆：同一问题复发"),
    Scenario("order", summary="已创建保养工单 SO123，到店日期 2026-10-20",
             query="上次那个工单怎么样了", expect=("SO123",),
             allow_keys=("episodic",), note="情景记忆：历史工单"),
    Scenario("city2", "我常住北京", query="保养去哪家店", expect=("北京",),
             allow_keys=("city",), note="城市 → 门店选择"),
    Scenario("ac", "每次空调都打开", query="空调怎么调", expect=("开启",),
             allow_keys=("pref:空调",), note="功能偏好（开启）"),
    # 负样本：不相关的问题不该注入任何记忆
    Scenario("unrelated", "我住在上海", query="今天天气怎么样", expect=(),
             allow_keys=(), note="负样本：不相关不注入"),
    # 负样本：过期记忆不该注入（城市 90 天 TTL、情景 30 天 TTL，这里过 120 天）
    Scenario("stale", "我住在上海", query="帮我约个保养", expect=(),
             allow_keys=(), age_days=120, note="负样本：过期不注入"),
]


class SpyPlanner(RuleBasedPlannerLLM):
    """记录 AgentGraph 实际发出的 messages —— 证明记忆进了 prompt，而不是停在字符串里。"""

    def __init__(self) -> None:
        super().__init__()
        self.seen: List[List[Dict[str, Any]]] = []

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024):   # type: ignore[override]
        self.seen.append([dict(m) for m in messages])
        return super().chat(messages, tools=tools, temperature=temperature, max_tokens=max_tokens)


def memory_block(llm: SpyPlanner) -> str:
    """只取 system 消息里「跨会话记忆」那一段。

    为什么要限定范围：`SYSTEM_PROMPT` 自带"胎压报警"这类示例词，全文匹配会把负样本判成命中
    （第一版就是这么错的：OFF 组也"命中"了 1/1）。
    """
    if not llm.seen:
        return ""
    system = "\n".join(str(m.get("content") or "") for m in llm.seen[0] if m.get("role") == "system")
    marker = "【跨会话记忆】"
    index = system.find(marker)
    return system[index:] if index >= 0 else ""


def run_once(scenario: Scenario, *, store: Optional[LongTermMemory], registry: ToolRegistry,
             now: float) -> Dict[str, Any]:
    memory = ConversationMemory()
    prompt_block = ""
    if store is not None:
        prompt_block = store.context_text(scenario.query, user_id="ab", now=now)
        memory.attach_longterm(prompt_block)

    llm = SpyPlanner()
    graph = AgentGraph(llm, registry, config=AgentConfig(), memory=memory, agent_name="ab")
    state = graph.run(scenario.query)

    actual = memory_block(llm)                       # 真实进入 prompt 的那一段
    facts_hit = [fact for fact in scenario.expect if fact in actual]
    injected = [line for line in actual.splitlines() if line.startswith("- ")]
    hints = _key_hints(scenario)
    relevant = [line for line in injected if any(hint in line for hint in hints)]
    return {
        "status": state.status,
        "context_chars": len(prompt_block),
        "injected_lines": injected,
        "false_inject": max(0, len(injected) - len(relevant)),
        "facts_hit": facts_hit,
        "facts_ok": len(facts_hit) == len(scenario.expect),
        "block_present": bool(actual),
    }


def _key_hints(scenario: Scenario) -> List[str]:
    """把 allow_keys 翻译成在记忆块里可检索的片段（用于判定"注入的是不是相关的那条"）。"""
    hints: List[str] = []
    for key in scenario.allow_keys:
        if key == "city":
            hints.extend(["常住城市"])
        elif key == "usual_center":
            hints.extend(["惯用门店"])
        elif key == "vehicle_model":
            hints.extend(["车型"])
        elif key == "episodic":
            hints.extend(["相关历史"])
        elif key.startswith("pref:"):
            hints.extend(["常用功能", key.split(":", 1)[1]])
    return hints


def main() -> int:
    kb = KnowledgeBase.load(FIXTURE_KB if os.path.exists(FIXTURE_KB) else None)
    registry = ToolRegistry(kb)
    tmpdir = tempfile.mkdtemp(prefix="memory-ab-")
    base_now = 1_760_000_000.0

    rows: List[Dict[str, Any]] = []
    print("=" * 78)
    print("长期记忆 AB 对照（每个场景独立记忆库 + 统一背景噪音）")
    print("=" * 78)

    for scenario in SCENARIOS:
        # 每个场景一个独立库：避免场景之间互相污染（共享一个 user 时"误注入"根本没法归因）
        store = LongTermMemory(os.path.join(tmpdir, f"{scenario.sid}.db"), clock=lambda: base_now)
        store.remember_episode(DECOY_EPISODE, user_id="ab", session_id="decoy")
        store.remember_preference(DECOY_PREF[0], DECOY_PREF[1], user_id="ab", label="常用功能·氛围灯")
        if scenario.first_turn:
            store.extract_and_remember(scenario.first_turn, user_id="ab")
        if scenario.summary:
            store.remember_episode(scenario.summary, user_id="ab", session_id=scenario.sid)

        now = base_now + scenario.age_days * DAY
        store.maintain(user_id="ab", now=now)
        on = run_once(scenario, store=store, registry=registry, now=now)
        off = run_once(scenario, store=None, registry=registry, now=now)
        store.close()

        rows.append({"scenario": scenario.sid, "note": scenario.note, "expect": list(scenario.expect),
                     "on": on, "off": off})
        if scenario.expect:
            passed = (on["facts_ok"] and on["block_present"] and not off["block_present"]
                      and on["false_inject"] == 0)
        else:
            passed = not on["block_present"] and not off["block_present"]
        print(f"  [{'OK' if passed else '!!'}] {scenario.sid:<10} ON 注入 {len(on['injected_lines'])} 行 / "
              f"命中 {len(on['facts_hit'])}/{len(scenario.expect)} / 误注入 {on['false_inject']}"
              f"　OFF 命中 {len(off['facts_hit'])}/{len(scenario.expect)}")

    positives = [r for r in rows if r["expect"]]
    negatives = [r for r in rows if not r["expect"]]
    on_recall = sum(1 for r in positives
                    if r["on"]["facts_ok"] and r["on"]["block_present"]) / max(1, len(positives))
    off_recall = sum(1 for r in positives
                     if r["off"]["facts_ok"] and r["off"]["block_present"]) / max(1, len(positives))
    prompt_proof = sum(1 for r in positives if r["on"]["block_present"]) / max(1, len(positives))
    false_inject = sum(r["on"]["false_inject"] for r in rows)
    neg_leak = sum(1 for r in negatives if r["on"]["block_present"])
    extra_chars = sum(r["on"]["context_chars"] for r in positives) / max(1, len(positives))

    summary = {
        "scenarios": len(rows), "positives": len(positives), "negatives": len(negatives),
        "recall_on": round(on_recall, 4), "recall_off": round(off_recall, 4),
        "prompt_evidence_rate": round(prompt_proof, 4),
        "false_inject_items": false_inject, "negative_leaks": neg_leak,
        "avg_context_chars_on": round(extra_chars, 1),
        "avg_token_delta_est": round(extra_chars / 1.6, 1),
    }
    print("-" * 78)
    print(f"跨会话召回率：记忆 ON {on_recall:.0%} ｜ OFF {off_recall:.0%}")
    print(f"记忆进入真实 prompt 的比例：{prompt_proof:.0%}（SpyPlanner 抓 AgentGraph 实发 messages）")
    print(f"误注入：{false_inject} 条 ｜ 负样本被注入：{neg_leak}/{len(negatives)}")
    print(f"平均上下文增量：+{extra_chars:.0f} 字（约 +{extra_chars / 1.6:.0f} token）")

    report = os.path.join(ROOT, "eval", "memory_ab_report.json")
    with open(report, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, ensure_ascii=False, indent=2)
    print(f"报告：{report}")

    ok = (on_recall >= 0.9 and off_recall == 0.0 and false_inject == 0
          and neg_leak == 0 and prompt_proof >= 0.9)
    print("[OK] AB 通过：记忆确实在决策路径上，且关掉后事实缺席" if ok else "[FAIL] AB 未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
