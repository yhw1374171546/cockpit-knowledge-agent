# -*- coding: utf-8 -*-
"""generate：程序化生成 100 车主 × 7 轮跨会话对话 + gold 事实 + 六题型题目。

为什么程序化而不是 LLM 生成（诚实取舍）
--------------------------------------
- **gold 可控**：题目答案直接从画像卡推导，零标注误差；LLM 生成的对话
  需要人工审核协议（LongMemEval 成品率约 5%），现阶段性价比低；
- **完全可复现**：同种子同产物，可进 CI 做版本化对比；
- 局限：句式模板化，"证据间接透露"的难度低于 LongMemEval 真人生成——
  后续可加 LLM 润色开关（--paraphrase），gold 不变、表面形式变。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.memory_bench.profiles import build_cards  # noqa: E402
from eval.memory_bench.schema import Question, Sample, Turn, save_jsonl, stats  # noqa: E402

OUT_JSONL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_v0.1.jsonl")


def build_sample(card: Dict[str, Any], idx: int) -> Sample:
    """把一张画像卡展开成：7 轮对话（3 会话、跨度 14 天）+ gold 事实 + 6 道题。"""
    base = date(2026, 10, 1) + timedelta(days=idx)
    d0, d7, d14 = base, base + timedelta(days=7), base + timedelta(days=14)
    order_id = f"SO{91000000 + idx}"
    sid = f"mb-{idx:03d}"

    turns = [
        Turn(d0.isoformat(), 1, "user",
             f"我住在{card['city']}，我的车是{card['model']}，车牌是{card['plate']}"),
        Turn(d0.isoformat(), 1, "user", f"我一般去{card['center']}，以后保养都去这家"),
        Turn(d7.isoformat(), 2, "user",
             f"帮我约 {d7.isoformat()} 到店保养，换机油机滤。我已经确认过了，直接办。"),
        Turn(d7.isoformat(), 2, "assistant",
             f"已按您的要求办好预约：单号 {order_id}，更换机油机滤，{d7.isoformat()} "
             f"到 {card['center']}，状态：已预约。"),
        Turn(d7.isoformat(), 2, "user", f"以后{card['pref_feature']}都{card['pref_action']}"),
    ]
    gold: List[Dict[str, Any]] = [
        {"key": "city", "value": card["city"], "source": "user_explicit", "turn": 1},
        {"key": "vehicle_model", "value": card["model"], "source": "user_explicit", "turn": 1},
        {"key": "plate_region", "value": card["plate"], "source": "user_explicit", "turn": 1},
        {"key": "usual_center", "value": card["center"], "source": "user_explicit", "turn": 2},
        {"key": f"pref:{card['pref_feature']}", "value": card["pref_action"],
         "source": "user_explicit", "turn": 5},
        {"key": "episode:order",
         "value": f"已建工单 {order_id}：更换机油机滤，{d7.isoformat()} 到 {card['center']}",
         "turn": 4},
    ]
    if card["drift"]:
        turns.append(Turn(d14.isoformat(), 3, "user",
                          f"不对，我一般去{card['alt_center']}，以后都去这家"))
        gold.append({"key": "usual_center", "value": card["alt_center"],
                     "source": "user_explicit", "turn": 6})
    else:
        turns.append(Turn(d14.isoformat(), 3, "user", "上次那个保养约的是哪天？"))
    turns.append(Turn(d14.isoformat(), 3, "user", f"以后千万别{card['redline']}"))
    gold.append({"key": "do_not", "value": card["redline"], "source": "user_explicit", "turn": 7})

    latest_center = card["alt_center"] if card["drift"] else card["center"]
    questions = [
        Question(f"{sid}-IE1", "IE", "我住在哪个城市？", card["city"]),
        Question(f"{sid}-IE2", "IE", "我的车是什么型号？", card["model"], "接受简称（如 领克08）"),
        Question(f"{sid}-KU", "KU", "我现在一般去哪家店做保养？", latest_center,
                 "答最新门店；同时提到旧门店但给出新门店也算对"),
        Question(f"{sid}-TR", "TR", "我最开始说去哪家店保养？后来有没有改？",
                 f"最初 {card['center']}" + (f"，后来改成 {card['alt_center']}"
                                             if card["drift"] else "，没有改过")),
        Question(f"{sid}-MR", "MR", "我上次预约的保养约的是哪天？", d7.isoformat()),
        Question(f"{sid}-ABS", "ABS", f"我之前说过我的车是{card['abs_model']}吗？",
                 "不知道/没有记录（必须拒答）"),
    ]
    return Sample(sid=sid, profile_card=card, turns=turns, gold_facts=gold,
                  questions=questions)


def generate(n: int = 100, seed: int = 42, out: str = OUT_JSONL) -> Tuple[List[Sample], Dict[str, Any]]:
    samples = [build_sample(card, i) for i, card in enumerate(build_cards(n=n, seed=seed))]
    save_jsonl(samples, out)
    return samples, stats(samples)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成座舱长期记忆 benchmark（程序化、确定性）")
    ap.add_argument("--n", type=int, default=100, help="车主数（默认 100 → 700 轮 / 600 题）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=OUT_JSONL)
    args = ap.parse_args()
    samples, st = generate(n=args.n, seed=args.seed, out=args.out)
    print(f"已生成 {st['samples']} 个车主样本 → {args.out}")
    print(f"  轮次 {st['turns']} ｜ 题目 {st['questions']} ｜ 题型分布 {st['by_type']}")
    print(f"  会话数 {st['sessions']}（每车主 3 会话、跨度 14 天）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
