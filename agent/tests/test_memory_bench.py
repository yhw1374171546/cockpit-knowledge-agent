# -*- coding: utf-8 -*-
"""memory_bench 测试：画像卡确定性 / 生成物 gold 完整性 / judge 解析 / 检索层命中。"""

from __future__ import annotations

import tempfile

from agent.llm import LLMResponse, ScriptedLLM
from eval.memory_backends import make_backend
from eval.memory_bench.generate import build_sample
from eval.memory_bench.judge import judge, meta_eval_report, parse_verdict
from eval.memory_bench.profiles import build_cards
from eval.memory_bench.runner import evaluate_sample, ingest, value_hit
from eval.memory_bench.schema import load_jsonl, save_jsonl, stats


def test_cards_are_deterministic_and_complete():
    cards1 = build_cards(n=100, seed=42)
    cards2 = build_cards(n=100, seed=42)
    assert cards1 == cards2
    assert len(cards1) == 100
    for card in cards1:
        for key in ("model", "city", "plate", "center", "drift", "alt_center",
                    "pref_feature", "pref_action", "redline", "abs_model"):
            assert key in card, f"缺字段 {key}"
        assert card["abs_model"] != card["model"]          # ABS 假前提必须与真值不同


def test_generated_sample_gold_integrity():
    card = build_cards(n=1, seed=42)[0]
    sample = build_sample(card, 0)
    assert len(sample.turns) == 7
    assert len(sample.questions) == 6
    types = {q.qtype for q in sample.questions}
    assert types == {"IE", "KU", "TR", "MR", "ABS"}
    # 漂移样本：usual_center 有两条演化史，KU 的 gold 是最新值
    centers = [f["value"] for f in sample.gold_facts if f["key"] == "usual_center"]
    if card["drift"]:
        assert centers == [card["center"], card["alt_center"]]
    else:
        assert centers == [card["center"]]
    ku = next(q for q in sample.questions if q.qtype == "KU")
    assert ku.gold == (card["alt_center"] if card["drift"] else card["center"])
    # gold 事实的位置标注必须指向真实轮次
    for fact in sample.gold_facts:
        assert 1 <= fact["turn"] <= len(sample.turns)
    # 时间推理的跨度：首尾轮次相差 14 天
    assert sample.turns[0].ts != sample.turns[-1].ts


def test_jsonl_roundtrip_and_stats():
    card = build_cards(n=3, seed=42)
    samples = [build_sample(c, i) for i, c in enumerate(card)]
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        path = tmp.name
    save_jsonl(samples, path)
    loaded = load_jsonl(path)
    assert [s.sid for s in loaded] == [s.sid for s in samples]
    st = stats(loaded)
    assert st["turns"] == 21 and st["questions"] == 18
    assert st["by_type"]["IE"] == 6


def test_value_hit_normalization():
    assert value_hit("领克08 EM-P", "车型=领克08")
    assert value_hit("上海", "常住城市=上海")
    assert not value_hit("上海", "常住城市=杭州")


def test_judge_parses_verdict():
    assert parse_verdict("理由：答案正确。\nVERDICT: yes") == "yes"
    assert parse_verdict("……\nVERDICT: No\n") == "no"
    assert parse_verdict("没有结论") is None


def test_judge_with_scripted_llm():
    llm = ScriptedLLM([LLMResponse(content="理由：命中。\nVERDICT: yes")])
    result = judge(llm, "KU", "我现在一般去哪家店？", "领克中心(浦东金桥店)",
                   "您现在一般去浦东金桥店")
    assert result["verdict"] == "yes"


def test_meta_eval_threshold():
    report = meta_eval_report([("yes", "yes")] * 9 + [("no", "yes")])
    assert report["agreement"] == 0.9 and report["usable"] is True
    assert meta_eval_report([("yes", "yes"), ("no", "yes")])["usable"] is False


def _run_sample(profile: str, sample) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = tmp.name
    backend = make_backend(profile, path, None)
    try:
        ingest(backend, sample)
        rows = evaluate_sample(backend, sample)
    finally:
        backend.close()
    return {r["qtype"]: r for r in rows}


def test_recall_layer_v2_passes_ie_ku_tr():
    card = build_cards(n=1, seed=42)[0]
    card["drift"] = True                                   # 强制漂移，测 TR
    sample = build_sample(card, 0)
    rows = _run_sample("v2", sample)
    assert all(r["hit"] for r in rows.values() if r["qtype"] in ("IE", "KU")), rows
    # TR：v2 有演化史，两个值都在
    assert rows["TR"]["hit"] is True


def test_recall_layer_v1_fails_tr_honestly():
    card = build_cards(n=1, seed=42)[0]
    card["drift"] = True
    sample = build_sample(card, 0)
    rows = _run_sample("v1", sample)
    # v1 的 overwrite 丢历史：TR（演化史）必然失败——如实记录，不是评测 bug
    assert rows["TR"]["hit"] is False
    # KU（最新值）v1 应通过（overwrite 恰好满足"只记最新"）
    assert rows["KU"]["hit"] is True
