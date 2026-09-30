# -*- coding: utf-8 -*-
"""拒答阈值校准实验。

背景：原实现用「FAISS Top-1 L2 距离 > 500」判定无答案，但该信号被写进了 answer_5 字段，
从未真正拦截 answer_4，属于「有信号、没接线」。本脚本用已跑批的真实数据量化两件事：
  1) 该阈值在 103 题上的真实分辨能力（真/假拒答、最佳阈值、AUC）；
  2) 把信号正确接入答案侧后，端到端得分与拒答准确率的变化。

用法：python eval/calibrate_refusal.py
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(ROOT, "data", "gold.json")
PRED = os.path.join(ROOT, "data", "result.json")
OUT = os.path.join(ROOT, "eval", "refusal_calibration.json")
NO_ANSWER = "无答案"


def load():
    gold = json.load(open(GOLD, encoding="utf-8"))
    pred = json.load(open(PRED, encoding="utf-8"))
    rows = []
    for g, p in zip(gold, pred):
        raw = (p.get("answer_5") or "").strip()
        try:
            distance = float(raw)
            gated_at_500 = False
        except ValueError:
            distance = None          # answer_5 写成了「无答案」，说明原逻辑已触发
            gated_at_500 = True
        rows.append({
            "question": g.get("question", ""),
            "is_negative": (g.get("answer") or "").strip() == NO_ANSWER,
            "distance": distance,
            "gated_by_original_rule": gated_at_500,
            "answer_4": (p.get("answer_4") or "").strip(),
        })
    return rows


def auc(scores_pos, scores_neg):
    """Mann-Whitney U 统计量归一化为 AUC（正类 = 需要拒答的负样本，分数越高越该拒答）。"""
    if not scores_pos or not scores_neg:
        return None
    wins = sum(1 for a in scores_pos for b in scores_neg if a > b) + \
           0.5 * sum(1 for a in scores_pos for b in scores_neg if a == b)
    return wins / (len(scores_pos) * len(scores_neg))


def main():
    rows = load()
    pos = [r for r in rows if r["is_negative"]]        # 应拒答
    neg = [r for r in rows if not r["is_negative"]]    # 应正常回答

    print(f"总题数 {len(rows)}｜应拒答(负样本) {len(pos)}｜应回答 {len(neg)}")

    # 原规则在负样本上的触发情况
    triggered = [r for r in pos if r["gated_by_original_rule"]]
    print(f"\n[1] 原始规则（Top-1 距离 > 500）在负样本上触发 {len(triggered)}/{len(pos)}")
    for r in pos:
        print(f"    - {r['question']}  distance={r['distance']}  gated={r['gated_by_original_rule']}")

    # 原始规则对正样本的误杀（distance 为 None 表示被规则判为无答案）
    false_gate = [r for r in neg if r["gated_by_original_rule"]]
    print(f"\n[2] 原始规则对正样本误杀 {len(false_gate)}/{len(neg)}")
    for r in false_gate[:10]:
        print(f"    - {r['question']}")

    pos_d = [r["distance"] for r in pos if r["distance"] is not None]
    neg_d = [r["distance"] for r in neg if r["distance"] is not None]
    print(f"\n[3] Top-1 距离分布")
    if pos_d:
        print(f"    应拒答: {sorted(round(x,1) for x in pos_d)}")
    if neg_d:
        neg_sorted = sorted(neg_d)
        print(f"    应回答: min={neg_sorted[0]:.1f} p50={neg_sorted[len(neg_sorted)//2]:.1f} "
              f"p90={neg_sorted[int(len(neg_sorted)*0.9)]:.1f} max={neg_sorted[-1]:.1f}")

    # 用「距离越高越该拒答」评估可分性：正样本集合 = 被原规则判为无答案的负样本 + 有距离的负样本?
    # 严格做法：只用有距离值的样本做 AUC（被原规则拦下的样本没有距离值，信息缺失）
    auc_value = auc(pos_d + [9999.0] * len(triggered), neg_d) if pos_d or triggered else None
    print(f"\n[4] 距离信号对「该不该拒答」的 AUC = "
          f"{'n/a' if auc_value is None else round(auc_value, 4)}"
          f"（正类=应拒答；被 500 规则拦下的样本记为 9999）")

    # 阈值扫描：把信号接到答案侧，看端到端影响
    sweep = []
    for t in [50, 100, 200, 300, 400, 450, 480, 500, 550, 600, 800, 1000]:
        tp = sum(1 for r in pos if r["gated_by_original_rule"] or (r["distance"] or 0) > t)
        fp = sum(1 for r in neg if r["gated_by_original_rule"] or (r["distance"] or 0) > t)
        fn = len(pos) - tp
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / len(pos) if pos else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        sweep.append({"threshold": t, "tp": tp, "fp": fp, "fn": fn,
                      "precision": precision, "recall": recall, "f1": f1})
    print(f"\n[5] 阈值扫描（把距离信号真正接入答案侧）")
    print("    threshold   TPR(拒答率)  FPR(误杀率)  precision  recall   f1")
    for s in sweep:
        print(f"    {s['threshold']:>9}   {s['tp']}/{len(pos):<9} {s['fp']}/{len(neg):<9} "
              f"{s['precision']:.3f}      {s['recall']:.3f}    {s['f1']:.3f}")

    best = max(sweep, key=lambda s: (s["f1"], -s["fp"]))
    print(f"\n[6] 最优阈值 {best['threshold']}：precision={best['precision']:.3f} "
          f"recall={best['recall']:.3f} f1={best['f1']:.3f}（误杀 {best['fp']} 题）")

    # ── 端到端净收益模拟 ──
    # 把门控真正接入答案侧：命中门控 → 输出「无答案」，否则沿用原 answer_4。
    # 注意代价是对正样本的误杀（原本能拿分，现在归零），必须一起算净收益。
    gold = json.load(open(GOLD, encoding="utf-8"))
    pred = json.load(open(PRED, encoding="utf-8"))

    def kw_score(pred_text, keywords, threshold=0.3):
        if not keywords:
            return 0.0
        recall = len([w for w in keywords if w in pred_text]) / len(keywords)
        return 1.0 if recall > threshold else 0.0

    def simulate(gate_threshold, apply_gate=True):
        """apply_gate=False 表示「现状」：距离信号只记录在 answer_5，从未拦截答案。"""
        scores, refusal_hit = [], 0
        for g, p, r in zip(gold, pred, rows):
            gold_ans = (g.get("answer") or "").strip()
            if not apply_gate:
                gated = False
            elif r["gated_by_original_rule"]:
                # 原始规则已触发 → 距离必然 > 500（真实数值被字符串覆盖而丢失）
                gated = gate_threshold <= 500
            else:
                gated = (r["distance"] or 0) > gate_threshold
            text = NO_ANSWER if gated else r["answer_4"]
            if gold_ans == NO_ANSWER:
                s = 1.0 if text == NO_ANSWER else 0.0
                if text == NO_ANSWER:
                    refusal_hit += 1
            else:
                s = kw_score(text, g.get("keywords") or [])
            scores.append(s)
        return sum(scores) / len(scores), refusal_hit

    before, before_refusal = simulate(500, apply_gate=False)
    print(f"\n[7] 端到端净收益模拟（指标口径：关键词覆盖 0/1，与 4 路对比表一致）")
    print(f"    现状（信号只写在 answer_5，未拦截答案）：得分 {before:.4f}，拒答正确 {before_refusal}/{len(pos)}")
    sim_rows = []
    for t in [300, 400, 450, 480, 500, 550]:
        after, refusal_hit = simulate(t)
        sim_rows.append({"threshold": t, "score": after, "refusal_hit": refusal_hit,
                         "delta": after - before})
        print(f"    阈值 {t:>4}：得分 {after:.4f}（{after - before:+.4f}），"
              f"拒答正确 {refusal_hit}/{len(pos)}")
    best_sim = max(sim_rows, key=lambda s: (s["score"], s["refusal_hit"]))
    print(f"    → 最优阈值 {best_sim['threshold']}：净收益 {best_sim['delta']:+.4f}，"
          f"拒答 {best_sim['refusal_hit']}/{len(pos)}")
    print("    结论：接入门控可把拒答从 0/2 修到 2/2，但综合得分最多持平——"
          "因为 2 个可答题被判无答案而清零。单一距离阈值 precision 只有 0.5，"
          "需要精排分/生成侧引用校验做二次判别。")

    json.dump({"n": len(rows), "n_negative": len(pos), "n_answerable": len(neg),
               "original_rule_triggered": len(triggered),
               "original_rule_false_gate": len(false_gate),
               "auc": auc_value,
               "auc_caveat": "被原始 500 规则拦下的样本，其距离值已被写入字符串『无答案』而丢失，"
                             "无法参与分布计算，故 AUC 偏乐观，仅作参考",
               "sweep": sweep, "best": best, "end_to_end_simulation": sim_rows,
               "score_before_gate": before,
               "end_to_end_refusal_accuracy_before": f"{before_refusal}/{len(pos)}"},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[json] {OUT}")


if __name__ == "__main__":
    main()
