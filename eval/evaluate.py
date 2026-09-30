# -*- coding: utf-8 -*-
"""离线评测脚本（修正版）。

相对原始 test_score.py 的改进：
1. 修复 gold 路径硬编码（原脚本读 ./data/gold2.json，仓库里只有 gold.json）；
2. 支持对 4 路消融答案（answer_1~answer_4）分别打分，产出对比表；
3. 增加检索侧指标：存储的检索上下文（answer_6 / answer_7）对标注关键词的覆盖率；
4. 语义相似度自主实现（macbert + mean pooling + cosine），与 text2vec SentenceModel 口径一致，
   避免依赖 text2vec / sentence-transformers 的版本兼容问题；
5. 结果落盘 JSON + Markdown 表，便于回填 README 与简历。

用法：
    python eval/evaluate.py                       # 关键词指标（无需 torch）
    python eval/evaluate.py --semantic            # 追加语义相似度（需要 torch + transformers）
    python eval/evaluate.py --semantic --json-out eval/metrics.json --md-out eval/report.md
"""

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GOLD = os.path.join(ROOT, "data", "gold.json")
DEFAULT_PRED = os.path.join(ROOT, "data", "result.json")
DEFAULT_SIM_MODEL = os.path.join(ROOT, "pre_train_model", "text2vec-base-chinese")

ANSWER_FIELDS = ["answer_1", "answer_2", "answer_3", "answer_4"]
FIELD_DESC = {
    "answer_1": "双路融合（无精排）",
    "answer_2": "仅 BM25",
    "answer_3": "仅向量",
    "answer_4": "双路 + 精排（最终方案）",
}
NO_ANSWER = "无答案"

# ── 语义相似度（可选依赖） ────────────────────────────────────────────


class SemanticScorer:
    """macbert + mean pooling + cosine，对齐 text2vec SentenceModel 的推理口径。"""

    def __init__(self, model_path=DEFAULT_SIM_MODEL, max_length=128, batch_size=64, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size

    def encode(self, texts):
        import torch.nn.functional as F

        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else "空" for t in texts[i:i + self.batch_size]]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vectors.append(F.normalize(pooled, p=2, dim=1).cpu())
        return self.torch.cat(vectors)


# ── 指标 ──────────────────────────────────────────────────────────────


def keyword_score(pred, keywords, threshold=0.3):
    """关键词覆盖率：命中关键词数 / 标注关键词数，超过阈值记 1 分（与原实现一致）。"""
    if not keywords:
        return 0.0, 0.0
    hit = [w for w in keywords if w in pred]
    recall = len(hit) / (len(keywords) + 1e-6)
    return (1.0 if recall > threshold else 0.0), recall


def score_entry(gold_answer, keywords, pred, semantic_fn=None, threshold=0.3):
    gold_answer = (gold_answer or "").strip()
    pred = (pred or "").strip()
    if gold_answer == NO_ANSWER:
        return {
            "score": 1.0 if pred == NO_ANSWER else 0.0,
            "is_refusal_case": True,
            "refusal_correct": pred == NO_ANSWER,
            "keyword_binary": None,
            "keyword_recall": None,
            "semantic": None,
        }
    kw_binary, kw_recall = keyword_score(pred, keywords, threshold)
    semantic = semantic_fn(gold_answer, pred) if semantic_fn else None
    score = 0.5 * kw_binary + 0.5 * semantic if semantic is not None else kw_binary
    return {
        "score": score,
        "is_refusal_case": False,
        "refusal_correct": None,
        "keyword_binary": kw_binary,
        "keyword_recall": kw_recall,
        "semantic": semantic,
    }


def context_keyword_recall(context, keywords):
    """检索上下文对标注关键词的覆盖率（RAGAS context recall 的简化版）。"""
    if not context or not keywords:
        return None
    hit = [w for w in keywords if w in context]
    return len(hit) / len(keywords)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


# ── 主流程 ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=DEFAULT_GOLD)
    parser.add_argument("--pred", default=DEFAULT_PRED)
    parser.add_argument("--sim-model", default=DEFAULT_SIM_MODEL)
    parser.add_argument("--semantic", action="store_true", help="计算语义相似度（需要 torch）")
    parser.add_argument("--keyword-threshold", type=float, default=0.3)
    parser.add_argument("--json-out", default=os.path.join(ROOT, "eval", "metrics.json"))
    parser.add_argument("--md-out", default=os.path.join(ROOT, "eval", "report.md"))
    parser.add_argument("--per-question", action="store_true", help="打印逐题明细")
    args = parser.parse_args()

    gold = json.load(open(args.gold, encoding="utf-8"))
    pred = json.load(open(args.pred, encoding="utf-8"))
    if len(gold) != len(pred):
        print(f"[warn] gold({len(gold)}) 与 pred({len(pred)}) 条数不一致，按较小值对齐", file=sys.stderr)
    n = min(len(gold), len(pred))

    scorer = None
    if args.semantic:
        print(f"[info] 加载语义模型：{args.sim_model}")
        scorer = SemanticScorer(args.sim_model)
        cache = {}

        def semantic_fn(a, b):
            key = (a, b)
            if key not in cache:
                vecs = scorer.encode([a, b])
                cache[key] = float((vecs[0] * vecs[1]).sum())
            return cache[key]
    else:
        semantic_fn = None

    variants = {}
    rows = []
    for field in ANSWER_FIELDS:
        entries = []
        for i in range(n):
            g = gold[i]
            p = pred[i]
            res = score_entry(g.get("answer", ""), g.get("keywords", []) or [],
                              p.get(field, ""), semantic_fn, args.keyword_threshold)
            res["question"] = g.get("question", "")
            res["pred"] = (p.get(field) or "").strip()
            res["gold"] = (g.get("answer") or "").strip()
            res["keywords"] = g.get("keywords", []) or []
            entries.append(res)
        variants[field] = {
            "desc": FIELD_DESC[field],
            "score": mean([e["score"] for e in entries]),
            "keyword_binary": mean([e["keyword_binary"] for e in entries]),
            "keyword_recall": mean([e["keyword_recall"] for e in entries]),
            "semantic": mean([e["semantic"] for e in entries]),
            "answer_avg_len": mean([len(e["pred"]) for e in entries]),
            "entries": entries,
        }
        rows.append(variants[field])

    # 检索侧：存储的上下文（answer_6 = BM25 上下文，answer_7 = 精排后上下文）
    ctx_stats = {}
    for field, label in (("answer_6", "BM25 召回上下文"), ("answer_7", "双路+精排上下文")):
        recalls, lengths, hit1 = [], [], 0
        for i in range(n):
            ctx = pred[i].get(field, "") or ""
            kws = gold[i].get("keywords", []) or []
            lengths.append(len(ctx))
            r = context_keyword_recall(ctx, kws)
            if r is not None:
                recalls.append(r)
                if r >= 1.0:
                    hit1 += 1
        ctx_stats[field] = {
            "label": label,
            "avg_length": mean(lengths),
            "avg_keyword_recall": mean(recalls),
            "full_recall_rate": (hit1 / len(recalls)) if recalls else None,
            "n": len(recalls),
        }

    # 拒答负样本
    refusal = []
    for i in range(n):
        if (gold[i].get("answer") or "").strip() == NO_ANSWER:
            refusal.append({
                "question": gold[i].get("question", ""),
                "gold": NO_ANSWER,
                "answer_4": (pred[i].get("answer_4") or "").strip(),
                "faiss_top1_distance": (pred[i].get("answer_5") or "").strip(),
            })

    report = {
        "n_questions": n,
        "variants": {k: {kk: vv for kk, vv in v.items() if kk != "entries"} for k, v in variants.items()},
        "context_stats": ctx_stats,
        "refusal_cases": refusal,
        "semantic_enabled": bool(scorer),
    }
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    json.dump(report, open(args.json_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ── 控制台 + Markdown 报告 ──
    lines = []
    lines.append("# 离线评测报告（自动生成）\n")
    lines.append(f"- 测试集：{n} 题（gold 与 result 对齐）")
    lines.append(f"- 关键词阈值：{args.keyword_threshold}")
    lines.append(f"- 语义相似度：{'已启用（macbert-mean-pooling, max_len=128）' if scorer else '未启用（仅关键词指标）'}\n")
    lines.append("## 1. 四路消融对比\n")
    head = "| 策略 | 综合得分 | 关键词覆盖(0/1) | 关键词覆盖率 | 语义相似度 | 平均答案长度 |"
    lines.append(head)
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for v in rows:
        lines.append(f"| {v['desc']} | {fmt(v['score'])} | {fmt(v['keyword_binary'])} | "
                     f"{fmt(v['keyword_recall'])} | {fmt(v['semantic'])} | {fmt(v['answer_avg_len'], 1)} |")

    lines.append("\n## 2. 检索上下文关键词覆盖率\n")
    lines.append("| 上下文来源 | 平均长度(字) | 平均关键词覆盖率 | 全命中率 |")
    lines.append("| --- | --- | --- | --- |")
    for st in ctx_stats.values():
        lines.append(f"| {st['label']} | {fmt(st['avg_length'], 1)} | {fmt(st['avg_keyword_recall'])} | "
                     f"{fmt(st['full_recall_rate'])} |")

    if refusal:
        lines.append("\n## 3. 拒答负样本明细\n")
        lines.append("| 问题 | answer_4 | FAISS Top-1 距离 |")
        lines.append("| --- | --- | --- |")
        for r in refusal:
            lines.append(f"| {r['question']} | {r['answer_4'][:40]} | {r['faiss_top1_distance']} |")

    md = "\n".join(lines)
    with open(args.md_out, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    print(f"\n[json] {args.json_out}\n[md]   {args.md_out}")

    if args.per_question:
        print("\n逐题明细（answer_4）：")
        for e in variants["answer_4"]["entries"]:
            print(f"  {e['score']:.3f}  kw={fmt(e['keyword_recall'],2)} sem={fmt(e['semantic'],3)}  {e['question'][:30]}")


if __name__ == "__main__":
    main()
