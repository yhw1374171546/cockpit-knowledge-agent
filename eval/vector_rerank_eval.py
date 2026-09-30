# -*- coding: utf-8 -*-
"""向量路 + RRF 融合 + Cross-Encoder 精排的检索侧评测（CPU 可跑）。

目的：补上 Agent 侧唯一没跑的实验——在 103 题测试集上，比较三种检索配置的证据覆盖率：
    A. 纯 BM25（Agent 默认，已测 0.8347）
    B. BM25 + 向量 RRF 融合
    C. B + bge-reranker-large 精排

对照基线（原 RAG 链路的存储上下文）：BM25 0.8314 / 双路+精排 0.9059

用法：
    python eval/vector_rerank_eval.py --bench        # 只测速，估算总耗时
    python eval/vector_rerank_eval.py --bench 64     # 用 64 个块测速
    python eval/vector_rerank_eval.py                # 全量评测
    python eval/vector_rerank_eval.py --limit 20     # 小样本快跑
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(ROOT, "data", "gold.json")
OUT_JSON = os.path.join(ROOT, "eval", "vector_rerank_metrics.json")
OUT_MD = os.path.join(ROOT, "eval", "vector_rerank_report.md")
M3E = os.path.join(ROOT, "pre_train_model", "m3e-large")
RERANKER = os.path.join(ROOT, "pre_train_model", "bge-reranker-large")


def load_cases(limit=0):
    gold = json.load(open(GOLD, encoding="utf-8"))
    cases = [{"question": g["question"], "keywords": g.get("keywords") or [],
              "negative": (g.get("answer") or "").strip() == "无答案"} for g in gold]
    return cases[:limit] if limit else cases


class VectorIndex:
    """m3e-large + FAISS 内积索引（向量已归一化，内积等价余弦）。"""

    def __init__(self, texts, model_path=M3E, batch_size=16, max_length=512):
        import faiss
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.np, self.torch, self.faiss = np, torch, faiss
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).eval().to(self.device)
        self.batch_size, self.max_length = batch_size, max_length
        self.texts = texts

        start = time.perf_counter()
        matrix = self.encode(texts)
        self.encode_seconds = time.perf_counter() - start
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)

    def encode(self, texts, log_every=0):
        import torch.nn.functional as F

        chunks = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t[:1500] if t else "空" for t in texts[i:i + self.batch_size]]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            chunks.append(F.normalize(pooled, p=2, dim=1).cpu().numpy().astype("float32"))
            if log_every and (i // self.batch_size) % log_every == 0:
                print(f"    encoded {min(i + self.batch_size, len(texts))}/{len(texts)}", flush=True)
        return self.np.vstack(chunks) if chunks else self.np.zeros((0, 1024), dtype="float32")

    def search(self, query, top_k=15):
        vec = self.encode([query])
        scores, ids = self.index.search(vec, top_k)
        return [(int(i), float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]


class Reranker:
    def __init__(self, model_path=RERANKER, max_length=512, batch_size=16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).eval().to(self.device)
        self.max_length, self.batch_size = max_length, batch_size

    def rank(self, query, docs):
        order = []
        for i in range(0, len(docs), self.batch_size):
            batch = docs[i:i + self.batch_size]
            inputs = self.tokenizer([(query, d[:1500]) for d in batch], padding=True,
                                    truncation=True, max_length=self.max_length,
                                    return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.model(**inputs).logits.view(-1).float().cpu().tolist()
            order.extend(zip(batch, logits))
        order.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in order]


def keyword_recall(texts, keywords):
    if not keywords:
        return None
    joined = "\n".join(texts)
    return sum(1 for w in keywords if w in joined) / len(keywords)


def mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=int, default=0, help="只对前 N 个块测速并退出")
    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 道题")
    ap.add_argument("--recall-k", type=int, default=15)
    ap.add_argument("--top-k", type=int, default=6)
    args = ap.parse_args()

    from agent.kb import KnowledgeBase

    print("[info] 加载知识库…", flush=True)
    kb = KnowledgeBase.load()
    texts = [c.text for c in kb.chunks]
    print(f"[info] {len(texts)} 个块", flush=True)

    if args.bench:
        n = min(args.bench, len(texts))
        print(f"[bench] 编码前 {n} 个块测速…", flush=True)
        index = VectorIndex(texts[:n])
        per_chunk = index.encode_seconds / n
        print(f"[bench] {index.encode_seconds:.1f}s / {n} 块 = {per_chunk * 1000:.0f} ms/块")
        print(f"[bench] 全量 {len(texts)} 块预计 {per_chunk * len(texts) / 60:.1f} 分钟")
        return

    cases = load_cases(args.limit)
    print(f"[info] 评测 {len(cases)} 题", flush=True)

    print("[info] 构建向量索引（m3e-large + FAISS）…", flush=True)
    index = VectorIndex(texts)
    print(f"[info] 向量索引完成：{index.encode_seconds:.1f}s", flush=True)

    reranker = Reranker()
    print("[info] 精排模型加载完成", flush=True)

    rows = []
    started = time.perf_counter()
    for i, case in enumerate(cases, start=1):
        q, kws = case["question"], case["keywords"]
        # A. 纯 BM25
        bm25_hits = kb.search_bm25(q, args.recall_k)
        # B. BM25 + 向量 RRF 融合
        vec_hits = index.search(q, args.recall_k)
        k = 60.0
        fused = {}
        for rank, ev in enumerate(bm25_hits, start=1):
            fused[ev.chunk_id] = fused.get(ev.chunk_id, 0.0) + 1.0 / (k + rank)
        for rank, (cid, _s) in enumerate(vec_hits, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
        hybrid_ids = [cid for cid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]
        # C. 对融合候选做精排
        cand_ids = (hybrid_ids[:args.recall_k] or
                    [ev.chunk_id for ev in bm25_hits][:args.recall_k])
        cand_texts = [kb.chunks[cid].text for cid in cand_ids]
        reranked_texts = reranker.rank(q, cand_texts)[:args.top_k]

        def texts_of(evs):
            return [e.text for e in evs[:args.top_k]]

        rows.append({
            "question": q,
            "bm25": keyword_recall(texts_of(bm25_hits), kws),
            "hybrid": keyword_recall([kb.chunks[c].text for c in hybrid_ids[:args.top_k]], kws),
            "hybrid_rerank": keyword_recall(reranked_texts, kws),
            "n_hybrid_candidates": len(hybrid_ids),
        })
        if i % 10 == 0:
            print(f"  {i}/{len(cases)} 已完成（{(time.perf_counter() - started):.0f}s）", flush=True)

    pos = [r for r, c in zip(rows, cases) if not c["negative"]]
    summary = {
        "n": len(cases), "recall_k": args.recall_k, "top_k": args.top_k,
        "vector_index_seconds": round(index.encode_seconds, 1),
        "device": index.device,
        "bm25_only": mean([r["bm25"] for r in pos]),
        "hybrid_rrf": mean([r["hybrid"] for r in pos]),
        "hybrid_rrf_rerank": mean([r["hybrid_rerank"] for r in pos]),
        "baseline_pipeline_bm25": 0.8314,
        "baseline_pipeline_rerank": 0.9059,
        "avg_candidates": mean([r["n_hybrid_candidates"] for r in rows]),
        "wall_clock_s": round(time.perf_counter() - started, 1),
    }
    json.dump({"summary": summary, "rows": rows},
              open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    md = ["# 向量路 + RRF 融合 + 精排 检索侧评测（自动生成）\n",
          f"- 设备：{summary['device']}｜向量索引耗时 {summary['vector_index_seconds']}s",
          f"- 召回 Top-{args.recall_k} → 精排 Top-{args.top_k}｜评测 {summary['n']} 题",
          f"- 平均融合候选数：{summary['avg_candidates']}\n",
          "## 证据关键词覆盖率对比\n",
          "| 检索配置 | 覆盖率 | 对照 |", "| --- | --- | --- |",
          f"| A. 纯 BM25（Top-{args.top_k}） | {summary['bm25_only']} | — |",
          f"| B. BM25 + 向量 RRF 融合 | {summary['hybrid_rrf']} | — |",
          f"| C. B + bge-reranker 精排 | {summary['hybrid_rrf_rerank']} | — |",
          f"| （原 RAG 链路）BM25 上下文 | {summary['baseline_pipeline_bm25']} | 历史跑批产物 |",
          f"| （原 RAG 链路）双路 + 精排上下文 | {summary['baseline_pipeline_rerank']} | 历史跑批产物 |",
          f"\n总耗时 {summary['wall_clock_s']}s"]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\n[json] {OUT_JSON}\n[md] {OUT_MD}")


if __name__ == "__main__":
    main()
