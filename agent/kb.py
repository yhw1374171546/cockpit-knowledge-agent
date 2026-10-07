# -*- coding: utf-8 -*-
"""知识库层：文档块存储 + 检索（BM25 为主，向量检索可选）。

设计要点
--------
1. **零依赖可运行**：BM25 自行实现（标准 k1/b 公式），分词优先 jieba，缺失时退化为
   中文字符二元组（bigram），因此没有 torch / rank_bm25 也能跑通全链路。
2. **引用溯源**：每个块都带 `id / source / page / header`，答案可以给出处。
3. **可切换向量路**：有 torch + faiss + m3e-large 时自动启用语义召回，与 BM25 做 RRF 融合。

用法：
    kb = KnowledgeBase.load()                       # 默认读 all_text.txt
    kb = KnowledgeBase.load("kb/chunks.jsonl")      # 带 page/header 元数据的版本
    for ev in kb.search("座椅加热怎么关", top_k=6):
        print(ev.cite(), ev.text[:60])
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_KB = os.path.join(ROOT, "all_text.txt")
DEFAULT_KB_JSONL = os.path.join(ROOT, "kb", "chunks.jsonl")
# 迷你语料：仓库不含受版权限制的真实手册，CI 与快速试用走这份合成语料
FIXTURE_KB = os.path.join(ROOT, "eval", "fixtures", "mini_corpus.jsonl")
KB_ENV_VAR = "AGENT_KB_PATH"


# ── 分词 ─────────────────────────────────────────────────────────────


class Tokenizer:
    """优先 jieba（搜索引擎模式），不可用时退化为字符二元组。"""

    def __init__(self, prefer_jieba: bool = True):
        self._jieba = None
        self.backend = "bigram"
        if prefer_jieba:
            try:
                import jieba  # type: ignore

                jieba.setLogLevel(60)
                self._jieba = jieba
                self.backend = "jieba-search"
            except Exception:
                self._jieba = None

    def cut(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        if self._jieba is not None:
            tokens = [t.strip() for t in self._jieba.cut_for_search(text) if t.strip()]
            # 中文单字信息量低，过滤掉纯单字噪声（保留英文/数字）
            return [t for t in tokens if len(t) > 1 or not _is_cjk(t)]
        return _bigrams(text)


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _bigrams(text: str) -> List[str]:
    """中文字符二元组 + 英文数字词，零依赖的中文检索兜底方案。"""
    text = re.sub(r"\s+", "", text)
    out: List[str] = []
    buf = ""
    for ch in text:
        if _is_cjk(ch):
            if buf:
                out.append(buf.lower())
                buf = ""
            out.append(ch)
        elif ch.isalnum():
            buf += ch
        else:
            if buf:
                out.append(buf.lower())
                buf = ""
    if buf:
        out.append(buf.lower())
    # 拼接相邻中文单字为 bigram，兼顾精度与召回
    merged: List[str] = []
    i = 0
    while i < len(out):
        if len(out[i]) == 1 and _is_cjk(out[i]) and i + 1 < len(out) and len(out[i + 1]) == 1 \
                and _is_cjk(out[i + 1]):
            merged.append(out[i] + out[i + 1])
            i += 1
        else:
            merged.append(out[i])
            i += 1
    return merged


# ── 数据模型 ──────────────────────────────────────────────────────────


@dataclass
class Chunk:
    id: int
    text: str
    source: str = "train_a.pdf"
    page: Optional[int] = None
    header: Optional[str] = None
    strategy: Optional[str] = None

    def cite(self) -> str:
        parts = [self.source]
        if self.page is not None:
            parts.append(f"第{self.page}页")
        if self.header:
            parts.append(self.header)
        return "·".join(parts) if len(parts) > 1 or parts[0] != "train_a.pdf" else f"块#{self.id}"

    def to_dict(self) -> Dict:
        return {"id": self.id, "text": self.text, "source": self.source,
                "page": self.page, "header": self.header, "strategy": self.strategy}


@dataclass
class Evidence:
    """一次检索命中的证据，供 Prompt 组装与引用校验使用。"""
    chunk_id: int
    text: str
    score: float
    source: str = "train_a.pdf"
    page: Optional[int] = None
    header: Optional[str] = None
    retriever: str = "bm25"

    @property
    def citation(self) -> str:
        if self.page is not None:
            return f"[{self.source} 第{self.page}页{('·' + self.header) if self.header else ''}]"
        return f"[{self.source} 块#{self.chunk_id}]"

    def to_dict(self) -> Dict:
        return {"chunk_id": self.chunk_id, "score": round(self.score, 4), "page": self.page,
                "header": self.header, "retriever": self.retriever,
                "source": self.source, "citation": self.citation, "text": self.text}


# ── BM25 ─────────────────────────────────────────────────────────────


class BM25Index:
    """标准 BM25（Okapi），自行实现以去除 rank_bm25 依赖。"""

    def __init__(self, tokenized_docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [list(d) for d in tokenized_docs]
        self.doc_len = [len(d) for d in self.docs]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.tf: List[Counter] = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf: Dict[str, float] = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def search(self, query_tokens: Sequence[str], top_k: int = 10) -> List[Tuple[int, float]]:
        scores: List[Tuple[int, float]] = []
        for i, tf in enumerate(self.tf):
            score = 0.0
            norm = 1 - self.b + self.b * (self.doc_len[i] / self.avg_len if self.avg_len else 1.0)
            for term in query_tokens:
                freq = tf.get(term)
                if not freq:
                    continue
                score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / (freq + self.k1 * norm)
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ── 知识库 ────────────────────────────────────────────────────────────


@dataclass
class KnowledgeBase:
    chunks: List[Chunk] = field(default_factory=list)
    tokenizer: Tokenizer = field(default_factory=Tokenizer)
    _bm25: Optional[BM25Index] = None
    _vector: object = None

    # -- 加载 ----------------------------------------------------------
    @classmethod
    def resolve_path(cls, path: Optional[str] = None) -> str:
        """确定知识库来源：显式路径 > 环境变量 AGENT_KB_PATH > kb/chunks.jsonl >
        all_text.txt > 迷你语料（仓库自带，保证克隆后即可运行）。"""
        if path:
            return path
        env = os.environ.get(KB_ENV_VAR)
        if env and os.path.exists(env):
            return env
        if os.path.exists(DEFAULT_KB_JSONL):
            return DEFAULT_KB_JSONL
        if os.path.exists(DEFAULT_KB):
            return DEFAULT_KB
        return FIXTURE_KB

    @classmethod
    def load(cls, path: Optional[str] = None, prefer_jieba: bool = True,
             with_vector: bool = False, vector_model: Optional[str] = None) -> "KnowledgeBase":
        path = cls.resolve_path(path)
        kb = cls(tokenizer=Tokenizer(prefer_jieba))
        if path.endswith(".jsonl"):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    kb.chunks.append(Chunk(id=d.get("id", len(kb.chunks)), text=d.get("text", ""),
                                           source=d.get("source", "train_a.pdf"),
                                           page=d.get("page"), header=d.get("header"),
                                           strategy=d.get("strategy")))
        else:
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    text = line.strip()
                    if len(text) < 5:
                        continue
                    kb.chunks.append(Chunk(id=i, text=text))
        kb.build_index()
        if with_vector:
            kb.enable_vector(vector_model)
        return kb

    def build_index(self) -> None:
        self._bm25 = BM25Index([self.tokenizer.cut(c.text) for c in self.chunks])

    # -- 可选向量路 ----------------------------------------------------
    def enable_vector(self, model_path: Optional[str] = None) -> bool:
        """启用 m3e-large + FAISS 语义召回；依赖缺失时静默降级为纯 BM25。"""
        model_path = model_path or os.path.join(ROOT, "pre_train_model", "m3e-large")
        try:
            import faiss  # type: ignore
            import numpy as np  # type: ignore
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path).eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            def encode(texts: Sequence[str]) -> "np.ndarray":
                vecs = []
                for i in range(0, len(texts), 16):
                    batch = list(texts[i:i + 16])
                    inputs = tokenizer(batch, padding=True, truncation=True,
                                       max_length=512, return_tensors="pt").to(device)
                    with torch.no_grad():
                        hidden = model(**inputs).last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    vecs.append(pooled.cpu().numpy().astype("float32"))
                return np.vstack(vecs)

            matrix = encode([c.text for c in self.chunks])
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            self._vector = (encode, index)
            return True
        except Exception:
            self._vector = None
            return False

    def search_vector(self, query: str, top_k: int = 15) -> List[Evidence]:
        if self._vector is None:
            return []
        encode, index = self._vector  # type: ignore
        vec = encode([query])
        scores, ids = index.search(vec, top_k)
        out = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            c = self.chunks[int(idx)]
            out.append(Evidence(c.id, c.text, float(score), c.source, c.page, c.header, "vector"))
        return out

    # -- 检索 ----------------------------------------------------------
    def search_bm25(self, query: str, top_k: int = 15) -> List[Evidence]:
        if self._bm25 is None:
            self.build_index()
        hits = self._bm25.search(self.tokenizer.cut(query), top_k=top_k)  # type: ignore
        out = []
        for idx, score in hits:
            c = self.chunks[idx]
            out.append(Evidence(c.id, c.text, float(score), c.source, c.page, c.header, "bm25"))
        return out

    def search(self, query: str, top_k: int = 6, recall_k: int = 15,
               use_vector: Optional[bool] = None, fusion: str = "rrf") -> List[Evidence]:
        """混合检索：BM25（+可选向量）→ RRF 融合 → 截断 top_k。

        RRF 只用排名不用分数，天然规避「向量分与 BM25 分量纲不同」的问题。
        """
        use_vector = bool(self._vector) if use_vector is None else (use_vector and bool(self._vector))
        bm25_hits = self.search_bm25(query, recall_k)
        if not use_vector:
            return bm25_hits[:top_k]

        vec_hits = self.search_vector(query, recall_k)
        k = 60.0
        fused: Dict[int, float] = {}
        keep: Dict[int, Evidence] = {}
        for rank, ev in enumerate(bm25_hits, start=1):
            fused[ev.chunk_id] = fused.get(ev.chunk_id, 0.0) + 1.0 / (k + rank)
            keep.setdefault(ev.chunk_id, ev)
        for rank, ev in enumerate(vec_hits, start=1):
            fused[ev.chunk_id] = fused.get(ev.chunk_id, 0.0) + 1.0 / (k + rank)
            keep.setdefault(ev.chunk_id, ev)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        out = []
        for cid, score in ranked:
            ev = keep[cid]
            ev.retriever = "hybrid-rrf" if ev.retriever == "bm25" else ev.retriever
            ev.score = score
            out.append(ev)
        return out

    # -- 工具方法 ------------------------------------------------------
    def get(self, chunk_id: int) -> Optional[Chunk]:
        for c in self.chunks:
            if c.id == chunk_id:
                return c
        return None

    def stats(self) -> Dict:
        lengths = [len(c.text) for c in self.chunks]
        return {
            "n_chunks": len(self.chunks),
            "total_chars": sum(lengths),
            "avg_len": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "with_page_meta": sum(1 for c in self.chunks if c.page is not None),
            "tokenizer": self.tokenizer.backend,
            "has_vector": bool(self._vector),
        }


if __name__ == "__main__":
    kb = KnowledgeBase.load()
    print(json.dumps(kb.stats(), ensure_ascii=False, indent=2))
    for q in ["座椅加热怎么关闭", "怎么打开危险警告灯", "保养周期"]:
        print(f"\nQ: {q}")
        for ev in kb.search(q, top_k=3):
            print(f"  {ev.score:.4f} {ev.citation} {ev.text[:50]}")
