# -*- coding: utf-8 -*-
"""可持久化、可增量的 FAISS 向量索引。

为什么要有这一层
----------------
原来的实现是「启动时把 9,301 块全部重新嵌入 + 建一个内存索引」：
1. **不能持久化**——每次进程重启都要重新过一遍模型（GPU 上是分钟级）；
2. **不能增量**——加一本手册要全量重建；
3. **不能删除**——旧内容留在索引里，会被检索到（安全与正确性问题）；
4. **没有版本与校验**——无法回答"这份索引对应哪份语料"，灰度/回滚也无从谈起。

设计取舍（写在这里，免得以后被"为什么不直接上 IVF/HNSW"问住）
------------------------------------------------------------
- **索引由 `vectors` 矩阵派生**：`add`/`remove` 只在向量矩阵上增删，然后重建 `IndexFlatIP`。
  好处是**永远不需要重新嵌入**（嵌入才是耗时大头），代码也简单到不会出错；
  代价是每次增删重建一次 flat 索引（9,301×1024 实测 ~10 ms，10 万块 ~1 s 量级）。
- **暂时不用 IVF/HNSW**：flat 是精确检索，规模在十万以内延迟可接受；
  上近似索引会引入 nprobe/efSearch 调参和召回率下降，属于"确定有需要再换"的优化。
  接口（`add/remove/search/save/load`）与索引类型解耦，换实现不影响调用方。
- **持久化三件套**：`index.faiss`（FAISS 原生）、`vectors.npy`（用于增删后重建）、
  `meta.json`（维度/模型/块数/语料 checksum/时间）。checksum 是**过期检测**的依据：
  语料变了而索引没重建，必须能被发现，而不是"悄悄用旧索引"。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# faiss / numpy 都是可选依赖：没有它们时向量路应当**静默降级为纯 BM25**，
# 而不是让整个 Agent 起不来（这是本项目一贯的"优雅降级"原则）。
try:                                                    # pragma: no cover
    import faiss  # type: ignore
    import numpy as np  # type: ignore

    HAS_FAISS = True
except Exception:                                       # noqa: BLE001  pragma: no cover
    faiss = None                                        # type: ignore
    np = None                                           # type: ignore
    HAS_FAISS = False

INDEX_FILE = "index.faiss"
VECTORS_FILE = "vectors.npy"
META_FILE = "meta.json"


def compute_checksum(chunks: Sequence[object]) -> str:
    """语料指纹：对 (id, text) 做 sha256。用于判断"索引是否对应当前语料"。"""
    digest = hashlib.sha256()
    for chunk in chunks:
        cid = getattr(chunk, "id", None)
        text = getattr(chunk, "text", None)
        if text is None and isinstance(chunk, dict):
            cid, text = chunk.get("id"), chunk.get("text")
        digest.update(f"{cid}\x00{text}\x00".encode("utf-8"))
    return digest.hexdigest()[:16]


@dataclass
class VectorMeta:
    dim: int
    n_vectors: int
    model_id: str = ""
    chunks_checksum: str = ""
    index_type: str = "flat-ip"
    built_at: float = field(default_factory=time.time)
    source: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {"dim": self.dim, "n_vectors": self.n_vectors, "model_id": self.model_id,
                "chunks_checksum": self.chunks_checksum, "index_type": self.index_type,
                "built_at": self.built_at, "source": self.source}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorMeta":
        return cls(dim=int(data.get("dim") or 0),
                   n_vectors=int(data.get("n_vectors") or 0),
                   model_id=str(data.get("model_id") or ""),
                   chunks_checksum=str(data.get("chunks_checksum") or ""),
                   index_type=str(data.get("index_type") or "flat-ip"),
                   built_at=float(data.get("built_at") or 0.0),
                   source=str(data.get("source") or ""))


class VectorStore:
    """可增删、可落盘的向量索引（flat 精确检索）。"""

    def __init__(self, vectors=None, meta: Optional[VectorMeta] = None):
        if not HAS_FAISS:
            raise RuntimeError("faiss/numpy 不可用，向量路应保持关闭")
        self.meta = meta or VectorMeta(dim=0, n_vectors=0)
        self._vectors = vectors if vectors is not None else np.zeros((0, self.meta.dim),
                                                                   dtype="float32")
        self._index: Any = None       # faiss.Index（faiss 是可选依赖，故不写具体类型）
        self._rebuild()

    # ── 构建 ──
    @classmethod
    def build(cls, vectors, *, model_id: str = "", chunks_checksum: str = "",
              source: str = "") -> "VectorStore":
        arr = np.asarray(vectors, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        meta = VectorMeta(dim=int(arr.shape[1]) if arr.size else 0, n_vectors=int(arr.shape[0]),
                          model_id=model_id, chunks_checksum=chunks_checksum, source=source)
        return cls(arr, meta)

    def _rebuild(self) -> None:
        if not HAS_FAISS:
            return
        dim = int(self._vectors.shape[1]) if self._vectors.size else self.meta.dim
        # 内积 + L2 归一化 = 余弦相似度。用 IndexIDMap2 直连位置 id，
        # 便于把检索结果直接映射回 chunks 列表下标。
        index = faiss.IndexFlatIP(dim or 1)
        if self._vectors.size:
            index.add(self._vectors)
        self._index = index
        self.meta.n_vectors = int(self._vectors.shape[0])
        if self._vectors.size:
            self.meta.dim = int(self._vectors.shape[1])

    # ── 增删 ──
    def add(self, vectors) -> int:
        """追加向量（顺序必须与 chunks 追加顺序一致）。返回新增数量。"""
        arr = np.asarray(vectors, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.size == 0:
            return 0
        if self._vectors.size == 0:
            self._vectors = arr
        else:
            if arr.shape[1] != self._vectors.shape[1]:
                raise ValueError(f"向量维度不一致：{arr.shape[1]} vs {self._vectors.shape[1]}")
            self._vectors = np.vstack([self._vectors, arr])
        self._rebuild()
        return int(arr.shape[0])

    def remove(self, positions: Sequence[int]) -> int:
        """按下标删除（对应 chunks 列表下标）；删除后**下标会重排**，调用方需同步列表。"""
        if not positions:
            return 0
        doomed = {int(p) for p in positions}
        keep = [i for i in range(int(self._vectors.shape[0])) if i not in doomed]
        removed = int(self._vectors.shape[0]) - len(keep)
        if removed:
            self._vectors = self._vectors[keep] if keep else np.zeros(
                (0, self.meta.dim), dtype="float32")
            self._rebuild()
        return removed

    # ── 检索 ──
    def search(self, query_vector, top_k: int = 15) -> Tuple[List[float], List[int]]:
        if self._index is None or not self.meta.n_vectors:
            return [], []
        query = np.asarray(query_vector, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        scores, ids = self._index.search(query, min(top_k, self.meta.n_vectors))
        return [float(s) for s in scores[0]], [int(i) for i in ids[0]]

    # ── 持久化 ──
    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, os.path.join(directory, INDEX_FILE))
        np.save(os.path.join(directory, VECTORS_FILE), self._vectors)
        self.meta.built_at = time.time()
        with open(os.path.join(directory, META_FILE), "w", encoding="utf-8") as fh:
            json.dump(self.meta.to_dict(), fh, ensure_ascii=False, indent=2)
        return directory

    @classmethod
    def load(cls, directory: str, *, expect_checksum: Optional[str] = None,
             expect_model: Optional[str] = None) -> Optional["VectorStore"]:
        """读取索引；不存在/过期/损坏时返回 None（由调用方决定重建）。

        `expect_checksum` / `expect_model` 是**过期检测**：不匹配说明"语料或模型变了"，
        此时必须重建，而不是拿旧索引凑合——旧索引会把已删除的内容继续检索出来。
        """
        if not HAS_FAISS:
            return None
        meta_path = os.path.join(directory, META_FILE)
        vectors_path = os.path.join(directory, VECTORS_FILE)
        if not (os.path.isfile(meta_path) and os.path.isfile(vectors_path)):
            return None
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = VectorMeta.from_dict(json.load(fh))
            if expect_model and meta.model_id and meta.model_id != expect_model:
                return None
            if expect_checksum and meta.chunks_checksum != expect_checksum:
                return None
            vectors = np.load(vectors_path)
            store = cls(vectors, meta)
            # index.faiss 存在就用文件里的（省一次 add），否则用 vectors 重建
            index_path = os.path.join(directory, INDEX_FILE)
            if os.path.isfile(index_path):
                loaded = faiss.read_index(index_path)
                if loaded.ntotal == meta.n_vectors:
                    store._index = loaded
            return store
        except Exception:                               # noqa: BLE001
            return None

    # ── 观测 ──
    def stats(self) -> Dict[str, object]:
        return {"n_vectors": self.meta.n_vectors, "dim": self.meta.dim,
                "index_type": self.meta.index_type, "model_id": self.meta.model_id,
                "chunks_checksum": self.meta.chunks_checksum,
                "built_at": self.meta.built_at}
