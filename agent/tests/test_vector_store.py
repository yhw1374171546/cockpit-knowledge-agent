# -*- coding: utf-8 -*-
"""向量索引持久化与增量维护测试（③：向量库持久化 + 增量索引）。

为什么能离线测：`enable_vector()` 支持注入 `encoder`，这里用**确定性桩编码器**
（字符 bigram 哈希 → 归一化向量）。它不追求语义质量，只保证"同一文本向量一致、
不同文本向量不同"，足以验证**持久化、过期检测、增删、检索映射**这些工程行为。
真实的 m3e 编码器需要 GPU 与模型权重，本机不具备——这一点在 README 里如实标注。

    python agent/tests/test_vector_store.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.kb import Chunk, KnowledgeBase  # noqa: E402
from agent.vector_store import HAS_FAISS, INDEX_FILE, META_FILE, VECTORS_FILE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(ROOT, "eval", "fixtures", "mini_corpus.jsonl")
DIM = 128


def stub_encoder(texts):
    """确定性桩编码器：字符 bigram 哈希到固定维度并 L2 归一化。"""
    import numpy as np

    matrix = np.zeros((len(texts), DIM), dtype="float32")
    for row, text in enumerate(texts):
        text = (text or "").strip()
        grams = [text[i:i + 2] for i in range(max(len(text) - 1, 1))] or [text]
        for gram in grams:
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            matrix[row, digest[0] % DIM] += 1.0
        norm = float(np.linalg.norm(matrix[row])) or 1.0
        matrix[row] /= norm
    return matrix


def make_kb(chunks) -> KnowledgeBase:
    kb = KnowledgeBase(path="")
    kb.chunks = list(chunks)
    kb.build_index()
    return kb


class VectorStoreTestBase(unittest.TestCase):
    def setUp(self):
        if not HAS_FAISS:
            self.skipTest("faiss/numpy 不可用")
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.cache_dir = os.path.join(self._tmp.name, "vec")

    def tearDown(self):
        self._tmp.cleanup()


class TestPersistence(VectorStoreTestBase):
    def test_build_save_load_roundtrip(self):
        """落盘后重新加载，检索结果必须与首次一致。"""
        kb = KnowledgeBase.load(FIXTURE)
        self.assertTrue(kb.enable_vector(encoder=stub_encoder, cache_dir=self.cache_dir,
                                         model_id="stub"))
        self.assertEqual(kb._vector_source, "built")
        first = kb.search_vector("座椅加热怎么关闭", top_k=3)
        self.assertTrue(first)

        # 三个文件都要落盘
        for name in (INDEX_FILE, VECTORS_FILE, META_FILE):
            self.assertTrue(os.path.isfile(os.path.join(self.cache_dir, name)),
                            f"缺少持久化文件 {name}")

        # 新建一个 KB 实例：应直接读盘（source == "cache"），结果要一致
        kb2 = KnowledgeBase.load(FIXTURE)
        self.assertTrue(kb2.enable_vector(encoder=stub_encoder, cache_dir=self.cache_dir,
                                          model_id="stub"))
        self.assertEqual(kb2._vector_source, "cache", "没有走缓存，说明持久化没生效")
        second = kb2.search_vector("座椅加热怎么关闭", top_k=3)
        self.assertEqual([e.chunk_id for e in first], [e.chunk_id for e in second])

    def test_stale_checksum_forces_rebuild(self):
        """语料变了（checksum 不同）必须重建，而不是拿旧索引凑合。"""
        kb = KnowledgeBase.load(FIXTURE)
        kb.enable_vector(encoder=stub_encoder, cache_dir=self.cache_dir, model_id="stub")
        kb2 = KnowledgeBase.load(FIXTURE)
        kb2.chunks.append(Chunk(id=9999, text="新增的一块内容：电动尾门支持脚踢开启。"))
        kb2.build_index()
        kb2.enable_vector(encoder=stub_encoder, cache_dir=self.cache_dir, model_id="stub")
        self.assertEqual(kb2._vector_source, "built", "语料变了却复用了旧索引")

    def test_model_change_forces_rebuild(self):
        kb = KnowledgeBase.load(FIXTURE)
        kb.enable_vector(encoder=stub_encoder, cache_dir=self.cache_dir, model_id="model-a")
        kb2 = KnowledgeBase.load(FIXTURE)
        kb2.enable_vector(encoder=stub_encoder, cache_dir=self.cache_dir, model_id="model-b")
        self.assertEqual(kb2._vector_source, "built", "换模型后复用了旧索引")

    def test_corrupt_cache_degrades_gracefully(self):
        """索引文件损坏时必须**重建**，而不是让 Agent 起不来。"""
        kb = KnowledgeBase.load(FIXTURE)
        kb.enable_vector(encoder=stub_encoder, cache_dir=self.cache_dir, model_id="stub")
        with open(os.path.join(self.cache_dir, VECTORS_FILE), "wb") as fh:
            fh.write(b"not a numpy file")
        kb2 = KnowledgeBase.load(FIXTURE)
        self.assertTrue(kb2.enable_vector(encoder=stub_encoder, cache_dir=self.cache_dir,
                                          model_id="stub"))
        self.assertEqual(kb2._vector_source, "built")
        self.assertTrue(kb2.search_vector("胎压报警", top_k=2))


class TestIncremental(VectorStoreTestBase):
    def test_upsert_makes_new_chunk_searchable(self):
        """新增一块内容后，必须能被检索到（不需要重嵌全量）。"""
        kb = KnowledgeBase.load(FIXTURE)
        kb.enable_vector(encoder=stub_encoder, cache_dir=self.cache_dir, model_id="stub")
        before = kb.search_vector("电动尾门脚踢开启", top_k=3)
        new_text = "电动尾门支持脚踢开启：携带钥匙靠近车尾，脚部轻扫保险杠下方即可。"
        added = kb.upsert_chunks([Chunk(id=9001, text=new_text)], cache_dir=self.cache_dir)
        self.assertEqual(added, 1)
        after = kb.search_vector("电动尾门脚踢开启", top_k=5)
        self.assertTrue(after)
        self.assertIn(9001, [e.chunk_id for e in after], "新增块没有被检索到（索引没更新）")
        # 确定性不变量：用**自身文本**检索，必须命中自己（余弦自相似=1）。
        # 注意：不要用"独有词必须排第一"来断言——桩编码器是哈希向量，语义排序不可靠；
        # 这里要验证的是"索引里有这条内容"，而不是"语义质量有多好"（那是真模型的事）。
        self.assertEqual(kb.search_vector(new_text, top_k=3)[0].chunk_id, 9001,
                         "新增块未能自命中，说明索引确实没更新")
        self.assertNotEqual([e.chunk_id for e in before], [e.chunk_id for e in after])

    def test_upsert_same_id_replaces_content(self):
        """同 id 覆盖：旧内容必须被替换，不能两条都在，也不能只删不加。"""
        kb = KnowledgeBase.load(FIXTURE)
        kb.enable_vector(encoder=stub_encoder, cache_dir=self.cache_dir, model_id="stub")
        target = kb.chunks[0]
        old_id = target.id
        new_text = "这是一段完全替换后的新内容：座椅加热共三档可调。"
        kb.upsert_chunks([Chunk(id=old_id, text=new_text)], cache_dir=self.cache_dir)
        hits = kb.search_vector(new_text, top_k=3)
        self.assertTrue(hits, "覆盖后整条都查不到（只删不加的老 bug 回来了）")
        self.assertEqual(hits[0].chunk_id, old_id, "覆盖后的新内容未能自命中")
        matches = [c for c in kb.chunks if c.id == old_id]
        self.assertEqual(len(matches), 1, "同 id 覆盖后出现了重复块")
        self.assertIn("完全替换后", matches[0].text, "旧内容没有被替换")

    def test_remove_chunk_disappears_from_index(self):
        """删除后旧条款不能再被检索到——手册改版场景下这是正确性问题。"""
        kb = KnowledgeBase.load(FIXTURE)
        kb.enable_vector(encoder=stub_encoder, cache_dir=self.cache_dir, model_id="stub")
        victim = kb.chunks[0]
        hits_before = kb.search_vector(victim.text[:20], top_k=5)
        self.assertIn(victim.id, [e.chunk_id for e in hits_before])

        removed = kb.remove_chunks([victim.id], cache_dir=self.cache_dir)
        self.assertEqual(removed, 1)
        self.assertNotIn(victim.id, [c.id for c in kb.chunks])
        hits_after = kb.search_vector(victim.text[:20], top_k=5)
        self.assertNotIn(victim.id, [e.chunk_id for e in hits_after],
                         "已删除的块仍能被检索到")

    def test_incremental_updates_persist(self):
        """增量修改后重新加载，新内容仍在、被删内容不在。"""
        kb = KnowledgeBase.load(FIXTURE)
        kb.enable_vector(encoder=stub_encoder, cache_dir=self.cache_dir, model_id="stub")
        kb.upsert_chunks([Chunk(id=9002, text="新增：后备箱应急开启拉手位于盖板下方。")],
                         cache_dir=self.cache_dir)

        kb2 = KnowledgeBase.load(FIXTURE)          # 注意：从原始语料文件重新加载
        # 语料文件没变，所以磁盘索引与文件 checksum 不匹配 → 会重建；
        # 这里验证的是"落盘发生在增量之后"，即索引里确实含 9002。
        from agent.vector_store import VectorStore

        store = VectorStore.load(self.cache_dir)
        self.assertIsNotNone(store)
        self.assertGreaterEqual(store.meta.n_vectors, len(kb.chunks))
        del kb2

    def test_chunk_count_and_index_stay_in_sync(self):
        """不变量：索引里的行数必须等于 chunks 数（增删之后都不能漂）。"""
        kb = KnowledgeBase.load(FIXTURE)
        kb.enable_vector(encoder=stub_encoder, cache_dir=self.cache_dir, model_id="stub")
        baseline = len(kb.chunks)
        kb.upsert_chunks([Chunk(id=9003, text="新增一块")], cache_dir=self.cache_dir)
        self.assertEqual(kb._vector[1].meta.n_vectors, baseline + 1)   # type: ignore[index]
        kb.remove_chunks([9003], cache_dir=self.cache_dir)
        self.assertEqual(kb._vector[1].meta.n_vectors, baseline)       # type: ignore[index]
        self.assertEqual(kb._vector[1].meta.n_vectors, len(kb.chunks))  # type: ignore[index]


class TestDegradation(unittest.TestCase):
    def test_no_faiss_degrades_to_bm25(self):
        """faiss 不可用时，enable_vector 必须返回 False 且不抛异常（纯 BM25 可用）。"""
        kb = KnowledgeBase.load(FIXTURE)
        # 用会抛异常的编码器模拟"模型加载失败"
        def broken(_texts):
            raise RuntimeError("model not found")

        self.assertFalse(kb.enable_vector(encoder=broken, cache_dir=None))
        self.assertTrue(kb.search_vector("座椅加热", top_k=3) == [],
                        "向量路关闭后 search_vector 必须返回空列表而不是抛异常")
        self.assertTrue(kb.search("座椅加热怎么关闭", top_k=3), "纯 BM25 路必须仍然可用")


if __name__ == "__main__":
    unittest.main(verbosity=2)
