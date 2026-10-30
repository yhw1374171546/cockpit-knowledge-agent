# -*- coding: utf-8 -*-
"""retrieve 测试：主题路由 / 词面兜底 / 时间衰减 / Top-k / 红线常注入 / 嵌入路径。"""

from __future__ import annotations

from agent.profile.consolidate import MemoryCandidate, consolidate
from agent.profile.retrieve import LocalEmbedder, cosine, overlap_score, retrieve
from agent.profile.store import ProfileStore


def _seed(store: ProfileStore, now: float) -> None:
    for cand in [
        MemoryCandidate("usual_center", "领克中心(徐汇店)", "user_explicit", 0.85),
        MemoryCandidate("city", "上海", "user_explicit", 0.9),
        MemoryCandidate("vehicle_model", "领克08", "user_explicit", 0.95),
        MemoryCandidate("pref:座椅加热", "关闭", "user_explicit", 0.9),
        MemoryCandidate("do_not", "不要关闭车身稳定系统", "user_explicit", 1.0),
    ]:
        consolidate(cand, store, user_id="u1", now=now)


def test_topic_routing_keeps_unrelated_out():
    with ProfileStore(":memory:") as store:
        _seed(store, 1000.0)
        ctx = retrieve("帮我约个保养", store, user_id="u1", now=1000.0)
        keys = {i.key for i in ctx.items if i.kind == "semantic"}
        assert "usual_center" in keys
        assert "pref:座椅加热" not in keys          # 主题不相关 → 不注入


def test_constraint_always_injected():
    with ProfileStore(":memory:") as store:
        _seed(store, 1000.0)
        ctx = retrieve("今天天气怎么样", store, user_id="u1", now=1000.0)
        reds = [i for i in ctx.items if i.kind == "constraint"]
        assert len(reds) == 1 and reds[0].text == "不要关闭车身稳定系统"


def test_ttl_expired_fact_not_recalled():
    with ProfileStore(":memory:") as store:
        consolidate(MemoryCandidate("usual_center", "徐汇店", "user_explicit", 0.85),
                    store, user_id="u1", now=1000.0)
        ctx = retrieve("帮我约保养", store, user_id="u1", now=1000.0 + 91 * 86400.0)
        assert ctx.items == []                       # 90 天 TTL 已过


def test_confidence_decay_drops_low_confidence():
    with ProfileStore(":memory:") as store:
        consolidate(MemoryCandidate("usual_center", "徐汇店", "user_explicit", 0.85),
                    store, user_id="u1", now=1000.0)
        ctx = retrieve("帮我约保养", store, user_id="u1", now=1000.0 + 30 * 86400.0)
        # 0.85 × 0.98^30 ≈ 0.464 > 0.25 → 仍召回（衰减但未清退）
        assert any(i.key == "usual_center" for i in ctx.items)


def test_top_k_caps_injected_items():
    with ProfileStore(":memory:") as store:
        _seed(store, 1000.0)
        ctx = retrieve("帮我约保养到店服务", store, user_id="u1", now=1000.0, k=2)
        semantic = [i for i in ctx.items if i.kind == "semantic"]
        assert len(semantic) == 2


def test_overlap_fallback_matches_v1_semantics():
    # v1 口径：查询侧覆盖率 + bigram；"工单"查询能命中含"工单"的事实
    assert overlap_score("上次那个工单怎么样了", "已建工单 SO123：更换机油机滤") > 0.0
    assert overlap_score("今天天气怎么样", "已建工单 SO123：更换机油机滤") == 0.0


def test_cosine_pure_python():
    assert abs(cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_embedder_absent_degrades_to_overlap():
    with ProfileStore(":memory:") as store:
        _seed(store, 1000.0)
        embedder = LocalEmbedder(model_dir="/nonexistent/path")
        assert not embedder.available                  # 环境缺失 → 词面兜底
        ctx = retrieve("帮我约个保养", store, user_id="u1", now=1000.0, embedder=embedder)
        assert any(i.key == "usual_center" for i in ctx.items)
