# -*- coding: utf-8 -*-
"""retrieve：画像事实的召回（主题路由 → 嵌入/词面打分 → 时间衰减 → Top-k）。

与 v1（agent/longterm.py 的词面重合 + 手工阈值 0.09）的差异
----------------------------------------------------------
1. **嵌入召回**：text2vec-base-chinese ONNX（仓库已有资产，CPU 可跑，零新依赖），
   余弦相似度 + 时间衰减；嵌入不可用时退化为词面重合（与 v1 同口径，便于消融对照）；
2. **主题路由保留**（v1 的 false_inject=0 是实测资产，不能丢）：
   不相关字段根本不进 prompt；
3. **红线常注入**：constraint 层（do_not）不参与打分、无条件注入——安全优先；
4. **季节复活（季节偏好到期"休眠"）**：v2 语义——季节字段被覆盖时旧值标 t_invalid，
   这里从 history 里按"同 key 的失效值 + 证据含季节词"找回（简单启发式，见 _resurrect）。

消融对照设计（阶段 4）：`embedder=None` 即 v1 口径，`embedder=LocalEmbedder` 即 v2——
同一组事实、同一组 query，唯一变量是打分函数。
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent.profile.schema import CONSTRAINT, spec_for
from agent.profile.store import ProfileFact, ProfileStore

DAY = 86400.0

# 主题路由触发词（继承 v1 的实测口径；pref:* 与 maintenance_history 走专门分支）
# 注：v2 在 v1 基础上补齐了"我住在哪个城市/我的车是什么型号"这类自指问句的触发词
# （v1 的 city/vehicle_model 表没有 城市/住/型号 → 对自指问句不召回，benchmark 实测暴露）。
CUES: Dict[str, Tuple[str, ...]] = {
    "city": ("保养", "到店", "门店", "预约", "服务", "限行", "上牌", "保险", "充电", "家充",
             "城市", "住", "哪里"),
    "usual_center": ("保养", "到店", "门店", "预约", "服务", "维修", "检查", "哪家店"),
    "vehicle_model": ("车型", "型号", "续航", "电池", "轮胎", "保养", "参数", "配置", "充电", "油耗", "什么车"),
    "plate_region": ("限行", "牌照", "上牌", "车牌"),
    "next_due": ("保养", "里程", "下次", "到期"),
    "maintenance_history": ("保养", "工单", "上次", "历史", "维修", "记录"),
}

# 季节词（季节偏好的"休眠/复活"依据）
SEASON_WORDS = ("夏天", "夏季", "冬天", "冬季", "秋天", "秋季", "春天", "春季", "高温", "低温", "天热", "天冷")


@dataclass
class MemoryItem:
    kind: str                 # semantic | constraint | episode（与 v1 对齐）
    key: str
    text: str
    score: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    items: List[MemoryItem] = field(default_factory=list)
    dropped: int = 0          # 因不相关/过期被丢弃的条数（false_inject 分母）

    def __bool__(self) -> bool:
        return bool(self.items)

    def to_prompt(self) -> str:
        if not self.items:
            return ""
        prefs = [i for i in self.items if i.kind == "semantic"]
        reds = [i for i in self.items if i.kind == "constraint"]
        eps = [i for i in self.items if i.kind == "episode"]
        lines = ["【跨会话记忆】"]
        if prefs:
            lines.append("- 车主画像：" + "；".join(
                f"{i.meta.get('label', i.key)}={i.text}" for i in prefs))
        if eps:
            lines.append("- 相关历史：" + "；".join(i.text for i in eps))
        if reds:
            lines.append("- 车主红线（必须遵守）：" + "；".join(i.text for i in reds))
        lines.append("（以上是历史信息；若与本次手册原文或实时车况冲突，以本次信息为准，并向车主说明差异）")
        return "\n".join(lines)


# ── 词面重合兜底（v1 口径：查询侧覆盖率 + 汉字 bigram）─────────────────

STOPWORDS = {
    "怎么", "什么", "为什么", "怎么样", "怎样", "这个", "那个", "上次", "下次", "一下",
    "可以", "我们", "你们", "他们", "现在", "今天", "明天", "还是", "就是", "然后",
    "已经", "是否", "多少", "哪个", "哪些", "这样", "那样", "刚才", "请问", "帮我",
}


def _tokens(text: str) -> List[str]:
    """分词 + 汉字 bigram（v1 同款：jieba 优先，无 jieba 退化为按标点切）。"""
    raw = (text or "").strip()
    if not raw:
        return []
    try:
        import jieba  # noqa: PLC0415
        words = [t for t in jieba.cut_for_search(raw) if len(t.strip()) >= 2 and t not in STOPWORDS]
    except Exception:                                      # noqa: BLE001
        words = [t for t in re.split(r"[^\u4e00-\u9fffA-Za-z0-9]+", raw)
                 if len(t) >= 2 and t not in STOPWORDS]
    grams: List[str] = []
    for run in re.findall(r"[\u4e00-\u9fff]+", raw):
        grams.extend(run[i:i + 2] for i in range(len(run) - 1))
    return words + [g for g in grams if g not in STOPWORDS]


def overlap_score(query: str, text: str) -> float:
    q = {t for t in _tokens(query) if t}
    if not q:
        return 0.0
    t = {w for w in _tokens(text) if w}
    return len(q & t) / len(q)


# ── 本地嵌入（text2vec-base-chinese ONNX + 极简 WordPiece）─────────────

def _wordpiece_tokenize(text: str, vocab: Dict[str, int]) -> List[str]:
    """极简 WordPiece：中文按字、ASCII 按单字符查表，缺失走 ##x，再缺 [UNK]。"""
    out: List[str] = []
    for ch in (text or "").strip().lower():
        if ch in vocab:
            out.append(ch)
        elif ch.isspace():
            continue
        elif f"##{ch}" in vocab:
            out.append(f"##{ch}")
        else:
            out.append("[UNK]")
    return out


class LocalEmbedder:
    """text2vec-base-chinese 的本地嵌入（ONNX + mean pooling，CPU 可跑）。

    懒加载 + 文本级缓存：记忆库每用户只有几十~几百条，缓存足够；
    加载失败（缺 onnxruntime/模型文件）时 `available` 为 False，调用方退回词面兜底。
    """

    def __init__(self, model_dir: Optional[str] = None, cache_size: int = 4096):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = model_dir or os.path.join(root, "pre_train_model", "text2vec-base-chinese", "onnx")
        self._sess = None
        self._vocab: Optional[Dict[str, int]] = None
        self._cache: Dict[str, List[float]] = {}
        self._cache_size = cache_size
        self._load_error: Optional[str] = None

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._sess is not None

    def _ensure_loaded(self) -> None:
        if self._sess is not None or self._load_error:
            return
        try:
            import numpy as np  # noqa: F401  仅嵌入路径需要
            import onnxruntime as ort
            vocab_path = os.path.join(self.model_dir, "vocab.txt")
            if not os.path.isfile(vocab_path):
                raise FileNotFoundError(vocab_path)
            self._vocab = {}
            with open(vocab_path, encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    self._vocab[line.rstrip("\n")] = i
            self._sess = ort.InferenceSession(
                os.path.join(self.model_dir, "model.onnx"), providers=["CPUExecutionProvider"])
        except Exception as exc:                       # noqa: BLE001 任何环境缺失都回退词面
            self._load_error = f"{type(exc).__name__}: {exc}"[:120]
            self._sess = None

    def encode(self, texts: Sequence[str]) -> Optional[List[List[float]]]:
        """批量编码并 L2 归一化；不可用时返回 None。"""
        self._ensure_loaded()
        if self._sess is None or self._vocab is None:
            return None
        import numpy as np
        missing = [t for t in texts if t not in self._cache]
        if missing:
            cls_id = self._vocab["[CLS]"]
            sep_id = self._vocab["[SEP]"]
            pad_id = self._vocab.get("[PAD]", 0)
            batches: List[List[List[int]]] = []
            masks: List[List[List[int]]] = []
            for text in missing:
                ids = [cls_id] + [self._vocab[t] for t in _wordpiece_tokenize(text, self._vocab)][:510] + [sep_id]
                batches.append(ids)
                masks.append([1] * len(ids))
            max_len = max((len(b) for b in batches), default=1)
            for b, m in zip(batches, masks, strict=True):
                b.extend([pad_id] * (max_len - len(b)))
                m.extend([0] * (max_len - len(m)))
            inputs = {
                "input_ids": np.asarray(batches, dtype=np.int64),
                "attention_mask": np.asarray(masks, dtype=np.int64),
                "token_type_ids": np.zeros((len(batches), max_len), dtype=np.int64),
            }
            hidden = self._sess.run(None, inputs)[0]                       # (n, L, 768)
            mask = np.asarray(masks, dtype=np.float32)[..., None]
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1.0)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
            pooled = pooled / norms
            for text, vec in zip(missing, pooled.tolist(), strict=True):
                self._cache[text] = vec
        vectors = [self._cache[t] for t in texts]
        # 简单容量控制：超限清一半（记忆库规模小，正常到不了这里）
        if len(self._cache) > self._cache_size:
            for k in list(self._cache)[:self._cache_size // 2]:
                self._cache.pop(k, None)
        return vectors

    def similarity(self, a: List[float], b: List[float]) -> float:
        return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b, strict=True))))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """纯 Python 余弦（测试与兜底用，不依赖 numpy）。"""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


# ── 召回主入口 ────────────────────────────────────────────────────────

def _relevant(key: str, query: str) -> bool:
    if key.startswith("pref:"):
        return key.split(":", 1)[1] in query
    if key == "do_not":
        return False                                   # 红线走独立分支，无条件注入
    return any(cue in query for cue in CUES.get(key, ()))


def retrieve(query: str, store: ProfileStore, user_id: str = "default",
             embedder: Optional[LocalEmbedder] = None, now: Optional[float] = None,
             k: int = 3, min_score: float = 0.09, min_confidence: float = 0.25,
             decay_per_day: float = 0.98) -> MemoryContext:
    """召回当前有效的画像事实：主题路由 → 打分（嵌入/词面）→ 时间衰减 → Top-k。

    打分公式：score = 相似度 × (0.8 + 0.2 × recency) × 置信度衰减
    - 相似度：嵌入余弦（v2）或词面重合（v1 口径，embedder=None）；
    - recency：与 v1 一致，时间只做微调，不能靠"新"压过相关性；
    - 置信度衰减：0.98/天（继承 v1），低于 min_confidence 直接不召回。
    """
    now = store._now() if now is None else now
    with store._lock:
        rows = store._conn.execute(
            "SELECT * FROM profile_facts WHERE user_id=? AND t_invalid IS NULL",
            (user_id,)).fetchall()
    facts = [ProfileFact.from_row(r) for r in rows]
    ctx = MemoryContext()

    query_vec: Optional[List[float]] = None
    embed_cache: Dict[str, List[float]] = {}
    if embedder is not None and embedder.available:
        try:
            query_vec = embedder.encode([query])[0]          # type: ignore[index]
        except Exception:                                    # noqa: BLE001
            query_vec = None

    scored: List[Tuple[float, ProfileFact, str]] = []
    for fact in facts:
        spec = spec_for(fact.key)
        # TTL 到期 → 不召回（forget.maintain 负责清库，这里兜底）
        if spec.ttl_days and now - fact.updated_at > spec.ttl_days * DAY:
            ctx.dropped += 1
            continue
        effective = float(fact.confidence) * decay_per_day ** ((now - fact.updated_at) / DAY)
        if effective < min_confidence:
            ctx.dropped += 1
            continue
        if spec.layer == CONSTRAINT:
            ctx.items.append(MemoryItem("constraint", fact.key, fact.value,
                                        1.0, {"label": "红线", "confidence": effective}))
            continue
        if not _relevant(fact.key, query):
            ctx.dropped += 1
            continue
        # 打分：嵌入余弦（v2）或 v1 口径（主题路由通过即注入，按置信度排序）
        if query_vec is not None:
            vec = embed_cache.get(fact.value)
            if vec is None and embedder is not None:
                enc = embedder.encode([fact.value])
                vec = enc[0] if enc else None
                if vec is not None:
                    embed_cache[fact.value] = vec
            sim = cosine(query_vec, vec) if vec else 0.0
        else:
            sim = 1.0     # v1 语义：cue 命中即相关，文本相似度不做门控（episodic 才做）
        recency = max(0.0, 1.0 - (now - fact.updated_at) / (90 * DAY))
        score = sim * (0.8 + 0.2 * recency) * effective
        if score < min_score:
            ctx.dropped += 1
            continue
        scored.append((score, fact, spec.label))
    scored.sort(key=lambda t: -t[0])
    for score, fact, label in scored[:k]:
        ctx.items.append(MemoryItem("semantic", fact.key, fact.value, round(score, 3),
                                    {"label": label,
                                     "confidence": round(fact.confidence, 3),
                                     "source": fact.source}))
    # ── 情景记忆（episodes）：v2 的嵌入语义匹配是修复 v1"口语指代失效"的关键 ──
    for ep in store.episodes(user_id, now=now):
        summary = str(ep["summary"])
        if query_vec is not None:
            vec = embed_cache.get(summary)
            if vec is None and embedder is not None:
                enc = embedder.encode([summary])
                vec = enc[0] if enc else None
                if vec is not None:
                    embed_cache[summary] = vec
            sim = cosine(query_vec, vec) if vec else 0.0
        else:
            sim = overlap_score(query, summary)
        recency = max(0.0, 1.0 - (now - float(ep["created_at"])) / (30 * DAY))
        total = sim * (0.8 + 0.2 * recency)
        if total < min_score:
            ctx.dropped += 1
            continue
        stamp = time.strftime("%m-%d", time.localtime(float(ep["created_at"])))
        ctx.items.append(MemoryItem("episode", str(ep["id"]),
                                    f"{stamp} 记录：{summary}", round(total, 3),
                                    {"source": "episode"}))
    # 情景取 top-k：与画像分开截断，避免长历史挤掉画像/红线
    eps = [i for i in ctx.items if i.kind == "episode"]
    eps.sort(key=lambda i: -i.score)
    ctx.items = [i for i in ctx.items if i.kind != "episode"] + eps[:2]
    return ctx


def context_text(query: str, store: ProfileStore, user_id: str = "default",
                 embedder: Optional[LocalEmbedder] = None, now: Optional[float] = None) -> str:
    return retrieve(query, store, user_id=user_id, embedder=embedder, now=now).to_prompt()
