# -*- coding: utf-8 -*-
"""语义缓存：高频问题直接命中，显著降低首字延迟与大模型调用量。

座舱场景里重复率极高（"怎么开空调""胎压多少"），缓存的价值远超通用问答。

设计要点
--------
1. **两级索引**：精确键（归一化后的原文）→ O(1) 命中；字符 n-gram 余弦 → 近似命中；
2. **零依赖**：不依赖向量模型也能工作（字符 bigram + 余弦），有 m3e 时可替换编码器；
3. **安全隔离**：缓存键包含「车况指纹 + 用户角色」，车况变了（例如胎压从 228 变成 148）
   必须 miss——否则会拿旧车况下的答案回复新状态，这在座舱里是**事故级错误**；
4. **LRU + TTL**：控制内存与陈旧度。
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


def normalize(text: str) -> str:
    text = (text or "").strip().lower()
    return re.sub(r"[\s，。！？、,.!?~]+", "", text)


def bigrams(text: str) -> List[str]:
    text = normalize(text)
    if len(text) < 2:
        return [text] if text else []
    return [text[i:i + 2] for i in range(len(text) - 1)]


def cosine(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    common = set(ca) & set(cb)
    num = sum(ca[t] * cb[t] for t in common)
    den = math.sqrt(sum(v * v for v in ca.values())) * math.sqrt(sum(v * v for v in cb.values()))
    return num / den if den else 0.0


def similarity(a: str, b: str) -> float:
    """归一化编辑距离相似度（字符级）：对"同义改写"敏感，对"反义替换"也会敏感，
    因此必须叠加 ANTONYM 守卫（见 antonym_conflict）。"""
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


# 反义对：座舱里"怎么打开"和"怎么关闭"的答案完全相反，
# 若把它们混为一次缓存命中，就是事故级错误。字符级相似度无法区分，必须显式守卫。
ANTONYM_PAIRS = (
    ("打开", "关闭"), ("开启", "关闭"), ("启动", "关闭"), ("开", "关"),
    ("加", "减"), ("上", "下"), ("高", "低"), ("热", "冷"), ("增", "减"),
)


def antonym_conflict(a: str, b: str) -> Optional[str]:
    """判断两个问法是否"只有反义项不同"（一个说打开、另一个说关闭）。"""
    a, b = normalize(a), normalize(b)
    for left, right in ANTONYM_PAIRS:
        a_left, a_right = left in a, right in a
        b_left, b_right = left in b, right in b
        a_only_left = a_left and not a_right
        a_only_right = a_right and not a_left
        b_only_left = b_left and not b_right
        b_only_right = b_right and not b_left
        if (a_only_left and b_only_right) or (a_only_right and b_only_left):
            return f"{left}/{right}"
    return None


@dataclass
class CacheEntry:
    key: str
    question: str
    vector: List[str]
    answer: str
    citations: List[str] = field(default_factory=list)
    status: str = "answered"
    fingerprint: str = ""
    created_at: float = field(default_factory=time.time)
    hits: int = 0


@dataclass
class CacheLookup:
    hit: bool = False
    kind: str = ""            # exact | fuzzy | miss | bypass
    similarity: float = 0.0
    blocked_by: str = ""      # 反义守卫拦截说明（可观测）
    entry: Optional[CacheEntry] = None


class SemanticCache:
    def __init__(self, threshold: float = 0.70, max_entries: int = 512, ttl_seconds: float = 3600,
                 encoder: Optional[Callable[[str], List[str]]] = None):
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.encoder = encoder or bigrams
        self._exact: Dict[str, str] = {}                 # 归一化问题 → 缓存键
        self._entries: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self.stats = {"lookups": 0, "exact_hits": 0, "fuzzy_hits": 0, "misses": 0,
                      "bypassed": 0, "evictions": 0, "expired": 0, "antonym_blocked": 0}

    # ── 缓存键 ──
    @staticmethod
    def fingerprint(telemetry: Optional[Dict[str, Any]] = None,
                    role: str = "driver") -> str:
        """车况指纹：把"会改变答案正确性"的状态纳入键，防止拿旧状态答新问题。"""
        telem = telemetry or {}
        parts = [
            role,
            str(telem.get("车型", "")),
            str(telem.get("告警灯", [])),
            str(telem.get("胎压告警", [])),
            ",".join(f"{k}{v}" for k, v in sorted((telem.get("胎压_kPa") or {}).items())),
            str(telem.get("剩余电量_%", "")),
            str(telem.get("里程_km", "")),
        ]
        return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def make_key(question: str, fingerprint: str) -> str:
        return hashlib.md5(f"{normalize(question)}|{fingerprint}".encode("utf-8")).hexdigest()

    # ── 读写 ──
    def get(self, question: str, fingerprint: str = "",
            cacheable: bool = True) -> CacheLookup:
        self.stats["lookups"] += 1
        if not cacheable:
            self.stats["bypassed"] += 1
            return CacheLookup(hit=False, kind="bypass")

        key = self.make_key(question, fingerprint)
        if key in self._entries:
            entry = self._touch(key)
            if entry is not None:
                self.stats["exact_hits"] += 1
                entry.hits += 1
                return CacheLookup(True, "exact", 1.0, entry)
            self.stats["expired"] += 1

        vector = self.encoder(question)
        best, best_sim, blocked = None, 0.0, ""
        for candidate in self._entries.values():
            if candidate.fingerprint != fingerprint:
                continue                      # 车况不同 → 不允许命中
            sim = max(similarity(question, candidate.question), cosine(vector, candidate.vector))
            # 反义守卫：问法相似但语义相反（打开 vs 关闭）一律不命中
            conflict = antonym_conflict(question, candidate.question)
            if conflict:
                blocked = blocked or conflict
                continue
            if sim > best_sim:
                best, best_sim = candidate, sim
        if best is not None and best_sim >= self.threshold:
            fresh = self._touch(best.key)
            if fresh is not None:
                self.stats["fuzzy_hits"] += 1
                fresh.hits += 1
                return CacheLookup(True, "fuzzy", round(best_sim, 4), entry=fresh)

        if blocked:
            self.stats["antonym_blocked"] += 1
        self.stats["misses"] += 1
        return CacheLookup(hit=False, kind="miss", similarity=round(best_sim, 4),
                           blocked_by=blocked)

    def put(self, question: str, answer: str, citations: Optional[List[str]] = None,
            status: str = "answered", fingerprint: str = "",
            cacheable: bool = True) -> Optional[str]:
        if not cacheable or not (answer or "").strip():
            return None
        key = self.make_key(question, fingerprint)
        entry = CacheEntry(key=key, question=question, vector=self.encoder(question),
                           answer=answer, citations=list(citations or []), status=status,
                           fingerprint=fingerprint)
        self._entries[key] = entry
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.stats["evictions"] += 1
        return key

    def _touch(self, key: str) -> Optional[CacheEntry]:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if self.ttl_seconds and (time.time() - entry.created_at) > self.ttl_seconds:
            self._entries.pop(key, None)
            return None
        self._entries.move_to_end(key)
        return entry

    # ── 统计 ──
    @property
    def hit_rate(self) -> float:
        total = self.stats["exact_hits"] + self.stats["fuzzy_hits"] + self.stats["misses"]
        return round((self.stats["exact_hits"] + self.stats["fuzzy_hits"]) / total, 4) if total else 0.0

    def report(self) -> Dict[str, Any]:
        return {**self.stats, "hit_rate": self.hit_rate, "size": len(self._entries),
                "threshold": self.threshold}

    def clear(self) -> None:
        self._entries.clear()


WHITELIST_CACHEABLE_INTENTS = ("manual_qa", "spec", "status", "maintenance")
