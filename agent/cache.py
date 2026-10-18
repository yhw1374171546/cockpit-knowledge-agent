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
import json
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


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

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "question": self.question, "vector": self.vector,
                "answer": self.answer, "citations": self.citations, "status": self.status,
                "fingerprint": self.fingerprint, "created_at": self.created_at,
                "hits": self.hits}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        return cls(key=str(data.get("key") or ""), question=str(data.get("question") or ""),
                   vector=list(data.get("vector") or []), answer=str(data.get("answer") or ""),
                   citations=list(data.get("citations") or []),
                   status=str(data.get("status") or "answered"),
                   fingerprint=str(data.get("fingerprint") or ""),
                   created_at=float(data.get("created_at") or 0.0),
                   hits=int(data.get("hits") or 0))


@dataclass
class CacheLookup:
    hit: bool = False
    kind: str = ""            # exact | fuzzy | miss | bypass
    similarity: float = 0.0
    blocked_by: str = ""      # 反义守卫拦截说明（可观测）
    entry: Optional[CacheEntry] = None


class SemanticCache:
    """语义缓存。

    `kv` 为 None 时是**进程内**实现（原来的行为，单实例/CI 用）；
    传入 KVStore 时切换到**共享模式**：条目与候选索引都落在 KV 里，
    多副本（多 worker / 多容器）能互相命中，且进程重启不丢。

    共享模式的两级设计（刻意不引入向量库）：
    - **精确命中**：1 次 KV 读（键 = 归一化问题 + 指纹 + 作用域）；
    - **近似命中**：1 次 KV 读拿到「该作用域下最近的候选列表」（有上限），
      在内存里算相似度 + 反义守卫，命中后再 1 次 KV 读取答案。
      最多 2 次 KV 读，且候选列表有硬上限，不会随缓存增长而变慢。
    """

    #: 共享模式下每个（作用域, 指纹）保留的候选数量上限
    INDEX_CAP = 64

    def __init__(self, threshold: float = 0.70, max_entries: int = 512, ttl_seconds: float = 3600,
                 encoder: Optional[Callable[[str], List[str]]] = None,
                 kv: Optional[Any] = None, namespace: str = "semcache"):
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.encoder = encoder or bigrams
        self.kv = kv
        self.namespace = namespace
        self._exact: Dict[str, str] = {}                 # 归一化问题 → 缓存键
        self._entries: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._shared_size = 0                            # 共享模式的写入计数（用于 report）
        self.stats = {"lookups": 0, "exact_hits": 0, "fuzzy_hits": 0, "misses": 0,
                      "bypassed": 0, "evictions": 0, "expired": 0, "antonym_blocked": 0,
                      "puts": 0}

    @property
    def shared(self) -> bool:
        return self.kv is not None

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
    def make_key(question: str, fingerprint: str, scope: str = "") -> str:
        """缓存键 = 归一化问题 + 车况指纹 + 作用域（租户/车型）。

        `scope` 是防**跨租户串答案**用的：不同车型租户即使问了同样的问题，
        也必须各自独立缓存（手册版本、政策口径都可能不同）。默认空串保持向后兼容。
        """
        return hashlib.md5(f"{normalize(question)}|{fingerprint}|{scope}".encode("utf-8")).hexdigest()

    # ── 读写 ──
    def get(self, question: str, fingerprint: str = "",
            cacheable: bool = True, scope: str = "") -> CacheLookup:
        self.stats["lookups"] += 1
        if not cacheable:
            self.stats["bypassed"] += 1
            return CacheLookup(hit=False, kind="bypass")
        if self.shared:
            return self._get_shared(question, fingerprint, scope)

        key = self.make_key(question, fingerprint, scope)
        if key in self._entries:
            entry = self._touch(key)
            if entry is not None:
                self.stats["exact_hits"] += 1
                entry.hits += 1
                # 必须用关键字：第 4 个位置参数是 blocked_by，不是 entry。
                # （这里原来写成位置参数，导致精确命中的 lookup.entry 永远是 None，
                #   调用方只能拿到"命中了但没答案"，于是又白跑一遍 Agent。）
                return CacheLookup(True, "exact", 1.0, entry=entry)
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

    # ── 共享模式（KV 后端）──
    def _entry_key(self, key: str) -> str:
        return f"{self.namespace}:e:{key}"

    def _index_key(self, fingerprint: str, scope: str) -> str:
        return f"{self.namespace}:i:{scope}|{fingerprint}"

    def _kv_ttl(self) -> Optional[int]:
        return int(self.ttl_seconds) if self.ttl_seconds and self.ttl_seconds > 0 else None

    def _read_json(self, key: str) -> Any:
        try:
            raw = self.kv.get(key)                      # type: ignore[union-attr]
        except Exception as exc:                        # noqa: BLE001  KV 故障时降级为 miss
            self._note_error("get", exc)
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError) as exc:
            self._note_error("decode", exc)
            return None

    def _note_error(self, stage: str, exc: BaseException) -> None:
        """缓存故障不抛给调用方，但**必须留痕**——否则"没命中"和"坏了"分不清。

        这条是被一次真实事故教育的：写路径的异常被 `except: pass` 吃掉，
        表现是"缓存永远不命中"，而指标上看起来一切正常。
        """
        self.stats["errors"] = self.stats.get("errors", 0) + 1
        self.last_error = f"{stage}: {type(exc).__name__}: {exc}"

    def _get_shared(self, question: str, fingerprint: str, scope: str) -> CacheLookup:
        key = self.make_key(question, fingerprint, scope)
        entry = self._read_entry(key)
        if entry is not None:
            if self.ttl_seconds and (time.time() - entry.created_at) > self.ttl_seconds:
                self.stats["expired"] += 1
                try:
                    self.kv.delete(self._entry_key(key))     # type: ignore[union-attr]
                except Exception:                            # noqa: BLE001
                    pass
            else:
                self.stats["exact_hits"] += 1
                return CacheLookup(True, "exact", 1.0, entry=entry)

        vector = self.encoder(question)
        best_key, best_sim, blocked = "", 0.0, ""
        for record in self._read_index(fingerprint, scope):
            if not isinstance(record, dict) or record.get("k") == key:
                continue
            other_q = str(record.get("q") or "")
            sim = max(similarity(question, other_q), cosine(vector, record.get("v") or []))
            conflict = antonym_conflict(question, other_q)
            if conflict:
                blocked = blocked or conflict
                continue
            if sim > best_sim:
                best_key, best_sim = str(record.get("k") or ""), sim
        if best_key and best_sim >= self.threshold:
            hit = self._read_entry(best_key)
            if hit is not None and not (
                    self.ttl_seconds and (time.time() - hit.created_at) > self.ttl_seconds):
                self.stats["fuzzy_hits"] += 1
                return CacheLookup(True, "fuzzy", round(best_sim, 4), entry=hit)

        if blocked:
            self.stats["antonym_blocked"] += 1
        self.stats["misses"] += 1
        return CacheLookup(hit=False, kind="miss", similarity=round(best_sim, 4),
                           blocked_by=blocked)

    def _read_entry(self, key: str) -> Optional[CacheEntry]:
        data = self._read_json(self._entry_key(key))
        if not isinstance(data, dict):
            return None
        return CacheEntry.from_dict(data)

    def _read_index(self, fingerprint: str, scope: str) -> List[Dict[str, Any]]:
        data = self._read_json(self._index_key(fingerprint, scope))
        return data if isinstance(data, list) else []

    def _write_shared(self, entry: CacheEntry, fingerprint: str, scope: str) -> None:
        kv = self.kv
        try:
            kv.set(self._entry_key(entry.key),                          # type: ignore[union-attr]
                   json.dumps(entry.to_dict(), ensure_ascii=False), ttl=self._kv_ttl())
            index = self._read_index(fingerprint, scope)
            index = [rec for rec in index
                     if isinstance(rec, dict) and rec.get("k") != entry.key]
            index.append({"k": entry.key, "q": entry.question, "v": entry.vector,
                          "t": entry.created_at})
            # 候选列表有硬上限：超了丢最旧的（近似的 LRU，够用且不会无限增长）
            index = index[-self.INDEX_CAP:]
            kv.set(self._index_key(fingerprint, scope),                 # type: ignore[union-attr]
                   json.dumps(index, ensure_ascii=False), ttl=self._kv_ttl())
            self._shared_size += 1
            self.stats["puts"] += 1
        except Exception as exc:                                        # noqa: BLE001
            # 缓存是"加速器"不是"正确性依赖"：写失败只丢性能，绝不能让请求失败。
            # 但异常必须留痕（见 _note_error 的注释）。
            self._note_error("set", exc)

    def put(self, question: str, answer: str, citations: Optional[List[str]] = None,
            status: str = "answered", fingerprint: str = "",
            cacheable: bool = True, scope: str = "") -> Optional[str]:
        if not cacheable or not (answer or "").strip():
            return None
        key = self.make_key(question, fingerprint, scope)
        entry = CacheEntry(key=key, question=question, vector=self.encoder(question),
                           answer=answer, citations=list(citations or []), status=status,
                           fingerprint=fingerprint)
        if self.shared:
            self._write_shared(entry, fingerprint, scope)
            return key
        self._entries[key] = entry
        self._entries.move_to_end(key)
        self.stats["puts"] += 1
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
        size = self._shared_size if self.shared else len(self._entries)
        return {**self.stats, "hit_rate": self.hit_rate, "size": size,
                "threshold": self.threshold, "shared": self.shared}

    def clear(self) -> None:
        """清空缓存。共享模式下只能清本地统计——KV 里的条目靠 TTL 自然过期。"""
        self._entries.clear()
        self._exact.clear()
        self._shared_size = 0


# 只有"正常作答"的答案可以进缓存。
# 拒答（refused）、待确认（needs_confirmation）、出错（error/blocked）、
# 步数耗尽（max_steps）都不缓存——把它们缓存起来等于把一次失败固化成长期答案。
# 同理，降级产物（degraded）也**不能**缓存：那是故障期间的兜底内容，
# 缓存它会把一次故障的影响延长到 TTL 结束。
#
# `limited`（可回答性校验后的**限定回答**，如"手册没说明本车配备冰箱 + 相关原文"）
# 可以缓存：它是**确定性的、带出处的、由知识库内容决定的**结论，
# 和 answered 一样稳定；缓存它能省下重复调用，且不会把错误固化（它本身就是谨慎表述）。
CACHEABLE_STATUSES = ("answered", "limited")
CACHEABLE_STATUS = CACHEABLE_STATUSES[0]        # 兼容旧名字


def cacheable_status(status: str) -> bool:
    """状态是否允许进缓存。"""
    return (status or "").strip() in CACHEABLE_STATUSES


WHITELIST_CACHEABLE_INTENTS = ("manual_qa", "spec", "status", "maintenance")
