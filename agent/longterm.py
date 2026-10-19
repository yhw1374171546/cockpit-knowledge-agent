# -*- coding: utf-8 -*-
"""长期记忆：工作记忆（会话 6 轮）之外的两层——情景记忆与语义记忆。

为什么需要它
------------
`agent/memory.py` 只解决"这一通会话里别忘事"：每请求从 `messages` 表回填最近 6 轮，
换一个会话就全忘了。真实车主的抱怨通常是跨会话的——"上次你说的那家店""还是那个胎压问题"。

三层划分（也是 JD 里最常见的问法）
----------------------------------
| 层 | 内容 | 存放 | 本模块 |
| --- | --- | --- | --- |
| 工作记忆 | 最近 6 轮对话、车辆槽位 | `messages` 表 + `ConversationMemory` | 已有，不改 |
| 情景记忆 | 会话摘要（问过什么、结论是什么）、历史工单 | SQLite `episodic` 表 | 本模块 |
| 语义记忆 | 车主偏好与档案（常住城市、惯用门店、常用功能） | SQLite `semantic` 表 | 本模块 |

四个必须写死的策略（否则记忆就是"越攒越脏"）
--------------------------------------------
1. **什么该记**：只记用户**明示**的信息（"我住上海""以后座椅加热都关掉"）。
   推断（例如从提问里猜偏好）默认**不写**，除非显式 `allow_inferred=True`——
   记忆错了比没记更糟，因为它会被当事实反复使用。
2. **冲突覆盖**：同一 `(用户, key)` 覆盖写；`user_explicit` **不允许被 inferred 覆盖**，
   覆盖前把旧值写进 `events` 审计表（"上次记的是朝阳店，这次换成浦东店"要查得到）。
3. **遗忘**：情景记忆按 TTL（默认 30 天）过期；语义记忆按天做置信度衰减（0.98/天），
   低于 0.25 直接清掉；每用户情景记忆有容量上限（默认 200 条，超了淘汰最旧的）。
4. **脱敏**：写入前复用 `agent.guardrails.policy.mask_sensitive()` 遮掉 VIN / 手机号 / 身份证——
   记忆是**跨会话持久**的，一旦把 PII 落库，泄露面比单轮日志大得多。

召回为什么不用向量库
--------------------
本机无 GPU、无中文嵌入权重；而单用户的情景记忆只有几十到几百条，
**BM25 风格的词面重合 + 时间衰减**足够，且和仓库现有检索用同一套 jieba 分词。
真实上线时换向量库只是替换 `_score_episodic()`，接口不变（诚实说明，不假装用了向量）。
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agent.guardrails.policy import mask_sensitive

DAY = 86400.0

# 语义记忆的触发词：只有问题真的和这个主题相关才注入，避免"每一轮都把全部档案塞进去"。
# 这也是 `false_inject` 指标能被压到 0 的原因——不相关的记忆根本不进 prompt。
SEMANTIC_CUES: Dict[str, Tuple[str, ...]] = {
    "city": ("保养", "到店", "门店", "预约", "服务", "限行", "上牌", "保险", "充电", "家充"),
    "usual_center": ("保养", "到店", "门店", "预约", "服务", "维修", "检查"),
    "vehicle_model": ("车型", "续航", "电池", "轮胎", "保养", "参数", "配置", "充电", "油耗"),
}

CITY_WORDS = ("北京", "上海", "深圳", "广州", "杭州", "成都", "合肥")

# 功能词不进召回分词：否则"今天天气怎么样"和"问过蓝牙怎么连接"会因为都含"怎么"而互相召回
# （AB 第一版就是这么把负样本判成误注入的）。这批词在任何手册/记忆摘要里都不携带信息。
STOPWORDS = {
    "怎么", "什么", "为什么", "怎么样", "怎样", "这个", "那个", "上次", "下次", "一下",
    "可以", "我们", "你们", "他们", "现在", "今天", "明天", "还是", "就是", "然后",
    "已经", "是否", "多少", "哪个", "哪些", "这样", "那样", "刚才", "请问", "帮我",
}

# 高精度抽取规则：宁可少记，也不要把"我在上海出差"记成"常住上海"
CITY_CUE = re.compile(r"(?:我住|我常住|我家在|我定居|我们住)(?:在)?\s*(" + "|".join(CITY_WORDS) + ")")
MODEL_CUE = re.compile(r"(?:我的车是|我开的是|我买的是|我这台是)\s*(领克\s*0?[1-9])")
CENTER_CUE = re.compile(r"(?:我(?:一般|通常|都|常|习惯)去|我就在)\s*([^\s，。！？、；]{2,30})")
PREF_CUE = re.compile(
    r"(?:以后|今后|默认|每次|一直|记住)[^。！？\n]{0,12}?"
    r"(座椅加热|座椅通风|空调|氛围灯|方向盘加热|后视镜加热)(?:都|就|要)?(打开|关闭|开|关|调低|调高|开启)"
)
ACTION_NORMALIZE = {"开": "开启", "关": "关闭", "打开": "开启", "关闭": "关闭",
                    "调低": "调低", "调高": "调高", "开启": "开启"}


@dataclass
class MemoryConfig:
    ttl_days_episodic: int = 30
    ttl_days_semantic_default: int = 180
    ttl_days_semantic: Dict[str, int] = field(default_factory=lambda: {"city": 90, "usual_center": 90})
    decay_per_day: float = 0.98
    min_confidence: float = 0.25
    max_episodes_per_user: int = 200
    k_episodic: int = 2
    k_semantic: int = 3
    # 阈值来自 eval/memory_ab.py 的标定：真匹配（"上次那个工单怎么样了"）≈0.10，
    # 不相关（"今天天气怎么样"）＝0.00。0.09 是这两类之间的分界线，不是拍脑袋。
    min_episodic_score: float = 0.09


@dataclass
class MemoryItem:
    kind: str                 # episodic | semantic
    key: str
    text: str
    score: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    items: List[MemoryItem] = field(default_factory=list)
    dropped: int = 0          # 因不相关/过期被丢弃的条数（用于 false_inject 分母）

    def __bool__(self) -> bool:
        return bool(self.items)

    def to_prompt(self) -> str:
        if not self.items:
            return ""
        preferences = [i for i in self.items if i.kind == "semantic"]
        episodes = [i for i in self.items if i.kind == "episodic"]
        lines = ["【跨会话记忆】"]
        if preferences:
            lines.append("- 车主偏好：" + "；".join(f"{i.meta.get('label', i.key)}={i.text}" for i in preferences))
        if episodes:
            lines.append("- 相关历史：" + "；".join(i.text for i in episodes))
        # 冲突处理指令：记忆是"过去的信息"，与当场检索/车况冲突时必须以后者为准，
        # 否则会出现"记忆说 92 号汽油、手册说 95 号，模型信记忆"这种事故。
        lines.append("（以上是历史信息；若与本次手册原文或实时车况冲突，以本次信息为准，并向车主说明差异）")
        return "\n".join(lines)


def _tokens(text: str) -> List[str]:
    """分词 + 汉字 bigram。

    为什么要加 bigram：jieba 会把「保养工单」切成「保养工」+「单」，于是"工单"这个词在
    情景记忆里根本不存在，问"上次那个工单怎么样了"就会召回失败（实测 0 命中）。
    bigram 是仓库里已有的老办法（`kb.Tokenizer` 在没有 jieba 时也这么退化），
    对"几十到几百条短摘要"这个规模足够，不需要上向量。
    """
    raw = (text or "").strip()
    if not raw:
        return []
    try:
        import jieba  # noqa: PLC0415  延迟导入：没有 jieba 时退化为按标点切
        words = [t for t in jieba.cut_for_search(raw) if len(t.strip()) >= 2 and t not in STOPWORDS]
    except Exception:                                          # noqa: BLE001
        words = [t for t in re.split(r"[^\u4e00-\u9fffA-Za-z0-9]+", raw)
                 if len(t) >= 2 and t not in STOPWORDS]
    grams: List[str] = []
    for run in re.findall(r"[\u4e00-\u9fff]+", raw):
        grams.extend(run[i:i + 2] for i in range(len(run) - 1))
    return words + [g for g in grams if g not in STOPWORDS]


def _overlap(query: Sequence[str], text: str) -> float:
    """召回视角的重合度：**问题里的词有多少能在记忆里找到**。

    不用 Jaccard（|A∩B|/|A∪B|）：摘要通常比问题长得多，Jaccard 会把"只提到一个关键词
    但正是要找的那条"压到 0.1 以下（实测"上次那个工单怎么样了"只有 0.085）。
    记忆召回关心的是"这条能不能回答这个问题"，所以按查询侧的覆盖率算。
    """
    q = {t for t in _tokens(" ".join(query)) if t}
    if not q:
        return 0.0
    t = {w for w in _tokens(text) if w}
    if not t:
        return 0.0
    return len(q & t) / len(q)


class LongTermMemory:
    """SQLite 支撑的三层记忆（工作记忆仍在 ConversationMemory，这里只管跨会话两层）。"""

    def __init__(self, path: str = ":memory:", config: Optional[MemoryConfig] = None,
                 clock: Optional[Callable[[], float]] = None):
        self.path = path
        self.config = config or MemoryConfig()
        self._now = clock or time.time
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")     # 多线程读 + 单写者，WAL 更稳
        except sqlite3.OperationalError:                      # :memory: 不支持 WAL，忽略
            pass
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS semantic (
                user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                label TEXT DEFAULT '', confidence REAL DEFAULT 0.9, source TEXT DEFAULT 'user_explicit',
                created_at REAL NOT NULL, updated_at REAL NOT NULL, expires_at REAL NOT NULL,
                hits INTEGER DEFAULT 0,
                PRIMARY KEY (user_id, key)
            );
            CREATE TABLE IF NOT EXISTS episodic (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT DEFAULT '',
                kind TEXT DEFAULT 'session', summary TEXT NOT NULL, topics TEXT DEFAULT '[]',
                created_at REAL NOT NULL, expires_at REAL NOT NULL, last_used_at REAL DEFAULT 0,
                dedupe_key TEXT UNIQUE
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, action TEXT, key TEXT,
                old_value TEXT, new_value TEXT, reason TEXT, at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_episodic_user ON episodic(user_id, created_at DESC);
            """
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "LongTermMemory":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ── 写入：偏好 ──
    def remember_preference(self, key: str, value: str, *, user_id: str = "default",
                            confidence: float = 0.9, source: str = "user_explicit",
                            label: str = "", allow_inferred: bool = False) -> bool:
        """写一条语义记忆。返回是否真的写入（被策略拒绝时返回 False，不静默吞掉）。"""
        key = (key or "").strip()
        value = mask_sensitive((value or "").strip())[:200]
        if not key or not value:
            return False
        if source != "user_explicit" and not allow_inferred:
            return False                       # 策略 1：推断默认不落库
        now = self._now()
        ttl_days = self.config.ttl_days_semantic.get(key, self.config.ttl_days_semantic_default)
        expires_at = now + ttl_days * DAY
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT value, confidence, source FROM semantic WHERE user_id=? AND key=?",
                (user_id, key)).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO semantic (user_id, key, value, label, confidence, source,"
                    " created_at, updated_at, expires_at, hits) VALUES (?,?,?,?,?,?,?,?,?,0)",
                    (user_id, key, value, label or key, float(confidence), source,
                     now, now, expires_at))
                self._event(user_id, "write", key, None, value, source, now)
                return True
            if row["value"] == value:
                # 同一件事再确认一次 → 置信度回升（复述是强信号）
                self._conn.execute(
                    "UPDATE semantic SET confidence=?, updated_at=?, expires_at=?, hits=hits+1"
                    " WHERE user_id=? AND key=?",
                    (max(float(confidence), float(row["confidence"])), now, expires_at, user_id, key))
                self._event(user_id, "reinforce", key, row["value"], value, source, now)
                return True
            if row["source"] == "user_explicit" and source != "user_explicit":
                # 策略 2：明示过的偏好不允许被推断改写
                self._event(user_id, "reject_override", key, row["value"], value, source, now)
                return False
            self._conn.execute(
                "UPDATE semantic SET value=?, label=?, confidence=?, source=?, updated_at=?,"
                " expires_at=?, hits=0 WHERE user_id=? AND key=?",
                (value, label or key, float(confidence), source, now, expires_at, user_id, key))
            self._event(user_id, "overwrite", key, row["value"], value, source, now)
            return True

    # ── 写入：情景 ──
    def remember_episode(self, summary: str, *, user_id: str = "default", session_id: str = "",
                         kind: str = "session", topics: Sequence[str] = (),
                         dedupe_key: Optional[str] = None) -> Optional[str]:
        """写一条情景记忆；`dedupe_key` 相同视为重复（幂等重试不会写两遍）。"""
        summary = mask_sensitive((summary or "").strip())[:500]
        if len(summary) < 4:
            return None
        now = self._now()
        episode_id = uuid.uuid4().hex[:16]
        dedupe = dedupe_key or f"{user_id}:{session_id}:{uuid.uuid4().hex[:8]}"
        with self._lock, self._conn:
            try:
                self._conn.execute(
                    "INSERT INTO episodic (id, user_id, session_id, kind, summary, topics,"
                    " created_at, expires_at, last_used_at, dedupe_key) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (episode_id, user_id, session_id, kind, summary,
                     json.dumps(list(topics), ensure_ascii=False),
                     now, now + self.config.ttl_days_episodic * DAY, now, dedupe))
            except sqlite3.IntegrityError:
                return None                                     # 重复写入：直接忽略
            self._event(user_id, "episode", episode_id, None, summary, kind, now)
        return episode_id

    # ── 写入：从一句话里抽高精度偏好 ──
    def extract_and_remember(self, text: str, *, user_id: str = "default",
                             source: str = "user_explicit") -> List[str]:
        """只抽**明示**信息（四类句式），抽到的返回 `key=value` 列表，便于日志与测试断言。"""
        written: List[str] = []
        text = text or ""
        city = CITY_CUE.search(text)
        if city and self.remember_preference("city", city.group(1), user_id=user_id,
                                             source=source, label="常住城市"):
            written.append(f"city={city.group(1)}")
        model = MODEL_CUE.search(text)
        if model:
            name = re.sub(r"\s+", "", model.group(1))
            digits = re.search(r"(\d{1,2})$", name)
            if digits:                                       # 领克8 / 领克 08 统一成 领克08
                name = f"领克{int(digits.group(1)):02d}"
            if self.remember_preference("vehicle_model", name, user_id=user_id,
                                        source=source, label="车型"):
                written.append(f"vehicle_model={name}")
        center = CENTER_CUE.search(text)
        if center:
            value = center.group(1).strip().rstrip("，。")     # 括号里的门店名不能被截断
            if ("店" in value or "中心" in value) and self.remember_preference(
                    "usual_center", value, user_id=user_id, source=source, label="惯用门店"):
                written.append(f"usual_center={value}")
        pref = PREF_CUE.search(text)
        if pref:
            feature, action = pref.group(1), ACTION_NORMALIZE.get(pref.group(2), pref.group(2))
            if self.remember_preference(f"pref:{feature}", action, user_id=user_id,
                                        source=source, label=f"常用功能·{feature}"):
                written.append(f"pref:{feature}={action}")
        return written

    # ── 召回 ──
    def recall(self, query: str, *, user_id: str = "default",
               now: Optional[float] = None) -> MemoryContext:
        now = self._now() if now is None else now
        tokens = _tokens(query)
        context = MemoryContext()

        # 语义记忆：先按触发词筛主题，再按置信度取前 k（不相关的一条都不注入）
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value, label, confidence, updated_at, expires_at FROM semantic"
                " WHERE user_id=?", (user_id,)).fetchall()
        candidates: List[MemoryItem] = []
        for row in rows:
            if row["expires_at"] < now:
                context.dropped += 1
                continue
            effective = float(row["confidence"]) * self.config.decay_per_day ** (
                (now - float(row["updated_at"])) / DAY)
            if effective < self.config.min_confidence:
                context.dropped += 1
                continue
            if not self._semantic_relevant(str(row["key"]), query, tokens):
                context.dropped += 1
                continue
            days = int((now - float(row["updated_at"])) / DAY)
            candidates.append(MemoryItem(
                kind="semantic", key=str(row["key"]), text=str(row["value"]), score=effective,
                meta={"label": row["label"] or row["key"], "age_days": days,
                      "confidence": round(effective, 3)}))
        candidates.sort(key=lambda i: -i.score)
        context.items.extend(candidates[:self.config.k_semantic])

        # 情景记忆：词面重合 + 时间衰减
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, summary, created_at, expires_at FROM episodic WHERE user_id=?"
                " ORDER BY created_at DESC LIMIT 100", (user_id,)).fetchall()
        scored: List[Tuple[float, MemoryItem]] = []
        for row in rows:
            if row["expires_at"] < now:
                context.dropped += 1
                continue
            score = _overlap(tokens, str(row["summary"]))
            if score <= 0:
                context.dropped += 1
                continue
            recency = max(0.0, 1.0 - (now - float(row["created_at"])) / (self.config.ttl_days_episodic * DAY))
            total = score * (0.8 + 0.2 * recency)          # 时间只做微调，不能靠"新"就压过相关性
            if total < self.config.min_episodic_score:
                context.dropped += 1
                continue
            scored.append((total, MemoryItem(kind="episodic", key=str(row["id"]),
                                             text=self._episode_line(str(row["summary"]), float(row["created_at"])),
                                             score=round(total, 3))))
        scored.sort(key=lambda pair: -pair[0])
        context.items.extend(item for _, item in scored[:self.config.k_episodic])

        hits = [i.key for i in context.items if i.kind == "episodic"]
        if hits:
            with self._lock, self._conn:
                for episode_id in hits:
                    self._conn.execute("UPDATE episodic SET last_used_at=? WHERE id=?",
                                       (now, episode_id))
        return context

    def _semantic_relevant(self, key: str, query: str, tokens: Sequence[str]) -> bool:
        if key.startswith("pref:"):
            return key.split(":", 1)[1] in query
        cues = SEMANTIC_CUES.get(key, ())
        return any(cue in query for cue in cues)

    @staticmethod
    def _episode_line(summary: str, created_at: float) -> str:
        stamp = time.strftime("%m-%d", time.localtime(created_at))
        return f"{stamp} 记录：{summary}"

    def context_text(self, query: str, *, user_id: str = "default",
                     now: Optional[float] = None) -> str:
        return self.recall(query, user_id=user_id, now=now).to_prompt()

    def preference(self, user_id: str, key: str, *, now: Optional[float] = None) -> Optional[str]:
        """取一条语义记忆的当前值（已过期或置信度不足则返回 None）。

        用途是**给工具填默认参数**（例如惯用门店）——这是长期记忆唯一能立刻变现的地方：
        用户说过一次"我一般去浦东店"，下次建工单就不用再说。
        """
        now = self._now() if now is None else now
        with self._lock:
            row = self._conn.execute(
                "SELECT value, confidence, updated_at, expires_at FROM semantic"
                " WHERE user_id=? AND key=?", (user_id, key)).fetchone()
        if row is None or float(row["expires_at"]) < now:
            return None
        effective = float(row["confidence"]) * self.config.decay_per_day ** (
            (now - float(row["updated_at"])) / DAY)
        if effective < self.config.min_confidence:
            return None
        return str(row["value"])

    # ── 维护：过期、衰减、容量 ──
    def maintain(self, *, user_id: Optional[str] = None, now: Optional[float] = None) -> Dict[str, int]:
        now = self._now() if now is None else now
        stats = {"expired_episodes": 0, "purged_semantics": 0, "evicted_episodes": 0, "decayed": 0}
        with self._lock, self._conn:
            where, params = ("WHERE expires_at < ?", (now,)) if user_id is None else \
                ("WHERE user_id=? AND expires_at < ?", (user_id, now))
            stats["expired_episodes"] = self._conn.execute(
                f"DELETE FROM episodic {where}", params).rowcount
            cur = self._conn.execute(f"DELETE FROM semantic {where}", params)
            stats["purged_semantics"] = cur.rowcount

            rows = self._conn.execute("SELECT user_id, key, confidence, updated_at FROM semantic").fetchall()
            for row in rows:
                effective = float(row["confidence"]) * self.config.decay_per_day ** (
                    (now - float(row["updated_at"])) / DAY)
                if effective < self.config.min_confidence:
                    self._conn.execute("DELETE FROM semantic WHERE user_id=? AND key=?",
                                       (row["user_id"], row["key"]))
                    self._event(str(row["user_id"]), "decay_purge", str(row["key"]),
                                None, None, f"confidence<{self.config.min_confidence}", now)
                    stats["purged_semantics"] += 1
                else:
                    stats["decayed"] += 1

            # 容量上限：每用户保留最近 N 条情景记忆
            users = self._conn.execute("SELECT DISTINCT user_id FROM episodic").fetchall()
            for row in users:
                uid = str(row["user_id"])
                excess = self._conn.execute(
                    "SELECT id FROM episodic WHERE user_id=? ORDER BY created_at DESC LIMIT -1 OFFSET ?",
                    (uid, self.config.max_episodes_per_user)).fetchall()
                for item in excess:
                    self._conn.execute("DELETE FROM episodic WHERE id=?", (item["id"],))
                    stats["evicted_episodes"] += 1
        return stats

    def _event(self, user_id: str, action: str, key: str, old_value: Optional[str],
               new_value: Optional[str], reason: str, at: float) -> None:
        self._conn.execute(
            "INSERT INTO events (user_id, action, key, old_value, new_value, reason, at)"
            " VALUES (?,?,?,?,?,?,?)", (user_id, action, key, old_value, new_value, reason, at))

    # ── 观测 ──
    def fingerprint(self, user_id: str = "default") -> str:
        """记忆指纹：进语义缓存的 scope。

        为什么必须有：缓存键只含"车型 + KB 版本 + 车况"，记忆一改（用户刚说"以后座椅加热都关掉"），
        旧答案就会被原样返回——"记住了但不生效"。把记忆指纹加进 scope，记忆一变缓存自然失效。
        """
        with self._lock:
            semantic = self._conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(MAX(updated_at), 0) AS m FROM semantic"
                " WHERE user_id=?", (user_id,)).fetchone()
            episodic = self._conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(MAX(created_at), 0) AS m FROM episodic"
                " WHERE user_id=?", (user_id,)).fetchone()
        raw = f"{semantic['n']}:{float(semantic['m']):.0f}|{episodic['n']}:{float(episodic['m']):.0f}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]

    def stats(self, user_id: str = "default") -> Dict[str, Any]:
        with self._lock:
            semantic = self._conn.execute(
                "SELECT COUNT(*) AS n, AVG(confidence) AS c FROM semantic WHERE user_id=?",
                (user_id,)).fetchone()
            episodic = self._conn.execute(
                "SELECT COUNT(*) AS n FROM episodic WHERE user_id=?", (user_id,)).fetchone()
            events = self._conn.execute(
                "SELECT COUNT(*) AS n FROM events WHERE user_id=?", (user_id,)).fetchone()
        return {"semantic": int(semantic["n"]), "episodic": int(episodic["n"]),
                "events": int(events["n"]),
                "avg_confidence": round(float(semantic["c"] or 0.0), 3)}

    def events(self, user_id: str = "default", limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT action, key, old_value, new_value, reason, at FROM events"
                " WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()
        return [dict(row) for row in rows]


def defaults_from_preferences(store: Optional[LongTermMemory], user_id: str) -> Dict[str, str]:
    """把偏好翻译成工具默认参数。只映射与工具契约对齐的键（目前是惯用门店）。"""
    if store is None:
        return {}
    out: Dict[str, str] = {}
    center = store.preference(user_id, "usual_center")
    if center:
        out["center"] = center
    return out


def order_event_recorder(store: Optional[LongTermMemory], user_id: str,
                         session_id: str = "orders",
                         query: str = "") -> Optional[Callable[[str, Dict[str, Any]], None]]:
    """工单写操作 → 情景记忆。服务层、CLI、评测共用这一份，避免三处各写一遍。

    `query` 是用户这句话：**带上它记忆才召回得到**。用户说"上次那个保养"，
    而工单摘要里写的是"更换机油机滤"——不存原话，词面匹配就对不上（实测召回 0 条）。
    """
    if store is None:
        return None
    prefix = f"用户提到「{query[:30]}」；" if query else ""

    def record(action: str, payload: Dict[str, Any]) -> None:
        order_id = str(payload.get("预约单号") or "")
        item = str(payload.get("项目") or "")
        date = str(payload.get("到店日期") or "")
        center = str(payload.get("门店") or "")
        if action == "order_created":
            text = f"{prefix}已建工单 {order_id}：{item}，{date} 到 {center}"
        elif action == "order_reused":
            text = f"{prefix}工单 {order_id} 已存在：{item}，{date} 到 {center}"
        elif action == "order_rescheduled":
            text = f"{prefix}工单 {order_id} 改期到 {date}" + (f"（{center}）" if center else "")
        elif action == "order_cancelled":
            text = f"{prefix}工单 {order_id} 已取消"
        else:
            return
        store.remember_episode(text, user_id=user_id, session_id=session_id,
                               dedupe_key=f"{session_id}:{action}:{order_id}:{date}")

    return record
