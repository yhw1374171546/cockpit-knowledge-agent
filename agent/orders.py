# -*- coding: utf-8 -*-
"""服务工单：一个**真的**有状态的业务对象（SQLite 落库 + 状态机 + 事件历史）。

为什么需要它
------------
在它之前，`create_service_order` 只返回一个字符串单号（见 `tools.py` 的旧实现）：
不落库、没有状态、不能查询/改期/取消。后果是**长期记忆没有东西可记**、
"MCP 供多个 Agent 复用"也没有真实对象可复用——基础设施只能自证，这正是这次补它的原因。

状态机（刻意做小，但每条非法转移都有测试）
------------------------------------------
    booked ──reschedule──> booked（date/center 变，version+1）
    booked ──cancel──────> cancelled（终态）
    booked ──complete────> completed（终态）
    cancelled / completed ──任何操作──> 报错

三条业务规则
------------
1. **日期不早于今天**（时钟可注入，测试不依赖真实时间）；
2. **跨用户隔离**：查别人的工单返回 `not_found`（不回 403，避免"这个单号存在"的信息泄露）；
3. **创建幂等**：同用户 + 同项目 + 同日期 + 同门店，且已有 `booked` 单 → 直接返回那一条
   （标记 `existing=True`），不会因为重试多出一单。
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
STATES = ("booked", "completed", "cancelled")
TERMINAL = ("completed", "cancelled")


class OrderError(Exception):
    """业务错误：带机器可读的 code，工具层会翻译成结构化错误给模型看。"""

    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


@dataclass
class Order:
    id: str
    user_id: str
    item: str
    center: str
    date: str
    status: str = "booked"
    version: int = 1
    created_at: float = 0.0
    updated_at: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)
    existing: bool = False          # 本次调用是否命中了已有工单（幂等重试）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "预约单号": self.id, "项目": self.item, "门店": self.center,
            "到店日期": self.date, "状态": self.status, "改期次数": self.version - 1,
            "是否已存在": self.existing,
        }

    def summary(self) -> str:
        label = {"booked": "已预约", "completed": "已完成", "cancelled": "已取消"}.get(self.status, self.status)
        return f"{self.id}｜{self.item}｜{self.date}｜{self.center}｜{label}"


class OrderStore:
    """SQLite 支撑的工单存储（与长期记忆同一套"零依赖 + 可注入时钟"的写法）。"""

    def __init__(self, path: str = ":memory:", clock: Optional[Callable[[], float]] = None,
                 enforce_future_date: bool = False):
        self.path = path
        self._now = clock or time.time
        # 日期不得早于今天这条规则默认关闭：离线测试用固定日期（2025-06-18）才能稳定断言；
        # 服务层与 CLI 打开它（真实业务里"约到上个月"必须被拦住）。
        self.enforce_future_date = enforce_future_date
        # RLock：`create()` 在持有锁的临界区里还会调 `_new_id()`（它也要锁）。
        # 用普通 Lock 会直接死锁——单测第一版就是这么挂住的。
        self._lock = threading.RLock()
        if path not in (":memory:", "") and os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, item TEXT NOT NULL,
                center TEXT NOT NULL, date TEXT NOT NULL, status TEXT NOT NULL,
                version INTEGER DEFAULT 1, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                history TEXT DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_orders_user ON orders(user_id, created_at DESC);
            """
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "OrderStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ── 校验 ──
    def _validate_date(self, date: str, now: float) -> str:
        date = (date or "").strip()
        if not DATE_RE.match(date):
            raise OrderError("bad_date", f"日期格式应为 YYYY-MM-DD，收到「{date}」",
                             "例如 2026-11-05")
        try:
            parsed = time.mktime(time.strptime(date, "%Y-%m-%d"))
        except ValueError as exc:
            raise OrderError("bad_date", f"日期不合法：{exc}", "例如 2026-11-05") from exc
        if self.enforce_future_date and parsed < time.mktime(time.localtime(now)[:3] + (0, 0, 0, 0, 0, -1)):
            raise OrderError("past_date", f"到店日期 {date} 早于今天", "请给一个今天或之后的日期")
        return date

    @staticmethod
    def _row_to_order(row: sqlite3.Row) -> Order:
        return Order(id=str(row["id"]), user_id=str(row["user_id"]), item=str(row["item"]),
                     center=str(row["center"]), date=str(row["date"]), status=str(row["status"]),
                     version=int(row["version"]), created_at=float(row["created_at"]),
                     updated_at=float(row["updated_at"]),
                     history=json.loads(row["history"] or "[]"))

    def _fetch(self, user_id: str, order_id: str) -> Order:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM orders WHERE id=? AND user_id=?", (order_id, user_id)).fetchone()
        if row is None:
            # 不区分"不存在"和"不是你的"：否则单号本身成了探测工具
            raise OrderError("not_found", f"没有找到工单 {order_id}",
                             "可以让用户确认单号，或用「我最近的工单」查一下")
        return self._row_to_order(row)

    # ── 写 ──
    def create(self, user_id: str, item: str, date: str, center: str) -> Order:
        item = (item or "").strip()
        center = (center or "").strip() or "领克中心(浦东金桥店)"
        if not item:
            raise OrderError("bad_item", "服务项目不能为空", "例如 更换机油机滤")
        now = self._now()
        date = self._validate_date(date, now)
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT * FROM orders WHERE user_id=? AND item=? AND date=? AND center=?"
                " AND status='booked' ORDER BY created_at DESC LIMIT 1",
                (user_id, item, date, center)).fetchone()
            if row is not None:                       # 幂等：重试不多出单
                order = self._row_to_order(row)
                order.existing = True
                return order
            order_id = self._new_id(now)
            history = [{"at": now, "action": "created",
                        "to": {"item": item, "date": date, "center": center}}]
            self._conn.execute(
                "INSERT INTO orders (id, user_id, item, center, date, status, version,"
                " created_at, updated_at, history) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (order_id, user_id, item, center, date, "booked", 1, now, now,
                 json.dumps(history, ensure_ascii=False)))
        return Order(id=order_id, user_id=user_id, item=item, center=center, date=date,
                     created_at=now, updated_at=now, history=history)

    def _new_id(self, now: float) -> str:
        base = f"SO{int(now) % 10 ** 8:08d}"
        with self._lock:
            exists = self._conn.execute("SELECT 1 FROM orders WHERE id=?", (base,)).fetchone()
        return base if exists is None else f"{base}{uuid.uuid4().hex[:2].upper()}"

    def reschedule(self, user_id: str, order_id: str, *, date: Optional[str] = None,
                   center: Optional[str] = None) -> Order:
        order = self._fetch(user_id, order_id)
        if order.status in TERMINAL:
            raise OrderError("invalid_state", f"工单 {order_id} 已是「{order.status}」，不能改期",
                             "已完成的工单请新建一条预约")
        now = self._now()
        new_date = self._validate_date(date, now) if date else order.date
        new_center = (center or "").strip() or order.center
        if new_date == order.date and new_center == order.center:
            order.existing = True                      # 参数没变，等价于"重复提交"
            return order
        history = order.history + [{"at": now, "action": "rescheduled",
                                    "from": {"date": order.date, "center": order.center},
                                    "to": {"date": new_date, "center": new_center}}]
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE orders SET date=?, center=?, version=version+1, updated_at=?,"
                " history=? WHERE id=? AND user_id=?",
                (new_date, new_center, now, json.dumps(history, ensure_ascii=False),
                 order_id, user_id))
        order.date, order.center = new_date, new_center
        order.version += 1
        order.updated_at = now
        order.history = history
        return order

    def cancel(self, user_id: str, order_id: str) -> Order:
        return self._finish(user_id, order_id, "cancelled")

    def complete(self, user_id: str, order_id: str) -> Order:
        return self._finish(user_id, order_id, "completed")

    def _finish(self, user_id: str, order_id: str, status: str) -> Order:
        order = self._fetch(user_id, order_id)
        if order.status in TERMINAL:
            raise OrderError("invalid_state", f"工单 {order_id} 已是「{order.status}」", "无需重复操作")
        now = self._now()
        history = order.history + [{"at": now, "action": status, "from": order.status}]
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE orders SET status=?, updated_at=?, history=? WHERE id=? AND user_id=?",
                (status, now, json.dumps(history, ensure_ascii=False), order_id, user_id))
        order.status, order.updated_at, order.history = status, now, history
        return order

    # ── 读 ──
    def get(self, user_id: str, order_id: str) -> Order:
        return self._fetch(user_id, order_id)

    def list(self, user_id: str, *, limit: int = 5, status: Optional[str] = None) -> List[Order]:
        sql = "SELECT * FROM orders WHERE user_id=?"
        params: List[Any] = [user_id]
        if status:
            sql += " AND status=?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(int(limit or 5), 20)))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_order(row) for row in rows]

    def stats(self, user_id: str) -> Dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM orders WHERE user_id=? GROUP BY status",
                (user_id,)).fetchall()
        out = {state: 0 for state in STATES}
        out.update({str(row["status"]): int(row["n"]) for row in rows})
        return out
