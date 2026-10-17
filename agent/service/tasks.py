# -*- coding: utf-8 -*-
"""轻量任务队列（BE3）：索引构建、批量评测这类**长任务**不该占着请求线程。

为什么不直接引 Celery / RQ
--------------------------
本项目刻意"零新增依赖"（README 里反复强调）：单机/单库场景下，用**数据库表 + 后台线程**
就能满足"提交 → 排队 → 执行 → 查结果"的全部需求，而且**不引入新的运维组件**。
真要上多机大规模，再换 Celery/Kafka 也只是替换 `TaskQueue` 的实现，接口不变。

关键设计（都是踩过坑才写对的）
------------------------------
1. **认领要原子**：`UPDATE ... WHERE status='queued'` 这一条语句同时完成"选中 + 置为 running"，
   不能先 SELECT 再 UPDATE——两个 worker 会同时选中同一条任务，**同一个索引被构建两次**。
   SQLite 用 `BEGIN IMMEDIATE`（本项目已有的做法）保证写锁；PG 上这条 UPDATE 本身就是原子的。
2. **失败要留痕且可重试**：记录 attempts / error / 最后一次堆栈摘要，超过上限置 `failed`；
   **不能静默吞掉**（否则用户看到任务一直 queued，永远不知道发生了什么）。
3. **任务状态必须能查**：`GET /v1/admin/tasks/{id}` 返回 status/attempts/error/结果摘要。

已知边界（如实写在这里）：当前是**单进程内后台线程**，多副本会各起一个 worker，
靠上面那条原子 UPDATE 保证同一任务只被一个 worker 认领——这已经够用；
但**没有优先级队列、没有定时任务、没有跨机负载均衡**，那属于下一阶段。
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from agent.service.db import Database, TaskRow

# 任务状态
QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"

MAX_ATTEMPTS = 3


@dataclass
class TaskInfo:
    id: str
    kind: str
    status: str
    attempts: int
    error: str = ""
    result: Optional[Dict[str, Any]] = None
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "status": self.status,
                "attempts": self.attempts, "error": self.error, "result": self.result,
                "created_at": self.created_at, "started_at": self.started_at,
                "finished_at": self.finished_at}


def _row_to_info(row: TaskRow) -> TaskInfo:
    result = None
    if row.result:
        try:
            result = json.loads(row.result)
        except (ValueError, TypeError):
            result = {"raw": row.result}
    return TaskInfo(id=row.id, kind=row.kind, status=row.status,
                    attempts=int(row.attempts or 0), error=row.error or "",
                    result=result,
                    created_at=row.created_at.timestamp() if row.created_at else None,
                    started_at=row.started_at.timestamp() if row.started_at else None,
                    finished_at=row.finished_at.timestamp() if row.finished_at else None)


class TaskQueue:
    """数据库支撑的极简任务队列（提交 / 原子认领 / 完成 / 失败重试 / 查询）。"""

    def __init__(self, db: Database) -> None:
        self.db = db

    # ── 提交 ──
    def enqueue(self, kind: str, payload: Optional[Dict[str, Any]] = None) -> TaskInfo:
        task_id = uuid.uuid4().hex[:16]
        with self.db.session() as session:
            session.add(TaskRow(id=task_id, kind=kind,
                                payload=json.dumps(payload or {}, ensure_ascii=False),
                                status=QUEUED, attempts=0))
        info = self.get(task_id)
        assert info is not None
        return info

    # ── 查询 ──
    def get(self, task_id: str) -> Optional[TaskInfo]:
        with self.db.session() as session:
            row = session.get(TaskRow, task_id)
            return _row_to_info(row) if row is not None else None

    def list(self, limit: int = 20, kind: str = "", status: str = "") -> list:
        with self.db.session() as session:
            query = session.query(TaskRow)
            if kind:
                query = query.filter(TaskRow.kind == kind)
            if status:
                query = query.filter(TaskRow.status == status)
            rows = query.order_by(TaskRow.created_at.desc()).limit(limit).all()
            return [_row_to_info(r).to_dict() for r in rows]

    def stats(self) -> Dict[str, int]:
        with self.db.session() as session:
            out = {"queued": 0, "running": 0, "succeeded": 0, "failed": 0}
            for row in session.query(TaskRow).all():
                if row.status in out:
                    out[row.status] += 1
            return out

    # ── 认领（原子）──
    def claim(self) -> Optional[tuple]:
        """认领一个排队任务；返回 (TaskInfo, payload dict)，没有则 None。

        **一条 UPDATE 完成"选中 + 占用"**，不做 SELECT-then-UPDATE：
        后者在多个 worker 同时轮询时会重复认领同一任务（索引被建两遍、评测被跑两遍）。
        """
        with self.db.session() as session:
            row = (session.query(TaskRow)
                   .filter(TaskRow.status == QUEUED)
                   .order_by(TaskRow.created_at.asc())
                   .first())
            if row is None:
                return None
            # 带上 status 条件再更新一次：如果有别的 worker 抢先改了，affected=0 → 本轮放弃
            affected = (session.query(TaskRow)
                        .filter(TaskRow.id == row.id, TaskRow.status == QUEUED)
                        .update({"status": RUNNING, "started_at": __import__("datetime").datetime.utcnow(),
                                 "attempts": (row.attempts or 0) + 1}, synchronize_session=False))
            if not affected:
                return None
            payload = {}
            if row.payload:
                try:
                    payload = json.loads(row.payload)
                except (ValueError, TypeError):
                    payload = {"raw": row.payload}
            return _row_to_info(row), payload

    # ── 完成 / 失败 ──
    def complete(self, task_id: str, result: Optional[Dict[str, Any]] = None) -> None:
        with self.db.session() as session:
            session.query(TaskRow).filter(TaskRow.id == task_id).update(
                {"status": SUCCEEDED,
                 "result": json.dumps(result or {}, ensure_ascii=False, default=str),
                 "finished_at": __import__("datetime").datetime.utcnow()})

    def fail(self, task_id: str, error: str) -> None:
        """失败：未到上限则退回 queued（可重试），否则置 failed。**错误必须落库**。"""
        with self.db.session() as session:
            row = session.get(TaskRow, task_id)
            if row is None:
                return
            attempts = int(row.attempts or 0)
            status = QUEUED if attempts < MAX_ATTEMPTS else FAILED
            row.status = status
            row.error = (error or "")[:2000]
            if status == FAILED:
                row.finished_at = __import__("datetime").datetime.utcnow()


class TaskWorker(threading.Thread):
    """后台 worker：轮询队列并执行已注册的 handler。

    刻意做成**可停的守护线程**（`stop()` + `join(timeout)`），
    这样优雅停机（BE8）时不会把一个正在建索引的任务硬切掉。
    """

    def __init__(self, queue: TaskQueue, handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
                 poll_s: float = 0.5) -> None:
        super().__init__(name="task-worker", daemon=True)
        self.queue = queue
        self.handlers = handlers
        self.poll_s = poll_s
        # 注意：不能叫 self._stop —— 那会覆盖 Thread 的私有方法 _stop()，
        # join() 内部会调用它，于是抛 TypeError: 'Event' object is not callable。
        self._stop_event = threading.Event()
        self.processed = 0
        self.failed = 0

    def stop(self) -> None:
        self._stop_event.set()

    def run_once(self) -> bool:
        """执行一个任务（供测试直接驱动，不必等轮询）。返回是否处理了任务。"""
        claimed = self.queue.claim()
        if claimed is None:
            return False
        info, payload = claimed
        handler = self.handlers.get(info.kind)
        if handler is None:
            self.queue.fail(info.id, f"未知任务类型：{info.kind}")
            self.failed += 1
            return True
        try:
            result = handler(payload) or {}
            self.queue.complete(info.id, result)
            self.processed += 1
        except Exception as exc:                                     # noqa: BLE001
            summary = f"{type(exc).__name__}: {exc}"
            detail = traceback.format_exc(limit=3)
            self.queue.fail(info.id, f"{summary}\n{detail}")
            self.failed += 1
        return True

    def run(self) -> None:                                          # pragma: no cover
        while not self._stop_event.is_set():
            try:
                handled = self.run_once()
            except Exception:                                       # noqa: BLE001
                handled = False
            if not handled:
                self._stop_event.wait(self.poll_s)


# ── 内置 handler ──

def make_kb_build_handler(registry) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """`kb_build`：把一份语料注册成新的知识库版本（耗时的加载放到队列里做）。

    这样"注册新版本"这个动作在请求线程里只是**入队**，不会因为加载 9,301 块
    而把 HTTP 请求卡上几十秒（尤其带向量索引时）。
    """

    def handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        path = str(payload.get("path") or "")
        vehicle_model = str(payload.get("vehicle_model") or "default")
        version = str(payload.get("version") or "")
        if not path:
            raise ValueError("缺少 path")
        record = registry.register(path, vehicle_model, version or None)
        return {"version": record.version, "checksum": record.checksum,
                "n_chunks": record.n_chunks, "vehicle_model": record.vehicle_model}

    return handler


def make_noop_handler(name: str = "noop") -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """占位 handler（用于验证队列本身，不产生副作用）。"""

    def handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"handler": name, "echo": payload, "at": time.time()}

    return handler
