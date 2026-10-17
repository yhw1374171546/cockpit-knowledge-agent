# -*- coding: utf-8 -*-
"""知识库版本化演示：注册新版本 → 灰度 50% → 观察分流 → 回滚。

这是"线上怎么换手册、改错了怎么回滚"的可执行答案。以前的做法是：改配置 + 重启，
没有版本、没有灰度、出问题只能再改回去重启。

    python eval/kb_versioning_demo.py
    python eval/kb_versioning_demo.py --pct 30      # 指定灰度比例

演示全程走**真实 HTTP 链路**（TestClient），并打印每一步的路由结果。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from agent.service.app import create_app  # noqa: E402
from agent.service.config import Settings  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "eval", "fixtures", "mini_corpus.jsonl")


def make_v2(tmpdir: str) -> str:
    """基于迷你语料造一个"新版手册"：改一块内容 + 加一块新内容。"""
    rows = []
    with open(FIXTURE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows[0]["text"] = "（2026 新版）" + rows[0]["text"]
    rows.append({"id": 9001, "text": "（2026 新版）电动尾门支持脚踢开启：携带钥匙靠近车尾，"
                                     "脚部轻扫保险杠下方即可，重复动作两次可关闭。",
                 "source": "manual_2026.pdf", "page": 88, "header": "尾门"})
    path = os.path.join(tmpdir, "corpus_2026.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=int, default=50, help="灰度比例（默认 50）")
    ap.add_argument("--users", type=int, default=20, help="用于观察分流的用户数")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        settings = Settings(
            database_url=f"sqlite:///{os.path.join(tmpdir, 'demo.db')}",
            kb_path=FIXTURE, jwt_secret="demo-secret-change-me-32bytes",
            auth_required=True, dev_token_enabled=True, kv_backend="memory",
            rate_limit_capacity=10_000, rate_limit_refill_per_sec=10_000.0,
            quota_tokens_per_day=10_000_000, llm_backend="planner",
            default_vehicle_model="lynk08", allowed_models=["lynk08", "lynk09"],
        )
        app = create_app(settings)
        with TestClient(app) as client:
            def token(user: str, role: str = "driver") -> str:
                return client.post("/v1/auth/token",
                                   json={"user_id": user, "role": role,
                                         "vehicle_model": "lynk08"}).json()["access_token"]

            def ask(user: str, question: str) -> str:
                headers = {"Authorization": f"Bearer {token(user)}"}
                body = client.post("/v1/chat", json={"message": question},
                                   headers=headers).json()
                return body.get("kb_version", "?")

            print("=" * 74)
            print("知识库版本化演示：注册 → 灰度 → 分流 → 回滚")
            print("=" * 74)

            base = ask("demo-u0", "座椅加热怎么关闭")
            print(f"\n[1] 当前生效版本：{base}")

            v2_path = make_v2(tmpdir)
            svc = {"Authorization": f"Bearer {token('demo-svc', 'service')}"}
            record = client.post("/v1/admin/kb/register",
                                 json={"path": v2_path, "vehicle_model": "lynk08"},
                                 headers=svc).json()
            print(f"[2] 注册新版本：{record['version']}  "
                  f"（{record['n_chunks']} 块，checksum={record['checksum']}）")
            print("    → 注册**不改变**线上路由，此时用户仍走旧版本：",
                  ask("demo-u0", "胎压多少算正常"))

            client.post("/v1/admin/kb/activate",
                        json={"version": record["version"], "traffic_pct": args.pct,
                              "vehicle_model": "lynk08"}, headers=svc)
            print(f"\n[3] 灰度 {args.pct}%：新版本只给部分用户")

            counts = {}
            for i in range(args.users):
                user = f"demo-u{i}"
                version = ask(user, "危险警告灯怎么用")
                counts[version] = counts.get(version, 0) + 1
            print("    分流结果：", json.dumps(counts, ensure_ascii=False))

            # 同一用户重复问，版本必须稳定（不能在同一会话里跳版）
            stable = {ask("demo-u7", q) for q in ("胎压多少算正常", "座椅加热怎么关闭",
                                                  "危险警告灯怎么用")}
            print(f"    同一用户连续三问的版本集合：{stable}"
                  f"  → {'✅ 稳定' if len(stable) == 1 else '❌ 跳版了'}")

            print("\n[4] 全量切到新版本，并验证新内容可检索")
            client.post("/v1/admin/kb/activate",
                        json={"version": record["version"], "traffic_pct": 100,
                              "vehicle_model": "lynk08"}, headers=svc)
            headers = {"Authorization": f"Bearer {token('demo-u0')}"}
            body = client.post("/v1/chat", json={"message": "电动尾门脚踢开启"},
                               headers=headers).json()
            print(f"    版本={body['kb_version']}  答案={body['answer'][:46]}…")

            print("\n[5] 回滚（一条命令，不需要重启）")
            rolled = client.post("/v1/admin/kb/rollback", json={"vehicle_model": "lynk08"},
                                 headers=svc).json()
            print(f"    回滚到：{rolled['version']}")
            print(f"    再次询问版本：{ask('demo-u0', '电动尾门脚踢开启')}")

            versions = client.get("/v1/admin/kb/versions?vehicle_model=lynk08",
                                  headers=svc).json()
            print("\n[6] 版本清单：")
            for item in versions["versions"]:
                print(f"    {item['version']:<14} {item['status']:<9} "
                      f"{item['n_chunks']:>3} 块  traffic={item['traffic_pct']}%  "
                      f"{os.path.basename(item['path'])}")
            print("    路由统计：", json.dumps(versions["stats"].get("routed", {}),
                                             ensure_ascii=False))

    print("\n" + "-" * 74)
    print("结论：换手册 = 注册新版本 + 灰度；出问题 = 一条 rollback，不需要改配置重启。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
