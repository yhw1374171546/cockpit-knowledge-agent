# -*- coding: utf-8 -*-
"""跑全部测试套件并汇总（CI 与本地都可用）。

为什么要这个脚本：手工一个个跑容易漏、也不方便统计总数；
CI 里逐条 `python xxx.py` 失败时定位够用，但**总览**（哪些套件、共多少项）没有。
本脚本用子进程逐个跑，汇总"通过/失败/项数"，失败时打印该套件的失败用例。

    python eval/run_all_tests.py
    python eval/run_all_tests.py --only service sse
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)          # 让 `from agent...` 可用
from agent.console import harden_stdout  # noqa: E402

harden_stdout()   # 中文 Windows 的 GBK 控制台编不出 [OK]，否则会崩在最后一行

TEST_DIR = os.path.join(ROOT, "agent", "tests")

# 顺序刻意从"最快/最基础"到"最慢/最集成"
SUITES = [
    "test_guardrails.py",
    "test_regressions.py",
    "test_offline.py",
    "test_fc_pipeline.py",
    "test_multiagent.py",
    "test_vector_store.py",
    "test_sse.py",
    "test_service.py",
    "test_tasks_and_traces.py",
    "test_migrations.py",
]

RAN_RE = re.compile(r"^Ran (\d+) tests", re.MULTILINE)


def run_one(name: str) -> tuple:
    path = os.path.join(TEST_DIR, name)
    result = subprocess.run([sys.executable, path], cwd=ROOT, capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    output = (result.stdout or "") + (result.stderr or "")
    match = RAN_RE.search(output)
    count = int(match.group(1)) if match else 0
    ok = result.returncode == 0
    failures = [line.strip() for line in output.splitlines()
                if line.startswith(("FAIL:", "ERROR:"))]
    return ok, count, failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="只跑名字里包含这些关键字的套件")
    args = ap.parse_args()

    suites = SUITES
    if args.only:
        suites = [s for s in SUITES if any(k in s for k in args.only)]

    total, failed = 0, []
    print("=" * 72)
    print("全量测试")
    print("=" * 72)
    for name in suites:
        ok, count, failures = run_one(name)
        total += count
        flag = "OK" if ok else "FAILED"
        print(f"  {name:<30} {flag:<8} {count:>3} 项")
        if not ok:
            failed.append(name)
            for line in failures[:4]:
                print(f"      {line}")

    print("-" * 72)
    print(f"合计 {total} 项 ｜ 套件 {len(suites) - len(failed)}/{len(suites)} 通过")
    if failed:
        print("失败套件: " + ", ".join(failed))
        return 1
    print("[OK] 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
