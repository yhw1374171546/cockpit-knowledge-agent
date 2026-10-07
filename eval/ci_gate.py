# -*- coding: utf-8 -*-
"""评测门禁：把「指标不得退化」变成可执行的检查，接进 CI。

用法：
    python eval/ci_gate.py                # 有指标文件就校验，没有就标记为 skipped
    python eval/ci_gate.py --require-all   # 缺文件即失败（本地全量回归时用）
退出码：0 = 通过（含全部 skipped）；1 = 有指标越界或缺文件（--require-all）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLDS = os.path.join(ROOT, "eval", "thresholds.json")


def dig(data: Any, path: str) -> Any:
    """按 a.b.c 取值；返回 None 表示缺失。"""
    current = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def check_file(rel_path: str, rules: Dict[str, Dict[str, Any]]) -> Tuple[str, List[str]]:
    full = os.path.join(ROOT, rel_path)
    if not os.path.exists(full):
        return "skipped", [f"{rel_path} 不存在（未跑过该项评测）"]
    try:
        data = json.load(open(full, encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return "fail", [f"{rel_path} 解析失败：{exc}"]

    messages, failed = [], False
    for metric, rule in rules.items():
        value = dig(data, metric)
        if value is None or not isinstance(value, (int, float)):
            messages.append(f"  ⚠️  {metric}: 缺失或非数值（{value!r}）")
            failed = True
            continue
        low, high = rule.get("min"), rule.get("max")
        if low is not None and value < low:
            messages.append(f"  ❌ {metric}: {value} < 下限 {low}（{rule.get('desc','')}）")
            failed = True
        elif high is not None and value > high:
            messages.append(f"  ❌ {metric}: {value} > 上限 {high}（{rule.get('desc','')}）")
            failed = True
        else:
            bound = f"min={low}" if low is not None else f"max={high}"
            messages.append(f"  ✅ {metric}: {value}（{bound}）")
    return ("fail" if failed else "pass"), messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-all", action="store_true",
                    help="缺少指标文件时判为失败（本地全量回归）")
    args = ap.parse_args()

    config = json.load(open(THRESHOLDS, encoding="utf-8"))
    thresholds = config.get("thresholds", {})
    overall_fail = False
    summary = []

    print("=" * 72)
    print("评测门禁（CI Gate）")
    print("=" * 72)
    for rel_path, rules in thresholds.items():
        status, messages = check_file(rel_path, rules)
        icon = {"pass": "PASS", "fail": "FAIL", "skipped": "SKIP"}[status]
        print(f"\n[{icon}] {rel_path}")
        for line in messages:
            print(line)
        if status == "fail" or (status == "skipped" and args.require_all):
            overall_fail = True
        summary.append({"file": rel_path, "status": status, "messages": messages})

    print("\n" + "-" * 72)
    counts = {k: sum(1 for s in summary if s["status"] == k) for k in ("pass", "fail", "skipped")}
    print(f"结果：pass={counts['pass']} fail={counts['fail']} skipped={counts['skipped']}"
          f"　→ {'门禁未通过' if overall_fail else '门禁通过'}")
    json.dump({"summary": summary, "counts": counts, "failed": overall_fail},
              open(os.path.join(ROOT, "eval", "ci_gate_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return 1 if overall_fail else 0


if __name__ == "__main__":
    sys.exit(main())
