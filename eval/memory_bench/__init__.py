# -*- coding: utf-8 -*-
"""memory_bench：座舱长期记忆专用 benchmark（阶段 4）。

四件套：
- schema.py：样本格式（画像卡 → 带时间戳对话 → gold 事实位置标注 → 六题型题目）
- profiles.py：100 张画像卡（程序化组合，确定性）
- generate.py：生成评测集（`python -m eval.memory_bench.generate`）
- runner.py：检索层指标（`python -m eval.memory_bench.runner --profile both`，离线 CI 可跑）
- judge.py：题型化 LLM 判定（答案层指标，启用前需 meta-eval ≥90% 人工一致）
"""

from eval.memory_bench.judge import judge, meta_eval_report, parse_verdict  # noqa: F401
from eval.memory_bench.schema import (  # noqa: F401
    BENCH_VERSION,
    QUESTION_TYPES,
    Question,
    Sample,
    Turn,
    load_jsonl,
    save_jsonl,
    stats,
)

__all__ = [
    "BENCH_VERSION", "QUESTION_TYPES", "Sample", "Turn", "Question",
    "load_jsonl", "save_jsonl", "stats", "judge", "parse_verdict", "meta_eval_report",
]
