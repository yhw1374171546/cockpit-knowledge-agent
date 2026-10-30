# 记忆检索层 benchmark（自动生成）

> 口径：gold 事实召回率（IE/MR/KU=上下文含 gold 值；TR=演化史完整；红线=无条件注入）。
> 生成：`python eval/memory_bench/generate.py`（程序化、确定性、零成本）；
> 重跑：`python eval/memory_bench/runner.py --profile both`。

## 1. 总体

| 后端 | 样本 | 题目 | 命中 | 召回率 | 红线注入 |
| --- | --- | --- | --- | --- | --- |
| v2 | 15 | 75 | 65 | **86.7%** | 40.0% |

## 2. 按题型

| 后端 | IE | MR | KU | KU干净 | TR |
| --- | --- | --- | --- | --- | --- |
| v2 | 100.0% | 100.0% | 80.0% | 80.0% | 53.3% |

## 3. 口径说明

- TR：v1 的 overwrite 不保留旧值，`history()` 如实返回空 → TR 必然失败；这是 v1/v2 的已知差异，不是评测 bug。
- KU 干净率：最新值在上下文且**旧值不在**。ADD-only 后端（mem0 语义档）会把新旧值一起返回 → 干净率低，答案层会被旧值干扰。
- mem0 语义档 = 按 Mem0 公开语义的本地复现（ADD-only + 无业务 schema + 无冲突消解 + threshold=0.1），**不是 Mem0 产品调用**——用来回答'如果开箱用会发生什么'。
- ABS（拒答）与答案层指标见 `answer_runner.py` / `judge.py`，检索层不适用。
