# legacy/ —— 课程原始 RAG 链路（对照基线，不参与主线）

> **一句话**：这里是本项目**改造之前的课程版本**代码，保留它只为两件事：
> ① **对照**——新链路的效果提升要能和它比（README 的消融表里"原始链路（课程实现）"那一栏就是它）；
> ② **溯源**——说明"哪些是我做的、哪些是原本就有的"，避免把基线代码算成自己的产出。
>
> **主线代码不在这里**：`agent/`（RAG 底层 / Agent / 护栏 / 多 Agent / 服务化）、`eval/`（评测与门禁）。
> **CI 与静态检查都不覆盖本目录**（`pyproject.toml` 的 `[tool.ruff] extend-exclude` 已排除），
> 因为这里的代码是上游原样保留，不打算改。

## 里面有什么

| 文件 | 作用 | 现状 |
| --- | --- | --- |
| `pdf_parse.py` | 原始 PDF 解析 → 知识库（8,785 块，**页码元数据 0%**） | 被 `agent/kb.py` 的三策略解析取代 |
| `bm25_retriever.py` | 原始 BM25 检索（调库，非自实现） | 被 `agent/kb.py` 的自实现 BM25 取代 |
| `faiss_retriever.py` | 原始向量检索（m3e + FAISS） | 保留思路，重写进 `agent/kb.py` 双路召回 |
| `rerank_model.py` | 原始精排（bge-reranker） | 保留，接进 `agent/kb.py` 的 Top-6 精排 |
| `vllm_model.py` | 原始 vLLM 推理封装 | 被 `agent/llm.py` 的 Provider 抽象取代（vllm/deepseek/openai） |
| `qwen_generation_utils.py` | Qwen 生成工具 | 同上 |
| `config.py` | 原始全局配置（含 `LLM_DEVICE` 等） | 被 `agent/service/config.py`（环境变量驱动）取代 |
| `run.py` | 原始端到端跑批（4 路消融 → `data/result.json`） | 被 `eval/run_eval.py` 等取代 |
| `test_score.py` | 原始评测（默认读不存在的 `gold2.json`，跑不通） | 被 `eval/evaluate.py`（参数化）取代 |
| `build.sh` / `run.sh` | 原始镜像构建与启动脚本 | 被 `Dockerfile.service` + `docker-compose.yml` 取代 |

## 如果要跑

这些脚本按**课程环境**写的（相对路径、依赖 `config.py`、部分需要 GPU）。需要在**本目录内**执行
（`from config import *` 依赖脚本目录在 `sys.path` 里）：

```bash
cd legacy
python pdf_parse.py     # 仅解析 PDF，无 GPU 也可跑
python run.py           # 端到端 4 路跑批，需要模型与 GPU
```

> ⚠️ 它们**不是本项目的主线入口**。主线入口见 README §快速开始：
> `python -m agent.run_agent ...`、`uvicorn agent.service.app:app`、`python eval/ci_gate.py`。
