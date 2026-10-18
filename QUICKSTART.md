# 快速开始（5 分钟跑起来）

> 这份是**怎么跑**；项目做了什么、为什么这么设计，见 [README.md](README.md)。
> 下面每条命令都**在 Windows + Python 3.13 上实测过**，输出也照实写了，方便你判断"我这边对不对"。

---

## 0. 前置条件

| 项目 | 要求 | 说明 |
| --- | --- | --- |
| Python | **3.10+**（实测 3.12 / 3.13） | 用到了 `zip(strict=)`，3.10 以下不支持 |
| GPU | **不需要** | 主线全 CPU 可跑；向量路/真模型是可选能力 |
| 语料 | 没准备也能跑 | 仓库自带 **24 块合成语料**，克隆即可跑 |
| 网络 | 不需要 | 所有测试/冒烟都是离线、零成本 |

### ⚠️ 关于中文乱码：**不要设 `PYTHONIOENCODING`**

这一条我一开始写错了，实测踩坑后改正。Windows 中文控制台的代码页是 **936（GBK）**：

| 你的设置 | Python 输出 | 控制台解码 | 结果 |
| --- | --- | --- | --- |
| **什么都不设（推荐）** | GBK | GBK | ✅ 中文正常 |
| 设 `PYTHONIOENCODING=utf-8`，但**没** `chcp 65001` | UTF-8 | GBK | ❌ 乱码：`鍏ㄩ噺娴嬭瘯` |
| `chcp 65001` **且** 设 `PYTHONIOENCODING=utf-8` | UTF-8 | UTF-8 | ✅ 正常（要和 CI 完全一致时用） |

```powershell
# 推荐：直接跑，什么环境变量都不用设
cd 综合项目实战项目一
python eval/run_all_tests.py
```

> 脚本内部已经做了兼容（`agent/console.py::harden_stdout`）：
> **终端**保持系统编码（中文正常显示）、**重定向/管道**强制 UTF-8（方便 CI 与日志工具解析），
> 并且把 GBK 编不出的符号（`✅` 这类）替换成 ASCII 标记，避免 `UnicodeEncodeError` 崩在最后一行。
> 所以你**不需要**为编码做任何配置。

---

## 1. 安装依赖（约 1 分钟）

```powershell
cd 综合项目实战项目一
python -m pip install -r requirements.txt
```

`requirements.txt` 只有 8 个包（jieba / pdfplumber / fastapi / uvicorn / sqlalchemy / alembic / pyjwt / httpx）。
**不需要 GPU**；想要向量召回再装可选的：

```powershell
python -m pip install -r requirements-optional.txt   # faiss-cpu + numpy
```

> ⚠️ 不要装 `requirements-gpu.txt`——那是**课程原始链路**（`legacy/`）和真 vLLM 推理的依赖，
> 体积大、对 CUDA 敏感，与主线无关。

**验证安装**：

```powershell
python -c "import jieba, pdfplumber, fastapi, sqlalchemy, alembic, jwt; print('依赖 OK')"
```

---

## 2. 三分钟建立信心：跑测试

```powershell
python eval/run_all_tests.py
```

**期望输出**（照抄实际运行结果）：

```
  test_guardrails.py             OK        37 项
  test_regressions.py            OK        29 项
  test_offline.py                OK        28 项
  test_fc_pipeline.py            OK        20 项
  test_multiagent.py             OK        26 项
  test_vector_store.py           OK        10 项
  test_sse.py                    OK        12 项
  test_service.py                OK        72 项
  test_tasks_and_traces.py       OK        13 项
  test_migrations.py             OK         6 项
------------------------------------------------------------
合计 253 项 ｜ 套件 10/10 通过
[OK] 全部通过
```

> **第一次跑会多等约 1 秒**：jieba 在构建分词缓存（`Loading model from cache ...`），属正常。

再跑两个"质量门禁"，确认指标没退化：

```powershell
python eval/ci_gate.py            # 期望：结果：pass=6 fail=0 → 门禁通过
python eval/ci_smoke.py           # 期望：冒烟结果：10/10 通过
python eval/eval_extra_smoke.py   # 期望：补充冒烟结果：26/26 通过
```

---

## 3. 问第一个问题（单轮问答，看完整链路）

```powershell
python -m agent.run_agent "座椅加热怎么关闭"
```

**期望输出**（节选）：

```
==============================================================================
Q: 座椅加热怎么关闭
路由: manual_qa   状态: answered   步数: 3   重试: 0   总耗时: 24.7 ms
工具调用: search_manual(ok)
A: 您可以通过中央显示屏，设置驾驶员/副驾驶员侧座椅加热强度或关闭座椅加热功能。
出处: [train_a.pdf 第114页] [train_a.pdf 第114页] [train_a.pdf 第113页] [train_a.pdf 第113页]
反思: grounded（接地率 1.0，动作 accept）
```

**这段输出本身就是个演示**：路由 → 调工具 → 回答 → **给出出处（页码）** → 接地校验通过。

再试几个能看出差别的问法：

```powershell
python -m agent.run_agent "胎压报警了怎么办"                 # 安全类：应给停驶指令
python -m agent.run_agent "中国足球的队长是谁"               # 域外问题：应拒答（无答案）
python -m agent.run_agent --multi-turn                       # 多轮：指代消解演示
python -m agent.run_agent                                    # 不传问题：跑内置演示集
```

---

## 4. 跑服务（真实 HTTP + SSE，最接近线上）

需要**两个终端**。

**终端 A —— 起服务**：

```powershell
$env:AGENT_KB_PATH = "eval/fixtures/mini_corpus.jsonl"     # 用内置合成语料，启动快
$env:SERVICE_JWT_SECRET = "dev-secret-change-me"           # 本地开发用；生产必须换
python -m uvicorn agent.service.app:app --port 8077
```

启动成功的标志（日志里能看到）：

```
服务启动完成 db=sqlite kv=memory llm=planner cache=shared
Uvicorn running on http://127.0.0.1:8077
```

**终端 B —— 端到端冒烟（36 项检查）**：

```powershell
python eval/service_smoke.py --url http://127.0.0.1:8077
```

**期望**：`冒烟结果：36/36 通过 → 通过`

这 36 项覆盖：健康检查、鉴权、租户隔离、限流 429、配额、幂等重放、SSE 事件顺序与 TTFT、
**语义缓存命中**、写操作确认、审计、指标、**知识库版本灰度**。

**手动玩一下（可选）**：

```powershell
# 取 token
$tok = (Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8077/v1/auth/token `
        -ContentType "application/json" `
        -Body '{"user_id":"me","role":"service","vehicle_model":"lynk08"}').access_token

# 普通问答
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8077/v1/chat `
  -Headers @{Authorization="Bearer $tok"} -ContentType "application/json" `
  -Body '{"message":"座椅加热怎么关闭"}'

# 流式（SSE，能看到事件带 id: 与 retry:）
curl.exe -N -X POST http://127.0.0.1:8077/v1/chat/stream `
  -H "Authorization: Bearer $tok" -H "Content-Type: application/json" `
  -d "{\"message\":\"胎压报警了怎么办\"}"
```

---

## 4.5 打开演示页面（推荐：比 curl 直观得多）

服务起来后，浏览器直接打开：

```
http://127.0.0.1:8077/
```

**不用登录**——页面会自动向 `/v1/auth/token` 申请 token（默认 `demo / driver / lynk08`）。
想用版本管理功能，点右上角「设置」把角色改成 `service`。

页面上有几块，都对应真实接口：

| 区域 | 能看出什么 |
| --- | --- |
| 对话区（打字机效果） | SSE 流式；顶部徽标显示 **TTFT**、`answered / refused / 待确认`、**命中语义缓存**、`降级` |
| **引用出处** | 每轮答案下方是 `文件名 · 第 N 页`，右侧面板同步列出——**可溯源** |
| 右侧「本轮追踪」 | 状态 / TTFT / 总延迟 / token / 成本 / 风险等级 / 注入标记 |
| 右侧「工具调用」 | 调了哪个工具、耗时、是否**被拦截**、参数是否**被修复** |
| 右侧「知识库版本」 | 当前版本、灰度比例（拖滑块 + 点版本行即可切换）、**回滚** |
| 快捷问题 | 含一条域外问题（"中国足球的队长是谁"）→ 用来演示**拒答** |

**会自动验证页面接口是否对得上**（不需要浏览器）：

```powershell
python eval/ui_contract_check.py --url http://127.0.0.1:8077
# 期望：接口契约：29/29 通过
```

> 这个脚本按页面里的调用顺序逐个请求，并断言页面实际读取的每个字段都存在。
> 纯静态页最容易出的问题是"页面读 `tool_error`、后端给 `reason`"，在浏览器里只表现为空白，很难查。

---

## 5. 两个"值得单独看"的演示

### 5.1 语义缓存到底省了多少（走真实 HTTP 链路）

```powershell
python eval/cache_bench.py
```

**期望**：

```
 冷启动     10 题   9 可答   0 命中   P50  9.12 ms   输出 token 455
 命中缓存   10 题   9 可答   9 命中   P50  6.96 ms   输出 token  94
缓存命中率（第二轮）：9/10 = 90.0%
结论： [OK] 缓存已真实生效（命中即跳过检索与大模型调用）
```

> **看 `输出 token` 那一列**：命中时归零——这说明它真的跳过了模型调用，而不只是"查了下缓存"。

### 5.2 换手册：注册 → 灰度 → 回滚（不用重启、不用改配置）

```powershell
python eval/kb_versioning_demo.py
```

**期望**：

```
[1] 当前生效版本：v1-7e8a811d
[2] 注册新版本：v2-3e522390 （25 块）→ 注册**不改变**线上路由
[3] 灰度 50%：分流结果 {"v1-7e8a811d": 12, "v2-3e522390": 8}
    同一用户连续三问的版本集合：{v1-7e8a811d} → [OK] 稳定（不会在会话中跳版）
[5] 回滚（一条命令，不需要重启）→ v1-7e8a811d
```

---

## 6. 数据库迁移（改表结构时用）

```powershell
python -m alembic upgrade head              # 建表 / 升到最新
python -m alembic upgrade head --sql        # 只生成 SQL，不连库（CI 里用它校验）
python -m alembic downgrade -1              # 回滚一版
python -m alembic current                   # 看当前版本
```

> 应用启动时默认会 `create_all()`（开发方便），但它**只建不改**。
> 生产请设 `SERVICE_AUTO_CREATE_SCHEMA=0`，只走 `alembic upgrade head`。

---

## 7. 常见问题（都是我实际踩过的）

| 现象 | 原因 | 怎么办 |
| --- | --- | --- |
| 输出中文全是乱码（如 `鍏ㄩ噺娴嬭瘯`） | **设了 `PYTHONIOENCODING=utf-8` 但控制台还是 GBK**（代码页 936） | **删掉这个环境变量**：`Remove-Item Env:PYTHONIOENCODING`；或改用 `chcp 65001`（两者必须配对） |
| `UnicodeEncodeError: 'gbk' codec can't encode character '\u2705'` | GBK 编不出 `✅` 这类符号，`print` 直接抛异常（旧脚本的老问题） | 已通过 `agent/console.py::harden_stdout` 修掉；如仍出现说明脚本没接入，告诉我脚本名 |
| `ModuleNotFoundError: sqlalchemy` | 没装依赖 | `python -m pip install -r requirements.txt` |
| `TypeError: zip() takes no keyword arguments` | Python < 3.10 | 升级到 3.10+ |
| 第一次跑卡 1~2 秒，日志有 `Loading model from cache` | jieba 构建分词缓存 | 正常，只发生一次 |
| `Address already in use` / 端口被占 | 8077 被占用 | 换端口：`--port 8078`，冒烟也换成对应 URL |
| `database is locked`（SQLite） | 多进程同时写同一个 `.db` | 停掉其他实例；或换 `SERVICE_DATABASE_URL` 指向新文件 |
| 服务起不来，报 `文件不存在：pre_train_model` | 想开向量路但没模型权重 | **不用管**：向量路是可选的，缺模型会自动降级为纯 BM25 |
| `alembic` 报 `UnicodeDecodeError: 'gbk'` | `alembic.ini` 被加了中文注释 | ini 必须纯 ASCII（本仓库已保证）；中文说明放代码注释里 |
| 冒烟里 `分多块下发` 失败 | 第二次问同一句命中了缓存 → 只发 1 个 delta | 这是**预期行为**；检查项已改为"多块 **或** 缓存命中" |

---

## 8. 想进一步玩（可选，需要额外准备）

| 想做的事 | 怎么做 | 需要什么 |
| --- | --- | --- |
| 用**完整语料**（9,301 块 / 579 万字） | 直接用仓库里的 `kb/chunks.jsonl`；或 `$env:AGENT_KB_PATH="kb/chunks.jsonl"` | 语料文件（未入库，见 `.gitignore`） |
| 用**真实大模型** | 配 `DEEPSEEK_API_KEY` 或指向 vLLM 的 `--backend openai --base-url ...` | API Key 或 GPU |
| 开**向量召回** | 准备 `pre_train_model/m3e-large`，然后 `KB.load(..., with_vector=True)` | 模型权重（CPU 也能跑，慢） |
| 跑**注入对抗集** | `python eval/injection_eval.py` | 无（离线） |
| 跑**多轮/口语化/边界**评测 | `python eval/multiturn_eval.py` / `colloquial_eval.py` / `edge_cases.py` | 完整语料更真实 |
| 压测 | `python eval/load_test.py --spawn --concurrency 1,10,25 --requests 200` | 无 |

---

## 9. 目录导航（先看这几个就够）

```
综合项目实战项目一/
├── agent/                    ← 主线代码
│   ├── kb.py                 ★ RAG 底座：三策略解析 / 双路召回 / RRF / 精排 / 增量索引
│   ├── graph.py              ★ Agent 状态机：Router→Planner→Executor→Reflector→Answer
│   ├── tools.py              ★ 工具注册表（Function Calling schema / 白名单隔离）
│   ├── memory.py             ★ 记忆与指代消解（错误继承 0%）
│   ├── reflection.py         ★ 接地校验 + 引用三元组校验
│   ├── cache.py              ★ 语义缓存（车况指纹 + 反义守卫）
│   ├── vector_store.py       ★ 向量索引持久化 + 增量
│   ├── orchestrator.py       ★ Supervisor 多 Agent 编排
│   ├── agents/               四个专家（手册 / 车控 / 服务 / 安全评审）
│   ├── guardrails/           注入检测 / 策略引擎 / 预算
│   ├── service/              ★ 服务层（FastAPI / SSE / 幂等 / 限流 / 任务队列 / 迁移）
│   └── tests/                10 个测试套件（253 项）
├── eval/                     ← 评测与门禁
│   ├── run_all_tests.py      ★ 一次跑完全部测试
│   ├── ci_gate.py            ★ 指标门禁（退化即拦）
│   ├── service_smoke.py      端到端冒烟（36 项）
│   ├── cache_bench.py        语义缓存真实收益
│   ├── kb_versioning_demo.py 换手册：灰度 + 回滚
│   └── fixtures/             内置合成语料（克隆即可跑）
├── migrations/               ← Alembic 迁移（3 个版本）
├── legacy/                   课程原始链路（对照基线，不参与主线，CI 不覆盖）
├── Dockerfile.service        Agent 服务镜像
├── docker-compose.yml        api + postgres + redis 编排
└── requirements.txt          核心依赖（8 个包，CPU 可跑）
```

---

## 10. 一分钟自检清单

复制粘贴跑完，全绿就说明环境没问题：

```powershell
# 不需要设任何编码环境变量（脚本自己处理；见第 0 节）
python -m pip install -q -r requirements.txt
python eval/run_all_tests.py           # 253 项全过
python eval/ci_gate.py                 # 门禁 6/6
python eval/ci_smoke.py                # 冒烟 10/10
python -m agent.run_agent "座椅加热怎么关闭"   # 能给出带页码的答案
python -m alembic upgrade head --sql   # 迁移脚本可解析
```

有问题先看第 7 节；还有问题把**完整报错**发我，我按这条链一路查下去。
