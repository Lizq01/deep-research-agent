# 深度调研 Agent（Deep Research）

输入一个调研主题，自动完成「规划 → 多路检索 → 撰写」，输出一份带来源引用的 Markdown 调研简报。

- 当前进度：**M3 已就绪**（评测体系 + 可观测追踪：20 条评测集 / LLM-as-Judge / 运行级 trace / Langfuse 预留接入）
- 里程碑：~~M2 多智能体（LangGraph）~~ → ~~M3-RAG 检索增强~~ → ~~M3 评测与可观测~~

---

## 一、运行环境（已配置完成）

| 项      | 值                               |
| ------ | ------------------------------- |
| 项目位置   | 桌面 `深度调研 Agent」（Deep Research）` |
| Python | 3.13（`.venv` 虚拟环境已创建）           |
| 依赖     | 已安装，清单见 `requirements.txt`      |

### 在 PyCharm 中打开

1. `File → Open`，选择本项目文件夹
2. 配置解释器：`File → Settings → Project → Python Interpreter → Add Interpreter → Add Local Interpreter → Existing`，选择  
   `.venv\Scripts\python.exe`
3. 新建运行配置：`Run → Edit Configurations → + → Python`
   - Script path：`main.py`
   - Parameters：`"2026 年 AI Agent 行业现状"`（换成你想调研的主题）
4. 点击运行

---

## 二、配置 API Key（第一次运行前必做）

把 `.env.example` 复制为 `.env`（注意文件名开头有个点），填入任意 OpenAI 兼容接口的 Key：

- **DeepSeek**（推荐，便宜）：<https://platform.deepseek.com>
- OpenAI / Kimi / 通义 等：改 `OPENAI_BASE_URL` 和 `MODEL_NAME` 即可（模板里有示例）

> 搜索默认走免费的 DuckDuckGo；配置 `TAVILY_API_KEY` 后自动切换到更稳定的 Tavily。

---

## 三、命令行运行

```bash
# 1. 检查环境（依赖 + 配置 + 真实连通性测试）
.venv\Scripts\python.exe main.py --check

# 2. 跑一次调研（默认 M2：LangGraph 多智能体）
.venv\Scripts\python.exe main.py "你的调研主题"

# 3. 非交互运行：人工审核自动通过（脚本 / CI 用）
.venv\Scripts\python.exe main.py --yes "你的调研主题"

# 4. 对照跑 M1 顺序流程
.venv\Scripts\python.exe main.py --m1 "你的调研主题"

# 5. 评测：跑 N 条主题（默认 5），产出质量报告与逐题数据
.venv\Scripts\python.exe main.py --eval          # 抽前 5 条
.venv\Scripts\python.exe main.py --eval --all    # 全量 20 条
.venv\Scripts\python.exe main.py --eval --compare  # 同题加跑 M1 对照
```

M2 流程中 Reviewer 评审通过后，会**暂停等待人工审核**：输入 `y` 保存报告、输入修改意见退回 Writer 重写、输入 `n` 放弃。

> 注意：**必须用 `.venv\Scripts\python.exe` 运行**。直接用系统 `python` 会报 `ModuleNotFoundError: No module named 'rich'`。  
> `--check` 的输出第 0 行会打印当前解释器路径，一眼就能看出是否用错了 Python。

生成的报告保存在 `outputs/` 目录。

---

## 四、目录结构

```
├── main.py                # 命令行入口（--check / "主题" / --yes / --m1 / --eval [N]）
├── requirements.txt       # 依赖清单
├── .env                   # 环境变量（含 Key，已 gitignore）
├── bge-small-zh-v1.5/     # 本地嵌入模型（已 gitignore，见第五节说明）
├── evals/
│   └── dataset.json       # 评测集：20 条主题 + 应覆盖要点清单
├── src/
│   ├── config.py          # 配置读取
│   ├── llm.py             # LLM 客户端（超时 + 重试 + 退避抖动 + trace 记录）
│   ├── tools.py           # 工具：搜索（Tavily/DuckDuckGo）+ 网页抓取
│   ├── utils.py           # JSON 解析等小工具
│   ├── tracing.py         # 可观测：运行级 trace（节点/LLM/汇总）+ Langfuse 可选上报
│   ├── pipeline.py        # M1 主流程（顺序执行，对照用）
│   ├── graph.py           # M2 主流程（LangGraph 多智能体图 + RAG 节点）
│   ├── agents/
│   │   ├── planner.py     # Planner：拆解检索子问题
│   │   ├── researcher.py  # Researcher：搜集资料 + 抓正文（M2 中并行执行）
│   │   ├── writer.py      # Writer：整合成带引用简报（支持带反馈修订）
│   │   └── reviewer.py    # Reviewer：5 维 rubric 结构化评审
│   ├── rag/               # RAG 子包（本地向量检索）
│   │   ├── chunking.py    # 段落感知分块（500 字 + 80 字重叠）
│   │   ├── embedder.py    # 本地 BGE 嵌入（bge-small-zh-v1.5，512 维）
│   │   ├── vectorstore.py # 向量索引 + 余弦检索（含 BM25 降级）
│   │   └── retriever.py   # 按子问题召回 → 聚合证据 + 统计
│   └── evals/             # 评测子包
│       ├── judge.py       # LLM-as-Judge 独立评审（与环内 Reviewer 分离）
│       └── runner.py      # 批量评测 → 确定性指标 + 评审聚合报告
└── outputs/               # 报告 / 评测产物 / traces（已 gitignore）
```

---

## 五、架构（LangGraph 多智能体 + RAG）

```mermaid
flowchart TD
  START --> planner[Planner 拆子问题]
  planner -- "Send ×N 动态并行" --> researcher[Researcher ×N<br/>搜索 + 抓正文]
  researcher --> aggregator[Aggregator<br/>URL 去重 + 截断]
  aggregator --> retriever[Retriever RAG 节点<br/>切块 → 向量化 → 按子问题召回]
  retriever --> writer[Writer 撰写 / 修订<br/>只用召回片段]
  writer --> reviewer[Reviewer 结构化评审<br/>passed / score / issues / feedback]
  reviewer -->|不通过且修订数 ≤ 2| writer
  reviewer -->|通过或达上限| human{interrupt 人工审核}
  human -->|y 通过| save[save_report 落盘] --> END
  human -->|修改意见| writer
  human -->|n 放弃| END
```

**设计要点（演进取舍）：**

- **五种模式组合**：Chaining（主链路）+ Parallelization（`Send` 动态并行检索）+ RAG（切块 / 向量化 / 语义召回）+ Evaluator-Optimizer（Writer↔Reviewer 回退修订）+ 人工介入（`interrupt`）
- **有界自主**：回退修订上限 `MAX_REVISIONS=2`（.env 可配），超限强制转人工；图执行 `recursion_limit=50` 兜底
- **优雅失败三层降级**：向量索引不可用 → BM25 关键词检索 → 全量资料拼接；搜索/抓取失败不抛异常，空证据也能走完全程并如实说明
- **LLM 调用加固**：60s 超时 + 2 次重试（指数退避 + 抖动）

### RAG 子系统（本地向量检索）

| 环节 | 实现                                                  | 参数（.env 可调）                           |
| -- | --------------------------------------------------- | ------------------------------------- |
| 分块 | 段落感知切分（句末标点优先），相邻块保留重叠                              | `CHUNK_SIZE=500` / `CHUNK_OVERLAP=80` |
| 嵌入 | 本地 bge-small-zh-v1.5（512 维，纯离线）；查询侧加 BGE 指令前缀做非对称检索 | `EMBEDDING_MODEL_PATH`                |
| 检索 | numpy 余弦相似度 Top-K，归一化后点积即相似度                        | `RAG_TOP_K=8` / `RAG_MIN_SCORE=0.35`  |
| 降级 | 模型缺失/加载失败 → BM25（标准库实现，中英混合粗切）                      | 自动                                    |
| 聚合 | 同一 URL 命中多片段合并为一条证据，保留最高分                           | 自动                                    |

**为什么索引不持久化**：调研信源是实时网页，持久化会把过期内容当成当前现状（脏数据风险），所以每次运行重建内存态索引。

**实测收益**（真实运行数据，已写入每份报告头部的 HTML 注释）：

- 运行一：8 个网页 → 105 片段 → 召回 5 条证据，上下文 38313 → 11314 字符（**省 70%**）
- 运行二：8 个网页 → 66 片段 → 召回 6 条证据，上下文 23426 → 9669 字符（**省 59%**）

**嵌入模型**：从 HuggingFace 下载 `BAAI/bge-small-zh-v1.5`（约 92MB），放到项目根目录的 `bge-small-zh-v1.5/` 文件夹即可（已 gitignore，不入库）。`RAG_ENABLED=false` 可整体关闭 RAG 退回全量拼接。

---

## 六、评测与可观测（M3）

### 评测体系（`evals/` + `src/evals/`）

自建 20 条主题评测集（`evals/dataset.json`，领域混合：技术 / 行业 / 协议 / 安全 / 算力），
每条主题带「应覆盖要点清单」。指标分两层，刻意区分**可复现性**：

| 层 | 指标 | 口径 |
|---|---|---|
| 确定性（无 LLM） | 引用有效率 | 正则校验 `[编号]` 是否落在 1~N 证据范围内 |
| 确定性 | 证据引用覆盖 / 证据数 / 报告长度 | 直接计数 |
| 确定性 | RAG 上下文压缩率 / 耗时 / token / 重试 | 来自 trace |
| LLM 评审 | 覆盖度 / 忠实度 / 结构 / 洞察 + 总分 | LLM-as-Judge |

**评审器独立性**：评测用的 Judge 与流水线内的 Reviewer 是两套 prompt、两个角色——
Reviewer 在环内反馈给 Writer 修订；Judge 在环外只打分，且拿着评测集的要点清单
逐项核对。避免"自己写的自己评"。

产物（`outputs/`）：`eval_report_*.md`（聚合报告）、`eval_results_*.json`（逐题明细）。

### 可观测（`src/tracing.py`）

每次运行自动产出 trace 文件（`outputs/traces/run_*.json`），记录：

- **节点级**：各图节点耗时与关键产物（RAG 统计、评审得分、修订轮次、报告路径）
- **LLM 级**：每次调用的模型、延迟、输入/输出 token、重试次数、错误
- **运行级**：总耗时、token 汇总、运行状态

运行结束会在控制台打印摘要（`本次运行：耗时 · LLM 调用 · token · trace 路径`）。

**Langfuse 接入（可选，预留位）**：在 `.env` 中配置以下三项即自动开启上报
（Ingestion API 直连，无需安装 SDK；上报失败只告警不阻塞主流程）：

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # 自部署则改成自己的地址
```

未配置时一切照常，本地 trace 独立工作。`TRACE_ENABLED=false` 可整体关闭落盘。

---

## 七、演进路线

- [x] **M1** 顺序流程：Plan → Research → Write
- [x] **M2** 升级 LangGraph 多智能体：Planner / Researcher（并行）/ Writer / Reviewer + 回退修订 + 人工介入
- [x] **M3-RAG** 本地向量检索：bge-small-zh-v1.5 嵌入 + 分块索引 + 按子问题语义召回（上下文省 59%~70%）
- [x] **M3 评测与可观测**：20 条评测集 + LLM-as-Judge 独立评审 + 运行级 trace（Langfuse 可选接入）

### 首轮评测实测（5 条主题，2026-10-14）

| 指标 | 结果 |
|---|---|
| 完成率 | 5/5（0 失败、0 评审解析失败） |
| Judge 总分 | 8.2/10（四维：覆盖 8.8 / 忠实 8.2 / 结构 9.2 / 洞察 7.8） |
| 引用有效率 | **257/257 = 100%**（确定性指标：[编号] 全部落在证据范围内） |
| RAG 上下文压缩 | 70%~75% |
| 平均耗时 / token | 37.5s / 15264 token（单主题端到端） |
| 回退修订 | t05 真实触发 2 轮修订闭环（7 次 LLM 调用，评审 7→通过） |

> 复现：`.venv\Scripts\python.exe main.py --eval`；全量 20 条加 `--all`；M1 对照加 `--compare`。
> 最弱维度是**洞察（7.8）**——评审摘录普遍指出"资料归纳多于增量分析"，这是后续
> 优化 Writer prompt 或引入多轮反思的方向，已有数据支撑决策。

---

## 八、常见问题

| 问题                                              | 解决                                                                                    |
| ----------------------------------------------- | ------------------------------------------------------------------------------------- |
| 报错"未找到 OPENAI_API_KEY"                          | 还没创建 `.env`，见上文第二部分                                                                   |
| 搜索一条都搜不到                                        | 检查是否配了 `TAVILY_API_KEY`（DuckDuckGo 在国内网络常被限流）；跑 `main.py --check` 看第 3 节连通性测试结果       |
| 报 `ModuleNotFoundError: No module named 'rich'` | 用错解释器了，改用 `.venv\Scripts\python.exe`，或在 PyCharm 里把解释器切到 `.venv`                       |
| 运行到一半"卡住不动"                                     | 正常现象：M2 在人工审核处等待输入（`y` / 修改意见 / `n`）。不想交互就加 `--yes`                                   |
| 填了 key 还是不生效                                    | 确认填的是项目根目录的 `.env`，填完保存后再跑 `main.py --check` 验证                                       |
| 日志显示"降级为 BM25"                                  | 嵌入模型目录找不到或加载失败：确认 `bge-small-zh-v1.5/` 在项目根目录（或 .env 里 `EMBEDDING_MODEL_PATH` 指向正确路径） |
| 首次运行 Retriever 阶段较慢                             | 嵌入模型首次加载约 30~40 秒（之后同进程内复用，毫秒级），属正常现象                                                 |
| 换了电脑 / 依赖丢失                                     | 重新执行 `.venv\Scripts\python.exe -m pip install -r requirements.txt`，并重新下载嵌入模型          |
| PyCharm 找不到 `.venv`                             | 手动指定解释器路径（见上文第一步）                                                                     |
