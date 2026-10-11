# 深度调研 Agent（Deep Research）

输入一个调研主题，自动完成「规划 → 多路检索 → 撰写」，输出一份带来源引用的 Markdown 调研简报。

- 当前进度：**M1 已就绪**（顺序流程 + 搜索/抓取工具）
- 里程碑：M2 多智能体（LangGraph）→ M3 评测与可观测

---

## 一、运行环境（已配置完成）

| 项 | 值 |
|---|---|
| 项目位置 | 桌面 `深度调研 Agent」（Deep Research）` |
| Python | 3.13（`.venv` 虚拟环境已创建） |
| 依赖 | 已安装，清单见 `requirements.txt` |

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

- **DeepSeek**（推荐，便宜）：https://platform.deepseek.com
- OpenAI / Kimi / 通义 等：改 `OPENAI_BASE_URL` 和 `MODEL_NAME` 即可（模板里有示例）

> 搜索默认走免费的 DuckDuckGo；配置 `TAVILY_API_KEY` 后自动切换到更稳定的 Tavily。

---

## 三、命令行运行

```bash
# 1. 检查环境（依赖 + 配置）
.venv\Scripts\python.exe main.py --check

# 2. 跑一次调研
.venv\Scripts\python.exe main.py "你的调研主题"
```

生成的报告保存在 `outputs/` 目录。

---

## 四、目录结构

```
├── main.py                # 命令行入口（--check / "主题"）
├── requirements.txt       # 依赖清单
├── .env.example           # 环境变量模板（复制为 .env 使用）
├── src/
│   ├── config.py          # 配置读取
│   ├── llm.py             # LLM 客户端（OpenAI 兼容接口）
│   ├── tools.py           # 工具：搜索（Tavily/DuckDuckGo）+ 网页抓取
│   ├── utils.py           # JSON 解析等小工具
│   ├── pipeline.py        # M1 主流程（对应未来 LangGraph 的图）
│   └── agents/
│       ├── planner.py     # Planner：拆解检索子问题
│       ├── researcher.py  # Researcher：搜集资料 + 抓正文
│       └── writer.py      # Writer：整合成带引用简报
└── outputs/               # 生成的报告（已 gitignore）
```

---

## 五、路线图（面试叙事线）

- [x] **M1** 顺序流程：Plan → Research → Write
- [ ] **M2** 升级 LangGraph 多智能体：Planner / Researcher（并行）/ Writer / Reviewer + 回退修订 + 人工介入
- [ ] **M3** 评测与可观测：自建评测集 + Langfuse 接入 + README 架构图与演示

---

## 六、常见问题

| 问题 | 解决 |
|---|---|
| 报错"未找到 OPENAI_API_KEY" | 还没创建 `.env`，见上文第二部分 |
| 搜索一条都搜不到 | DuckDuckGo 可能限流，重试；或配一个免费 Tavily key |
| 换了电脑 / 依赖丢失 | 重新执行 `.venv\Scripts\python.exe -m pip install -r requirements.txt` |
| PyCharm 找不到 `.venv` | 手动指定解释器路径（见上文第一步） |
