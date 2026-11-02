"""全局配置：统一从 .env 读取环境变量。"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]  # src/config/settings.py → 项目根目录
load_dotenv(ROOT_DIR / ".env")

# ===== LLM（任何 OpenAI 兼容接口）=====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat").strip()

# ===== 搜索 =====
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

# ===== 运行参数 =====
MAX_QUERIES = int(os.getenv("MAX_QUERIES", "4"))
# 信源预算：实测（2026-10-14，5 条评测题）放宽到 12 页后，证据数并未增加
# （7.4 → 7.2，被 RAG_TOP_K=8 的召回上限卡住），而 token +67%、耗时 +65%。
# 结论：单独放宽抓页数是纯成本，真正的瓶颈在"召回上限"，见 RAG_TOP_K。
MAX_PAGES = int(os.getenv("MAX_PAGES", "8"))
PAGE_MAX_CHARS = int(os.getenv("PAGE_MAX_CHARS", "6000"))

# ===== M2 多智能体 =====
PAGES_PER_QUERY = int(os.getenv("PAGES_PER_QUERY", "2"))  # 每个并行 Researcher 抓几页正文
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))      # Writer↔Reviewer 回退修订上限（C5 有界自主）
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))       # 单次 LLM 调用超时（秒）
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))  # LLM 调用失败重试次数

# ===== M3 RAG =====
# 本地嵌入模型目录（相对项目根目录，或填绝对路径）
EMBEDDING_MODEL_PATH = (
    os.getenv("EMBEDDING_MODEL_PATH", "bge-small-zh-v1.5").strip() or "bge-small-zh-v1.5"
)
RAG_ENABLED = os.getenv("RAG_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))       # 分块目标字符数
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))  # 相邻块重叠字符数
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))           # 每个查询召回的段落数
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.35"))  # 相似度下限，低于此值不召回

# ===== M3 可观测 =====
TRACE_ENABLED = os.getenv("TRACE_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
# Langfuse 上报（可选）：两个 key 都配置后才启用；未配置时只写本地 trace 文件
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()

# ===== MCP 知识库工具层 =====
# 开关：关掉后 knowledge 节点直接跳过（图结构不变，等价于没有内部知识库）
MCP_ENABLED = os.getenv("MCP_ENABLED", "true").strip().lower() in {"1", "true", "yes"}

# 服务端项目目录：默认取本项目的同级目录 rag-knowledge-chat
MCP_SERVER_CWD = (
    os.getenv("MCP_SERVER_CWD", "").strip() or str(ROOT_DIR.parent / "rag-knowledge-chat")
)
# 启动命令：留空则用服务端项目自己的 .venv 解释器（服务端依赖 fastembed / faiss，
# 不能用本项目 venv 的 python，否则 import 失败）
MCP_SERVER_COMMAND = os.getenv("MCP_SERVER_COMMAND", "").strip()
# 启动参数：默认 `python -m mcp_server`（stdio）
MCP_SERVER_ARGS = tuple(os.getenv("MCP_SERVER_ARGS", "-m mcp_server").split())

MCP_CONNECT_TIMEOUT = float(os.getenv("MCP_CONNECT_TIMEOUT", "30"))  # 握手超时（秒）
MCP_CALL_TIMEOUT = float(os.getenv("MCP_CALL_TIMEOUT", "60"))        # 单次工具调用超时

# 检索预算：子问题数 / 每个问题召回片段数 / 最多并入多少条内部证据
MCP_MAX_QUERIES = int(os.getenv("MCP_MAX_QUERIES", "3"))
MCP_TOP_K = int(os.getenv("MCP_TOP_K", "3"))
MCP_MAX_EVIDENCE = int(os.getenv("MCP_MAX_EVIDENCE", "3"))

# 服务端 stderr 的去处：默认落文件（协议流与日志分离），设 MCP_STDERR=inherit 直接打屏
MCP_STDERR_INHERIT = os.getenv("MCP_STDERR", "").strip().lower() in {"1", "true", "yes", "inherit"}
MCP_SERVER_LOG = Path(
    os.getenv("MCP_SERVER_LOG", "").strip() or str(ROOT_DIR / "outputs" / "mcp_server.log")
)


def resolve_mcp_server_command() -> str:
    """解析服务端启动命令：显式配置优先，否则找服务端项目的 .venv。"""
    if MCP_SERVER_COMMAND:
        return MCP_SERVER_COMMAND
    cwd = Path(MCP_SERVER_CWD)
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        candidate = cwd / rel
        if candidate.exists():
            return str(candidate)
    return sys.executable


OUTPUT_DIR = ROOT_DIR / "outputs"
TRACE_DIR = OUTPUT_DIR / "traces"
EVAL_DATASET = ROOT_DIR / "evals" / "dataset.json"

# 评测基线（入库，用于回归对比：改一次代码就知道是变好还是变坏）
BASELINE_DIR = ROOT_DIR / "evals" / "baselines"
BASELINE_FILE = BASELINE_DIR / "baseline.json"


def resolve_embedding_model_path() -> Path:
    """把 EMBEDDING_MODEL_PATH 解析为绝对路径（相对路径按项目根目录解析）。"""
    p = Path(EMBEDDING_MODEL_PATH)
    return p if p.is_absolute() else (ROOT_DIR / p)
