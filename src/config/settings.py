"""全局配置：统一从 .env 读取环境变量。"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# ===== LLM（任何 OpenAI 兼容接口）=====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat").strip()

# ===== 搜索 =====
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

# ===== 运行参数 =====
MAX_QUERIES = int(os.getenv("MAX_QUERIES", "4"))
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

OUTPUT_DIR = ROOT_DIR / "outputs"
TRACE_DIR = OUTPUT_DIR / "traces"
EVAL_DATASET = ROOT_DIR / "evals" / "dataset.json"


def resolve_embedding_model_path() -> Path:
    """把 EMBEDDING_MODEL_PATH 解析为绝对路径（相对路径按项目根目录解析）。"""
    p = Path(EMBEDDING_MODEL_PATH)
    return p if p.is_absolute() else (ROOT_DIR / p)
