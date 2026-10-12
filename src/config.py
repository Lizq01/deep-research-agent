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

OUTPUT_DIR = ROOT_DIR / "outputs"
