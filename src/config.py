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

OUTPUT_DIR = ROOT_DIR / "outputs"
