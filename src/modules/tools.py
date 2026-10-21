"""Agent 的两把工具：联网搜索 + 网页正文抓取。

设计要点：
- 搜索做了双通道：有 Tavily key 用 Tavily，没有就降级到免费的 DuckDuckGo；
- 抓取失败不抛异常，返回空字符串，让上层退回摘要兜底 —— Agent 必须能优雅失败。
"""
import httpx
from bs4 import BeautifulSoup

from src.config import PAGE_MAX_CHARS, TAVILY_API_KEY

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DeepResearchAgent/0.1)"}


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """联网搜索，返回 [{title, url, snippet}, ...]。"""
    if TAVILY_API_KEY:
        return _search_tavily(query, max_results)
    return _search_ddg(query, max_results)


def _search_tavily(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=TAVILY_API_KEY)
    resp = client.search(query=query, max_results=max_results)
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in resp.get("results", [])
    ]


def _search_ddg(query: str, max_results: int) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # 旧包名，兼容用
        except ImportError:
            return []

    try:
        with DDGS() as ddgs:
            rows = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href") or r.get("url") or "",
            "snippet": r.get("body", ""),
        }
        for r in rows
    ]


def fetch_page(url: str, max_chars: int = PAGE_MAX_CHARS) -> str:
    """抓取网页并提取纯文本，失败返回空字符串。"""
    try:
        resp = httpx.get(url, headers=_HEADERS, timeout=15, follow_redirects=True)
        resp.raise_for_status()
    except Exception:
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text[:max_chars]
