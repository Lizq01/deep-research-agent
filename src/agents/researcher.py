"""Researcher：按子问题搜集资料、抓取网页正文，产出证据列表。

M2 升级点：这里会改成并行检索（多个 Researcher 同时跑），
并接入向量库做去重与召回。
"""
from src.config import MAX_PAGES
from src.tools import fetch_page, web_search


def gather_evidence(queries: list[str], max_pages: int = MAX_PAGES) -> list[dict]:
    """对所有子问题做搜索 → 按 URL 去重 → 抓正文 → 返回证据列表。"""
    seen_urls: set[str] = set()
    candidates: list[dict] = []

    for q in queries:
        for r in web_search(q, max_results=4):
            url = r["url"]
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            candidates.append(r)

    evidence: list[dict] = []
    for r in candidates[:max_pages]:
        content = fetch_page(r["url"])
        evidence.append(
            {
                "title": r["title"],
                "url": r["url"],
                "content": content or r["snippet"],  # 抓不到正文就退回摘要兜底
            }
        )

    return evidence
