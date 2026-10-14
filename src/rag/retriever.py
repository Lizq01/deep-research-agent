"""RAG · 检索器：按调研子问题召回相关段落，替代原先的全文拼接。

对外只暴露两个函数：
- build_index(docs, log)：把抓到的网页切块建索引（含降级逻辑）
- retrieve_context(store, queries, log)：按每个子问题召回段落，聚合去重后返回

降级契约：任何一步失败都退回"原始网页正文"，保证 Writer 永远有料可写。
"""
import logging

from src.config import RAG_ENABLED, RAG_TOP_K
from src.rag.chunking import chunk_documents
from src.rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)


def build_index(docs: list[dict], log=print) -> VectorStore | None:
    """切块 + 建索引。RAG 关闭或没有资料时返回 None（上层走全量拼接）。"""
    if not RAG_ENABLED:
        log("      RAG 已关闭（.env 里 RAG_ENABLED=false），使用全量资料拼接")
        return None
    if not docs:
        return None

    chunks = chunk_documents(docs)
    store = VectorStore()
    store.build(chunks)

    mode_label = {
        "embedding": "向量检索",
        "bm25": "BM25 关键词检索（降级）",
        "empty": "空索引",
    }.get(store.mode, store.mode)
    log(f"      RAG 索引：{len(chunks)} 个片段（来自 {len(docs)} 个网页）· 模式：{mode_label}")
    return store


def retrieve_context(
    store: VectorStore | None,
    queries: list[str],
    docs: list[dict],
    top_k: int = RAG_TOP_K,
    log=print,
) -> tuple[list[dict], dict]:
    """按子问题召回，返回 (evidence, stats)。

    evidence 保持既有 Writer 契约：[{title, url, content}]。
    同一 URL 命中多个片段时合并为一个证据条目，避免 Writer 看到重复内容。
    stats 供观测使用：{mode, chunks, hits, input_chars, output_chars}。
    """
    if store is None or len(store) == 0:
        total = sum(len(d.get("content", "")) for d in docs)
        return docs, {
            "mode": "full",
            "chunks": 0,
            "hits": len(docs),
            "input_chars": total,
            "output_chars": total,
        }

    # 按 URL 聚合命中片段，保留最高分
    by_url: dict[str, dict] = {}
    order: list[str] = []
    for query in queries:
        for hit in store.search(query, top_k=top_k):
            url = hit.get("url", "")
            if url not in by_url:
                by_url[url] = {"title": hit.get("title", ""), "url": url, "parts": [], "score": hit["score"]}
                order.append(url)
            entry = by_url[url]
            entry["score"] = max(entry["score"], hit["score"])
            if hit["text"] not in entry["parts"]:
                entry["parts"].append(hit["text"])

    evidence = [
        {
            "title": by_url[u]["title"],
            "url": u,
            "content": "\n…\n".join(by_url[u]["parts"]),
        }
        for u in order
    ]

    input_chars = sum(len(d.get("content", "")) for d in docs)
    output_chars = sum(len(e["content"]) for e in evidence)
    stats = {
        "mode": store.mode,
        "chunks": len(store),
        "hits": len(evidence),
        "input_chars": input_chars,
        "output_chars": output_chars,
    }

    saved = input_chars - output_chars
    pct = (saved / input_chars * 100) if input_chars else 0.0
    log(
        f"      召回 {len(evidence)} 条证据（{len(store)} 片段中命中）"
        f"· 上下文 {input_chars} → {output_chars} 字符（省 {pct:.0f}%）"
    )
    return evidence, stats
