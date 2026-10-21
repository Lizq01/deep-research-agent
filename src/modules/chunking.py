"""RAG · 文档分块：把网页正文切成适合向量检索的段落。

设计要点：
- 段落感知优先：先在句末/换行处切，避免把一个完整论断拦腰砍断；
- 相邻块保留重叠区（overlap），防止跨块的上下文信息丢失；
- 超长段落再降级为按句号切分，保证任何输入都能终止（不会死循环）。
"""
import re

from src.config import CHUNK_OVERLAP, CHUNK_SIZE

# 中文句末标点 + 英文句末标点 + 换行，作为优先切分点
_SENTENCE_END = re.compile(r"(?<=[。！？；!?;])\s*|\n+")


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """把长文本切成若干段，每段不超过 chunk_size 字符，相邻段重叠 overlap 字符。"""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if overlap >= chunk_size:
        # 防呆：重叠不能大于等于块长，否则会无限推进
        overlap = max(0, chunk_size // 5)

    chunks: list[str] = []
    buf = ""
    for sentence in _split_sentences(text):
        # 单句就超长（如无标点的长文本），先冲刷缓冲，再硬切这一句
        if len(sentence) > chunk_size:
            if buf:
                chunks.append(buf.strip())
                buf = ""
            chunks.extend(_hard_split(sentence, chunk_size, overlap))
            continue

        if len(buf) + len(sentence) <= chunk_size:
            buf += sentence
        else:
            chunks.append(buf.strip())
            # 新块以尾部 overlap 字符开头，保留跨块上下文
            buf = (buf[-overlap:] if overlap else "") + sentence

    if buf.strip():
        chunks.append(buf.strip())

    return [c for c in chunks if c]


def _split_sentences(text: str) -> list[str]:
    """按句末标点切分，保留标点本身。"""
    parts: list[str] = []
    for piece in _SENTENCE_END.split(text):
        if not piece:
            continue
        # 标点被分隔符吃掉时补回（re.split 的 lookbehind 保留标点在上一段）
        parts.append(piece if piece.endswith(("。", "！", "？", "；", "!", "?", ";")) else piece)
    return parts


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """无有效标点时的兜底硬切。"""
    step = max(1, chunk_size - overlap)
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


def chunk_documents(docs: list[dict]) -> list[dict]:
    """把 [{title,url,content,...}] 切块，产出带溯源信息的 chunk 列表。

    返回 [{"text", "url", "title", "chunk_idx", "query"}, ...]
    """
    chunks: list[dict] = []
    for doc in docs:
        pieces = chunk_text(doc.get("content", ""))
        for idx, piece in enumerate(pieces):
            chunks.append(
                {
                    "text": piece,
                    "url": doc.get("url", ""),
                    "title": doc.get("title", ""),
                    "chunk_idx": idx,
                    "query": doc.get("query", ""),
                }
            )
    return chunks
