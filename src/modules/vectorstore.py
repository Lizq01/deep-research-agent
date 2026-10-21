"""RAG · 向量库：内存态向量索引 + 余弦相似度检索，附 BM25 关键词降级。

设计要点：
- 索引为纯内存态（每次运行重建）：调研信源是实时网页，持久化只会引入
  "把过期内容当成当前现状" 的脏数据风险；
- 向量检索不可用（模型缺失）时退化到 BM25，保证功能可用而非整体失败；
- BM25 只依赖标准库 + 简单计数，无第三方依赖。
"""
import logging
import math
import re
from collections import Counter

from src.config import RAG_MIN_SCORE, RAG_TOP_K

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    """中英混合粗切：英文按词，中文按单字（BM25 兜底够用，无需分词器）。"""
    return _TOKEN_RE.findall((text or "").lower())


class VectorStore:
    """内存向量索引。vectors 与 chunks 一一对应。"""

    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self._vectors = None  # numpy.ndarray, shape=(n, dim)
        self._mode: str = "empty"  # empty | embedding | bm25
        self._bm25_cache: dict | None = None

    @property
    def mode(self) -> str:
        """当前检索模式，供上层日志展示。"""
        return self._mode

    def __len__(self) -> int:
        return len(self.chunks)

    def build(self, chunks: list[dict]) -> None:
        """建索引：优先做向量索引，失败自动降级为 BM25。"""
        self.chunks = list(chunks)
        self._bm25_cache = None
        if not self.chunks:
            self._mode = "empty"
            return

        try:
            import numpy as np

            from src.rag.embedder import embed_texts

            vectors = embed_texts([c["text"] for c in self.chunks])
            self._vectors = np.asarray(vectors, dtype="float32")
            self._mode = "embedding"
            logger.info("向量索引已建立：%d 块 / %d 维", *self._vectors.shape)
        except Exception as e:  # 模型缺失等，降级而非失败
            self._vectors = None
            self._mode = "bm25"
            logger.warning("向量索引不可用，降级为 BM25 关键词检索：%s", e)

    def search(self, query: str, top_k: int = RAG_TOP_K) -> list[dict]:
        """检索最相关的 top_k 个片段，返回 chunk 字典并附加 score 字段。"""
        if not self.chunks:
            return []
        if self._mode == "embedding":
            return self._search_vector(query, top_k)
        if self._mode == "bm25":
            return self._search_bm25(query, top_k)
        return []

    # ---------- 向量检索 ----------

    def _search_vector(self, query: str, top_k: int) -> list[dict]:
        import numpy as np

        from src.rag.embedder import embed_query

        try:
            q = np.asarray(embed_query(query), dtype="float32")
        except Exception as e:
            logger.warning("查询向量化失败，退回 BM25：%s", e)
            return self._search_bm25(query, top_k)

        # 文档向量已归一化，点积即余弦相似度
        scores = self._vectors @ q
        k = min(top_k, len(self.chunks))
        idx = np.argsort(-scores)[:k]

        hits: list[dict] = []
        for i in idx:
            score = float(scores[i])
            if score < RAG_MIN_SCORE:
                continue
            hits.append({**self.chunks[int(i)], "score": round(score, 4)})
        return hits

    # ---------- BM25 降级 ----------

    def _build_bm25(self) -> dict:
        if self._bm25_cache is not None:
            return self._bm25_cache

        docs_tokens = [_tokenize(c["text"]) for c in self.chunks]
        doc_len = [len(t) for t in docs_tokens]
        avgdl = sum(doc_len) / len(doc_len) if doc_len else 0.0

        df: Counter = Counter()
        for tokens in docs_tokens:
            df.update(set(tokens))

        n = len(docs_tokens)
        idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self._bm25_cache = {
            "docs_tokens": docs_tokens,
            "doc_len": doc_len,
            "avgdl": avgdl,
            "idf": idf,
        }
        return self._bm25_cache

    def _search_bm25(self, query: str, top_k: int, k1: float = 1.5, b: float = 0.75) -> list[dict]:
        data = self._build_bm25()
        q_terms = _tokenize(query)
        if not q_terms or not data["avgdl"]:
            return []

        scored: list[tuple[int, float]] = []
        for i, tokens in enumerate(data["docs_tokens"]):
            tf = Counter(tokens)
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                freq = tf[term]
                denom = freq + k1 * (1 - b + b * data["doc_len"][i] / data["avgdl"])
                score += data["idf"].get(term, 0.0) * freq * (k1 + 1) / denom
            if score > 0:
                scored.append((i, score))

        scored.sort(key=lambda x: -x[1])
        return [
            {**self.chunks[i], "score": round(s, 4)}
            for i, s in scored[:top_k]
        ]
