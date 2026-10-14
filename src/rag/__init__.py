"""RAG 子包：本地向量检索增强。

模块职责：
- chunking   ：网页正文 → 段落片段
- embedder   ：片段/查询 → 向量（本地 BGE 模型）
- vectorstore：向量索引与余弦检索（含 BM25 降级）
- retriever  ：按子问题召回并聚合为 Writer 可用的证据
"""
from src.rag.retriever import build_index, retrieve_context

__all__ = ["build_index", "retrieve_context"]
