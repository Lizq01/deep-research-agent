"""core · 向量库接口。

契约刻意只留三个动作：建索引、检索、报规模，外加一个降级模式上报。
这样内存索引、FAISS、BM25、云端向量库都能用同一张脸面对上层。
"""
from abc import ABC, abstractmethod


class BaseVectorStore(ABC):
    """向量库抽象：建索引 + 检索 + 自身状态上报。"""

    @property
    @abstractmethod
    def mode(self) -> str:
        """当前检索模式（如 embedding / bm25 / empty），供上层日志与观测展示。"""

    @abstractmethod
    def build(self, chunks: list[dict]) -> None:
        """用 chunk 列表重建索引。chunks 与内部向量一一对应。

        索引不可用时由实现自行降级（如退回 BM25），不允许整体失败。
        """

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[dict]:
        """检索最相关的 top_k 个片段，返回带 score 字段的 chunk 字典。"""

    @abstractmethod
    def __len__(self) -> int:
        """已索引的片段数（0 表示索引为空，上层据此走全量兜底）。"""
