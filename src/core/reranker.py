"""core · 重排器接口。

召回（向量 / BM25）负责"别漏"，重排负责"别乱"：
对召回候选做精排，输入输出都是 chunk 字典列表，方便串成 pipeline。
"""
from abc import ABC, abstractmethod


class BaseReranker(ABC):
    """重排器抽象：对召回候选做精排，不改变 chunk 的数据结构。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """重排器标识（如 bge-reranker-v2-m3），用于 trace 与评测报告归因。"""

    @abstractmethod
    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """按与 query 的相关度重排 candidates，返回前 top_k 条（含 score 字段）。

        实现不可用时由上层决定跳过重排，而不是中断主流程。
        """
