"""RAG · 嵌入器：用本地 BGE 模型把文本转成向量。

设计要点：
- 模型懒加载且全局复用（首次加载约 2~5 秒，之后秒级）；
- 归一化向量 → 相似度退化为点积，检索更快；
- 模型缺失/加载失败时抛 EmbeddingUnavailable，由上层降级为 BM25，
  不允许静默返回零向量（那会让检索结果变成随机噪声）。
"""
import logging

from src.config import resolve_embedding_model_path

logger = logging.getLogger(__name__)

_MODEL = None
# BGE 官方建议：查询侧加指令前缀（非对称检索），文档侧不加
_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class EmbeddingUnavailable(RuntimeError):
    """嵌入模型不可用时抛出，供上层捕获并降级。"""


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    path = resolve_embedding_model_path()
    if not path.exists():
        raise EmbeddingUnavailable(
            f"未找到本地嵌入模型目录：{path}\n"
            f"请下载 BAAI/bge-small-zh-v1.5 放到该目录，"
            f"或在 .env 里用 EMBEDDING_MODEL_PATH 指定路径。"
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise EmbeddingUnavailable(
            "缺少依赖 sentence-transformers，请执行："
            ".venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from e

    try:
        # local_files_only=True：强制离线加载，避免误连外网
        _MODEL = SentenceTransformer(str(path), local_files_only=True)
    except Exception as e:
        raise EmbeddingUnavailable(f"加载本地模型失败（{path}）：{e}") from e

    logger.info("嵌入模型已加载：%s", path)
    return _MODEL


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """批量向量化文档片段，返回归一化后的向量（float list）。"""
    if not texts:
        return []
    model = _load_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


def embed_query(query: str) -> list[float]:
    """向量化检索查询（BGE 非对称检索：查询侧加指令前缀）。"""
    model = _load_model()
    vector = model.encode(
        _QUERY_PREFIX + query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.tolist()


def embedding_dim() -> int:
    """返回向量维度（bge-small-zh-v1.5 为 512）。"""
    model = _load_model()
    # sentence-transformers 6.x 把 get_sentence_embedding_dimension 改名为
    # get_embedding_dimension，这里做双版本兼容（避免 warning 与未来断崖式失效）
    for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        fn = getattr(model, name, None)
        if fn is not None:
            return int(fn())
    return len(embed_query("维度探测"))
