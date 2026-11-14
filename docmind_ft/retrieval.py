"""Dense retrieval for the QASPER data engine (F1.2).

Embeddings use fastembed's ONNX MiniLM, the same model as Project 7. Ranking
is pure: exact cosine, deterministic order, ties broken by chunk id.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
TOP_K = 10


@dataclass(frozen=True)
class Retrieved:
    chunk_id: str
    score: float


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and every row of ``matrix``."""
    query = np.asarray(query, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if query.ndim != 1 or matrix.ndim != 2 or matrix.shape[1] != query.shape[0]:
        raise ValueError(f"shape mismatch: query {query.shape}, matrix {matrix.shape}")
    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    dots = matrix @ query
    return np.divide(dots, denominator, out=np.zeros_like(dots), where=denominator > 0)


def rank_chunks(
    scores: Sequence[float], chunk_ids: Sequence[str], top_k: int = TOP_K
) -> list[Retrieved]:
    """Top ``top_k`` chunks by score, ties broken by chunk id (ascending)."""
    if len(scores) != len(chunk_ids):
        raise ValueError(f"{len(scores)} scores for {len(chunk_ids)} chunk ids")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    ordered = sorted(zip(scores, chunk_ids, strict=True), key=lambda pair: (-pair[0], pair[1]))
    return [Retrieved(chunk_id, float(score)) for score, chunk_id in ordered[:top_k]]


def retrieve(
    query: np.ndarray,
    matrix: np.ndarray,
    chunk_ids: Sequence[str],
    top_k: int = TOP_K,
) -> list[Retrieved]:
    """Exact cosine top-k over one candidate pool (normally one paper's chunks)."""
    if matrix.shape[0] != len(chunk_ids):
        raise ValueError(f"{matrix.shape[0]} vectors for {len(chunk_ids)} chunk ids")
    return rank_chunks(cosine_scores(query, matrix), chunk_ids, top_k)


def insert_gold(retrieved: Sequence[Retrieved], gold: Sequence[Retrieved]) -> list[Retrieved]:
    """Merge gold chunks into a ranked list, still ordered by score then chunk id."""
    by_id = {item.chunk_id: item for item in retrieved}
    for item in gold:
        by_id.setdefault(item.chunk_id, item)
    return sorted(by_id.values(), key=lambda item: (-item.score, item.chunk_id))


class FastEmbedder:
    """fastembed ONNX embedder; rows are L2-normalized like Project 7."""

    dim = EMBED_DIM

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        *,
        cache_dir: str | None = None,
        threads: int = 1,
    ):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=threads)

    def embed(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = np.asarray(
            list(self._model.embed(texts, batch_size=batch_size)), dtype=np.float32
        )
        if vectors.shape != (len(texts), self.dim):
            raise RuntimeError(
                f"embedder returned {vectors.shape}, expected {(len(texts), self.dim)}"
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)
