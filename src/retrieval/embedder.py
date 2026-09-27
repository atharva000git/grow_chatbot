"""Shared sentence-transformers encoder for ingestion and query time.

Both sides of the retrieval pipeline must use the identical encoder, otherwise
document and query vectors live in different spaces. This module is the single
place that model is constructed.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import settings


class EmbedderUnavailable(RuntimeError):
    """The sentence-transformers model could not be loaded."""


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    try:
        return SentenceTransformer(settings.EMBEDDING_MODEL)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a named error
        raise EmbedderUnavailable(
            f"Could not load {settings.EMBEDDING_MODEL}: {exc}. "
            "Run once with network access to populate the model cache, or use the "
            "precomputed vectors in data/embeddings/."
        ) from exc


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = get_embedder().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    return [vector.tolist() for vector in vectors]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
