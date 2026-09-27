"""Stage 4 - Embedding. Satisfies PRD FR-3.

Vectorizes every Phase 3 chunk with all-MiniLM-L6-v2 and caches the result to
disk. The cache key is derived from the chunk ids and the model name, so
re-chunking or switching models invalidates it automatically instead of silently
serving stale vectors.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from config import settings
from src.models import REPO_ROOT, Chunk
from src.retrieval.embedder import EmbedderUnavailable, embed_texts

log = logging.getLogger(__name__)

EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings"

# Re-exported so callers can catch it from the ingestion side without knowing
# which module defines it.
__all__ = [
    "EmbedderUnavailable",
    "cache_path",
    "embed_chunks",
    "load_or_build_embeddings",
    "load_manifest",
    "read_cache",
]

DEFAULT_BATCH_SIZE = 16


class EmbeddingError(RuntimeError):
    pass


def _cache_key(chunks: list[Chunk]) -> str:
    """sha256 over the chunk ids plus the model name.

    Ids alone would be wrong: re-chunking can change the text while keeping the
    same `source_id::cN` shape, and the model can change while ids stay put.
    """
    digest = hashlib.sha256()
    digest.update(settings.EMBEDDING_MODEL.encode("utf-8"))
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()[:16]


def cache_path(chunks: list[Chunk] | None = None) -> Path:
    if chunks is None:
        chunks = load_chunks()
    return EMBEDDINGS_DIR / f"{_cache_key(chunks)}.npy"


def load_chunks():
    from src.ingest.chunk import load_chunks as _load

    return _load()


def embed_chunks(
    chunks: list[Chunk], batch_size: int = DEFAULT_BATCH_SIZE
) -> tuple[list[str], list[list[float]]]:
    if batch_size < 1:
        raise EmbeddingError(f"batch_size must be >= 1, got {batch_size}")
    ids = [chunk.chunk_id for chunk in chunks]
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors.extend(embed_texts([chunk.text for chunk in batch]))
        log.info("embedded %d/%d chunks", min(start + batch_size, len(chunks)), len(chunks))

    wrong = [ids[i] for i, v in enumerate(vectors) if len(v) != settings.EMBEDDING_DIM]
    if wrong:
        raise EmbeddingError(
            f"{len(wrong)} vector(s) are not {settings.EMBEDDING_DIM}-dim "
            f"(expected from {settings.EMBEDDING_MODEL}); first offender {wrong[0]}"
        )
    return ids, vectors


def read_cache(path: Path) -> tuple[list[str], list[list[float]]]:
    payload = np.load(path, allow_pickle=True)
    ids = [str(item["chunk_id"]) for item in payload]
    vectors = [item["vector"].tolist() for item in payload]
    if any(len(v) != settings.EMBEDDING_DIM for v in vectors):
        raise EmbeddingError(f"cache {path.name} holds vectors that are not {settings.EMBEDDING_DIM}-dim")
    return ids, vectors


def load_manifest(path: Path) -> dict:
    return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


def _save_cache(
    path: Path, ids: list[str], vectors: list[list[float]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.empty(
        len(ids), dtype=object
    )
    for index, (chunk_id, vector) in enumerate(zip(ids, vectors)):
        payload[index] = {"chunk_id": chunk_id, "vector": np.asarray(vector, dtype=np.float32)}
    np.save(path, payload, allow_pickle=True)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "model": settings.EMBEDDING_MODEL,
                "dim": settings.EMBEDDING_DIM,
                "chunk_count": len(ids),
                "chunk_ids_sha256": path.stem,
                "normalized": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_or_build_embeddings(
    chunks: list[Chunk], batch_size: int = DEFAULT_BATCH_SIZE
) -> tuple[list[str], list[list[float]]]:
    path = cache_path(chunks)
    if path.exists():
        try:
            ids, vectors = read_cache(path)
        except (EmbeddingError, ValueError, OSError) as exc:
            log.warning("ignoring unreadable cache %s: %s", path.name, exc)
        else:
            if ids == [chunk.chunk_id for chunk in chunks]:
                log.info("cache hit %s (%d vectors)", path.name, len(vectors))
                return ids, vectors
            log.warning("cache %s has mismatched ids; rebuilding", path.name)

    started = time.perf_counter()
    ids, vectors = embed_chunks(chunks, batch_size=batch_size)
    _save_cache(path, ids, vectors)
    log.info(
        "built %d vectors in %.1fs -> %s", len(vectors), time.perf_counter() - started, path.name
    )
    return ids, vectors


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from src.ingest.chunk import load_chunks

    corpus = load_chunks()
    chunk_ids, vecs = load_or_build_embeddings(corpus)
    print(f"vectors {len(vecs)}  dim {len(vecs[0])}  expect {settings.EMBEDDING_DIM}")
    print(f"cache {cache_path(corpus)}")
