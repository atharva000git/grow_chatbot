"""Stage 5 - Vector store. Satisfies PRD FR-4.

A numpy matrix on disk, not a vector database.

Chroma was here first and was removed on purpose. It cost 84 MB of resident
memory on every process that touched it, which is the difference between this
app fitting and not fitting Render's 512 MB free-tier limit (measured peak:
525 MB with Chroma, ~446 MB without). It also cost a large install and an
approximate-search failure mode: Chroma's HNSW silently dropped strong
neighbours, and the ELSS lock-in chunk scored 0.555 at rank 1 of 273 yet went
unreturned at n_results=8. That only worked around by pinning hnsw:search_ef
to 256.

None of that is worth 84 MB for 349 vectors. A dense matrix multiply over 349
x 384 float32 is 536 KB and takes well under a millisecond, and it is
*exhaustive* - there is no candidate list to tune and no recall cliff. The
exhaustiveness is the point: the previous HNSW bug class cannot recur.

Two invariants:

1. A build must never accumulate. `build_index` writes a fresh directory and
   swaps it in, so running the pipeline twice cannot leave orphaned vectors
   from a previous corpus behind.
2. A half-written index must never be served. The swap is what guarantees
   this: readers see either the old index or the new one, never a partial file.
   A Streamlit process is reading this index while `run_all` rewrites it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np

from config import settings
from src.models import REPO_ROOT, Chunk

log = logging.getLogger(__name__)

# settings.INDEX_DIR is the documented relative value ("index"). Resolve it
# against the repo root so the store is found no matter the working directory.
INDEX_PATH = Path(settings.INDEX_DIR)
if not INDEX_PATH.is_absolute():
    INDEX_PATH = REPO_ROOT / INDEX_PATH
VECTORS_PATH = INDEX_PATH / "vectors.npy"
CHUNKS_PATH = INDEX_PATH / "chunks.jsonl"

# Fields persisted per row, in the order Chunk declares them. `text` is stored
# because the prompt needs the chunk body, not just its vector.
CHUNK_FIELDS = ("chunk_id", "text", "source_url", "scheme", "category", "section", "chunk_index", "fetched_at")


class StoreError(RuntimeError):
    pass


def reset_index() -> None:
    if INDEX_PATH.exists():
        shutil.rmtree(INDEX_PATH)
        log.info("deleted index at %s", INDEX_PATH)


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to unit length so a dot product *is* the cosine.

    The embedder already normalises, but making the invariant explicit here
    means the search layer can rely on it unconditionally instead of trusting
    an upstream detail. Costs 536 KB of arithmetic.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise StoreError("index contains a zero vector; cosine similarity is undefined for it")
    return (matrix / norms).astype(np.float32)


def build_index(chunks: list[Chunk], vectors: list[list[float]] | None = None) -> int:
    """Write the index from scratch. Returns the stored count."""
    if not chunks:
        raise StoreError("refusing to build an index from zero chunks")
    if vectors is None:
        from src.ingest.embed import load_or_build_embeddings

        _, vectors = load_or_build_embeddings(chunks)
    if len(vectors) != len(chunks):
        raise StoreError(f"{len(vectors)} vectors for {len(chunks)} chunks; they must align")

    matrix = _normalise(np.asarray(vectors, dtype=np.float32))
    if matrix.shape[1] != settings.EMBEDDING_DIM:
        raise StoreError(
            f"vectors are {matrix.shape[1]}-dimensional but EMBEDDING_DIM is "
            f"{settings.EMBEDDING_DIM}; re-run after changing the embedding model"
        )

    # Build beside the target, then swap. A crash mid-write leaves the previous
    # index intact and readable instead of a truncated vectors.npy.
    staging = INDEX_PATH.with_name(INDEX_PATH.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    ids = [chunk.chunk_id for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise StoreError("duplicate chunk_id; row order could not be trusted")

    with (staging / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            row = {field: getattr(chunk, field) for field in CHUNK_FIELDS}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    np.save(staging / "vectors.npy", matrix, allow_pickle=False)

    reset_index()
    os.replace(staging, INDEX_PATH)
    log.info("indexed %d chunks into %s", len(chunks), INDEX_PATH)
    return len(chunks)


@lru_cache(maxsize=1)
def load_index() -> tuple[np.ndarray, list[dict]]:
    """Return (matrix, rows). Row i of the matrix is row i of the list.

    Cached for the process lifetime: 536 KB of vectors plus the chunk bodies,
    loaded once per Streamlit session rather than per query.
    """
    if not VECTORS_PATH.exists() or not CHUNKS_PATH.exists():
        raise StoreError(
            f"no index at {INDEX_PATH}. Build it with:\n"
            "    python -m src.ingest.run_all --rebuild"
        )
    matrix = np.load(VECTORS_PATH, allow_pickle=False)
    rows = [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if matrix.shape[0] != len(rows):
        raise StoreError(
            f"index is inconsistent: {matrix.shape[0]} vectors but {len(rows)} chunk rows. "
            "Rebuild it with python -m src.ingest.run_all --rebuild"
        )
    return matrix, rows


def collection_count() -> int:
    """Named for continuity with the Chroma era; the sidebar calls it."""
    try:
        matrix, _ = load_index()
    except StoreError:
        return 0
    return int(matrix.shape[0])
