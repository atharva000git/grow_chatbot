"""Stage 5 - Vector store. Satisfies PRD FR-4.

Wraps ChromaDB. Two invariants matter here:

1. A build must never accumulate. `build_index` deletes and recreates the
   collection, so running the pipeline twice cannot leave orphaned chunks from a
   previous corpus behind.
2. Chroma only accepts str/int/float/bool metadata, so every field is coerced
   and anything else is dropped rather than passed through and rejected.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import chromadb

from config import settings
from src.models import REPO_ROOT, Chunk

log = logging.getLogger(__name__)

# settings.CHROMA_DIR is the documented relative value ("chroma"). Resolve it
# against the repo root so the store is found no matter the working directory.
CHROMA_PATH = Path(settings.CHROMA_DIR)
if not CHROMA_PATH.is_absolute():
    CHROMA_PATH = REPO_ROOT / CHROMA_PATH

METADATA_FIELDS = ("source_url", "scheme", "category", "section", "chunk_index", "fetched_at")


class StoreError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def _coerce(value: object) -> str | int | float | bool:
    """Reduce a value to a type Chroma accepts, or drop it."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value
    return str(value)


def build_metadatas(chunks: list[Chunk]) -> list[dict[str, str | int | float | bool]]:
    metadatas: list[dict[str, str | int | float | bool]] = []
    for chunk in chunks:
        row = {field: _coerce(getattr(chunk, field)) for field in METADATA_FIELDS}
        metadatas.append(row)
    return metadatas


@lru_cache(maxsize=1)
def get_collection() -> chromadb.CollectionAPI:
    client = get_client()
    return client.get_or_create_collection(
        name=settings.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def reset_index() -> None:
    client = get_client()
    try:
        client.delete_collection(settings.COLLECTION_NAME)
        log.info("deleted collection %s", settings.COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - absent collection is not an error
        log.info("collection %s did not exist", settings.COLLECTION_NAME)


def build_index(chunks: list[Chunk], vectors: list[list[float]] | None = None) -> int:
    """Rebuild the collection from scratch. Returns the stored count."""
    if not chunks:
        raise StoreError("refusing to build an index from zero chunks")
    if vectors is None:
        from src.ingest.embed import load_or_build_embeddings

        _, vectors = load_or_build_embeddings(chunks)
    if len(vectors) != len(chunks):
        raise StoreError(f"{len(vectors)} vectors for {len(chunks)} chunks; they must align")

    reset_index()
    get_collection.cache_clear()
    collection = get_collection()

    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=build_metadatas(chunks),
        embeddings=[list(v) for v in vectors],
    )
    count = collection.count()
    log.info("indexed %d chunks into %s", count, settings.COLLECTION_NAME)
    return count


def peek(limit: int = 1) -> dict:
    return get_collection().peek(limit=limit)


def collection_count() -> int:
    return get_collection().count()
