"""Corpus manifest - proof of what is inside the vector store.

The manifest exists so a demo can state exactly which documents, which chunk
count and which model produced the answers, and so a re-run can detect that the
index is stale.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from src.models import REPO_ROOT, Chunk, SourceDoc

log = logging.getLogger(__name__)

MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"


def corpus_hash(docs: list[SourceDoc]) -> str:
    """sha256 over the sorted content hashes, so ordering cannot change it."""
    digest = hashlib.sha256()
    for content_hash in sorted(doc.content_hash for doc in docs if doc.content_hash):
        digest.update(content_hash.encode("utf-8"))
    return digest.hexdigest()[:16]


def build_manifest(docs: list[SourceDoc], chunks: list[Chunk]) -> dict:
    sources: dict[str, str] = {}
    for doc in docs:
        sources.setdefault(doc.source_url, doc.fetched_at)

    return {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "n_sources": len(sources),
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dim": settings.EMBEDDING_DIM,
        "chunk_strategy": settings.CHUNK_STRATEGY,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "top_k": settings.TOP_K,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD,
        "vector_store": "numpy",
        "corpus_hash": corpus_hash(docs),
        "sources": sources,
    }


def write_manifest(manifest: dict, path: Path = MANIFEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def read_manifest(path: Path = MANIFEST_PATH) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
