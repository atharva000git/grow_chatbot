"""Stage 5 CLI - runs load -> chunk -> embed -> index -> manifest end to end.

    python -m src.ingest.run_all --rebuild
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from config import settings
from src.ingest.chunk import CHUNKS_PATH, chunk_all, save_chunks
from src.ingest.embed import load_or_build_embeddings
from src.ingest.load import load_all
from src.ingest.manifest import build_manifest, write_manifest
from src.ingest.store import build_index, reset_index

log = logging.getLogger(__name__)


class _Stages:
    """Collects (stage, in, out, seconds) rows for the summary table."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, int, int, float]] = []

    def record(self, name: str, inputs: int, outputs: int, seconds: float) -> None:
        self.rows.append((name, inputs, outputs, seconds))

    def table(self) -> str:
        header = f"{'stage':<26}{'in':>6}{'out':>7}{'seconds':>10}"
        lines = [header, "-" * len(header)]
        for name, inputs, outputs, seconds in self.rows:
            lines.append(f"{name:<26}{inputs:>6}{outputs:>7}{seconds:>10.2f}")
        return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the HDFC facts-only RAG index.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="delete the existing index before building",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.rebuild:
        reset_index()

    stage = _Stages()

    started = time.perf_counter()
    docs = load_all()
    stage.record("load documents", len(docs), len(docs), time.perf_counter() - started)

    started = time.perf_counter()
    chunks = chunk_all(docs)
    stage.record("chunk documents", len(docs), len(chunks), time.perf_counter() - started)

    started = time.perf_counter()
    save_chunks(chunks, CHUNKS_PATH)
    stage.record("save chunks", len(chunks), len(chunks), time.perf_counter() - started)

    started = time.perf_counter()
    chunk_ids, vectors = load_or_build_embeddings(chunks)
    stage.record("embed chunks", len(chunks), len(vectors), time.perf_counter() - started)

    started = time.perf_counter()
    indexed = build_index(chunks, vectors)
    stage.record("build index", len(chunks), indexed, time.perf_counter() - started)

    started = time.perf_counter()
    manifest = build_manifest(docs, chunks)
    manifest_path = write_manifest(manifest)
    stage.record("write manifest", len(chunks), len(manifest), time.perf_counter() - started)

    print(stage.table())
    print(
        f"\ncorpus_hash {manifest['corpus_hash']}  "
        f"docs {manifest['n_docs']}  chunks {manifest['n_chunks']}  "
        f"sources {manifest['n_sources']}  vectors {settings.INDEX_DIR}/*.npy (numpy)"
    )
    print(f"vectors {len(vectors)}  ids aligned {chunk_ids == [c.chunk_id for c in chunks]}")
    print(f"manifest {manifest_path.relative_to(manifest_path.parents[1])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
