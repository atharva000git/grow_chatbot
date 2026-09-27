"""Stage 3 - Chunking. Satisfies PRD FR-2.

The strategy is a config value, not a code fork: `settings.CHUNK_STRATEGY`
selects between the recursive splitter and the semantic splitter. Both were run
over the real corpus; the comparison is recorded in notes/chunking.md.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from src.models import REPO_ROOT, Chunk, SourceDoc

log = logging.getLogger(__name__)

CHUNKS_PATH = REPO_ROOT / "data" / "chunks" / "chunks.jsonl"
SEPARATORS = ["\n## ", "\n\n", "\n", ". ", " "]
MERGE_CEILING = 1.3
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n{2,}")

SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("lock_in", re.compile(r"lock[-\s]?in", re.I)),
    ("exit_load", re.compile(r"exit[-\s]?load", re.I)),
    ("riskometer", re.compile(r"riskometer|risk\s+category|rated\s+\w+\s+risk|risk\s+profile", re.I)),
    ("fees", re.compile(r"expense\s+ratio|statutory\s+levies\s+on\s+expenses|additional\s+expense", re.I)),
    ("benchmark", re.compile(r"benchmark", re.I)),
    ("statements", re.compile(r"capital\s+gains?\s+statement|tax\s+report|account\s+statement|download\s+statement", re.I)),
    ("sip", re.compile(r"\bSIP\b|systematic\s+investment|monthly\s+installment|\bmin\.?\s+for\s+sip", re.I)),
    ("nav", re.compile(r"\bNAV\b|net\s+asset\s+value", re.I)),
    ("faq", re.compile(r"\bFAQ\b|frequently\s+asked", re.I)),
)

MID_LINE_START = re.compile(r"^[a-z,.;:%)\]]")
PAGE_MARKER = re.compile(r"\.{2,}\s*Contd on next page", re.I)


class ChunkingError(RuntimeError):
    pass


def infer_section(text: str) -> str:
    """Label a chunk by the highest-priority fact it carries.

    The whole chunk is scanned, not just its opening characters: the factsheet
    packs expense ratio, benchmark, lock-in and exit load into one dense block,
    and a 200-character window labelled that chunk `fees`, burying the ELSS
    lock-in that the corpus was extended to capture.
    """
    for name, pattern in SECTION_PATTERNS:
        if pattern.search(text):
            return name
    return "overview"


def _mid_line_starts(chunks: list[str]) -> int:
    return sum(1 for c in chunks if MID_LINE_START.match(c.lstrip()))


def _merge_mid_line_starts(chunks: list[str]) -> tuple[list[str], int, int]:
    """Fold a chunk that begins mid-sentence into its predecessor so a figure is
    never separated from the label that introduces it."""
    merged: list[str] = []
    merges = 0
    ceiling = int(settings.CHUNK_SIZE * MERGE_CEILING)
    for chunk in chunks:
        if merged and MID_LINE_START.match(chunk.lstrip()):
            combined = f"{merged[-1]}\n{chunk}"
            if len(combined) <= ceiling:
                merged[-1] = combined
                merges += 1
                continue
        merged.append(chunk)
    return merged, merges, _mid_line_starts(merged)


def chunk_recursive(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=SEPARATORS,
        keep_separator=True,
    )
    return splitter.split_text(text)


def chunk_semantic(text: str, breakpoint_threshold: float = 85.0) -> list[str]:
    """Group sentences by embedding-distance breakpoints.

    Implemented directly on sentence-transformers rather than
    langchain_experimental.SemanticChunker: no 0.2.x release of that package
    exists to match langchain==0.2.2, and 0.3.x would force a langchain and
    numpy upgrade that breaks the rest of the pins.
    """
    from src.retrieval.embedder import embed_texts

    sentences = [s for s in SENTENCE_BREAK.split(text) if s and s.strip()]
    if len(sentences) < 2:
        return [text] if text.strip() else []

    vectors = embed_texts(sentences)
    distances = [1.0 - float(sum(a * b for a, b in zip(vectors[i], vectors[i + 1])))
                 for i in range(len(vectors) - 1)]
    if not distances:
        return [text]

    ordered = sorted(distances)
    index = min(int(len(ordered) * breakpoint_threshold / 100.0), len(ordered) - 1)
    threshold = ordered[index]

    groups: list[list[str]] = [[sentences[0]]]
    for offset, sentence in enumerate(sentences[1:]):
        if distances[offset] > threshold:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)

    ceiling = int(settings.CHUNK_SIZE * MERGE_CEILING)
    chunks: list[str] = []
    for group in groups:
        joined = " ".join(group)
        if len(joined) <= ceiling:
            chunks.append(joined)
        else:
            chunks.extend(chunk_recursive(joined))
    return chunks


def _normalize(chunk: str) -> str:
    """Collapse the factsheet's hard line breaks into single spaces.

    The August 2026 factsheet is laid out in narrow columns, so pypdf emits one
    fact across several physical lines:

        LOCK-IN PERIOD\\n3 years from the date of allotment of the\\nrespective Units

    Left alone that label and value are never a contiguous span, which is what
    the encoder needs to embed and what the answer generator needs to quote. Run
    after _merge_mid_line_starts, which needs the real line structure to detect
    a mid-sentence start.
    """
    return " ".join(PAGE_MARKER.sub(" ", chunk).split())


def chunk_document(doc: SourceDoc, strategy: str | None = None) -> list[Chunk]:
    text = Path(doc.clean_path).read_text(encoding="utf-8")
    chosen = (strategy or settings.CHUNK_STRATEGY).lower()
    if chosen == "semantic":
        raw = chunk_semantic(text)
    elif chosen == "recursive":
        raw = chunk_recursive(text)
    else:
        raise ChunkingError(f"unknown CHUNK_STRATEGY {chosen!r}; use 'recursive' or 'semantic'")

    kept = [c for c in raw if len(c.strip()) >= settings.MIN_CHUNK_CHARS]
    merged, merges, remaining = _merge_mid_line_starts(kept)
    log.info(
        "%s: %d raw -> %d kept (>=%d chars) -> %d chunks; %d mid-line start(s) merged, "
        "%d kept because merging would exceed %d chars",
        doc.source_id, len(raw), len(kept), settings.MIN_CHUNK_CHARS, len(merged),
        merges, remaining, int(settings.CHUNK_SIZE * MERGE_CEILING),
    )

    return [
        Chunk(
            chunk_id=f"{doc.source_id}::c{index}",
            text=_normalize(chunk),
            source_url=doc.source_url,
            scheme=doc.scheme,
            category=doc.category,
            section=infer_section(chunk),
            chunk_index=index,
            fetched_at=doc.fetched_at,
        )
        for index, chunk in enumerate(merged)
    ]


def chunk_all(docs: list[SourceDoc], strategy: str | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, strategy=strategy))
    return chunks


def save_chunks(chunks: list[Chunk], path: Path = CHUNKS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")
    return path


def load_chunks(path: Path = CHUNKS_PATH) -> list[Chunk]:
    with path.open(encoding="utf-8") as handle:
        return [Chunk(**json.loads(line)) for line in handle if line.strip()]


def _stats(chunks: list[Chunk]) -> dict:
    lengths = [len(c.text) for c in chunks]
    return {
        "chunks": len(chunks),
        "min": min(lengths) if lengths else 0,
        "mean": round(sum(lengths) / len(lengths)) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "sections": dict(Counter(c.section for c in chunks)),
    }


def study(docs: list[SourceDoc]) -> None:
    for strategy in ("recursive", "semantic"):
        started = time.perf_counter()
        chunks = chunk_all(docs, strategy=strategy)
        elapsed = time.perf_counter() - started
        s = _stats(chunks)
        print(f"\n=== {strategy} ===  {elapsed:.1f}s")
        print(f"  chunks={s['chunks']}  chars min/mean/max={s['min']}/{s['mean']}/{s['max']}")
        print(f"  per doc: {dict(Counter(c.chunk_id.split('::')[0] for c in chunks))}")
        print(f"  sections: {s['sections']}")
        empty = [k for k, v in s["sections"].items() if v == 0]
        print(f"  empty sections: {empty or 'none'}")


if __name__ == "__main__":
    from src.ingest.load import load_all

    study(load_all())
