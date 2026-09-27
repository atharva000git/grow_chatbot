"""Core typed models for the HDFC facts-only RAG chatbot. Satisfies PRD FR-1..FR-10."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES_CSV = REPO_ROOT / "config" / "sources.csv"

REQUIRED_COLUMNS = ("source_id", "scheme", "category", "source_url")

Intent = Literal["ANSWER", "REFUSAL", "INSUFFICIENT_CONTEXT", "PII_BLOCKED"]


class SourceConfigError(RuntimeError):
    """config/sources.csv is missing, empty, or has unexpected columns."""


@dataclass(frozen=True)
class SourceDoc:
    source_id: str
    scheme: str
    category: str
    source_url: str
    fetched_at: str
    content_hash: str
    raw_path: str
    clean_path: str
    source_type: str = "html"
    page_start: int = 0
    page_end: int = 0


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source_url: str
    scheme: str
    category: str
    section: str
    chunk_index: int
    fetched_at: str = ""


@dataclass
class RetrievedChunk:
    chunk: Chunk
    similarity: float


@dataclass
class ChatResponse:
    intent: Intent
    answer_text: str
    citation_url: str | None
    source_scheme: str | None
    last_updated: str
    retrieval_hits: int
    # Top similarity of the hits behind this answer. Carried so the UI can show
    # that an answer was actually grounded rather than asserting it. Defaulted so
    # every existing construction site keeps working.
    top_similarity: float | None = None


@dataclass
class PIIVerdict:
    blocked: bool
    sanitized_text: str
    matched_kinds: list[str]


@dataclass
class AdviceVerdict:
    is_advice: bool
    matched_pattern: str


@lru_cache(maxsize=1)
def load_sources() -> tuple[dict[str, str], ...]:
    """Read config/sources.csv. The only place a permitted URL is defined."""
    if not SOURCES_CSV.exists():
        raise SourceConfigError(f"Source registry not found at {SOURCES_CSV}")

    with SOURCES_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SourceConfigError(f"{SOURCES_CSV} is empty")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise SourceConfigError(
                f"{SOURCES_CSV} is missing required columns: {', '.join(missing)}"
            )
        rows = tuple({k: (v or "").strip() for k, v in row.items() if k} for row in reader)

    if not rows:
        raise SourceConfigError(f"{SOURCES_CSV} contains no data rows")
    return rows


def permitted_urls() -> frozenset[str]:
    """Citation allow-list. Nothing outside this set may ever reach the UI."""
    return frozenset(row["source_url"] for row in load_sources())


def build_source_docs() -> list[SourceDoc]:
    """SourceDoc stubs for the ingestion run. Phase 2 fills hash and paths."""
    today = date.today().isoformat()
    docs: list[SourceDoc] = []
    for row in load_sources():
        page_start = int(row.get("page_start") or 0)
        page_end = int(row.get("page_end") or 0)
        docs.append(
            SourceDoc(
                source_id=row["source_id"],
                scheme=row["scheme"],
                category=row["category"],
                source_url=row["source_url"],
                fetched_at=today,
                content_hash="",
                raw_path="",
                clean_path="",
                source_type=(row.get("source_type") or "html").lower(),
                page_start=page_start,
                page_end=page_end or page_start,
            )
        )
    return docs
