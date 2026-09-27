"""Stage 6 - Retrieval. Satisfies PRD FR-5.

Turns a question into ranked, thresholded chunks with their metadata intact.

Two behaviours worth knowing about:

* The scheme pre-filter is a precision boost, not a gate. A filter that returns
  nothing is retried once unfiltered, so a phrasing the token matcher missed
  degrades to ordinary search rather than to a false "no answer".
* `is_sufficient` is an any-hit gate on the single best chunk, not an all-hit
  gate. One strong chunk is enough to answer a three-sentence question, and
  gating on every hit would reject good answers whenever a weak second result
  rode along in the top-k.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from config import settings
from src.models import Chunk, RetrievedChunk
from src.retrieval.embedder import embed_query

log = logging.getLogger(__name__)


class IndexNotBuilt(RuntimeError):
    """The Chroma collection is missing or empty."""


# Distinctive tokens per scheme, matched against the scheme column of
# config/sources.csv. "equity fund" is kept as an alias because the scheme was
# renamed from HDFC Equity Fund to HDFC Flexi Cap Fund and both names are still
# in circulation.
SCHEME_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("balanced advantage", ("balanced advantage", "balanced advantage fund")),
    ("small cap", ("small cap",)),
    ("large cap", ("large cap",)),
    ("flexi cap", ("flexi cap", "equity fund", "flexi")),
    ("elss", ("elss", "tax saver")),
)


@dataclass(frozen=True)
class _SchemeEntry:
    scheme: str
    category: str
    tokens: tuple[str, ...]


def _known_schemes() -> list[_SchemeEntry]:
    from src.models import SourceConfigError, build_source_docs

    try:
        docs = build_source_docs()
    except (FileNotFoundError, SourceConfigError):
        return []
    seen: dict[str, _SchemeEntry] = {}
    for doc in docs:
        category = doc.category.strip().lower()
        if category not in seen:
            seen[category] = _SchemeEntry(doc.scheme, doc.category, ())
    for category, entry in seen.items():
        tokens: tuple[str, ...] = ()
        for name, aliases in SCHEME_TOKENS:
            if name == category or any(a in category for a in aliases):
                tokens = aliases
                break
        seen[category] = _SchemeEntry(entry.scheme, entry.category, tokens)
    return list(seen.values())


def detect_scheme_filter(query: str) -> str | None:
    """Return the scheme string a Chroma `where` filter should use, or None."""
    lowered = query.lower()
    for entry in _known_schemes():
        if any(token in lowered for token in entry.tokens):
            return entry.scheme
    return None


# Metric intent: a question that names a fact constrains the search to chunks
# labelled with that fact. The `where` filter on scheme alone was not enough,
# because every chunk in a scheme's page repeats the fund's name, so a generic
# description ("HDFC Large Cap Fund Direct Growth is a Equity Mutual Fund
# Scheme launched by HDFC Mutual Fund.") matched "What is the exit load of HDFC
# Large Cap Fund?" at 0.684 and buried the chunk that actually states the exit
# load. Restricting to the section removes the decoys structurally instead of
# hoping the embedding ranks past them.
SECTION_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("lock_in", ("lock-in", "lock in", "locked in", "redeem after", "3 year", "three year")),
    ("exit_load", ("exit load", "exit-load", "redemption charge", "charge on redemption")),
    ("benchmark", ("benchmark",)),
    ("management", ("fund manager", "who manages", "fund management", "portfolio manager")),
    ("fees", ("expense ratio", "expense", "ter")),
    ("aum", ("aum", "fund size", "asset under management", "assets under management")),
    ("sip", ("sip", "systematic investment", "installment", "instalment", "monthly investment")),
    ("nav", ("nav", "net asset value")),
    ("riskometer", ("riskometer", "risk category", "risk-ometer", "rated")),
)


def _section_patterns() -> tuple[tuple[str, tuple[re.Pattern[str], ...]], ...]:
    """Compile SECTION_TOKENS into word-boundary matchers, once, at first use.

    Plain substring matching is wrong here, and not hypothetically: the `fees`
    tokens include "ter" (total expense ratio), and "riskome-ter" contains it.
    Every riskometer question was therefore filtered to the fees section, which
    is the one section guaranteed not to contain the riskometer. Word
    boundaries stop the collision; the optional plural keeps "SIPs" and
    "expenses" matching.
    """
    return tuple(
        (
            name,
            tuple(re.compile(rf"\b{re.escape(token)}s?\b", re.IGNORECASE) for token in tokens),
        )
        for name, tokens in SECTION_TOKENS
    )


def detect_section_filter(query: str) -> str | None:
    """Return the section label a question is asking about, or None."""
    for name, patterns in _section_patterns():
        if any(pattern.search(query) for pattern in patterns):
            return name
    return None


def _collection():
    from src.ingest.store import get_client, get_collection

    try:
        collection = get_collection()
    except Exception as exc:  # noqa: BLE001 - surfaced as a named error
        raise IndexNotBuilt(
            "Could not open the Chroma index. Build it with:\n"
            "    python -m src.ingest.run_all --rebuild\n"
            f"underlying error: {exc}"
        ) from exc
    if collection.count() == 0:
        raise IndexNotBuilt(
            "The Chroma collection is empty. Build it with:\n"
            "    python -m src.ingest.run_all --rebuild"
        )
    return collection


def _to_hits(ids: list[str], docs: list[str], metadatas: list[dict], distances: list[float]) -> list[RetrievedChunk]:
    hits: list[RetrievedChunk] = []
    for chunk_id, text, metadata, distance in zip(ids, docs, metadatas, distances):
        # Chroma cosine distance is 1 - cosine_similarity; clamp because a
        # float32 round-trip can nudge it just outside the unit interval.
        similarity = min(1.0, max(0.0, 1.0 - float(distance)))
        hits.append(
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    source_url=str(metadata.get("source_url", "")),
                    scheme=str(metadata.get("scheme", "")),
                    category=str(metadata.get("category", "")),
                    section=str(metadata.get("section", "overview")),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    fetched_at=str(metadata.get("fetched_at", "")),
                ),
                similarity=similarity,
            )
        )
    hits.sort(key=lambda hit: hit.similarity, reverse=True)
    return hits


def search(
    query: str,
    k: int | None = None,
    scheme: str | None = None,
) -> list[RetrievedChunk]:
    """Rank chunks for a question, most similar first."""
    collection = _collection()
    top_k = k or settings.TOP_K
    embedding = embed_query(query)
    detected = scheme or detect_scheme_filter(query)
    if scheme is not None:
        detected = scheme
    section = detect_section_filter(query)

    def run(filter_scheme: str | None, filter_section: str | None, limit: int) -> list[RetrievedChunk]:
        clauses: list[dict] = []
        if filter_scheme:
            clauses.append({"scheme": {"$eq": filter_scheme}})
        if filter_section:
            clauses.append({"section": {"$eq": filter_section}})
        where = clauses[0] if len(clauses) == 1 else ({"$and": clauses} if clauses else None)
        result = collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return _to_hits(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )

    # Narrowest first, then relax. Each step is a recall safety net, not a
    # ranking trick: a filter that returns nothing must never become a false
    # "no answer", so the query is retried one clause looser before giving up.
    #
    # Steps are accumulated rather than short-circuited, because returning the
    # first non-empty attempt made TOP_K a ceiling in name only - a section
    # filter that matched 2 chunks stopped the search there and the prompt saw
    # 2, not TOP_K. Dedupe by chunk_id so a chunk reachable through two filters
    # is not counted twice, and order is preserved narrowest-first rather than
    # re-sorted by similarity: hits[0] must stay the precise match, because the
    # sufficiency gate, the citation and the answer's anchor all read it.
    #
    # Relaxing to (None, None) is a last resort, not the default filler. It is
    # the only step that admits other schemes' chunks, and a cross-scheme fact
    # ("the exit load is X" from the wrong fund) is worse than a narrow answer.
    # A single scheme holds 40+ chunks, so the scheme-anchored steps fill the
    # window on their own for any real question.
    attempts: list[tuple[str | None, str | None]] = [(detected, section), (detected, None)]
    if not detected:
        attempts = [(None, section)]
    attempts.append((None, None))

    collected: list[RetrievedChunk] = []
    seen: set[str] = set()
    for index, (filter_scheme, filter_section) in enumerate(attempts):
        added = 0
        for hit in run(filter_scheme, filter_section, top_k - len(collected)):
            if hit.chunk.chunk_id in seen:
                continue
            seen.add(hit.chunk.chunk_id)
            collected.append(hit)
            added += 1
            if len(collected) >= top_k:
                break
        if added and index:
            log.info(
                "relaxed to scheme=%r section=%r and added %d hit(s) to fill k=%d",
                filter_scheme, filter_section, added, top_k,
            )
        if len(collected) >= top_k:
            break
    return collected


def best_similarity(hits: list[RetrievedChunk]) -> float:
    return hits[0].similarity if hits else 0.0


def is_sufficient(hits: list[RetrievedChunk]) -> bool:
    """Any-hit gate: one chunk at or above the threshold is enough."""
    return best_similarity(hits) >= settings.SIMILARITY_THRESHOLD


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    QUERIES = [
        "What is the exit load of HDFC Large Cap Fund?",
        "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
        "What is the minimum SIP for HDFC Small Cap Fund?",
        "What is the benchmark of HDFC Balanced Advantage Fund?",
        "Who is the prime minister of India?",
    ]
    for question in QUERIES:
        started = time.perf_counter()
        found = search(question)
        elapsed = (time.perf_counter() - started) * 1000
        print(f"\n{question}")
        print(f"  filter={detect_scheme_filter(question)!r} best={best_similarity(found):.3f} "
              f"sufficient={is_sufficient(found)} {elapsed:.0f} ms")
        for hit in found[:3]:
            print(f"    {hit.similarity:.3f} [{hit.chunk.section:10}] {hit.chunk.scheme[:38]:38} {hit.chunk.source_url[:58]}")
