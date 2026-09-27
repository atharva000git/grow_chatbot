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

import numpy as np

from config import settings
from src.models import Chunk, RetrievedChunk
from src.retrieval.embedder import embed_query

log = logging.getLogger(__name__)


class IndexNotBuilt(RuntimeError):
    """The vector index is missing, empty, or unreadable."""


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


def _index() -> tuple[np.ndarray, list[dict]]:
    from src.ingest.store import load_index

    try:
        matrix, rows = load_index()
    except Exception as exc:  # noqa: BLE001 - surfaced as a named error
        raise IndexNotBuilt(
            "Could not open the vector index. Build it with:\n"
            "    python -m src.ingest.run_all --rebuild\n"
            f"underlying error: {exc}"
        ) from exc
    if matrix.shape[0] == 0:
        raise IndexNotBuilt(
            "The vector index is empty. Build it with:\n"
            "    python -m src.ingest.run_all --rebuild"
        )
    return matrix, rows


def _mask(rows: list[dict], filter_scheme: str | None, filter_section: str | None) -> np.ndarray:
    """Rows matching every supplied clause, mirroring Chroma's `$and` of `$eq`."""
    mask = np.ones(len(rows), dtype=bool)
    if filter_scheme:
        mask &= np.fromiter((row["scheme"] == filter_scheme for row in rows), bool, len(rows))
    if filter_section:
        mask &= np.fromiter((row["section"] == filter_section for row in rows), bool, len(rows))
    return mask


def _to_hits(rows: list[dict], indices: np.ndarray, scores: np.ndarray) -> list[RetrievedChunk]:
    hits: list[RetrievedChunk] = []
    for index in indices:
        row = rows[int(index)]
        # Clamp because a float32 dot product can land a hair outside the unit
        # interval, and callers compare against a fixed threshold.
        similarity = min(1.0, max(0.0, float(scores[int(index)])))
        hits.append(
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id=str(row["chunk_id"]),
                    text=str(row["text"]),
                    source_url=str(row.get("source_url", "")),
                    scheme=str(row.get("scheme", "")),
                    category=str(row.get("category", "")),
                    section=str(row.get("section", "overview")),
                    chunk_index=int(row.get("chunk_index", 0)),
                    fetched_at=str(row.get("fetched_at", "")),
                ),
                similarity=similarity,
            )
        )
    return hits


def search(
    query: str,
    k: int | None = None,
    scheme: str | None = None,
) -> list[RetrievedChunk]:
    """Rank chunks for a question, most similar first."""
    matrix, rows = _index()
    top_k = k or settings.TOP_K
    embedding = np.asarray(embed_query(query), dtype=np.float32)
    # The stored rows are unit vectors (store._normalise), so a dot product is
    # the cosine. Normalising the query closes the same contract from this side.
    norm = float(np.linalg.norm(embedding))
    if norm == 0.0:
        raise IndexNotBuilt("query produced a zero-length embedding; cannot score it")
    scores = matrix @ (embedding / norm)

    detected = scheme or detect_scheme_filter(query)
    if scheme is not None:
        detected = scheme
    section = detect_section_filter(query)

    def run(
        filter_scheme: str | None,
        filter_section: str | None,
        limit: int,
        exclude: set[str],
    ) -> list[RetrievedChunk]:
        candidates = np.flatnonzero(_mask(rows, filter_scheme, filter_section))
        if candidates.size == 0:
            return []
        # Sort only the surviving rows, so a filter narrows the pool and the
        # best of what is left still wins. `stable` keeps insertion order for
        # equal scores, which a plain argsort does not guarantee.
        order = np.argsort(-scores[candidates], kind="stable")
        # Skip rows an earlier, narrower attempt already contributed, and keep
        # scanning past them. Truncating to `limit` first and then discarding
        # duplicates cannot backfill, so a heavily-overlapping relaxation left
        # the window short - 8 of 10 with two filters sharing a chunk.
        picked = [
            index
            for index in candidates[order]
            if str(rows[int(index)]["chunk_id"]) not in exclude
        ][:limit]
        return _to_hits(rows, np.asarray(picked, dtype=np.int64), scores)

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
        added = run(filter_scheme, filter_section, top_k - len(collected), seen)
        if added and index:
            log.info(
                "relaxed to scheme=%r section=%r and added %d hit(s) to fill k=%d",
                filter_scheme, filter_section, len(added), top_k,
            )
        for hit in added:
            seen.add(hit.chunk.chunk_id)
            collected.append(hit)
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
