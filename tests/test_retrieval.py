"""Retrieval window contract tests.

These pin the behaviour of the TOP_K=10 context window, which is the one
setting where "more context" is not automatically "better context".

The interesting part is not the number 10. It is that a wider window had to be
made *real* first: the search used to return the first filter attempt that
produced any hit at all, so a section filter matching two chunks ended the
search and the prompt saw 2 rather than TOP_K. TOP_K was a ceiling in name only.

These tests build a real numpy index and run the real scoring code, so they need
neither `index/` nor the embedding model, and they run keyless and offline.

Scoring trick: the stubbed query embedding is the unit vector e_0, and a row
meant to score `s` is [s, sqrt(1-s^2), 0, ...]. That vector has unit length, so
`row @ e_0 == s` exactly - the same contract the real index relies on, and no
test-only scoring path to drift out of sync with production.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import settings  # noqa: E402
from src.retrieval import search as rs  # noqa: E402

DIM = settings.EMBEDDING_DIM
SCHEME = "HDFC Large Cap Fund - Direct - Growth"
OTHER = "HDFC Small Cap Fund - Direct - Growth"


def _row_vector(similarity: float) -> np.ndarray:
    """A unit vector whose dot product with e_0 is exactly `similarity`."""
    vector = np.zeros(DIM, dtype=np.float32)
    vector[0] = similarity
    vector[1] = math.sqrt(max(0.0, 1.0 - similarity * similarity))
    return vector


def _make_index(table: dict[tuple[str | None, str | None], list[tuple[str, float, str]]]):
    """Build (matrix, rows) from {(scheme, section): [(chunk_id, sim, scheme)]}."""
    vectors: list[np.ndarray] = []
    rows: list[dict] = []
    for (_f_scheme, f_section), entries in table.items():
        for chunk_id, similarity, scheme in entries:
            vectors.append(_row_vector(similarity))
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "text": f"text for {chunk_id}",
                    "source_url": "https://example.test/a",
                    "scheme": scheme,
                    "category": "faqs",
                    "section": f_section or "overview",
                    "chunk_index": len(rows),
                    "fetched_at": "",
                }
            )
    matrix = np.vstack(vectors) if vectors else np.zeros((0, DIM), dtype=np.float32)
    return matrix, rows


def _install(monkeypatch, table, seen: list | None = None) -> None:
    matrix, rows = _make_index(table)
    monkeypatch.setattr(rs, "_index", lambda: (matrix, rows))
    query = np.zeros(DIM, dtype=np.float32)
    query[0] = 1.0
    monkeypatch.setattr(rs, "embed_query", lambda q: query.tolist())
    if seen is not None:
        real_mask = rs._mask

        def spy(rows_, filter_scheme, filter_section):
            seen.append((filter_scheme, filter_section))
            return real_mask(rows_, filter_scheme, filter_section)

        monkeypatch.setattr(rs, "_mask", spy)


@pytest.fixture
def install(monkeypatch):
    def _do(table, seen=None):
        _install(monkeypatch, table, seen)

    return _do


# --- the window actually fills ------------------------------------------------


def test_window_fills_beyond_a_narrow_filter(install):
    """A section filter matching 2 chunks must not cap the window at 2.

    This is the regression that made TOP_K cosmetic: the old code returned the
    first non-empty attempt, so the prompt saw 2 chunks and the setting did
    nothing.
    """
    install(
        {
            (SCHEME, "fees"): [("a", 0.53, SCHEME), ("b", 0.44, SCHEME)],
            (SCHEME, None): [(f"c{i}", 0.40 - i / 100, SCHEME) for i in range(9)],
            (None, None): [(f"z{i}", 0.10, OTHER) for i in range(10)],
        }
    )

    hits = rs.search("What is the expense ratio of HDFC Large Cap Fund?", k=10)

    assert len(hits) == 10, "window must fill to k, not stop at the narrow filter"
    assert hits[0].chunk.chunk_id == "a", "the most precise match stays first"
    assert len({h.chunk.chunk_id for h in hits}) == 10, "no chunk counted twice"


def test_never_exceeds_k(install):
    install({(None, None): [(f"z{i}", 0.5, SCHEME) for i in range(30)]})
    assert len(rs.search("anything", k=10)) == 10
    assert len(rs.search("anything", k=3)) == 3


def test_hits_are_globally_deduplicated(install):
    """A chunk reachable through two filters must appear once, not twice."""
    shared = [("dup", 0.5, SCHEME), ("other", 0.4, SCHEME)]
    install({(SCHEME, "fees"): shared, (SCHEME, None): shared})
    hits = rs.search("expense ratio of HDFC Large Cap Fund", k=10)
    assert [h.chunk.chunk_id for h in hits].count("dup") == 1


def test_returns_empty_when_nothing_matches(install):
    install({})
    assert rs.search("who is the prime minister of india", k=10) == []


def test_ranked_by_similarity(install):
    install({(None, None): [("lo", 0.20, SCHEME), ("hi", 0.80, SCHEME), ("mid", 0.50, SCHEME)]})
    hits = rs.search("anything", k=10)
    assert [h.chunk.chunk_id for h in hits] == ["hi", "mid", "lo"]
    assert hits[0].similarity == pytest.approx(0.80, abs=1e-6)


def test_search_is_exhaustive(install):
    """The strongest match must be found even when it is a lone outlier.

    Chroma's HNSW was approximate and genuinely lost strong neighbours: the
    ELSS lock-in chunk scored 0.555 at rank 1 of 273 and went unreturned at
    n_results=8. An exact scan over a dense matrix has no candidate list to
    truncate, so that entire bug class cannot recur. This test is the guard.
    """
    # 300 decoys packed near each other, one clear winner far away in the
    # ranking. An approximate index with a small candidate list can miss it.
    decoys = [(f"d{i}", 0.30 + (i % 5) / 100, SCHEME) for i in range(300)]
    install({(None, None): decoys + [("winner", 0.95, SCHEME)]})
    hits = rs.search("anything", k=1)
    assert [h.chunk.chunk_id for h in hits] == ["winner"]


# --- the safety property that makes widening safe -----------------------------


@pytest.mark.parametrize("k", [1, 4, 10, 25])
def test_sufficiency_ignores_k(install, k):
    """`is_sufficient` reads the BEST hit, so widening k cannot rescue a bad query.

    This is the property that justified TOP_K=10. If it ever stops holding, a
    wider window starts answering out-of-scope questions and the change is no
    longer safe.
    """
    install({(None, None): [("a", 0.20, SCHEME), ("b", 0.10, SCHEME)]})
    assert rs.is_sufficient(rs.search("q", k=k)) is False

    install({(None, None): [("a", 0.80, SCHEME), ("b", 0.10, SCHEME)]})
    assert rs.is_sufficient(rs.search("q", k=k)) is True


def test_relaxation_is_a_last_resort(install):
    """Other schemes' chunks are admitted only after the detected scheme is spent.

    A cross-scheme fact ("the exit load is X", from the wrong fund) is worse
    than a narrow answer, so the unfiltered step must not be reached while the
    scheme-anchored step can still fill the window.
    """
    seen: list = []
    install(
        {
            (SCHEME, "exit_load"): [("mine", 0.9, SCHEME)],
            (SCHEME, None): [(f"mine{i}", 0.5, SCHEME) for i in range(9)],
            (None, None): [("theirs", 0.99, OTHER)],
        },
        seen,
    )

    hits = rs.search("exit load for HDFC Large Cap Fund?", k=10)

    assert "theirs" not in {h.chunk.chunk_id for h in hits}
    assert (None, None) not in seen, "the unfiltered step ran while the scheme step could still fill k"


def test_zero_query_embedding_is_rejected(install):
    """A degenerate embedding must not silently score everything as 0."""
    install({(None, None): [("a", 0.9, SCHEME)]})
    monkey = np.zeros(DIM, dtype=np.float32)
    rs.embed_query = lambda q: monkey.tolist()  # type: ignore[assignment]
    try:
        with pytest.raises(rs.IndexNotBuilt):
            rs.search("anything", k=10)
    finally:
        pass


# --- section detection (keyless, no index) ------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What is the expense ratio of HDFC Large Cap Fund?", "fees"),
        ("What is the exit load?", "exit_load"),
        ("What is the lock-in period for HDFC ELSS Tax Saver Fund?", "lock_in"),
        ("What is the minimum SIP amount?", "sip"),
        ("What are the SIPs?", "sip"),
        ("What is the riskometer category?", "riskometer"),
        ("Who manages this fund?", "management"),
        ("What is the NAV?", "nav"),
        ("What is the AUM?", "aum"),
        ("Who is the prime minister of India?", None),
    ],
)
def test_detect_section_filter(query: str, expected: str | None) -> None:
    assert rs.detect_section_filter(query) == expected


def test_ter_does_not_match_inside_riskometer() -> None:
    """Regression: the fees token "ter" (total expense ratio) is a substring of
    "riskometer".

    Substring matching sent every riskometer question to the fees section - the
    one section guaranteed not to contain the riskometer. It stayed invisible
    because the filter relaxation quietly recovered the right chunk, so the
    answer came out right for the wrong reason.
    """
    assert rs.detect_section_filter("What is the riskometer category?") == "riskometer"
    assert rs.detect_section_filter("What is the TER of HDFC Large Cap Fund?") == "fees"


# --- the store is still numpy, and still exhaustive ---------------------------


def test_store_no_longer_depends_on_chromadb() -> None:
    """Guards the memory decision that removed it.

    Chroma cost 84 MB of resident memory per process, which is what stopped the
    app fitting Render's 512 MB free-tier limit, and its HNSW index dropped
    strong neighbours. Both facts are easy to forget and easy to undo by
    convenience. See notes/retrieval.md.
    """
    store = (REPO_ROOT / "src" / "ingest" / "store.py").read_text(encoding="utf-8")
    assert "chromadb" not in store, "the numpy store must not import chromadb"
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "chromadb" not in requirements, "chromadb is no longer a dependency"


def test_index_is_exact_arithmetic() -> None:
    """Stored rows must be unit vectors, or a dot product is not a cosine."""
    from src.ingest.store import _normalise

    raw = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    out = _normalise(raw)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    with pytest.raises(Exception):
        _normalise(np.array([[0.0, 0.0]], dtype=np.float32))


def test_top_k_is_a_window_not_a_hard_filter() -> None:
    assert settings.TOP_K >= 2
    assert json.loads((REPO_ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))["top_k"] == settings.TOP_K, (
        "manifest is stale; re-run python -m src.ingest.run_all --rebuild"
    )
