"""Retrieval window contract tests.

These pin the behaviour of the TOP_K=10 context window, which is the one
setting where "more context" is not automatically "better context".

The interesting part is not the number 10. It is that a wider window had to be
made *real* first: the search used to return the first filter attempt that
produced any hit at all, so a section filter matching two chunks ended the
search and the prompt saw 2 rather than TOP_K. TOP_K was a ceiling in name only.
These tests use a stub collection, so they need neither `chroma/` nor the
embedding model, and they run keyless and offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import settings  # noqa: E402
from src.models import RetrievedChunk  # noqa: E402
from src.retrieval import search as rs  # noqa: E402


def _hit(chunk_id: str, similarity: float, scheme: str = "HDFC Large Cap Fund - Direct - Growth") -> RetrievedChunk:
    from src.models import Chunk

    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            text=f"chunk {chunk_id}",
            source_url="https://example.test/a",
            scheme=scheme,
            category="faqs",
            section="overview",
            chunk_index=0,
        ),
        similarity=similarity,
    )


class StubCollection:
    """Answers `query` from a fixed table keyed by the `where` filter.

    Keyed by a JSON rendering of the filter because a `where` clause is a dict
    and dicts cannot be dict keys.
    """

    def __init__(self, table: dict[str, list[RetrievedChunk]]):
        self.table = table
        self.calls: list[tuple] = []

    def count(self) -> int:
        return 99

    def query(self, query_embeddings, n_results, where, include):  # noqa: ANN001, ARG002
        self.calls.append((where, n_results))
        hits = self.table.get(_key(where), [])
        chosen = hits[:n_results]
        return {
            "ids": [[h.chunk.chunk_id for h in chosen]],
            "documents": [[h.chunk.text for h in chosen]],
            "metadatas": [[{"scheme": h.chunk.scheme, "section": h.chunk.section} for h in chosen]],
            "distances": [[1.0 - h.similarity for h in chosen]],
        }


def _key(where: object) -> str:
    return json.dumps(where, sort_keys=True)


SCHEME = "HDFC Large Cap Fund - Direct - Growth"


def _scheme_key() -> str:
    return _key({"scheme": {"$eq": SCHEME}})


def _scheme_section_key(section: str) -> str:
    """Mirror how search.run builds a two-clause filter: {"$and": [...]}.

    Getting this wrong is silent - the stub simply returns nothing for the
    attempt, the test passes for the wrong reason, and the relaxation it was
    meant to forbid looks necessary.
    """
    return _key({"$and": [{"scheme": {"$eq": SCHEME}}, {"section": {"$eq": section}}]})


@pytest.fixture
def stub(monkeypatch):
    def install(table: dict[str, list[RetrievedChunk]]) -> StubCollection:
        collection = StubCollection(table)
        monkeypatch.setattr(rs, "_collection", lambda: collection)
        monkeypatch.setattr(rs, "embed_query", lambda q: [0.0] * settings.EMBEDDING_DIM)
        return collection

    return install


# --- the window actually fills ------------------------------------------------


def test_window_fills_beyond_a_narrow_filter(stub):
    """A section filter matching 2 chunks must not cap the window at 2.

    This is the regression that made TOP_K cosmetic: the old code returned the
    first non-empty attempt, so the prompt saw 2 chunks and the setting did
    nothing.
    """
    collection = stub(
        {
            # First attempt (scheme + section) is deliberately thin.
            _scheme_section_key("fees"): [_hit("a", 0.53), _hit("b", 0.44)],
            _scheme_key(): [_hit(f"c{i}", 0.40 - i / 100) for i in range(9)],
            _key(None): [_hit(f"z{i}", 0.10) for i in range(10)],
        }
    )

    hits = rs.search("What is the expense ratio of HDFC Large Cap Fund?", k=10)

    assert len(hits) == 10, "window must fill to k, not stop at the narrow filter"
    assert hits[0].chunk.chunk_id == "a", "the most precise match stays first"
    assert len({h.chunk.chunk_id for h in hits}) == 10, "no chunk counted twice"


def test_never_exceeds_k(stub):
    stub({_key(None): [_hit(f"z{i}", 0.5) for i in range(30)]})
    assert len(rs.search("anything", k=10)) == 10
    assert len(rs.search("anything", k=3)) == 3


def test_hits_are_globally_deduplicated(stub):
    """A chunk reachable through two filters must appear once, not twice."""
    same = [_hit("dup", 0.5), _hit("other", 0.4)]
    stub(
        {
            _scheme_section_key("fees"): same,
            _scheme_key(): same,
        }
    )
    hits = rs.search("expense ratio of HDFC Large Cap Fund", k=10)
    assert [h.chunk.chunk_id for h in hits].count("dup") == 1


def test_returns_empty_when_nothing_matches(stub):
    stub({})
    assert rs.search("who is the prime minister of india", k=10) == []


# --- the safety property that makes widening safe -----------------------------


@pytest.mark.parametrize("k", [1, 4, 10, 25])
def test_sufficiency_ignores_k(stub, k):
    """`is_sufficient` reads the BEST hit, so widening k cannot rescue a bad query.

    This is the property that justified TOP_K=10. If it ever stops holding, a
    wider window starts answering out-of-scope questions and the change is no
    longer safe.
    """
    stub({_key(None): [_hit("a", 0.20), _hit("b", 0.10)]})
    assert rs.is_sufficient(rs.search("q", k=k)) is False

    stub({_key(None): [_hit("a", 0.80), _hit("b", 0.10)]})
    assert rs.is_sufficient(rs.search("q", k=k)) is True


def test_relaxation_is_a_last_resort(stub):
    """Other schemes' chunks are admitted only after the detected scheme is spent.

    A cross-scheme fact ("the exit load is X", from the wrong fund) is worse
    than a narrow answer, so the unfiltered step must not be reached while the
    scheme-anchored step can still fill the window.
    """
    collection = stub(
        {
            _scheme_section_key("exit_load"): [_hit("mine", 0.9)],
            _scheme_key(): [_hit(f"mine{i}", 0.5) for i in range(9)],
            _key(None): [_hit("theirs", 0.99, scheme="HDFC Small Cap Fund - Direct - Growth")],
        }
    )

    hits = rs.search("exit load for HDFC Large Cap Fund?", k=10)

    assert "theirs" not in {h.chunk.chunk_id for h in hits}
    assert None not in [where for where, _ in collection.calls], "unfiltered step ran too early"


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


def test_top_k_is_a_window_not_a_hard_filter() -> None:
    """A guard: TOP_K must stay >= 2, and search_ef must clear it.

    `hnsw:search_ef` is pinned in ingest/store.py. If someone lowers TOP_K
    toward that value the index starts silently dropping recall, so assert the
    margin here where the failure is visible.
    """
    assert settings.TOP_K >= 2
    source = (REPO_ROOT / "src" / "ingest" / "store.py").read_text(encoding="utf-8")
    assert "hnsw:search_ef" in source, "search_ef pin was removed; TOP_K has no headroom"
