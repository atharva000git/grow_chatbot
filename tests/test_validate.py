"""Generation post-validation tests. Keyless, no index, no network (Phase 11).

`validate(raw, hits)` is the layer that makes the output contract true, and it is
the only safety-critical piece that can be tested with no API key at all: it
takes the raw model string plus the hits, so every case here is exactly what the
model might have emitted.

The five cases are Phase 8's acceptance set. The rest are the failure modes a
model actually produces, which the spec's five do not name: run-on answers, bullet lists, a citation that is nearly right, and a refusal
that reaches for a number.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Chunk, RetrievedChunk  # noqa: E402

# Import the module, not the re-exported function: `src.generation` exports a
# function called `validate`, which shadows the submodule of the same name.
validate_mod = importlib.import_module("src.generation.validate")
validate = validate_mod.validate

FACTSHEET = "https://files.hdfcfund.com/s3fs-public/2026-09/HDFC%20MF%20Factsheet%20-%20August%202026.pdf"
LARGE_CAP = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
SMALL_CAP = "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"


import json as _json

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks" / "chunks.jsonl"


@pytest.fixture(scope="module")
def corpus_chunks() -> list[dict]:
    """The committed corpus. Not the index, so this stays keyless and offline."""
    if not CHUNKS_PATH.exists():
        pytest.skip("no built corpus; run python -m src.ingest.run_all")
    return [_json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def hit(url: str = LARGE_CAP, scheme: str = "HDFC Large Cap Fund - Direct - Growth",
        similarity: float = 0.62) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id="c1",
            text="EXIT LOAD: 1% if redeemed within 1 year from the date of investment.",
            source_url=url,
            scheme=scheme,
            category="Large Cap",
            section="Exit Load",
            chunk_index=0,
            fetched_at="2026-09-01",
        ),
        similarity=similarity,
    )


def hits(*urls: str) -> list[RetrievedChunk]:
    if not urls:
        return [hit()]
    return [hit(url) for url in urls]


# --- Phase 8 acceptance cases -------------------------------------------------

def test_case1_answer_with_valid_citation() -> None:
    r = validate(f"The exit load is 1% within 1 year. Source: {LARGE_CAP}", hits())
    assert r.intent == "ANSWER"
    assert r.citation_url in {h.chunk.source_url for h in hits()}
    assert r.citation_url and r.citation_url.startswith("https://")


def test_case2_performance_claim_refused() -> None:
    r = validate(
        f"The fund returned 32% in 5 years. Source: https://example.com/fake",
        hits(),
    )
    assert r.intent == "REFUSAL"
    assert "example.com" not in r.answer_text
    assert "example.com" not in (r.citation_url or "")


def test_case3_no_context_refused() -> None:
    r = validate("INSUFFICIENT_CONTEXT", hits())
    assert r.intent == "INSUFFICIENT_CONTEXT"
    assert r.citation_url is None
    assert len(r.answer_text.split(".")) <= 3


def test_case4_advice_refused() -> None:
    r = validate("You should buy this fund. Source: https://example.com/x", hits())
    assert r.intent == "REFUSAL"
    assert "example.com" not in r.answer_text


def test_case5_five_sentences_truncated_to_three() -> None:
    raw = (
        f"One. Two. Three. Four. Five. Source: {LARGE_CAP}"
    )
    r = validate(raw, hits())
    assert r.intent == "ANSWER"
    assert validate_mod.count_sentences(r.answer_text) <= 3


def test_every_answer_carries_a_freshness_line() -> None:
    r = validate(f"The exit load is 1%. Source: {LARGE_CAP}", hits())
    assert validate_mod.FRESHNESS_RE.search(r.answer_text), r.answer_text
    assert r.last_updated == "2026-09-01"


# --- Citation allow-list ------------------------------------------------------

def test_invented_citation_is_replaced_or_refused() -> None:
    """The model is told the allow-list and ignores it. Never emit its URL."""
    r = validate("The exit load is 1%. Source: https://evil.example/random", hits())
    assert r.intent in {"ANSWER", "REFUSAL"}
    assert "evil.example" not in r.answer_text
    if r.citation_url:
        assert r.citation_url in {h.chunk.source_url for h in hits()}


def test_citation_allowlist_is_the_hit_urls_not_the_whole_registry() -> None:
    """A real source, but not one behind this answer, is still not allowed.

    The allow-list is built per call from the retrieved chunks, not from
    sources.csv, so the model cannot cite a genuine HDFC page that this answer
    did not actually come from.
    """
    allow = set(validate_mod.citation_allowlist(hits(LARGE_CAP)))
    assert allow == {LARGE_CAP}
    assert SMALL_CAP not in allow, "an un-retrieved source must not become citable"


def test_no_hits_means_nothing_is_citable() -> None:
    assert list(validate_mod.citation_allowlist([])) == []


# --- Shape and framing --------------------------------------------------------

def test_a_bulleted_answer_still_meets_the_sentence_contract() -> None:
    """The contract is at most 3 sentences, whatever shape the model chose.

    Phase 8 does not require bullets to be flattened, so this asserts the rule
    that does exist rather than one invented here: the answer is capped and
    cites a permitted source even when the model formatted it as a list.
    """
    r = validate(
        f"- The exit load is 1%\n- It applies within 1 year\n- Source: {LARGE_CAP}",
        hits(),
    )
    assert validate_mod.count_sentences(r.answer_text) <= 3
    assert r.citation_url in {h.chunk.source_url for h in hits()}
    assert "1%" in r.answer_text


def test_single_sentence_answer_survives_intact() -> None:
    r = validate(f"The lock-in period is 3 years. Source: {LARGE_CAP}", hits())
    assert "3 years" in r.answer_text
    assert r.intent == "ANSWER"


def test_top_similarity_is_carried_for_the_ui() -> None:
    """The UI shows this to evidence grounding, so it must not be lost."""
    r = validate(f"The exit load is 1%. Source: {LARGE_CAP}", hits(LARGE_CAP, SMALL_CAP))
    assert r.top_similarity == 0.62
    assert r.retrieval_hits == 2


def test_refusals_carry_no_provenance() -> None:
    """A refusal is not a fact from a document, so no source date is attached."""
    r = validate("The fund returned 32% in 5 years. Source: " + FACTSHEET, hits())
    assert r.intent == "REFUSAL"
    assert r.last_updated == ""


def test_performance_detector_separates_facts_from_claims() -> None:
    """The distinction that the spec's own regex got wrong.

    "Exit load is 1% for 12 months" is a fact and "returned 32% in 5 years" is a
    claim. A bare percentage only counts as performance with in/over plus a
    word-bounded unit.
    """
    assert not validate_mod.has_performance_language("The exit load is 1% for 12 months.")
    assert validate_mod.has_performance_language("The fund returned 32% in 5 years.")
    assert validate_mod.has_performance_language("It gave 12.4% over 3 years.")
    assert not validate_mod.has_performance_language("The expense ratio is 1.5%.")


def test_advice_language_detector_ignores_asset_allocation() -> None:
    """"Balanced asset allocation" is the BA fund's mandate, not a suggestion."""
    assert not validate_mod.has_advice_language(
        "The fund follows a balanced asset allocation strategy across equity and debt."
    )
    assert validate_mod.has_advice_language("You should allocate 60% to equity.")


def test_the_real_returns_table_cannot_be_quoted(corpus_chunks: list[dict]) -> None:
    """The corpus does contain a returns table; the output guard is what stops it.

    Worth being explicit about, because the design intent is "no performance data
    by answer". The Groww pages carry a "Returns and rankings" table and a
    "Return calculator" widget, both of which are chunked into the corpus - a
    fact section boundary even runs the returns table into the exit-load section,
    so an exit-load query retrieves it as its top hit. The only thing preventing
    those numbers from reaching a user is this layer, so it is tested against the
    real chunk rather than an invented example.
    """
    tables = [c for c in corpus_chunks if "Fund returns" in c["text"] or "Historic returns" in c["text"]]
    assert tables, "expected the returns tables to be present in the corpus"

    for row in tables:
        url = row["source_url"]
        for quoted in (f"The fund returned {row['text'][:60]}. Source: {url}", row["text"][:120]):
            r = validate(quoted, hits(url, row["scheme"]))
            assert r.intent == "REFUSAL", (
                f"a returns table from {row['chunk_id']} was answered as {r.intent}: {r.answer_text[:80]!r}"
            )
            assert "%" not in r.answer_text, f"a percentage leaked into a refusal: {r.answer_text[:80]!r}"


def test_empty_hits_cannot_produce_a_citation() -> None:
    r = validate(f"The exit load is 1%. Source: {LARGE_CAP}", [])
    assert r.citation_url is None
    assert r.retrieval_hits == 0


@pytest.mark.parametrize(
    "raw",
    [
        f"The exit load is 1%. Source: {LARGE_CAP}",
        f"The exit load is 1%. Source: {FACTSHEET}",
        f"Exit load: 1% if redeemed within 1 year. Source: {LARGE_CAP}",
        f"The NAV was 62.4831. Source: {LARGE_CAP}",
    ],
    ids=["groww", "factsheet", "field", "nav"],
)
def test_well_formed_answers_pass_through(raw: str) -> None:
    """A correct answer must not be damaged by the validator."""
    r = validate(raw, hits(LARGE_CAP, FACTSHEET))
    assert r.intent == "ANSWER"
    assert validate_mod.count_sentences(r.answer_text) <= 3
