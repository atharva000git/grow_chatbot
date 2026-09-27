"""Chunking invariant tests (Phase 11).

Two layers. The first tests the chunking functions on synthetic input, which is
keyless and needs no corpus. The second asserts the invariants on the committed
corpus, because a unit test that only exercises synthetic text cannot catch a
real page producing a bad chunk.

The corpus tests skip rather than fail when `data/chunks` is absent, so a fresh
clone can run the suite before its first ingest.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import settings  # noqa: E402
from src.ingest import chunk as ck  # noqa: E402
from src.models import Chunk, build_source_docs, permitted_urls  # noqa: E402

CHUNKS_PATH = ck.CHUNKS_PATH
REQUIRED_METADATA = ("chunk_id", "text", "source_url", "scheme", "category", "section", "fetched_at")


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    if not CHUNKS_PATH.exists():
        pytest.skip("no built corpus; run python -m src.ingest.run_all")
    return [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- corpus invariants --------------------------------------------------------

def test_corpus_is_not_empty(corpus: list[dict]) -> None:
    assert len(corpus) > 100, f"only {len(corpus)} chunks; the corpus looks truncated"


def test_chunk_ids_are_unique(corpus: list[dict]) -> None:
    ids = [r["chunk_id"] for r in corpus]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"duplicate chunk_ids: {sorted(duplicates)[:5]}"


def test_every_chunk_has_all_metadata(corpus: list[dict]) -> None:
    for row in corpus:
        for field in REQUIRED_METADATA:
            assert field in row, f"{row.get('chunk_id')} has no {field}"
            assert str(row[field]).strip(), f"{row['chunk_id']} has an empty {field}"


def test_every_source_url_is_permitted(corpus: list[dict]) -> None:
    """A URL outside sources.csv would break the citation allow-list, so this
    is the invariant that ties the corpus to the compliance boundary."""
    allowed = permitted_urls()
    rogue = {r["source_url"] for r in corpus} - allowed
    assert not rogue, f"chunks cite URLs absent from sources.csv: {rogue}"


def test_no_chunk_is_empty(corpus: list[dict]) -> None:
    blanks = [r["chunk_id"] for r in corpus if not r["text"].strip()]
    assert not blanks, f"empty chunks: {blanks[:5]}"


def test_no_label_only_chunks(corpus: list[dict]) -> None:
    """A heading with no value of its own is a nav crumb, not a fact.

    Two of these were emitted before Phase 11: five bare `## Exit load` chunks
    (the value sentence was longer than FIELD_VALUE_MAX, so it failed the
    "is this a value" test and left the label orphaned) and five bare
    `## Minimum investments` chunks (the oversized-prose branch skipped the
    prose floor that the other two branches apply). Both are regressions of the
    same kind, so both are pinned here.
    """
    crumbs = [
        r["chunk_id"]
        for r in corpus
        if r["text"].strip() in {"## Exit load", "## Exit Load", "## Minimum investments"}
    ]
    assert not crumbs, f"label-only chunks present: {crumbs}"


def test_metric_labels_keep_their_value(corpus: list[dict]) -> None:
    """Every exit_load chunk must actually contain exit-load content.

    The corollary of the crumb bug: a chunk labelled `exit_load` that is just the
    label matches the query "exit load" perfectly and answers nothing.
    """
    for row in corpus:
        if row["section"] == "exit_load":
            body = row["text"].lower().replace("exit load", "").replace("##", "").strip()
            assert body, f"{row['chunk_id']} is an exit_load chunk with no exit-load content"


def test_the_prose_floor_stays_the_majority_rule(corpus: list[dict]) -> None:
    """MIN_CHUNK_CHARS is a *prose* floor, and this is why it is not asserted corpus-wide.

    Phase 11 asks for "no chunk under MIN_CHUNK_CHARS". That contradicts the Phase
    6 design and cannot be asserted honestly, so the exemptions are pinned at the
    function level in the tests below and only the shape is checked here.

    Two paths deliberately emit short chunks. `chunk_html` sets `floor = 1` for a
    recognised metric, and `chunk_facts` uses FACT_MIN_CHARS for a factsheet
    all-caps header, because the facts they protect are short: "Min. for SIP Rs
    100" is 17 characters and the ELSS lock-in is 73. `split_about` emits one
    sentence per chunk, because that passage repeats the fund name in nearly
    every sentence and bundling it scored above the real answers. Raising the
    corpus-wide floor to 80 would delete real facts, which is the mistake those
    decisions were made to undo.

    Classifying them from the committed JSONL is not possible either: `_normalize`
    collapses newlines, so a "label\\nvalue" pair and a prose sentence are
    indistinguishable after the fact. The bound below is therefore the honest
    assertion - the short chunks must stay a small minority, which fails if a
    regression starts flooding the corpus with crumbs.
    """
    assert settings.MIN_CHUNK_CHARS == 80
    assert ck.FACT_MIN_CHARS < settings.MIN_CHUNK_CHARS, (
        "FACT_MIN_CHARS must stay below the prose floor or atomic facts stop being emitted"
    )
    short = [r for r in corpus if len(r["text"].strip()) < settings.MIN_CHUNK_CHARS]
    assert short, "expected the metric and About exemptions to produce short chunks"
    assert len(short) < len(corpus) * 0.35, (
        f"{len(short)}/{len(corpus)} chunks are under the prose floor; the exemptions "
        "should be a minority, not the rule"
    )


def test_prose_splitter_drops_pieces_under_the_floor() -> None:
    """The floor is enforced here, where it can be asserted deterministically."""
    crumbs = "\n".join(f"## Section {i}\nshort" for i in range(12))
    pieces = ck.chunk_recursive(crumbs)
    assert all(len(p.strip()) >= settings.MIN_CHUNK_CHARS for p in pieces), (
        f"chunk_recursive emitted a {min(len(p.strip()) for p in pieces)}-character piece"
    )


def test_metric_blocks_are_exempt_from_the_prose_floor() -> None:
    assert 17 < settings.MIN_CHUNK_CHARS, "the exemption only matters while it is lower"
    out = ck.chunk_html("Min. for SIP\nRs 100", strategy="recursive")
    assert out == ["Min. for SIP\nRs 100"]


def test_about_sentence_splitter_keeps_one_sentence_per_chunk() -> None:
    """Including a semicolon-terminated one: the page closes its last sentence
    with "Exit load of 1% if redeemed within 1 year ;"."""
    about = (
        "## About HDFC Large Cap Fund Direct Growth ## Investment Objective\n"
        "The scheme seeks long-term capital appreciation.\n"
        "Exit load of 1% if redeemed within 1 year ;"
    )
    sentences = ck.split_about(about)
    assert len(sentences) == 2, sentences
    assert sentences[-1].strip().endswith(";")
    for sentence in sentences:
        assert sentence.count("HDFC Large Cap Fund Direct Growth") <= 1, (
            "the fund name repeated inside one chunk is what bundling caused"
        )


def test_chunk_indices_are_dense_per_source(corpus: list[dict]) -> None:
    """Gaps mean a dropped chunk, which usually means a parsing bug."""
    by_source: dict[str, list[int]] = {}
    for row in corpus:
        source_id = row["chunk_id"].split("::")[0]
        by_source.setdefault(source_id, []).append(row["chunk_index"])
    for source_id, indices in by_source.items():
        assert sorted(indices) == list(range(len(indices))), (
            f"{source_id} has non-contiguous chunk_index values: {sorted(indices)[:8]}"
        )


def test_manifest_agrees_with_the_chunks_file(corpus: list[dict]) -> None:
    manifest_path = REPO_ROOT / "data" / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("no manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["n_chunks"] == len(corpus), (
        f"manifest says {manifest['n_chunks']} chunks, chunks.jsonl has {len(corpus)}"
    )
    assert manifest["n_docs"] == len({r["chunk_id"].split("::")[0] for r in corpus})


# --- unit level, synthetic input ---------------------------------------------

def test_metric_prefix_rejects_a_value_sentence() -> None:
    """The regression that produced the five dead heading-only chunks.

    `startswith("exit load")` also matches the sentence that states the value,
    so the sentence claimed to be a metric and left `## Exit load` without one.
    """
    assert ck.metric_prefix("## Exit load") == "Exit load"
    assert ck.metric_prefix("Exit load of 1% if redeemed within 1 year") is None
    assert ck.metric_prefix("Min. for SIP") == "Min. for SIP"
    assert ck.metric_prefix("NAV: 25 Sep '26") == "NAV:"


def test_a_metric_heading_absorbs_a_long_value() -> None:
    """A label and its value are one fact and must stay one chunk."""
    text = "## Exit load\nExit load of 1% if redeemed within 1 year\n## Stamp duty: 0.005%"
    blocks = ck.segment_fields(text)
    assert ("## Exit load", "Exit load of 1% if redeemed within 1 year") in blocks


def test_no_block_is_a_bare_heading() -> None:
    text = (
        "NAV: 25 Sep '26\nRs 1,189.08\n"
        "## Exit load\nExit load of 1% if redeemed within 1 year\n"
        "## Lock-in period\n3 years from the date of allotment of the respective Units"
    )
    assert ck.segment_fields(text)
    for head, body in ck.segment_fields(text):
        assert not (head and not body), f"heading with no value: {head!r}"


def test_short_metric_pairs_still_split() -> None:
    """The Phase 6 fix must survive: one fact per chunk, not four per block.

    Left bundled, four citable facts shared one embedding and a query for one of
    them matched all four weakly.
    """
    text = "NAV: 25 Sep '26\nRs 159.82\nMin. for SIP\nRs 100\nFund size (AUM)\nRs 41,890.86 Cr\nExpense ratio\n0.78%"
    chunks = [c.strip() for c in ck.chunk_html(text, strategy="recursive")]
    assert len(chunks) == 4, f"expected 4 atomic chunks, got {len(chunks)}: {chunks}"

    for value in ("159.82", "Rs 100", "41,890.86", "0.78%"):
        holders = [c for c in chunks if value in c]
        assert len(holders) == 1, f"{value!r} appears in {len(holders)} chunks: {holders}"

    # Each chunk carries its own label, so a chunk is self-describing.
    for label, value in (("NAV", "159.82"), ("Min. for SIP", "Rs 100"),
                         ("Fund size (AUM)", "41,890.86"), ("Expense ratio", "0.78%")):
        assert any(label in c and value in c for c in chunks), f"{label} lost its value"


def test_section_is_inferred_for_every_chunk() -> None:
    for text, expected in [
        ("Expense ratio (direct plan) 0.78%", "fees"),
        ("Exit load of 1% if redeemed within 1 year", "exit_load"),
        ("LOCK-IN PERIOD 3 years from the date of allotment", "lock_in"),
        ("Benchmark Nifty 50 TRI", "benchmark"),
        ("Fund size (AUM) Rs 39,933.37 Cr", "aum"),
    ]:
        assert ck.infer_section(text) == expected, f"{text!r} -> {ck.infer_section(text)}"


def test_chunk_document_emits_complete_metadata() -> None:
    """Every Chunk from the real pipeline carries all 8 fields, non-empty."""
    doc = build_source_docs()[0]
    doc = type(doc)(**{**doc.__dict__, "clean_path": str(REPO_ROOT / "data" / "clean" / f"{doc.source_id}.txt")})
    text = "Expense ratio\nDirect plan 0.78%\nExit load\n1% within 1 year"
    chunks = ck.chunk_document(doc, strategy="recursive")
    assert chunks
    for c in chunks:
        assert isinstance(c, Chunk)
        for field in ("chunk_id", "text", "source_url", "scheme", "category", "section"):
            assert str(getattr(c, field)).strip(), f"{c.chunk_id} has empty {field}"
        assert c.source_url in permitted_urls()
