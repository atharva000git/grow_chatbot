"""Deploy-build tests: the index build a hosting platform does not run.

The deployed app failed with `IndexNotBuilt` because `index/` is a gitignored
build artifact and nothing on Streamlit Cloud invokes the ingest. These tests
cover the two things that fix had to get right:

1. the build runs when - and only when - the index is missing, and
2. a rebuild from committed data does not restamp the freshness date, because
   that date is quoted to the user on every single answer.

The date tests are the important ones. `load_source` defaults a cached
document's `fetched_at` to today, which is correct for a fresh fetch and wrong
for a rebuild: without the override a redeploy silently moves every figure
forward to the deploy date, and the app's freshness line becomes a lie that
still looks correct.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.ingest import bootstrap  # noqa: E402
from src.ingest.load import IngestError, load_fetched_at, load_source  # noqa: E402
from src.models import SourceDoc  # noqa: E402

TODAY = date.today().isoformat()


def _doc(source_id: str = "hdfc-large-cap") -> SourceDoc:
    return SourceDoc(
        source_id=source_id,
        scheme="HDFC Large Cap Fund - Direct - Growth",
        category="groww",
        source_url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        fetched_at=TODAY,
        content_hash="",
        raw_path="",
        clean_path="",
        source_type="html",
    )


# --- load_fetched_at ----------------------------------------------------------


def test_reads_dates_from_a_chunks_file(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    rows = [
        {"chunk_id": "hdfc-large-cap::c0", "fetched_at": "2026-09-27"},
        {"chunk_id": "hdfc-large-cap::c1", "fetched_at": "2026-09-27"},
        {"chunk_id": "hdfc-elss::c0", "fetched_at": "2026-08-01"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    assert load_fetched_at(path) == {"hdfc-large-cap": "2026-09-27", "hdfc-elss": "2026-08-01"}


def test_reads_dates_from_a_plain_json_object(tmp_path: Path) -> None:
    path = tmp_path / "dates.json"
    path.write_text(json.dumps({"hdfc-large-cap": "2026-09-27"}), encoding="utf-8")

    assert load_fetched_at(path) == {"hdfc-large-cap": "2026-09-27"}


def test_missing_file_is_not_an_error(tmp_path: Path) -> None:
    assert load_fetched_at(tmp_path / "absent.jsonl") == {}


def test_a_json_array_is_rejected_rather_than_silently_empty(tmp_path: Path) -> None:
    path = tmp_path / "dates.json"
    path.write_text(json.dumps(["2026-09-27"]), encoding="utf-8")

    with pytest.raises(IngestError):
        load_fetched_at(path)


def test_rows_without_a_date_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        json.dumps({"chunk_id": "hdfc-large-cap::c0"}) + "\n" + json.dumps({"chunk_id": "x::c0", "fetched_at": ""}) + "\n",
        encoding="utf-8",
    )

    assert load_fetched_at(path) == {}


# --- load_source and the override --------------------------------------------


def test_a_cached_document_without_an_override_is_stamped_today(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = _doc()
    monkeypatch.setattr("src.ingest.load._paths", lambda *_: (tmp_path / "raw.html", _cached(tmp_path)))

    assert load_source(doc).fetched_at == TODAY


def test_the_override_wins_over_today_on_the_cached_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = _doc()
    monkeypatch.setattr("src.ingest.load._paths", lambda *_: (tmp_path / "raw.html", _cached(tmp_path)))

    assert load_source(doc, "2026-09-27").fetched_at == "2026-09-27"


def _cached(tmp_path: Path) -> Path:
    """A non-empty cleaned document, so `load_source` takes the cached path."""
    path = tmp_path / "clean.txt"
    path.write_text("Expense ratio 1.03% " * 20, encoding="utf-8")
    return path


def test_the_committed_corpus_dates_are_not_todays_date() -> None:
    """The corpus of record must carry real fetch dates, not build dates."""
    if not bootstrap.CORPUS_OF_RECORD.exists():
        pytest.skip("no committed corpus")
    dates = set(load_fetched_at(bootstrap.CORPUS_OF_RECORD).values())

    assert dates, "committed chunks carry no fetched_at"
    assert len(dates) == 1, f"documents were fetched on {len(dates)} different dates: {sorted(dates)}"


# --- ensure_index -------------------------------------------------------------


def test_ensure_index_is_a_no_op_when_the_index_exists() -> None:
    bootstrap._RESOLVED = None
    try:
        if not bootstrap.index_is_built():
            pytest.skip("no built index")
        assert bootstrap.ensure_index() is True
    finally:
        bootstrap._RESOLVED = None


def test_ensure_index_answers_from_cache_on_the_second_call() -> None:
    """The second call must not touch disk: Streamlit reruns on every keystroke."""
    calls: list[int] = []
    original = bootstrap.index_is_built
    bootstrap._RESOLVED = None

    def counting() -> bool:
        calls.append(1)
        return original()

    bootstrap.index_is_built = counting  # type: ignore[assignment]
    try:
        bootstrap.ensure_index()
        bootstrap.ensure_index()
    finally:
        bootstrap.index_is_built = original  # type: ignore[assignment]
        bootstrap._RESOLVED = None

    assert len(calls) == 1


def test_the_lock_file_lives_outside_the_index_directory() -> None:
    """`build_index` deletes and replaces the index directory wholesale."""
    assert bootstrap.LOCK_PATH.parent == bootstrap.INDEX_PATH.parent
    assert bootstrap.LOCK_PATH != bootstrap.INDEX_PATH
