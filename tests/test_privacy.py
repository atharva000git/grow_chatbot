"""NFR-5: the API key must never be logged, echoed, or written to disk.

Added after a live verification run. The key existed only in the gitignored
`.env` - no commit, no log, no tracked file contained it - so there was nothing
to clean up. This file exists so that stays true rather than being true once.

These are contract tests, not leak tests: nothing here greps a corpus for a
secret, because the guarantee is structural. The key is read in exactly one
place (`llm._client_config`) and the assertions below check that every path a
response or a log record can take omits it.
"""

from __future__ import annotations

import ast
import json

import pytest

from config import settings
from src.generation import llm as llm_mod
from src.generation.validate import validate
from src.models import Chunk, RetrievedChunk
import src.logging_utils as logging_utils

# Deliberately NOT key-shaped. An earlier value began "sk-not-a-real-key-", which
# is secret-scanner bait: GitHub push protection and most CI scanners match on
# the prefix, not on whether the key works, so publishing it would raise a
# false positive on a file that contains no secret.
SECRET = "SENTINELCANARY0123456789abcdef"


def _is_key_literal(node: ast.AST) -> bool:
    """True for the string "LLM_API_KEY" wherever it appears as a value."""
    return isinstance(node, ast.Constant) and node.value == "LLM_API_KEY"


def _dotted(node: ast.AST) -> str:
    """`os.environ.get` -> 'os.environ.get', so a nested call is nameable.

    Taking only the outermost attribute reports `get`, which matches none of the
    readers being looked for and let a real violation through.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


@pytest.fixture
def hit() -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="hdfc-large-cap::c0",
        text="Exit load of 1.00% if redeemed within 1 year from the date of allotment.",
        source_url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        scheme="HDFC Large Cap Fund - Direct - Growth",
        category="Large Cap",
        section="exit_load",
        chunk_index=0,
        fetched_at="2026-09-27T09:00:00+00:00",
    )
    return RetrievedChunk(chunk=chunk, similarity=0.62)


def test_no_module_logs_the_settings_object_or_the_key() -> None:
    """The easy accident is one `log.info("cfg=%s", settings)` and the key is on disk.

    A previous version of this test asserted the canary was absent from
    `repr(settings)`, which could never fail - a module's repr is its import
    path and none of its attributes.

    Parsed with `ast`, not regex: a regex cannot tell code from a string, and
    `LLM_API_KEY` legitimately appears inside user-facing text ("set
    `LLM_API_KEY` in `.env`"), in the `os.getenv` call that defines it, and in
    attribute access wrapped across lines. All three produced false positives.
    Two shapes are actually forbidden:

    1. the settings module handed to a log/print/streamlit sink as a bare Name;
    2. the key read through anything other than `settings.LLM_API_KEY`.

    Violated deliberately during development to confirm this fails.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    self_name = Path(__file__).name
    offenders: list[str] = []
    checked = 0

    for path in sorted(root.rglob("*.py")):
        if any(part in (".venv", "__pycache__", ".git") for part in path.parts):
            continue
        if path.name == self_name:  # this file names the patterns it forbids
            continue
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(root)
        is_settings_module = rel.as_posix() == "config/settings.py"
        tree = ast.parse(source, filename=str(path))
        checked += 1

        for node in ast.walk(tree):
            # 1. settings object passed whole to a sink
            if isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "attr", None) or getattr(func, "id", "")
                if name.startswith(("log", "print")) or name in {"error", "warning", "info", "markdown", "caption", "write", "exception"}:
                    if any(isinstance(a, ast.Name) and a.id == "settings" for a in node.args):
                        offenders.append(f"{rel}:{node.lineno}: passes the settings object to {name}()")

            # 2. the key read through anything but the settings module. This
            #    covers `settings.LLM_API_KEY` being bypassed for an env
            #    subscript, `os.environ.get("LLM_API_KEY")` or `os.getenv(...)`,
            #    each of which is a second, unlogged path to the secret.
            if isinstance(node, ast.Attribute) and node.attr == "LLM_API_KEY":
                owner = node.value
                if not (isinstance(owner, ast.Name) and owner.id == "settings"):
                    offenders.append(f"{rel}:{node.lineno}: reads LLM_API_KEY via {ast.dump(owner)[:40]}")
            if isinstance(node, ast.Subscript) and _is_key_literal(node.slice):
                if not is_settings_module:
                    offenders.append(f"{rel}:{node.lineno}: reads the key from an env subscript")
            if isinstance(node, ast.Call) and not is_settings_module:
                flat = _dotted(node.func)
                if any(
                    reader in flat for reader in ("environ", "getenv", "dotenv_values", "dotenv")
                ) and any(
                    _is_key_literal(arg) or _is_key_literal(kw.value)
                    for arg in (*node.args, *(kw.value for kw in node.keywords))
                ):
                    offenders.append(f"{rel}:{node.lineno}: reads the key via {flat or 'a call'}")

    assert checked > 10, f"only scanned {checked} files - the glob is wrong"
    assert not offenders, "possible key disclosure:\n  " + "\n  ".join(offenders)


def test_missing_key_error_never_echoes_a_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LLM_API_KEY", "")
    with pytest.raises(llm_mod.LLMUnavailable) as excinfo:
        llm_mod.get_client()
    message = str(excinfo.value)
    assert "LLM_API_KEY" in message, "the error must name the variable"
    assert ".env" in message, "the error must say where to set it"
    assert SECRET not in message
    assert "SENTINELCANARY" not in message
    assert "sk-" not in message, "the error must not echo anything key-shaped"


def test_client_config_is_the_only_reader_of_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """`get_client` is the single funnel into the HTTP call.

    It returns the key because `requests` needs it, so the contract is that it
    is the *only* thing that does - not that the key is unreadable.
    """
    monkeypatch.setattr(settings, "LLM_API_KEY", SECRET)
    config = llm_mod.get_client()
    assert config["api_key"] == SECRET
    assert config["model"] and config["base_url"]


@pytest.mark.parametrize(
    "raw",
    [
        "The exit load is 1.00% if redeemed within 1 year.",
        "The NAV was 62.4831.",
        "Last updated from sources: 2024-01-01",
    ],
    ids=["answer", "nav", "invented-date"],
)
def test_no_response_field_carries_the_key(
    raw: str, hit: RetrievedChunk, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever the model returns, the ChatResponse cannot echo the key back."""
    monkeypatch.setattr(settings, "LLM_API_KEY", SECRET)
    response = validate(raw, [hit])
    blob = " ".join(
        [
            response.answer_text,
            response.citation_url or "",
            response.source_scheme or "",
            response.last_updated,
            str(response.intent),
        ]
    )
    assert SECRET not in blob


def test_query_log_writes_only_sanitised_text(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A log record is `{ts, query, intent}` - three keys, no settings, no headers.

    The PII guard is what makes `query` safe; this pins the record's *shape*, so
    a future field cannot quietly widen what reaches disk.
    """
    log_file = tmp_path / "query_log.jsonl"
    monkeypatch.setattr(logging_utils, "QUERY_LOG_PATH", log_file)
    monkeypatch.setattr(settings, "LOG_QUERIES", True)
    monkeypatch.setattr(settings, "LLM_API_KEY", SECRET)

    logging_utils.log_query("expense ratio of HDFC Large Cap Fund", "ANSWER")

    record = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert set(record) == {"ts", "query", "intent"}
    assert record["intent"] == "ANSWER"
    assert SECRET not in log_file.read_text(encoding="utf-8")


def test_query_log_is_a_no_op_when_disabled(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_file = tmp_path / "query_log.jsonl"
    monkeypatch.setattr(logging_utils, "QUERY_LOG_PATH", log_file)
    monkeypatch.setattr(settings, "LOG_QUERIES", False)
    logging_utils.log_query("expense ratio of HDFC Large Cap Fund", "ANSWER")
    assert not log_file.exists(), "logging must stay off by default (NFR-5)"


def test_dotenv_is_gitignored() -> None:
    """The file holding the key must not be committable.

    Guards the `.gitignore` entry itself: a well-meaning `git add -A` during
    cleanup is how secrets usually escape, and this is the cheapest place to
    notice.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    ignore = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    patterns = {line.strip() for line in ignore if line.strip() and not line.startswith("#")}
    assert ".env" in patterns, ".env must stay in .gitignore"
    assert (root / ".env.example").exists(), "the tracked template must still exist"
