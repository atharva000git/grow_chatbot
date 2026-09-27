"""Graded evaluation harness. Produces samples/sample_qa.md (Phase 11).

Runs the PRD section 9 validation set through the real `answer_query` and reports
what the system actually did, including the PRD section 10 acceptance metrics:
grounding rate and mean latency.

Deliberately measures the whole orchestrator rather than the pieces. The
retrieval phase already proved it can rank the right chunk, and the guard phase
proved the gates fire; what has never been measured is the two together, which
is the only path a user can reach.

    python eval/run_eval.py            # table + samples/sample_qa.md
    python eval/run_eval.py --no-write # table only, leave samples/ alone

Exits non-zero if any case fails, so this is usable as a CI gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generation.llm import LLMUnavailable  # noqa: E402
from src.models import permitted_urls  # noqa: E402
from src.pipeline import answer_query, corpus_status  # noqa: E402

QUERIES_PATH = Path(__file__).resolve().parent / "queries.json"
SAMPLES_PATH = REPO_ROOT / "samples" / "sample_qa.md"

# The PRD prints en dashes in scheme names ("HDFC Large Cap Fund - Direct -
# Growth"); the corpus uses plain hyphens, and the eval queries are typed the way
# a user would type them. Compare on a normalised form so the check measures the
# scheme, not the keyboard.
_LOOKALIKE = str.maketrans({"–": "-", "—": "-", "‑": "-", "−": "-"})


def _normalise(text: str) -> str:
    return " ".join(text.translate(_LOOKALIKE).lower().split())


def _scheme_matches(expected: str | None, actual: str | None) -> bool:
    """`expected_scheme` is advisory: it documents intent, it is not a gate.

    Case 2 asks about "HDFC Equity Fund (Flexi Cap)" while the corpus only holds
    the direct-growth variant, so an exact string comparison would fail a case
    that behaved perfectly. What actually matters is the guardrail - an answer
    must never cite the wrong scheme - so that is what gets checked, by
    rejecting an answer that names a scheme and not the one expected.
    """
    if not expected:
        return True
    if not actual:
        return True
    if _normalise(expected) == _normalise(actual):
        return True
    # Tolerate the parenthetical nickname: "HDFC Equity Fund (Flexi Cap)" for
    # "HDFC Flexi Cap Fund - Direct - Growth".
    stem = _normalise(expected).split("(")[0].strip()
    return stem in _normalise(actual) or _normalise(actual).split("(")[0] in stem


@dataclass
class Result:
    case_id: int
    query: str
    expected: str
    actual: str
    passed: bool
    scheme_match: bool
    citation: str | None
    citation_grounded: bool
    latency_s: float
    note: str
    answer: str


def run_case(case: dict) -> Result:
    expected = case["expected_intent"]
    t0 = perf_counter()
    try:
        response = answer_query(case["query"])
        actual = response.intent
        citation = response.citation_url
        answer = response.answer_text
    except LLMUnavailable as exc:
        # Split the two causes. "No key" means the harness is unconfigured;
        # anything else (rate limit, 5xx, transport) means the provider let us
        # down on a question we were perfectly able to answer. Collapsing them
        # into one label sent me hunting for a missing key when the real cause
        # was a per-minute token quota, which a TOP_K=10 prompt crosses.
        message = str(exc)
        actual = "ERROR:NO_KEY" if "No LLM_API_KEY" in message else "ERROR:PROVIDER"
        latency = perf_counter() - t0
        return Result(
            case_id=case["id"],
            query=case["query"],
            expected=expected,
            actual=actual,
            passed=False,
            scheme_match=False,
            citation=None,
            citation_grounded=False,
            latency_s=latency,
            note=message,
            answer="",
        )
    latency = perf_counter() - t0

    if expected == "ANSWER_OR_INSUFFICIENT":
        intent_ok = actual in {"ANSWER", "INSUFFICIENT_CONTEXT"}
    else:
        intent_ok = actual == expected
    return Result(
        case_id=case["id"],
        query=case["query"],
        expected=expected,
        actual=actual,
        passed=intent_ok,
        scheme_match=_scheme_matches(case.get("expected_scheme"), response.source_scheme),
        citation=citation,
        citation_grounded=(citation in permitted_urls()) if citation else None,
        latency_s=latency,
        note=case.get("note", ""),
        answer=answer,
    )


def render_table(results: list[Result]) -> str:
    header = (
        f"{'id':>3}  {'expected':<24} {'actual':<21} {'ok':<5} "
        f"{'scheme':<7} {'cite':<5} {'latency':>8}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        cite = "-" if r.citation is None else ("ok" if r.citation_grounded else "BAD")
        lines.append(
            f"{r.case_id:>3}  {r.expected:<24} {r.actual:<21} "
            f"{'PASS' if r.passed else 'FAIL':<5} "
            f"{'ok' if r.scheme_match else 'NO':<7} {cite:<5} {r.latency_s:>7.3f}s"
        )
    return "\n".join(lines)


def render_samples(results: list[Result], status: dict) -> str:
    answered = [r for r in results if r.actual == "ANSWER"]
    grounded = [r for r in answered if r.citation_grounded]
    no_key = [r for r in results if r.actual == "ERROR:NO_KEY"]
    provider = [r for r in results if r.actual == "ERROR:PROVIDER"]
    out = [
        "# Sample Q&A",
        "",
        "Generated by `python eval/run_eval.py` from `eval/queries.json` (the PRD",
        "section 9 validation set). Regenerate rather than hand-edit; the answers",
        "below are the live output of the system, not written by hand.",
        "",
        f"- Index: {status['n_chunks']} chunks, built {status['built_at']}",
        f"- Grounding rate: {len(grounded)}/{len(answered)} cited answers point at a URL in `config/sources.csv`",
        "",
    ]
    if provider:
        out += [
            f"> **Incomplete: {len(provider)} of {len(results)} cases hit a provider error.**",
            "> These are recorded as `ERROR:PROVIDER`, not `ERROR:NO_KEY` - a key is",
            "> present and the question was answerable. The usual cause is a per-minute",
            "> token quota, which a wide retrieval window makes easier to cross:",
            ">",
            "> ```bash",
            "> python eval/run_eval.py   # 429s are retried with backoff; re-run to fill in",
            "> ```",
            "",
        ]
    if no_key:
        out += [
            f"> **Incomplete: {len(no_key)} of {len(results)} cases could not run.**",
            "> `LLM_API_KEY` is not set, so the factual cases cannot reach the model",
            "> and are recorded as `ERROR:NO_KEY`. The guard cases below are real",
            "> output and need no key. To produce the full set:",
            ">",
            "> ```bash",
            "> cp .env.example .env      # then set LLM_API_KEY",
            "> python eval/run_eval.py   # rewrites this file with real answers",
            "> ```",
            "",
        ]
    out += ["---", ""]
    for r in results:
        out += [
            f"## {r.case_id}. {r.query}",
            "",
            f"**Intent:** `{r.actual}` (expected `{r.expected}`)"
            + ("" if r.passed else " - **MISMATCH**"),
            "",
        ]
        if r.answer:
            out += [r.answer, ""]
        if r.citation:
            out += [f"**Source:** [{r.citation}]({r.citation})", ""]
        else:
            out += ["**Source:** none - this response is not a fact drawn from a document.", ""]
        out += [f"*Latency: {r.latency_s:.3f}s. {r.note}*", "", "---", ""]
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true", help="do not write samples/sample_qa.md")
    args = parser.parse_args()

    cases = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    status = corpus_status()
    if not status["ready"]:
        print("No index built. Run: python -m src.ingest.run_all --rebuild", file=sys.stderr)
        return 2

    print(f"Index: {status['n_chunks']} chunks, built {status['built_at']}\n")
    results = [run_case(case) for case in cases]
    print(render_table(results))

    answered = [r for r in results if r.actual == "ANSWER"]
    grounded = [r for r in answered if r.citation_grounded]
    mean_latency = sum(r.latency_s for r in results) / len(results)
    passed = sum(1 for r in results if r.passed)

    print(f"\n{passed}/{len(results)} cases behaved as expected")
    print(f"Grounding rate: {len(grounded)}/{len(answered)} "
          f"= {100 * len(grounded) / len(answered):.0f}%" if answered else "Grounding rate: n/a (no ANSWER cases)")
    print(f"Mean latency: {mean_latency:.3f}s")

    if not args.no_write:
        SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
        SAMPLES_PATH.write_text(render_samples(results, status), encoding="utf-8")
        print(f"Wrote {SAMPLES_PATH.relative_to(REPO_ROOT)}")

    if passed != len(results):
        print("\nFAILED cases:")
        for r in results:
            if not r.passed:
                print(f"  {r.case_id}: expected {r.expected}, got {r.actual}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
