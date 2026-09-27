"""PII guard tests. Keyless, no index, no network (Phase 11).

Every pattern gets a positive case and a negative case. The negatives are the
part that matters: a guard that blocks "SIP of 500" or "expense ratio 1.5%" is
worse than useless, because those are the questions the product exists to
answer. A false positive there reads to the user as "this bot is broken" rather
than "this bot protected my data".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.guards import pii  # noqa: E402

# (kind, must_block, text). Every one of these is a real format a person types.
POSITIVES: list[tuple[str, str]] = [
    ("pan", "My PAN is ABCDE1234F"),
    ("pan", "pan: abcdE1234f"),
    ("aadhaar", "Aadhaar 2345 6789 0123"),
    ("aadhaar", "aadhaar number 987654321099"),
    ("account_no", "account number 123456789012"),
    ("account_no", "my account no. is 98765432101234"),
    ("otp", "my otp is 482913"),
    ("otp", "One time password: 1234"),
    ("email", "reach me at ravi.sharma@gmail.com"),
    ("phone", "call me on 9876543210"),
    ("phone", "+91 98765 43210"),
    ("phone", "+919876543210"),
]

# The spec's three mandated negatives, plus the general rule: financial figures
# are not personal data. These are the highest-value regression cases in the
# suite and the first thing a reviewer will check.
NEGATIVES: list[str] = [
    "SIP of 500",
    "What is the expense ratio of HDFC Large Cap Fund?",
    "the expense ratio is 1.5%",
    "Nifty 50 TRI",
    "minimum SIP amount is Rs 500 per month",
    "exit load of 1% if redeemed within 1 year",
    "lock-in period of 3 years",
    "benchmark is Nifty 50 TRI",
    "direct plan expense ratio 1.05% versus regular 1.5%",
    "NAV was 62.4831 on the last working day",
    "What is the exit load of HDFC Flexi Cap Fund?",
    "Aadhaar and PAN are not needed for this query",
]


@pytest.mark.parametrize(("kind", "text"), POSITIVES, ids=[f"{k}:{t[:28]}" for k, t in POSITIVES])
def test_positive_is_blocked_and_kind_reported(kind: str, text: str) -> None:
    verdict = pii.scan(text)
    assert verdict.blocked, f"expected a block for {text!r}"
    assert kind in verdict.matched_kinds, f"{text!r} matched {verdict.matched_kinds}, want {kind}"


@pytest.mark.parametrize("text", NEGATIVES, ids=lambda t: t[:34])
def test_negative_is_not_blocked(text: str) -> None:
    verdict = pii.scan(text)
    assert not verdict.blocked, (
        f"false positive on {text!r} (kinds={verdict.matched_kinds}). Financial "
        "figures must never be treated as personal identifiers."
    )
    assert verdict.sanitized_text == text


def test_every_pattern_has_a_positive_and_a_negative() -> None:
    """Guards a pattern being added with no coverage on either side."""
    positives = {k for k, _ in POSITIVES}
    missing = set(pii.PII_PATTERNS) - positives
    assert not missing, f"patterns with no positive case: {sorted(missing)}"

    for text in NEGATIVES:
        assert not pii.scan(text).matched_kinds


def test_every_match_is_redacted() -> None:
    """No identifier may survive into the sanitized text, in whole or in part."""
    text = "PAN ABCDE1234F, phone 9876543210, email a.b@c.com, aadhaar 2345 6789 0123"
    verdict = pii.scan(text)
    assert verdict.blocked
    for leaked in ("ABCDE1234F", "9876543210", "a.b@c.com", "2345 6789 0123"):
        assert leaked not in verdict.sanitized_text, f"{leaked!r} survived redaction"
    assert verdict.sanitized_text.count("[REDACTED]") == 4


def test_redaction_does_not_corrupt_neighbouring_text() -> None:
    """Offset corruption is the classic redaction bug: sequential substitution
    would shift later matches and leave a partial identifier behind."""
    text = "Reach me at 9876543210 or email ravi.sharma@gmail.com today"
    verdict = pii.scan(text)
    assert verdict.blocked
    assert "Reach me at [REDACTED] or email [REDACTED] today" == verdict.sanitized_text
    # Nothing partial survives: no trailing digits of the phone, no domain.
    assert "543210" not in verdict.sanitized_text
    assert "@gmail.com" not in verdict.sanitized_text


def test_clean_text_passes_through_untouched() -> None:
    text = "What is the minimum SIP for HDFC Small Cap Fund?"
    verdict = pii.scan(text)
    assert not verdict.blocked
    assert verdict.sanitized_text == text
    assert verdict.matched_kinds == []


def test_block_response_carries_no_provenance() -> None:
    """A refusal is not a fact from a document, so it must not cite one."""
    response = pii.block_response()
    assert response.intent == "PII_BLOCKED"
    assert response.citation_url is None
    assert response.source_scheme is None
    assert response.last_updated == ""
    assert response.retrieval_hits == 0
    assert "PAN" in response.answer_text
