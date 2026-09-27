"""PII guard. Satisfies PRD FR-8.

Blocks a query that carries a personal identifier before retrieval runs. This is
the outer gate: nothing downstream - retrieval, the LLM, the query log - ever
sees the raw text, which is why `block_response` carries no citation and
`log_query` is only ever handed the sanitized string.
"""

from __future__ import annotations

import re

from src.models import ChatResponse, PIIVerdict

# Order is significant. `pan` must be tested before `account_no` because a PAN
# such as ABCDE1234F contains four digits, and the digit patterns are greedy
# enough to claim a span out of it. `aadhaar` before `account_no` for the same
# reason: a 12-digit Aadhaar is also a valid 12-digit account number, and the
# narrower, more specific pattern has to win so `matched_kinds` is honest.
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    # Case-insensitive: people type PANs in lower case, and an identifier is an
    # identifier regardless of the case it was typed in.
    "pan": re.compile(r"\b[a-z]{5}[0-9]{4}[a-z]\b", re.I),
    "aadhaar": re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b"),
    "account_no": re.compile(r"\b[0-9]{8,18}\b"),
    "otp": re.compile(r"\b(otp|one time password)\b.{0,15}\b\d{4,6}\b", re.I),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    # implementation.md gives `(\+91[\s-]?)?\b[6-9]\d{9}\b`, which needs ten
    # consecutive digits and therefore misses the format people actually type:
    # "+91 98765 43210". A phone number written with a space is still a phone
    # number, so one internal separator is allowed after the first five digits.
    # `\b` cannot sit between "+91" and the first digit - both are word
    # characters, so there is no boundary there and "+919876543210" fell
    # through to account_no and reported the wrong kind. A plain `(?<!\d)` in
    # that position is also wrong: it rejects "+919876543210" because the
    # preceding character is the "1" of the country code, while correctly
    # refusing to read ten digits out of the middle of a twelve-digit account
    # number. So the country code is consumed with a lookahead - "+91" already
    # establishes the boundary - and only the bare-number branch gets the
    # lookbehind.
    "phone": re.compile(r"(?:\+91[\s-]?(?=\d)|(?<!\d))[6-9]\d{4}[\s-]?\d{5}(?!\d)"),
}

BLOCKED_MESSAGE = (
    "I can't process personal identifiers, so please don't share a PAN, Aadhaar, "
    "bank account number, OTP, email address or phone number here. "
    "Ask me about the funds' published facts instead and I'll answer from the official factsheet."
)


def scan(text: str) -> PIIVerdict:
    """Detect identifiers, returning the text with every match redacted.

    All patterns are applied to the original string and their spans merged before
    redaction. Redacting sequentially instead would corrupt the offsets of later
    matches and could leave a partial identifier - a phone number with its first
    two digits intact - visible in the output.
    """
    spans: list[tuple[int, int]] = []
    kinds: list[str] = []
    for kind, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            spans.append(match.span())
            if kind not in kinds:
                kinds.append(kind)
    if not spans:
        return PIIVerdict(blocked=False, sanitized_text=text, matched_kinds=[])

    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    out: list[str] = []
    cursor = 0
    for start, end in merged:
        out.append(text[cursor:start])
        out.append("[REDACTED]")
        cursor = end
    out.append(text[cursor:])
    return PIIVerdict(blocked=True, sanitized_text="".join(out), matched_kinds=kinds)


def block_response() -> ChatResponse:
    """The fixed reply for a blocked query.

    No citation and no `last_updated`: a refusal is not a fact drawn from a
    source, and attaching a source URL to it would imply the answer came from
    the factsheet. `retrieval_hits=0` records that nothing was retrieved.
    """
    return ChatResponse(
        intent="PII_BLOCKED",
        answer_text=BLOCKED_MESSAGE,
        citation_url=None,
        source_scheme=None,
        last_updated="",
        retrieval_hits=0,
    )
