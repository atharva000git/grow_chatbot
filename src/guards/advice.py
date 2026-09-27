"""Advice guard. Satisfies PRD FR-7.

Refuses "should I buy X" style questions and redirects to the published facts.
SEBI's position is that a mutual-fund facts interface may state what a scheme
discloses but must not recommend or compare schemes for a user, so this gate is
not a stylistic preference - it is the compliance boundary of the product.

The patterns target *intent*, not topic. "What is the exit load of HDFC Large Cap
Fund?" mentions exit load and is a perfectly answerable fact; "Should I buy HDFC
Small Cap Fund now?" asks for a judgement. Only the second is refused.
"""

from __future__ import annotations

import re

from src.models import AdviceVerdict, ChatResponse, permitted_urls

ADVICE_PATTERNS: list[str] = [
    r"\bshould\s+(i|we)\b",
    r"\bdo you (think|recommend)\b",
    # implementation.md gives `which (fund|scheme|etf) is (the )?best`, which does
    # not match the spec's own acceptance examples - "Which of these funds is
    # best?" and Phase 9's "Which of these five funds is best?" - because of the
    # plural and the intervening words. Up to three arbitrary words are allowed
    # between "which" and the noun so qualifiers and counts both pass
    # ("of these five", "of the 5", "of my two"). Still anchored on
    # "which ... fund/scheme/etf/option" with "best" within 20 characters, so a
    # factual question that merely contains "best" is unaffected.
    r"\bwhich\s+(?:\w+\s+){0,3}?(fund|scheme|etf|option)s?\b.{0,20}?\b(?:best|better)\b",
    # PRD section 9 item 10, "Is HDFC ELSS a good fit for me?", is a must-refuse
    # and the spec's list has no pattern for it. Anchored on the trailing "for
    # me/my" so a factual suitability sentence without a first-person reference
    # is not swept up.
    r"\b(?:good|right|better|ideal) (?:fit|choice|option|alternative) for (?:me|my)\b",
    r"\bis (now|it) a good time\b",
    r"\bgood time to invest\b",
    r"\bhow much should i\b",
    r"\bbest for (me|my)\b",
    # `\bbuild (me )?a portfolio\b` missed "Can you build me a balanced\n    # portfolio?" because the adjective sits between "a" and "portfolio". A
    # compliance pattern that a single adjective defeats is not one.
    r"\bbuild (?:me|us) an?[\w\s]{0,25}?\bportfolio\b",
    r"\bsuitable for me\b",
    r"\bhighest returns?\b",
    # "What is the best performing fund?" is a ranking request, and the spec's
    # `which ... is best` pattern cannot see it because the question opens with
    # "what". Matched on the phrase itself, which no factual query needs.
    r"\bbest[- ]performing\b",
    r"\b(better|safer) (option|choice|fund)\b",
    r"\bworth (buying|investing)\b",
]

_COMPILED = tuple((pattern, re.compile(pattern, re.I)) for pattern in ADVICE_PATTERNS)

REFUSAL_MESSAGE = (
    "I can share what these funds publish - exit load, lock-in, benchmark, minimum "
    "SIP and expense ratio - but I can't tell you whether to buy or hold anything. "
    "That decision needs your own goals, horizon and risk tolerance, so please treat "
    "it as a facts lookup rather than advice."
)

# intent keyword -> scheme category in config/sources.csv. A refusal still cites
# somewhere real, so the link is looked up from the registry rather than written
# out here: a hard-coded URL could drift from sources.csv and would then point the
# user at a page the citation allow-list no longer permits.
EDUCATIONAL_CATEGORIES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("elss", "lock in", "lock-in", "tax saver", "80c", "section 80"), "ELSS"),
    (("sip", "systematic", "installment", "instalment", "monthly investment", "minimum"), "Small Cap"),
    (("riskometer", "risk", "volatility", "benchmark", "asset allocation", "balanced"), "Balanced Advantage"),
    (("exit", "tax", "redemption", "stamp duty", "short term", "capital gains"), "Large Cap"),
)

DEFAULT_CATEGORY = "Large Cap"


def detect(query: str) -> AdviceVerdict:
    """Flag a request for a recommendation or a buy/sell judgement."""
    for pattern, compiled in _COMPILED:
        if compiled.search(query):
            return AdviceVerdict(is_advice=True, matched_pattern=pattern)
    return AdviceVerdict(is_advice=False, matched_pattern="")


def _category_url(category: str) -> str:
    """The HTML page for a scheme category, from sources.csv."""
    from src.ingest.load import load_all

    for doc in load_all():
        if doc.category.strip().lower() == category.lower() and doc.source_type == "html":
            return doc.source_url
    raise KeyError(f"no HTML source for category {category!r} in config/sources.csv")


def pick_educational_link(query: str) -> str:
    """Map a refused question to a real page from the citation allow-list.

    Raises if the result would not be a permitted URL, rather than returning it.
    A fabricated or drifted link on a compliance refusal is worse than an
    exception, because it is the one response where the user is most likely to
    follow the link.
    """
    lowered = query.lower()
    category = DEFAULT_CATEGORY
    for tokens, mapped in EDUCATIONAL_CATEGORIES:
        if any(token in lowered for token in tokens):
            category = mapped
            break
    url = _category_url(category)
    if url not in permitted_urls():
        raise ValueError(f"refusing to emit a URL absent from sources.csv: {url!r}")
    return url


def refusal_response(query: str) -> ChatResponse:
    """The fixed reply for an advice request, with one real citation.

    `retrieval_hits=0` because no retrieval ran. `last_updated` is blank for the
    same reason the PII block is uncited: the refusal text is not drawn from a
    document, so attaching a source date to it would be a false provenance
    claim.
    """
    return ChatResponse(
        intent="REFUSAL",
        answer_text=REFUSAL_MESSAGE,
        citation_url=pick_educational_link(query),
        source_scheme=None,
        last_updated="",
        retrieval_hits=0,
    )
