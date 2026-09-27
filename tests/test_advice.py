"""Advice guard tests. Keyless, no index, no network (Phase 11).

Covers every pattern in `advice.ADVICE_PATTERNS` individually, so a new pattern
that never fires shows up as a failing name rather than as a silently dead
regex. Then the two-sided reality check: the PRD section 9 must-answer questions
must not fire, because the failure mode of a compliance guard is not a missed
refusal so much as a bot that refuses to answer anything.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.guards import advice  # noqa: E402
from src.models import permitted_urls  # noqa: E402

# One representative query per pattern, in the same order as ADVICE_PATTERNS.
# The queries are chosen so that no *earlier* pattern also matches, because
# `detect` returns the first hit and these cases assert the pattern is reached.
# Two of them could not be deconflicted and say so in their docstring; the
# regex-liveness test below covers those.
PER_PATTERN: list[tuple[str, str]] = [
    (r"\bshould\s+(i|we)\b", "Should I buy HDFC Small Cap Fund now?"),
    (r"\bdo you (think|recommend)\b", "Do you think HDFC ELSS is a good investment?"),
    (r"\bwhich\s+(?:\w+\s+){0,3}?(fund|scheme|etf|option)s?\b.{0,20}?\b(?:best|better)\b",
     "Which of these 5 funds is the best for long-term wealth creation?"),
    (r"\b(?:good|right|better|ideal) (?:fit|choice|option|alternative) for (?:me|my)\b",
     "Is HDFC ELSS a good fit for me?"),
    (r"\bis (now|it) a good time\b", "Is now a good time to invest in small caps?"),
    (r"\bgood time to invest\b", "Is this a good time to invest in the market?"),
    (r"\bhow much should i\b", "How much should I invest in HDFC Small Cap Fund?"),
    (r"\bbest for (me|my)\b", "Is HDFC ELSS the best for my situation?"),
    (r"\bbuild (?:me|us) an?[\w\s]{0,25}?\bportfolio\b", "Can you build me a balanced portfolio?"),
    (r"\bsuitable for me\b", "Is HDFC Balanced Advantage Fund suitable for me?"),
    (r"\bhighest returns?\b", "Which fund gives the highest returns?"),
    (r"\bbest[- ]performing\b", "What is the best performing fund?"),
    (r"\b(better|safer) (option|choice|fund)\b", "Which is the safer option, ELSS or a Flexi Cap fund?"),
    (r"\bworth (buying|investing)\b", "Is HDFC Large Cap Fund worth buying?"),
]

# Patterns shadowed by an earlier one, so `detect` can never return them.
# `should i/we` always wins over `how much should i`, which is harmless: both
# refuse. Listed so the ordering stays a documented decision rather than an
# accident, and so deleting the earlier pattern does not silently orphan a rule.
SHADOWED: dict[str, tuple[str, str]] = {
    r"\bhow much should i\b": (
        r"\bshould\s+(i|we)\b",
        "How much should I invest in HDFC Small Cap Fund?",
    ),
}

# PRD section 9 items 8, 9, 10 plus the variants that broke the guard in Phase 9.
MUST_REFUSE: list[str] = [
    "Should I buy HDFC Small Cap Fund now?",
    "Which of these 5 funds is the best for long-term wealth creation?",
    "Is HDFC ELSS a good fit for me?",
    "Which of these five funds is best?",
    "Which of my two funds is better?",
    "What is the best performing fund?",
    "Should I wait for a correction before investing?",
    "How much should I put into a small cap fund each month?",
    "Do you recommend HDFC Large Cap Fund?",
    "Can you build me a balanced portfolio?",
    "Is HDFC Balanced Advantage Fund suitable for me?",
    "Which scheme has the highest returns?",
    "Is HDFC Flexi Cap a better option than a large cap fund?",
    "Is HDFC Large Cap Fund worth investing in?",
    "Is now a good time to invest?",
    "Which fund is best for my retirement?",
]

# PRD section 9 items 1-6: the questions the product must actually answer.
MUST_ANSWER: list[str] = [
    "What is the expense ratio of HDFC Large Cap Fund - Direct - Growth?",
    "What is the exit load on HDFC Equity Fund (Flexi Cap)?",
    "What is the minimum SIP amount for HDFC Small Cap Fund - Direct - Growth?",
    "What is the lock-in period for HDFC ELSS Tax Saver Fund, and when can I exit?",
    "What is the benchmark and riskometer category of HDFC Balanced Advantage Fund?",
    "How do I download my capital-gains statement?",
    "What is the NAV of HDFC Large Cap Fund?",
    "What is the direct plan expense ratio of HDFC Flexi Cap Fund?",
    "What is the minimum lump sum amount for HDFC ELSS Tax Saver Fund?",
    "What is the exit load of HDFC Small Cap Fund - Direct - Growth?",
    "How long is the lock-in period for HDFC ELSS Tax Saver Fund?",
    "What is the AUM of HDFC Balanced Advantage Fund?",
    "Who manages HDFC Large Cap Fund?",
]


@pytest.mark.parametrize(
    ("pattern", "query"),
    [(p, q) for p, q in PER_PATTERN if p not in SHADOWED],
    ids=[q[:38] for _, q in PER_PATTERN if _ not in SHADOWED],
)
def test_every_pattern_is_reached(pattern: str, query: str) -> None:
    """A pattern nothing can reach is dead code in a compliance guard."""
    verdict = advice.detect(query)
    assert verdict.is_advice, f"{query!r} was not flagged as advice at all"
    assert verdict.matched_pattern == pattern, (
        f"{query!r} matched {verdict.matched_pattern!r} rather than {pattern!r}"
    )


@pytest.mark.parametrize(("pattern", "query"), PER_PATTERN, ids=[q[:38] for _, q in PER_PATTERN])
def test_every_regex_is_live(pattern: str, query: str) -> None:
    """Compile the pattern itself, ignoring `detect`'s precedence.

    Two patterns are shadowed by an earlier one and can never be returned by
    `detect`, so this is the only assertion that covers them.
    """
    assert re.search(pattern, query, re.I), f"{pattern!r} does not match {query!r}"


def test_pattern_list_and_cases_stay_aligned() -> None:
    """A new pattern with no case here is a pattern nobody has ever seen fire."""
    assert len(PER_PATTERN) == len(advice.ADVICE_PATTERNS), (
        f"{len(advice.ADVICE_PATTERNS)} patterns in the guard but "
        f"{len(PER_PATTERN)} cases here"
    )
    for (expected, _), actual in zip(PER_PATTERN, advice.ADVICE_PATTERNS, strict=True):
        assert expected == actual, f"guard pattern is {actual!r}, test case written for {expected!r}"


@pytest.mark.parametrize("pattern", sorted(SHADOWED), ids=lambda p: p[:30])
def test_shadowed_patterns_are_still_shadowed(pattern: str) -> None:
    """Documents the precedence. If this fails, SHADOWED needs updating."""
    shadower, query = SHADOWED[pattern]
    order = advice.ADVICE_PATTERNS
    assert order.index(shadower) < order.index(pattern)
    assert advice.detect(query).matched_pattern == shadower


@pytest.mark.parametrize("query", MUST_REFUSE, ids=lambda q: q[:38])
def test_must_refuse(query: str) -> None:
    assert advice.detect(query).is_advice, f"advice request not refused: {query!r}"


@pytest.mark.parametrize("query", MUST_ANSWER, ids=lambda q: q[:38])
def test_must_not_flag_factual_questions(query: str) -> None:
    verdict = advice.detect(query)
    assert not verdict.is_advice, (
        f"false positive on a factual question: {query!r} "
        f"(matched {verdict.matched_pattern!r})"
    )


def test_refusal_cites_a_permitted_source() -> None:
    """A refusal still points somewhere real, and the link must be allow-listed.

    A fabricated link on a compliance refusal is the single worst failure this
    guard could have, because it is the response the user is most likely to act
    on.
    """
    for query in MUST_REFUSE:
        response = advice.refusal_response(query)
        assert response.intent == "REFUSAL"
        assert response.citation_url is not None, f"no educational link for {query!r}"
        assert response.citation_url in permitted_urls()
        assert response.last_updated == "", "a refusal is not a fact from a document"
        assert response.retrieval_hits == 0, "no retrieval runs on a refusal"


def test_refusal_links_the_relevant_scheme() -> None:
    """The educational link should point at the scheme the user asked about."""
    els = advice.pick_educational_link("Should I invest in HDFC ELSS for tax saving?")
    large = advice.pick_educational_link("Should I buy HDFC Large Cap Fund?")
    assert els in permitted_urls() and large in permitted_urls()
    assert els != large, "ELSS and Large Cap questions linked to the same page"


def test_refusal_does_not_leak_the_question() -> None:
    """The reply is fixed text; it must not echo the user's words back."""
    response = advice.refusal_response("Should I buy HDFC Small Cap Fund now?")
    assert "Small Cap" not in response.answer_text
    assert "buy" not in response.answer_text.lower().replace("buy or hold", "")
    assert "can't tell you whether to buy or hold" in response.answer_text
