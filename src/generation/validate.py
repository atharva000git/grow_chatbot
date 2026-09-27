"""Post-validation. Satisfies PRD FR-6, and is the reason FR-6 is a code
guarantee rather than a hope.

The model is asked for at most three sentences, one citation and no advice. It
is not trusted to comply. Every one of those rules is re-checked here and
enforced mechanically, so the output contract holds even if the model returns
something that ignores it entirely.

Fully offline and deterministic: `validate` takes the raw string and the hits,
so it can be tested without a key or a network call. That is deliberate - the
layer that makes the safety guarantees is the layer most worth testing.
"""

from __future__ import annotations

import re

from config import settings
from src.models import ChatResponse, RetrievedChunk
from src.generation.prompt import citation_allowlist, newest_source_date

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
SOURCE_LINE_RE = re.compile(r"^\s*Source:\s*(\S+)\s*$")
FRESHNESS_RE = re.compile(r"Last updated from sources:\s*(\d{4}-\d{2}-\d{2})")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+")

PERFORMANCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\breturn(s|ed|ing)?\b", re.I),
    re.compile(r"\bCAGR\b", re.I),
    re.compile(r"\bhas (?:beaten|outperformed|outperformed)\b", re.I),
    re.compile(r"\btop performing\b|\bbest performing\b", re.I),
    re.compile(r"\bNAV trend", re.I),
    # A bare percentage needs a return word nearby to count, so this rule only
    # fires on "32% in 5 years", never on "exit load of 1% for 12 months" - an
    # exit-load table states a percentage per period and is a fact, not a
    # performance claim. "for" is excluded from the period preposition for the
    # same reason. The unit carries a trailing \b so "months" cannot match a
    # bare "month", and the return words above still catch "returned 32%".
    re.compile(r"\d+(?:\.\d+)?\s*%\s*(?:in|over)\s*(?:the\s*)?(?:\d+\s*)?(?:years?|months?|yrs?)\b", re.I),
)

ADVICE_OUTPUT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\byou should\b", re.I),
    re.compile(r"\bI recommend\b", re.I),
    re.compile(r"\bsuitable for you\b", re.I),
    re.compile(r"\bbuy this\b", re.I),
    re.compile(r"\b(?:I|we) would (?:buy|sell|invest)\b", re.I),
    re.compile(r"\byou could (?:buy|invest in)\b", re.I),
)

# "asset allocation" is a legitimate factsheet term - it is the Balanced
# Advantage fund's whole mandate - so it is removed before the allocation check
# runs, rather than the pattern being weakened to the point of missing a real
# recommendation.
ALLOCATE_RE = re.compile(r"\ballocat(?:e|ing|ion)\b", re.I)
ASSET_ALLOCATION_RE = re.compile(r"\basset allocation\b", re.I)

PERFORMANCE_REFUSAL = (
    "I can't state or compare returns or performance. The official factsheet "
    "publishes those figures, so please check it for the numbers and dates. "
    "I'm happy to share the scheme's published facts such as exit load, lock-in, "
    "benchmark, minimum SIP and expense ratio."
)

ADVICE_REFUSAL = (
    "I can't tell you whether to buy or hold a scheme. Please treat this as a "
    "facts lookup only, and make any decision using your own goals, horizon and "
    "risk tolerance alongside the official factsheet."
)

INSUFFICIENT_MESSAGE = (
    "I don't have that in the sources I can cite. I only answer from the "
    "official HDFC scheme pages and factsheets for these five funds."
)


def count_sentences(text: str) -> int:
    """Count sentences in the answer body, ignoring the Source/freshness lines."""
    body = _strip_meta_lines(text)
    return len([s for s in SENTENCE_RE.findall(body) if s.strip()])


def extract_urls(text: str) -> list[str]:
    return URL_RE.findall(text)


def _strip_meta_lines(text: str) -> str:
    """Drop the trailing Source/freshness lines before counting sentences.

    Without this a compliant two-sentence answer is counted as three, because
    the citation line is not a sentence and truncating to `MAX_ANSWER_SENTENCES`
    would then cut a real sentence to make room for the citation.
    """
    return "\n".join(
        line for line in text.split("\n")
        if not SOURCE_LINE_RE.match(line) and not FRESHNESS_RE.search(line)
    )


def truncate_to_sentences(text: str, n: int) -> str:
    """Keep the first `n` sentences of the body, preserving any meta lines."""
    lines = text.split("\n")
    body_lines = [
        line for line in lines
        if not SOURCE_LINE_RE.match(line) and not FRESHNESS_RE.search(line)
    ]
    meta_lines = [line for line in lines if line not in body_lines]
    body = "\n".join(body_lines)
    kept = [s.strip() for s in SENTENCE_RE.findall(body) if s.strip()][:n]
    trailing = body.strip()[sum(len(s) for s in kept):] if kept else ""
    remainder = trailing.strip()
    if remainder and len(SENTENCE_RE.findall(body)) <= n:
        kept.append(remainder)
    return "\n".join([*kept, *meta_lines]).strip()


def has_performance_language(text: str) -> bool:
    return any(pattern.search(text) for pattern in PERFORMANCE_PATTERNS)


def has_advice_language(text: str) -> bool:
    if ADVICE_OUTPUT_PATTERNS and any(p.search(text) for p in ADVICE_OUTPUT_PATTERNS):
        return True
    stripped = ASSET_ALLOCATION_RE.sub(" ", text)
    return bool(ALLOCATE_RE.search(stripped))


def _refusal(message: str, url: str, hits: list[RetrievedChunk]) -> ChatResponse:
    return ChatResponse(
        intent="REFUSAL",
        answer_text=message,
        citation_url=url or None,
        source_scheme=hits[0].chunk.scheme if hits else None,
        last_updated=newest_source_date(hits),
        retrieval_hits=len(hits),
        top_similarity=hits[0].similarity if hits else None,
    )


def validate(raw: str, hits: list[RetrievedChunk]) -> ChatResponse:
    """Turn a raw model string into a contract-compliant ChatResponse."""
    allow = citation_allowlist(hits)
    top_url = allow[0] if allow else ""
    text = (raw or "").strip()

    # 1. The model reporting it had nothing is a first-class return, not an error.
    if "INSUFFICIENT_CONTEXT" in text:
        return ChatResponse(
            intent="INSUFFICIENT_CONTEXT",
            answer_text=INSUFFICIENT_MESSAGE,
            citation_url=None,
            source_scheme=None,
            last_updated="",
            retrieval_hits=0,
        )

    # 2. Cap the length.
    if count_sentences(text) > settings.MAX_ANSWER_SENTENCES:
        text = truncate_to_sentences(text, settings.MAX_ANSWER_SENTENCES)

    # 3. Strip invented URLs. Substitution would be "fixing" a fabrication with
    #    another fabrication; removal leaves a gap that check 4 fills from real
    #    context, which is the honest outcome.
    for url in extract_urls(text):
        if url not in allow:
            text = text.replace(url, "")
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Exactly one valid citation.
    existing = [m.group(1) for m in (SOURCE_LINE_RE.match(l) for l in text.split("\n")) if m]
    existing = [u for u in existing if u in allow]
    text = "\n".join(l for l in text.split("\n") if not SOURCE_LINE_RE.match(l)).strip()
    citation = existing[0] if existing else top_url
    if citation:
        text = f"{text}\nSource: {citation}".strip()

    # 5/6. Performance and advice language both fall back to a refusal, checked
    #      after citation repair so a refusal can still carry a real citation.
    body = _strip_meta_lines(text)
    if has_performance_language(body):
        return _refusal(PERFORMANCE_REFUSAL, top_url, hits)
    if has_advice_language(body):
        return _refusal(ADVICE_REFUSAL, top_url, hits)

    # 7. Freshness line.
    if not FRESHNESS_RE.search(text) and citation:
        date = newest_source_date(hits)
        if date:
            text = f"{text}\nLast updated from sources: {date}".strip()

    return ChatResponse(
        intent="ANSWER",
        answer_text=text,
        citation_url=citation or None,
        source_scheme=hits[0].chunk.scheme if hits else None,
        last_updated=newest_source_date(hits),
        retrieval_hits=len(hits),
        top_similarity=hits[0].similarity if hits else None,
    )
