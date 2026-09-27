"""Prompt assembly. Satisfies PRD FR-6.

The context block is deliberately structured so the model can only cite a URL
that is physically present in the text it was given: each chunk carries its
`source_url` on the line above it and is numbered. A model that invents a
citation has to invent the *number* too, which `validate` then catches.
"""

from __future__ import annotations

from src.models import RetrievedChunk

# Verbatim from architecture.md section 5.4. This is the contract the model is
# asked to follow; `validate.py` is what makes it true regardless of what the
# model does. Changing the wording here changes the UI's behaviour, so the two
# are kept in step deliberately.
SYSTEM_RULES = """You are a mutual fund FACTS assistant for HDFC AMC schemes.

Rules (non-negotiable):
1. Use ONLY the numbered CONTEXT below. Nothing else is a permitted source.
2. Answer in at most 3 sentences. No bullet lists.
3. End with exactly ONE citation line: "Source: <url>" using a url present in CONTEXT.
4. Do NOT write a "Last updated from sources:" line. The application adds it from the
   real fetch date. Any date you write is discarded, and an invented one is a
   fabricated figure presented as a system guarantee.
5. If CONTEXT does not contain the fact, reply exactly: INSUFFICIENT_CONTEXT
6. Never state or compare returns, NAV trends, or performance. If asked, say the figure
   is in the official factsheet and cite it.
7. Never advise. No "should", "recommend", "suitable for you", buy/sell language.
8. Never invent a URL, date, or number. No PII requests."""


def build_context_block(hits: list[RetrievedChunk]) -> str:
    """Number the retrieved chunks and attach each one's real source URL."""
    if not hits:
        return "(no context retrieved)"
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(f"[{index}] source_url: {hit.chunk.source_url}\n{index}. {hit.chunk.text}")
    return "\n\n".join(blocks)


def citation_allowlist(hits: list[RetrievedChunk]) -> list[str]:
    """The only URLs a citation may name, de-duplicated in hit order."""
    seen: list[str] = []
    for hit in hits:
        url = hit.chunk.source_url
        if url and url not in seen:
            seen.append(url)
    return seen


def newest_source_date(hits: list[RetrievedChunk]) -> str:
    """The newest `fetched_at` across the hits, as an ISO date."""
    dates = [hit.chunk.fetched_at for hit in hits if hit.chunk.fetched_at]
    return max(dates) if dates else ""


def build_user_prompt(query: str, hits: list[RetrievedChunk]) -> str:
    return (
        f"{SYSTEM_RULES}\n\n"
        f"CONTEXT:\n{build_context_block(hits)}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer using only CONTEXT."
    )
