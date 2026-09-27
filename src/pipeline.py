"""Orchestrator. Ties S0-S5 into one call, satisfying FR-5 to FR-8.

Control flow, and the order is the architecture:

    0. empty / whitespace query           -> INSUFFICIENT_CONTEXT   (no LLM call)
    1. pii.scan(query).blocked            -> pii.block_response()   (no LLM call)
    2. advice.detect(sanitized).is_advice -> advice refusal        (no LLM call)
    3. retrieval.search + sufficiency      -> INSUFFICIENT_CONTEXT   (no LLM call)
    4. generate_raw(sanitized, hits)       -> LLM call
    5. validate(raw, hits)                -> contract-enforced answer
    6. log_query(sanitized, intent)       -> only for non-blocked paths

Steps 0-3 are the cheap half of the system and they are the reason the expensive
half is safe: four of the five ways this product must behave correctly - ignore
blank input, refuse PII, refuse advice, admit ignorance - are decided before a
token is spent. A guard that ran after generation would be a guard that had
already paid for the answer it was going to throw away.
"""

from __future__ import annotations

import json
import logging

from config import settings
from src.generation.llm import LLMUnavailable, generate_raw
from src.generation.validate import validate
from src.guards import advice, pii
from src.logging_utils import log_query
from src.models import ChatResponse, RetrievedChunk
from src.retrieval import search as retrieval

log = logging.getLogger(__name__)

INSUFFICIENT_MESSAGE = (
    "I don't have a citable answer for that in the HDFC scheme pages and "
    "factsheets I can use. Ask me about exit load, lock-in, benchmark, minimum "
    "SIP, expense ratio or NAV for these five funds."
)

INTERNAL_ERROR_MESSAGE = (
    "Something went wrong while answering that. Please try rephrasing the "
    "question, or check the index is built."
)


def _insufficient() -> ChatResponse:
    return ChatResponse(
        intent="INSUFFICIENT_CONTEXT",
        answer_text=INSUFFICIENT_MESSAGE,
        citation_url=None,
        source_scheme=None,
        last_updated="",
        retrieval_hits=0,
    )


def _internal_error() -> ChatResponse:
    return ChatResponse(
        intent="INSUFFICIENT_CONTEXT",
        answer_text=INTERNAL_ERROR_MESSAGE,
        citation_url=None,
        source_scheme=None,
        last_updated="",
        retrieval_hits=0,
    )


def answer_query(query: str, k: int | None = None) -> ChatResponse:
    """Answer any input, running the guards in order and never leaking a trace."""
    if not query or not query.strip():
        return _insufficient()

    # S0 - PII. The guard is outermost so an identifier never reaches retrieval,
    # the model, or the query log.
    verdict = pii.scan(query)
    if verdict.blocked:
        log.info("PII blocked, kinds=%s", verdict.matched_kinds)
        return pii.block_response()
    sanitized = verdict.sanitized_text

    # S1 - advice.
    if advice.detect(sanitized).is_advice:
        return advice.refusal_response(sanitized)

    # S2/S3 - retrieval, and the honesty gate.
    hits: list[RetrievedChunk] = retrieval.search(sanitized, k=k)
    if not retrieval.is_sufficient(hits):
        response = _insufficient()
        response.top_similarity = retrieval.best_similarity(hits)
        log_query(sanitized, response.intent)
        return response

    # S4 - generation. LLMUnavailable is allowed to escape so the UI can show a
    # banner rather than pretending the corpus had nothing to say.
    try:
        raw = generate_raw(sanitized, hits)
    except LLMUnavailable:
        raise
    except Exception:  # noqa: BLE001 - never surface a traceback to the UI
        log.exception("unexpected failure during generation")
        return _internal_error()

    # S5 - enforce the output contract, then log the sanitized text only.
    try:
        response = validate(raw, hits)
    except Exception:  # noqa: BLE001
        log.exception("unexpected failure during validation")
        return _internal_error()

    response.top_similarity = retrieval.best_similarity(hits)
    log_query(sanitized, response.intent)
    return response


def corpus_status() -> dict:
    """Sidebar status, read from the manifest written by the ingest run."""
    status = {
        "ready": False,
        "n_chunks": 0,
        "built_at": "",
        "embedding_model": settings.EMBEDDING_MODEL,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD,
    }
    from src.ingest.manifest import MANIFEST_PATH

    if not MANIFEST_PATH.exists():
        return status
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return status
    if not isinstance(data, dict):
        return status
    status["n_chunks"] = int(data.get("n_chunks") or 0)
    status["built_at"] = str(data.get("built_at") or "")
    status["ready"] = status["n_chunks"] > 0
    return status
