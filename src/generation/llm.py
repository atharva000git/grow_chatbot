"""LLM client. Satisfies PRD FR-6.

Deliberately raises rather than degrading. `architecture.md` section 9 is
explicit that a missing or failing provider must surface a banner in the UI
instead of silently producing an ungrounded answer, so every failure path here
becomes `LLMUnavailable` and the orchestrator lets it propagate.
"""

from __future__ import annotations

import logging
import time

import requests

from config import settings
from src.models import RetrievedChunk
from src.generation.prompt import build_user_prompt

log = logging.getLogger(__name__)

# OpenAI-compatible endpoint. `langchain-openai` is not in requirements.txt and
# adding it for one call would pull an untestable dependency into a project
# whose whole point is that every layer is verifiable offline; `requests` is
# already pinned, and this endpoint shape works for OpenAI, Groq, Together,
# OpenRouter and any other compatible provider.
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
TIMEOUT_SECONDS = 30
# A 429 is retried rather than surfaced. See generate() for why a wide
# retrieval window makes it common enough to be worth handling.
RATE_LIMIT_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 8


class LLMUnavailable(RuntimeError):
    """No key, no provider, or a provider call that failed."""


def _base_url() -> str:
    return (settings.LLM_BASE_URL or DEFAULT_BASE_URL).rstrip("/")


def _model() -> str:
    return settings.LLM_MODEL or DEFAULT_MODEL


def get_client() -> dict:
    """Return the provider config, or raise if no key is configured."""
    if not settings.LLM_API_KEY:
        raise LLMUnavailable(
            "No LLM_API_KEY configured. Create a .env with LLM_API_KEY=<key> "
            "(and optionally LLM_MODEL) and restart."
        )
    return {"api_key": settings.LLM_API_KEY, "base_url": _base_url(), "model": _model()}


def generate(prompt: str) -> str:
    """One completion at temperature 0, or raise LLMUnavailable."""
    client = get_client()
    payload = {
        "model": client["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
    }
    for attempt in range(RATE_LIMIT_ATTEMPTS):
        try:
            response = requests.post(
                f"{client['base_url']}/chat/completions",
                headers={
                    "Authorization": f"Bearer {client['api_key']}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - transport failures are retryable
            if attempt == RATE_LIMIT_ATTEMPTS - 1:
                log.warning("LLM transport failure: %s", exc)
                raise LLMUnavailable(f"could not reach the provider: {exc}") from exc
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
            continue

        if response.status_code == 429:
            # A wider retrieval window makes this routine rather than exotic:
            # at TOP_K=10 one prompt is ~2x the tokens of TOP_K=4, and a batch
            # of eval questions crosses the provider's per-minute budget
            # partway through. Retrying beats failing the answer, because the
            # user's question was fine - the quota was not.
            if attempt < RATE_LIMIT_ATTEMPTS - 1:
                wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
                log.warning("rate limited (429), retrying in %ss (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
                continue
            raise LLMUnavailable(
                "rate limited by the provider (HTTP 429) after "
                f"{RATE_LIMIT_ATTEMPTS} attempts. Wait a minute and retry, or lower "
                "TOP_K in config/settings.py to shrink the prompt."
            )

        if response.status_code >= 400:
            raise LLMUnavailable(
                f"provider returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            body = response.json()
            return str(body["choices"][0]["message"]["content"] or "").strip()
        except Exception as exc:  # noqa: BLE001 - malformed success body
            log.warning("LLM returned an unreadable body: %s", exc)
            raise LLMUnavailable(f"provider returned an unreadable response: {exc}") from exc
    raise LLMUnavailable("LLM call failed for an unclassified reason")


def generate_raw(query: str, hits: list[RetrievedChunk]) -> str:
    """Assemble the full prompt for a question and its hits, then call."""
    return generate(build_user_prompt(query, hits))
