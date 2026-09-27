"""Stage 8 - Generation. Satisfies PRD FR-6."""

from src.generation.llm import LLMUnavailable, generate, generate_raw, get_client
from src.generation.prompt import (
    SYSTEM_RULES,
    build_context_block,
    build_user_prompt,
    citation_allowlist,
    newest_source_date,
)
from src.generation.validate import (
    count_sentences,
    extract_urls,
    has_advice_language,
    has_performance_language,
    truncate_to_sentences,
    validate,
)

__all__ = [
    "LLMUnavailable",
    "SYSTEM_RULES",
    "build_context_block",
    "build_user_prompt",
    "citation_allowlist",
    "count_sentences",
    "extract_urls",
    "generate",
    "generate_raw",
    "get_client",
    "has_advice_language",
    "has_performance_language",
    "newest_source_date",
    "truncate_to_sentences",
    "validate",
]
