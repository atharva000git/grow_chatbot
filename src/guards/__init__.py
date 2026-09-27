"""Stage 7 - Guards. Satisfies PRD FR-7 (advice) and FR-8 (PII).

Two deterministic gates that run *before* any LLM call, so a request that must
be refused never reaches a model, never costs a token and never leaves the
process. Both are regex-based on purpose: they are auditable, they are
testable without a network call, and their behaviour cannot drift with a model
upgrade.
"""

from src.guards.advice import ADVICE_PATTERNS, detect, pick_educational_link, refusal_response
from src.guards.pii import PII_PATTERNS, block_response, scan

__all__ = [
    "ADVICE_PATTERNS",
    "PII_PATTERNS",
    "block_response",
    "detect",
    "pick_educational_link",
    "refusal_response",
    "scan",
]
