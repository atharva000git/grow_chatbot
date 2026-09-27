"""Query logging. Complements the PII guard (FR-8).

Logging is off by default (`settings.LOG_QUERIES = False`) and this module is a
no-op unless it is switched on. The guard that matters is not the flag but the
contract: `log_query` only ever receives sanitized text. The Phase 9
orchestrator is responsible for passing `PIIVerdict.sanitized_text` and must
never pass the raw query, so an identifier cannot reach disk even if logging is
enabled later.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from src.models import REPO_ROOT

log = logging.getLogger(__name__)

QUERY_LOG_PATH = REPO_ROOT / "data" / "query_log.jsonl"


def log_query(sanitized_text: str, intent: str) -> None:
    """Append one JSONL record, or do nothing when logging is disabled."""
    if not settings.LOG_QUERIES:
        return
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query": sanitized_text,
        "intent": intent,
    }
    QUERY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUERY_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
