"""Tunable configuration. Every value here is quoted in architecture.md section 7."""

import os

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHROMA_DIR = "chroma"
COLLECTION_NAME = "hdfc_faqs"
CHUNK_STRATEGY = "recursive"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
MIN_CHUNK_CHARS = 80
MIN_DOC_CHARS = 1000
# Retrieval context window: how many chunks reach the prompt.
#
# Raised 4 -> 10. Two things make a wide window safe here, and one that does not.
#
# Safe: `is_sufficient` is an any-hit gate on the BEST similarity, and hits come
# back ranked descending, so candidates 5-10 can only ever be lower-scoring.
# Widening the window therefore cannot turn an out-of-scope question into an
# answerable one - the top hit decides that, and it does not move.
#
# Safe: `hnsw:search_ef` is pinned to 256 in `ingest/store.py`, well clear of 10.
# At the default 10 it would sit exactly at k and start dropping recall.
#
# Not free: deeper context reaches chunks that were previously invisible,
# including the Groww returns tables. More context is not automatically better -
# see the measured effect in notes/retrieval.md.
TOP_K = 10
SIMILARITY_THRESHOLD = 0.35
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 220
MAX_ANSWER_SENTENCES = 3
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LOG_QUERIES = False
EDUCATIONAL_LINKS: dict[str, str] = {}
