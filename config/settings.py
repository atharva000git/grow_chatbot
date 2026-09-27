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
TOP_K = 4
SIMILARITY_THRESHOLD = 0.35
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 220
MAX_ANSWER_SENTENCES = 3
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LOG_QUERIES = False
EDUCATIONAL_LINKS: dict[str, str] = {}
