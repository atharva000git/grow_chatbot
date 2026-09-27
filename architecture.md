# Architecture — Mutual Fund Facts-Only RAG Chatbot (HDFC AMC)

**Version:** 1.0
**Status:** Draft for class demo
**Companion doc:** `PRD.md`
**Owner:** NextLeap PM cohort

---

## 1. Architecture Goals

1. **Faithful to the PRD.** Every PRD functional requirement (`FR-1`…`FR-10`) maps to a named module below.
2. **Full RAG pipeline visible.** Data ingestion and data retrieval are separate, explicit, inspectable stages — nothing hidden inside a framework default.
3. **Grounding over fluency.** The system is designed so that *not answering* is a first-class, reachable outcome.
4. **Offline-safe demo.** Corpus is cached to disk; the live demo performs zero network crawling and only one LLM call per query.
5. **Rebuildable.** The entire index is a pure function of the persisted corpus + pinned config.

### 1.1 Architecture principles

| Principle | Consequence in the design |
|---|---|
| No source, no answer | Retrieval threshold + explicit `INSUFFICIENT_CONTEXT` branch before generation |
| One answer, one citation | Citation is emitted from **chunk metadata**, never authored by the LLM |
| Advice is a separate class of query | A guard runs *before* retrieval/generation and can short-circuit the chain |
| PII never touches disk | Redaction happens at the boundary, before any logging or vectorization |
| Determinism | Temperature 0, fixed k, fixed threshold, no sampling anywhere |

---

## 2. System Context

```
        ┌──────────────┐
        │  End User    │  (retail investor / support agent)
        └──────┬───────┘
               │ natural language query
               ▼
        ┌──────────────────────────────────┐
        │   Streamlit UI  (Presentation)   │
        │  welcome · 3 examples · chat      │
        │  disclaimer · citation · sources  │
        └──────┬───────────────────────────┘
               │ ChatRequest { text, session_id }
               ▼
        ┌──────────────────────────────────┐
        │        RAG Orchestrator          │  src/pipeline.py
        │  guard → retrieve → prompt → LLM  │
        │  → post-validate → ChatResponse   │
        └──┬────────┬────────┬────────┬────┘
           │        │        │        │
           ▼        ▼        ▼        ▼
    ┌──────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
    │  Guards  │ │ChromaDB│ │  LLM   │ │ Logger   │
    │ PII      │ │ vector │ │ API    │ │ redacted │
    │ Advice   │ │ store  │ │(temp 0)│ │ queries  │
    └──────────┘ └────────┘ └────────┘ └──────────┘
        ▲            ▲
        │            │  built offline by the Ingestion Pipeline
   ┌────┴─────┐  ┌───┴──────────────┐
   │ Corpus   │  │ Embedding Model  │
   │ (5 pages)│  │ all-MiniLM-L6-v2  │
   └──────────┘  └──────────────────┘
```

---

## 3. Stage Map — PRD Requirement Traceability

| PRD | Stage | Module | Key artefact |
|---|---|---|---|
| `FR-1` | 1. Sources | `config/sources.csv` | 5 public URLs |
| `FR-1` | 2. Loading / Cleaning | `src/ingest/load.py` | `data/raw/*.html`, `data/clean/*.txt` |
| `FR-2` | 3. Chunking | `src/ingest/chunk.py` | `data/chunks/*.jsonl` |
| `FR-3` | 4. Embedding | `src/ingest/embed.py` | `data/embeddings/*.npy` |
| `FR-4` | 5. Vector store | `src/ingest/store.py` | `chroma/` |
| `FR-5` | 6. Retrieval | `src/retrieval/search.py` | `RetrievedChunk[]` |
| `FR-8` | 7. PII guard | `src/guards/pii.py` | `PIIVerdict` |
| `FR-7` | 7. Advice guard | `src/guards/advice.py` | `AdviceVerdict` |
| `FR-6` | 8. Generation | `src/generation/prompt.py`, `llm.py` | `DraftAnswer` |
| `FR-6` | 8. Post-validation | `src/generation/validate.py` | `ValidatedAnswer` |
| `FR-9` | 9. UI | `app.py` | Streamlit page |
| `FR-10` | 10. Docs / samples | `README.md`, `samples/sample_qa.md` | deliverables |

**Ingestion (offline, batch):** stages 1–5.
**Retrieval (online, per query):** stages 6–8.

---

## 4. Data Ingestion Pipeline (Offline)

Runs once (or on demand via `python -m src.ingest.run_all`). Emits a manifest so the UI can report index freshness.

```
sources.csv
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ STAGE 2 — LOADING                       src/ingest/load.py │
│  fetch(url) → HTML                                        │
│  boilerplate strip (nav, footer, ads, cookie banner)       │
│  main-content select → clean text                          │
│  sanity gate: len(clean_text) >= MIN_CHARS else FAIL loudly │
│  write data/raw/<scheme>.html, data/clean/<scheme>.txt     │
└───────────────────────────┬────────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────────┐
│ STAGE 3 — CHUNKING                      src/ingest/chunk.py│
│  strategy decided by inspecting real corpus (see §4.3)     │
│  recursive-character | semantic-similarity                 │
│  never split: fee table rows, FAQ Q/A pairs               │
│  emit chunk + metadata                                     │
└───────────────────────────┬────────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────────┐
│ STAGE 4 — EMBEDDING                    src/ingest/embed.py│
│  sentence-transformers/all-MiniLM-L6-v2  (dim 384)         │
│  batched, normalized embeddings → cache                    │
└───────────────────────────┬────────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────────┐
│ STAGE 5 — VECTOR STORE                   src/ingest/store.py│
│  ChromaDB PersistentClient(dir="chroma")                  │
│  collection "hdfc_faqs", cosine space                      │
│  ids: "<scheme_slug>::c<idx>"                              │
│  metadatas: source_url, scheme, category, section,        │
│             chunk_index, fetched_at, content_hash         │
└───────────────────────────┬────────────────────────────────┘
                            ▼
                    data/manifest.json
              { built_at, n_docs, n_chunks, model, dim,
                embedding_model_id, corpus_hash }
```

### 4.1 Ingestion invariants

- **Idempotent.** Re-running produces the same chunk ids and (given the same model) the same vectors.
- **Fail loud, fail early.** A source that yields too little text aborts the build with a named error — a silently empty page would poison retrieval.
- **Corpus hash.** `manifest.corpus_hash` = hash of the sorted `content_hash` values. If it changes, the UI shows a "sources updated" note.
- **No live crawl at query time.** Demo depends only on `chroma/` and `data/`.

### 4.2 Document model

```python
@dataclass(frozen=True)
class SourceDoc:
    source_id: str          # "hdfc-large-cap"
    scheme: str             # "HDFC Large Cap Fund - Direct - Growth"
    category: str           # "Large Cap" | "Flexi Cap" | "ELSS" | "Small Cap" | "Balanced Advantage"
    source_url: str         # exact public URL (also the citation string)
    fetched_at: str         # ISO-8601 date
    content_hash: str       # sha256 of cleaned text
    raw_path: str
    clean_path: str

@dataclass(frozen=True)
class Chunk:
    chunk_id: str           # "<source_id>::c<idx>"
    text: str
    source_url: str
    scheme: str
    category: str
    section: str            # "fees" | "exit_load" | "lock_in" | "riskometer" | "benchmark" | "faq" | ...
    chunk_index: int
```

### 4.3 Chunking strategy decision (FR-2)

The strategy is **not** hard-coded. `src/ingest/chunk.py` exposes both, and `notes/chunking.md` records the comparison performed on the actual corpus.

| | Recursive character split | Semantic split |
|---|---|---|
| Determinism | High | Medium (depends on model + threshold) |
| Keeps fee tables intact | Good with section-aware separators | Risk of mid-table split |
| Latency | Low | ~N× corpus passes |
| Fit for 5 short pages | Likely sufficient | May be overkill |

Decision rule: build with recursive splitting tuned to observed section boundaries; escalate to semantic only if the chunk study shows a metric is systematically separated from its scheme name. Either way, the strategy is a **config value** (`settings.chunk_strategy`), not a code fork.

### 4.4 Chunking parameters (initial)

| Param | Value | Rationale |
|---|---|---|
| `chunk_size` | 800 chars | A scheme's fee/exit-load block fits in one chunk |
| `chunk_overlap` | 120 chars | Preserves the sentence bridging two splits |
| `separators` | `\n## ` → `\n\n` → `\n` → `. ` → ` ` | Respect page section headings first |
| `min_chunk_chars` | 80 | Drop nav crumbs and orphan fragments |

---

## 5. Query Pipeline (Online)

```
user query (raw text)
      │
      ▼
┌──────────────────────────────────────────┐
│ S0. BOUNDARY — PII REDACTION             │  guards/pii.py
│    regex scan: PAN, Aadhaar, account no, │
│    OTP, email, phone                     │
│    HIT  → PII_BLOCKED response, log OFF  │  (query never persisted)
│    MISS → sanitized_text continues       │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│ S1. INTENT — ADVICE GUARD               │  guards/advice.py
│    pattern + keyword classifier          │
│    ADVICE → REFUSAL response + 1         │
│             educational link             │  (short-circuits S2–S4)
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│ S2. RETRIEVAL                           │  retrieval/search.py
│    embed(sanitized_text) with same model │
│    Chroma similarity_search(k=4)         │
│    + scheme metadata pre-filter (opt.)  │
│    + similarity threshold τ              │
│    ALL < τ  → INSUFFICIENT_CONTEXT       │  (short-circuits S3–S4)
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│ S3. PROMPT ASSEMBLY                     │  generation/prompt.py
│    system rules (facts-only, ≤3 sent.)   │
│    + numbered context blocks             │
│    + citation allow-list = context URLs  │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│ S4. GENERATION                          │  generation/llm.py
│    temperature 0, max_tokens ~200        │
│    POST {chat/completions} via requests │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│ S5. POST-VALIDATION                     │  generation/validate.py
│    1. sentence count <= 3                │
│    2. exactly one citation, and it MUST  │
│       be in the allow-list               │
│    3. no performance/return language     │
│    4. no advice language                 │
│    5. strip any LLM-invented URLs        │
│    any FAIL → degrade to safe template   │
└──────────────────┬───────────────────────┘
                   ▼
            ChatResponse { answer_text,
                           citation_url, source_scheme,
                           last_updated, intent, retrieval_hits }
                   │
                   ▼
              UI render + redacted log
```

### 5.1 Response envelope

Every terminal state is the same shape, so the UI never special-cases backend logic:

```python
Intent = Literal["ANSWER", "REFUSAL", "INSUFFICIENT_CONTEXT", "PII_BLOCKED"]

@dataclass
class ChatResponse:
    intent: Intent
    answer_text: str          # <= 3 sentences
    citation_url: str | None  # exactly one, from corpus
    source_scheme: str | None
    last_updated: str         # ISO date, max(fetched_at) of used docs
    retrieval_hits: int
```

### 5.2 Guard details

**PII guard (`S0`).** Regex families: PAN (`[a-z]{5}[0-9]{4}[a-z]`, case-insensitive), Aadhaar (12-digit with optional spaces), account number (8–18 digits), OTP (4–6 digit codes near "otp"), email, phone (+91 / 10-digit). On hit: fixed response, input cleared, **nothing written to the query log**. The log records the intent and a redacted placeholder only.

Two of these patterns were wrong until Phase 11 wrote the tests that would have caught them, and both are now pinned:

* The PAN pattern was `[A-Z]{5}[0-9]{4}[A-Z]`, so `abcde1234f` was not detected. Lower-case PANs are a real thing — an ARN, a Form 26A acknowledgement or a WhatsApp forward all routinely renders it that way.
* The country-code strip for phone numbers used a lookbehind. `\b` cannot sit between `+91` and the first digit — there is no word boundary inside `+919876543210` — so a number written in the common international form escaped detection, while the same lookbehind correctly spared a 10-digit suffix of a 12-digit account number. A **lookahead** on `(?:\+?91[- ]?)` fixes the first without reintroducing the second, and the negative test for account numbers is what proves it.

**Advice guard (`S1`).** Two signals, either sufficient:
- Regex/keyword families: `should i (buy|sell|invest)`, `which (fund|scheme) is best`, `good time to`, `how much should i invest`, `build me a portfolio`, `is (this|it) right for me`, `highest returns`.
- Lightweight intent label from the same embedding model (prototype-level, no extra LLM call).

Refusal text is fixed, ≤2 sentences, plus one **educational link** chosen from a small in-corpus map (e.g. exit-load/exit-tax guide → relevant scheme page). Refusal is a terminal state — no generation call, so the LLM cannot talk its way into advice.

### 5.3 Retrieval details (FR-5)

| Param | Value | Rationale |
|---|---|---|
| `top_k` | 10 | Filled, not capped: 8-10 chunks in practice. Costs 2.1x prompt tokens — see `notes/retrieval.md` |
| `distance_metric` | cosine | Consistent with MiniLM training |
| `threshold τ` | 0.35 similarity (unchanged; see below) | Below this the corpus does not cover the question |
| scheme pre-filter | on when the query names a scheme | Boosts precision for single-scheme questions |
| section pre-filter | on when the query names a metric | Excludes entity-matching decoys; see below |

**Why the section pre-filter exists.** Filtering on scheme alone was not enough,
because every chunk on a scheme's page repeats that scheme's name. A generic
description ("HDFC Large Cap Fund Direct Growth is a Equity Mutual Fund Scheme
launched by HDFC Mutual Fund.") scored **0.684** against "What is the exit load
of HDFC Large Cap Fund?" and buried the chunk that actually states the exit load.
Chunk-level `section` labels, made reliable in Phase 3, now constrain the search
the same way scheme labels do. Filters are tried narrowest-first and relaxed one
clause at a time, so an over-eager filter degrades to a wider search and never
into a false "no answer".

**Threshold evidence.** τ is **unchanged at 0.35**, and the ranges do *not*
separate cleanly:

| | min | mean | max |
|---|---|---|---|
| 11 in-scope questions | 0.350 | 0.564 | 0.843 |
| 12 out-of-scope questions | 0.009 | 0.285 | 0.654 |

The spec's out-of-scope probe, "Who is the prime minister of India?", scores
**0.009** and is correctly rejected. But three general-knowledge probes sharing a
token with the corpus still clear 0.35 — "Who is the CEO of HDFC Bank?" 0.654,
"Recommend a restaurant in Delhi" 0.436, "What is the GDP of India?" 0.422. An
unfiltered query always has a best match in *some* corpus, so similarity alone
cannot police scope. Raising τ to exclude those would reject the in-scope
"lock-in period" question at exactly 0.350, and the spec explicitly forbids
tuning τ to make results look good. τ is therefore left at 0.35, and the residual
false positives are left to the Phase 7 grounding and citation guards, which can
check whether the retrieved context actually answers the question. Raising τ
later requires re-measuring this table.

Chunk text sent to the LLM is prefixed with an index (`[1]`, `[2]`, …) and each context block carries its `source_url`, so the model can only *point at* a real URL.

### 5.3a Provider transport (deviation, recorded in Phase 8)

`S4` posts to `{LLM_BASE_URL or https://api.openai.com/v1}/chat/completions`
directly, with `requests`, and parses `choices[0].message.content` itself. It
issues **one** HTTP call and **one** `choices[0]` read.

The natural alternative is `langchain-openai`'s `ChatOpenAI`, but
`langchain-core==0.2.2` is pinned in `requirements.txt` while the matching
`langchain-openai` version is not installable in this environment. Adopting the
wrapper for a single endpoint call would have added a version constraint and a
second parsing layer for no gain, so `langchain_core` is used for the ported
message types and the HTTP call is explicit. Swapping in `ChatOpenAI` later is a
change to `src/generation/llm.py` alone — no other module imports it.

A missing key raises `LLMUnavailable` rather than degrading to an extractive
answer, so the failure is visible in the UI and the eval harness reports
`ERROR:NO_KEY` instead of a passing score.

### 5.4 Prompt contract (FR-6)
System rules, verbatim in `generation/prompt.py`:

```
You are a mutual fund FACTS assistant for HDFC AMC schemes.

Rules (non-negotiable):
1. Use ONLY the numbered CONTEXT below. Nothing else is a permitted source.
2. Answer in at most 3 sentences. No bullet lists.
3. End with exactly ONE citation line: "Source: <url>" using a url present in CONTEXT.
4. End with: "Last updated from sources: <YYYY-MM-DD>" using the newest date in CONTEXT.
5. If CONTEXT does not contain the fact, reply exactly: INSUFFICIENT_CONTEXT
6. Never state or compare returns, NAV trends, or performance. If asked, say the figure
   is in the official factsheet and cite it.
7. Never advise. No "should", "recommend", "suitable for you", buy/sell language.
8. Never invent a URL, date, or number. No PII requests.
```

Response parsing treats `INSUFFICIENT_CONTEXT` as a first-class return, not an error.

### 5.5 Post-validation and safe degradation (FR-6, NFR-1)

`validate.py` is the last gate before the UI. On any violation it substitutes a safe template and records the reason for the demo narration:

| Check | On failure |
|---|---|
| > 3 sentences | truncate to 3 |
| 0 or >1 citation | inject the top-ranked context URL |
| citation not in allow-list | replace with top-ranked context URL |
| performance/advice vocabulary | fall back to `REFUSAL` template |
| `INSUFFICIENT_CONTEXT` | emit `INSUFFICIENT_CONTEXT` template |

---

## 6. Module Layout

```
.
├── app.py                          # Streamlit UI (FR-9)
├── config/
│   ├── sources.csv                 # FR-1: the 5 public URLs + metadata
│   └── settings.py                 # thresholds, k, chunk params, model ids
├── src/
│   ├── models.py                   # SourceDoc, Chunk, ChatResponse, Intent
│   ├── pipeline.py                 # RAG orchestrator (online path)
│   ├── ingest/
│   │   ├── load.py                 # FR-1 fetch + clean + sanity gate
│   │   ├── chunk.py                # FR-2 recursive | semantic
│   │   ├── embed.py                # FR-3 MiniLM embedder (cached)
│   │   ├── store.py                # FR-4 Chroma build
│   │   ├── manifest.py             # build metadata + corpus hash
│   │   └── run_all.py              # stage runner CLI
│   ├── retrieval/
│   │   ├── embedder.py             # shared singleton embedder
│   │   └── search.py               # FR-5 top-k + threshold + pre-filter
│   ├── guards/
│   │   ├── pii.py                  # FR-8
│   │   └── advice.py               # FR-7
│   ├── generation/
│   │   ├── prompt.py               # FR-6 templates
│   │   ├── llm.py                  # LLM client, temperature 0
│   │   └── validate.py             # FR-6 post-validation
│   └── logging_utils.py            # redacted query log
├── data/
│   ├── raw/  clean/  chunks/  embeddings/  manifest.json
├── chroma/                          # FR-4 persisted index
├── notes/chunking.md                # FR-2 strategy study
├── samples/sample_qa.md             # FR-10 5–10 Q&A with links
├── eval/run_eval.py                 # validation set runner
├── tests/                           # guard + validation unit tests
├── README.md
└── requirements.txt                 # NFR-4 pinned
```

**Dependency rule:** `ingest/*` never imports `retrieval/*`, `guards/*`, or `generation/*`. The ingestion path is usable standalone as a notebook, which satisfies the PRD's "app or notebook" deliverable option.

---

## 7. Configuration

`config/settings.py` (env-overridable via `python-dotenv`, see `.env.example`):

```python
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM     = 384
CHROMA_DIR        = "chroma"
COLLECTION_NAME   = "hdfc_faqs"
CHUNK_STRATEGY    = "recursive"      # or "semantic"
CHUNK_SIZE        = 800
CHUNK_OVERLAP     = 120
MIN_CHUNK_CHARS   = 80
MIN_DOC_CHARS     = 1000             # sanity gate, FR-1
TOP_K             = 10
SIMILARITY_THRESHOLD = 0.35
LLM_TEMPERATURE   = 0.0
LLM_MAX_TOKENS    = 220
MAX_ANSWER_SENTENCES = 3
LLM_API_KEY       = os.getenv("LLM_API_KEY")
LLM_MODEL         = os.getenv("LLM_MODEL", "")       # "" -> gpt-4o-mini in llm.py
LLM_BASE_URL      = os.getenv("LLM_BASE_URL", "")     # "" -> https://api.openai.com/v1
LOG_QUERIES       = False            # NFR-5: off by default
```

`config/sources.csv` — 10 documents across 6 public URLs. `source_type` is `html` or `pdf`; `page_start`/`page_end` give a 1-indexed inclusive page slice for PDFs.

```csv
source_id,scheme,category,source_url,source_type,page_start,page_end
hdfc-large-cap,HDFC Large Cap Fund - Direct - Growth,Large Cap,https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth,html,0,0
hdfc-flexi-cap,HDFC Equity Fund - Direct - Growth,Flexi Cap,https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth,html,0,0
hdfc-elss,HDFC ELSS Tax Saver Fund - Direct - Plan - Growth,ELSS,https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth,html,0,0
hdfc-small-cap,HDFC Small Cap Fund - Direct - Growth,Small Cap,https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth,html,0,0
hdfc-balanced-advantage,HDFC Balanced Advantage Fund - Direct - Growth,Balanced Advantage,https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth,html,0,0
hdfc-fs-large-cap,HDFC Large Cap Fund - Direct - Growth,Large Cap,<FACTSHEET_URL>,pdf,12,13
hdfc-fs-flexi-cap,HDFC Equity Fund - Direct - Growth,Flexi Cap,<FACTSHEET_URL>,pdf,7,8
hdfc-fs-elss,HDFC ELSS Tax Saver Fund - Direct - Plan - Growth,ELSS,<FACTSHEET_URL>,pdf,63,64
hdfc-fs-small-cap,HDFC Small Cap Fund - Direct - Growth,Small Cap,<FACTSHEET_URL>,pdf,16,17
hdfc-fs-balanced-advantage,HDFC Balanced Advantage Fund - Direct - Growth,Balanced Advantage,<FACTSHEET_URL>,pdf,44,47
```

`<FACTSHEET_URL>` = `https://files.hdfcfund.com/s3fs-public/2026-09/HDFC%20MF%20Factsheet%20-%20August%202026.pdf` (HDFC MF master factsheet, 144 pages, August 2026). Each scheme gets a 2–4 page slice so a chunk is never detached from its scheme name. The PDF is fetched once and cached by URL hash; all five slices share that one file on disk.

---

## 8. Sequence — One Factual Query

```
Streamlit        pipeline        pii      advice     search     chroma      llm      validate
    │ query         │             │          │          │         │          │          │
    ├──────────────►│             │          │          │         │          │          │
    │               ├────────────►│          │          │         │          │          │
    │               │◄────────────┤ verdict  │          │         │          │          │
    │               ├────────────────────────►│          │         │          │          │
    │               │◄────────────────────────┤ ALLOW    │          │          │          │
    │               ├──────────────────────────────────────►│         │          │          │
    │               │                                       ├────────►│          │          │
    │               │◄──────────────────────────────────────────────────────┤          │
    │               │  (hits, scores, metadata)             │         │          │          │
    │               │  if max(sim) < τ → INSUFFICIENT_CONTEXT (no LLM call)        │
    │               ├───────────────────────────────────────────────────────────────►│
    │               │◄───────────────────────────────────────────────────────────────┤
    │               ├──────────────────────────────────────────────────────────────────────►│
    │               │◄──────────────────────────────────────────────────────────────────────┤
    │               │  ChatResponse                                                        │
    │◄──────────────┤                                                                      │
    │ render answer + 1 citation + "Last updated from sources: ..." + disclaimer
```

---

## 9. Failure Modes & Handling

| Failure | Detection | Behaviour |
|---|---|---|
| Source page JS-rendered / blocked | clean text < `MIN_DOC_CHARS` | `IngestError` naming the URL; fall back to official HDFC factsheet PDF (public) or manual text export |
| Corpus missing at query time | `chroma/` absent | UI shows "index not built — run `python -m src.ingest.run_all`" instead of a traceback |
| Embedding model unavailable offline | load failure at startup | Use the cached `data/embeddings/*.npy` for retrieval; LLM call still required |
| LLM API down / no key | client exception | Hard stop with a clear banner: "generation unavailable, cannot answer facts-only" — **never** fall back to an ungrounded answer |
| All chunks below τ | `max(similarity) < τ` | `INSUFFICIENT_CONTEXT`: "I couldn't find that in the HDFC pages I have. Sources: …" |
| Citation hallucinated by LLM | not in allow-list | Replaced with top-ranked context URL; reason logged |
| PII in query | regex hit | `PII_BLOCKED`, input cleared, nothing logged |
| Advice question | guard hit | `REFUSAL` + educational link, no LLM call |

**Anti-pattern explicitly forbidden:** an "I don't know" fallback that calls the LLM without context. Grounding failure must never degrade into ungrounded generation.

---

## 10. Performance & Resource Profile (NFR-2, NFR-6)

| Operation | Cost |
|---|---|
| Ingestion build (5 docs) | < 60 s on CPU, one-off |
| Query embedding (MiniLM, CPU) | ~30–80 ms |
| Chroma top-k (k=4) | < 20 ms |
| LLM generation | ~1.5–4 s (dominant term) |
| **End-to-end warm** | **< 5 s** |

Embedder is a module-level singleton; the corpus is 5 short pages, so the working set is a few MB. No GPU, no external vector service.

---

## 11. Security & Privacy Controls (NFR-5, PRD constraints)

| Control | Implementation |
|---|---|
| PII containment | Redaction at `S0`, before logging and before embedding |
| No PII persistence | `LOG_QUERIES = False` by default; when enabled, only redacted text + intent |
| Public sources only | URLs are whitelisted in `config/sources.csv`; the fetcher refuses any host not listed |
| Secrets | `LLM_API_KEY` from `.env`; `.env` gitignored; no key in prompts or logs |
| Citation integrity | Only whitelisted `source_url` values can ever reach the UI |
| No telemetry | No analytics, no outbound calls except the configured LLM endpoint |

---

## 12. Deployment Topology

**Primary (demo):** local Streamlit against a prebuilt `chroma/`.

```bash
pip install -r requirements.txt
python -m src.ingest.run_all     # build corpus → chunks → embeddings → index
streamlit run app.py
```

**Optional (hosted link):** Streamlit Community Cloud with `chroma/`, `data/`, and `config/sources.csv` committed; `LLM_API_KEY` supplied as a platform secret. Embeddings are precomputed so the cloud container does not need to run `all-MiniLM-L6-v2` at boot for retrieval — though the embedder still loads locally to encode the query.

**Fallback:** a notebook walking stages 1→8 with printed retrieval scores and citations, plus a ≤3-minute screen recording. The ingestion path is importable without the UI for exactly this reason.

---

## 13. Architectural Decisions

| # | Decision | Rationale | Alternative rejected |
|---|---|---|---|
| AD-1 | ChromaDB persistent client, cosine space | Zero-ops, ships with metadata filtering, matches PRD | Pinecone/Qdrant — external service, no benefit at 5 docs |
| AD-2 | `all-MiniLM-L6-v2` | 384-dim, fast on CPU, strong retrieval quality | OpenAI embeddings — needs network, cost, no offline demo |
| AD-3 | Guards before generation | Advice/PII become terminal states the LLM cannot override | Post-hoc filtering — the model has already produced the text |
| AD-4 | Citation from metadata, not model output | Makes an uncited or fabricated answer structurally impossible | Letting the LLM write the link — hallucinates URLs |
| AD-5 | Similarity threshold gate | Enables a truthful "not in corpus" answer | Always top-k — confident wrong answers |
| AD-6 | Cache corpus + index to disk | Deterministic demo; no rate-limit/ToS risk mid-presentation | Live crawl at query time |
| AD-7 | Strategy as config, not code branch | PRD requires deciding from the real corpus; keeps both paths testable | Hard-coding one splitter |
| AD-8 | Streamlit over a custom frontend | Fastest path to a credible demo UI | React/Next — no added value for a ≤3-min demo |
| AD-9 | Ingestion decoupled from online path | Enables the notebook deliverable and index-only rebuilds | Single monolithic script |

---

## 14. Validation & Test Architecture

| Level | Target | Method |
|---|---|---|
| Unit | PII patterns (PAN, Aadhaar, account, OTP, email, phone) | `tests/test_pii.py` — positive + negative cases |
| Unit | Advice refusal families | `tests/test_advice.py` — all §9 PRD items refuse |
| Unit | Post-validation rules (sentence cap, citation allow-list, perf-language) | `tests/test_validate.py` |
| Integration | End-to-end on the 7 factual validation queries | `eval/run_eval.py` — prints intent, answer, citation, score, latency |
| Integration | Negative cases (2 advice + 1 PII + 1 out-of-scope) | same runner, asserts `REFUSAL` / `PII_BLOCKED` / `INSUFFICIENT_CONTEXT` |
| Manual | Citation links open and contain the stated fact | checklist in `README.md` |

Acceptance gates (PRD §10): groundedness 100%, exactly-one-citation 100%, ≤3 sentences 100%, refusals 100%, zero performance claims, warm latency < 5 s.

---

## 15. Architecture Risks

| # | Risk | Architectural mitigation |
|---|---|---|
| A-1 | JS-rendered sources yield empty shells | `MIN_DOC_CHARS` sanity gate fails the build loudly; documented factsheet-PDF fallback in §9 |
| A-2 | Threshold too strict → too many "not found" | `SIMILARITY_THRESHOLD` tuned against the validation set; surfaced in the UI so the failure is explainable |
| A-3 | Threshold too loose → confident wrong answers | Post-validation + single-citation rule; eval runner reports the retrieval score for every demo query |
| A-4 | Chunking separates a metric from its scheme name | Section-aware separators + `scheme` metadata pre-filter; chunk study in `notes/chunking.md` |
| A-5 | LLM ignores the ≤3-sentence rule | `validate.py` truncates deterministically rather than trusting the model |
| A-6 | Index drift vs. source dates | `manifest.json` corpus hash + `fetched_at` surfaced as "Last updated from sources" |
| A-7 | Offline demo breaks on model download | Embeddings precomputed; embedder load failure degrades to cached vectors with a visible warning |
| A-8 | Scope creep beyond 5 schemes | Source allow-list is a single CSV; adding a scheme is a config change, not a code change |

---

## 16. Demo Walkthrough Order (narrative for the ≤3-min video)

1. **Corpus** — show `config/sources.csv`, the 5 HDFC schemes.
2. **Loading** — show `data/clean/*.txt`, note the boilerplate strip.
3. **Chunking** — show 3 sample chunks with metadata; cite `notes/chunking.md` for the strategy choice.
4. **Embedding** — show `all-MiniLM-L6-v2`, 384-dim, batch run.
5. **Vector store** — show the Chroma collection and a chunk count.
6. **Retrieval** — show top-4 chunks + similarity scores for one query.
7. **Generation** — show the ≤3-sentence answer, the one citation, the freshness stamp.
8. **Guard: advice** — "Should I buy HDFC Small Cap Fund now?" → polite refusal + educational link.
9. **Guard: PII** — feed a PAN + phone → blocked, nothing stored.
10. **Out of scope** — "What is the 5-year return?" → no number, points to the official factsheet.
11. **Closing** — deliverable list: prototype, source list, README, sample Q&A, disclaimer.
