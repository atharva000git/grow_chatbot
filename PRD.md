# PRD — Mutual Fund Facts-Only RAG Chatbot (HDFC AMC)

**Version:** 1.0
**Status:** Draft for class demo
**Owner:** NextLeap PM cohort
**Deliverable type:** Working prototype (app or notebook) + ≤3-min demo video fallback

---

## 1. Problem Statement

Retail investors and support/content teams repeatedly ask the same factual questions about mutual fund schemes — expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, and how to download statements. These answers live across a scattered set of public pages, are easy to misquote, and are frequently confused with investment *advice*.

We will build a **Retrieval-Augmented Generation (RAG) chatbot** that answers **facts only** about a scoped set of HDFC AMC schemes, grounded exclusively in a small corpus of public pages. Every answer must carry **one source citation**. The bot must **refuse** opinionated or portfolio questions.

### 1.1 Target Users

| User | Need |
|---|---|
| Retail users comparing schemes | Fast, trustworthy factual comparison across a small scheme set |
| Support / content teams | Deflect repetitive MF queries with consistent, sourced answers |

### 1.2 Why RAG (and not a plain LLM)

An LLM alone will hallucinate fee figures and lock-in periods. Retrieval grounding + mandatory citation means every factual claim is traceable to a crawled public page, and a "no source, no answer" rule keeps the bot inside its lane.

---

## 2. Goals & Non-Goals

### 2.1 Goals

1. Answer factual scheme queries using **only** retrieved content from the scoped corpus.
2. Show **one clear citation link** in every factual answer.
3. Politely refuse advice/portfolio questions with a facts-only message + an educational link.
4. Enforce hard constraints: public sources only, no PII, no performance claims, ≤3 sentences per answer, freshness stamp.
5. Demonstrate the **full RAG pipeline** end-to-end: Loading → Chunking → Embedding → Vector Store → Retrieval → Generation.

### 2.2 Non-Goals

- No performance/return computation, comparison, or projection. If asked, link the official factsheet.
- No account servicing, no logins, no transaction execution.
- No PII handling of any kind (PAN, Aadhaar, account numbers, OTPs, emails, phone numbers).
- No third-party blog sources, no screenshots of back-end internals as evidence.
- No multi-AMC scope for this milestone.

---

## 3. Scope

### 3.1 AMC

**HDFC Asset Management (HDFC AMC)** — one AMC only.

### 3.2 Schemes (5, all Direct–Growth)

| # | Category | Scheme | URL |
|---|---|---|---|
| 1 | Large Cap | HDFC Large Cap Fund – Direct – Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| 2 | Flexi Cap | HDFC Equity Fund (Flexi Cap) – Direct – Growth | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| 3 | ELSS | HDFC ELSS Tax Saver Fund – Direct – Plan – Growth | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| 4 | Small Cap | HDFC Small Cap Fund – Direct – Growth | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| 5 | Balanced Advantage (Hybrid) | HDFC Balanced Advantage Fund – Direct – Growth | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

Source type: public scheme pages.

**Supplementary official source (required, not optional).** A corpus audit during Phase 2 showed the 5 Groww pages do **not** contain: the ELSS lock-in period, the SEBI riskometer rating, or capital-gains statement guidance. The official HDFC MF factsheet does contain the lock-in, benchmark, expense ratio and exit load. It is therefore a second required source, sliced per scheme:

| Scheme | HDFC MF Factsheet (August 2026), pages |
|---|---|
| HDFC Flexi Cap Fund | 7–8 |
| HDFC Large Cap Fund | 12–13 |
| HDFC Small Cap Fund | 16–17 |
| HDFC Balanced Advantage Fund | 44–47 |
| HDFC ELSS – Tax Saver Fund | 63–64 |

Factsheet URL: `https://files.hdfcfund.com/s3fs-public/2026-09/HDFC%20MF%20Factsheet%20-%20August%202026.pdf`

Note: `www.hdfcfund.com` returns HTTP 403 to automated requests, so the AMC's own scheme pages cannot be scraped. The `files.hdfcfund.com` document host is publicly fetchable and is used instead.

**Known corpus limits (documented, not hidden).**
- The riskometer *rating* is a vector graphic in the factsheet and is not text-extractable. The Groww page carries the rating in prose ("rated Very High risk"), so the risk level is answerable but sourced from Groww, not SEBI's own scale.
- Capital-gains statement download guidance is absent from all 6 URLs. The bot returns "not in my sources" and points the user to the AMC's investor-services page.

### 3.3 Out of scope

- Other AMCs, other plan variants (IDCW, Regular), direct-vs-regular comparison.
- NAV history, return series, tax computation.

---

## 4. Functional Requirements

### FR-1 — Corpus Ingestion (Loading)

- Crawl/fetch the 5 scoped public pages as static HTML → clean text.
- Strip navigation, ads, cookie banners, footer boilerplate.
- Persist raw + cleaned text to disk so re-runs are deterministic (no live crawling at query time).
- Record, per document: `source_url`, `scheme`, `category`, `fetched_at`, `content_hash`.

### FR-2 — Chunking

- Strategy is **decided after inspecting the real corpus** (not fixed up front).
- Candidate strategies: **recursive character/token chunking** (size + overlap tuned to observed section boundaries) and **semantic chunking** (embedding-similarity split at topic shifts).
- Rules: chunks must not split a fee table row or a FAQ Q/A pair; target chunk size must keep a scheme name + its metric in the same chunk wherever possible.
- Every chunk retains metadata: `source_url`, `scheme`, `section`, `chunk_index`.

### FR-3 — Embedding

- Model: **`sentence-transformers/all-MiniLM-L6-v2`** (Hugging Face).
- One embedding pass over all chunks; embeddings cached to disk.
- Model name and dimension recorded in the run log for reproducibility.

### FR-4 — Vector Store

- **ChromaDB**, persistent client directory.
- One collection per milestone (e.g. `hdfc_faqs`), cosine distance space.
- Metadata stored alongside each embedding for citation rendering.

### FR-5 — Retrieval

- Query embedded with the same model; top-k semantic search.
- k default 4 (tunable, ≤6).
- Distance/similarity threshold to reject weak matches — below threshold, the bot must not answer.
- Optional: metadata pre-filter by `scheme` when the user names a scheme.

### FR-6 — Answer Generation

- Prompt is facts-only, hard-constrained:
  - Use **only** the provided context.
  - **≤3 sentences.**
  - Every answer ends with exactly **one** citation link from context metadata.
  - Append `Last updated from sources: <date of the newest source doc used>`.
  - If context is insufficient, say so and refuse rather than guess.
  - No performance claims, no return figures, no buy/sell language, no advice.
- No fabricated URLs: the citation must be a `source_url` present in the retrieved chunk metadata.

### FR-7 — Refusal / Scope Guard

- Detect and refuse: "Should I buy…", "Which is better?", "Is now a good time to invest?", "Build me a portfolio", "How much should I invest?", "Which fund gives highest returns?".
- Refusal template (≤2 sentences) + one **relevant educational link** drawn from the corpus, e.g. a factsheet/valuation guide.
- Applies *before* or *in parallel with* retrieval; refusal wins.

### FR-8 — PII Guard

- Input filter rejects/ignores PAN, Aadhaar, account numbers, OTPs, emails, phone numbers.
- Pattern-based detection on the query; on hit, respond with a fixed "do not share personal identifiers" message and **do not** store the query text.
- Query log stores only non-PII, redacted queries.

### FR-9 — UI (tiny)

- Single page, minimal.
- Welcome line + **3 example questions** + note: **"Facts-only. No investment advice."**
- Message thread: user query, assistant answer, citation link, freshness stamp.
- Optional: "Sources" expander showing the 5 source URLs.

### FR-10 — Deliverables & Docs

| Deliverable | Requirement |
|---|---|
| Prototype | Working app or notebook link; ≤3-min demo video if hosting impossible |
| Source list | CSV/MD of the 5 URLs used |
| README | Setup steps, scope (AMC + schemes), known limits |
| Sample Q&A | 5–10 queries with assistant answers + links |
| Disclaimer snippet | Exact text used in the UI |

---

## 5. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | Answer groundedness: every factual answer traceable to a retrieved chunk; zero uncited answers |
| NFR-2 | Latency: < 5 s per query end-to-end on demo hardware (warm index) |
| NFR-3 | Determinism: temperature 0; index rebuildable from persisted artifacts |
| NFR-4 | Reproducibility: pinned model + library versions in `requirements.txt` |
| NFR-5 | Privacy: zero PII persisted; corpus is public-only; no telemetry |
| NFR-6 | Portability: runs locally on CPU; no GPU required |
| NFR-7 | Transparency: refusal reasons and sources visible in UI |

---

## 6. System Architecture

```
┌──────────────┐
│  1. SOURCES  │  5 public Groww/HDFC pages (+ optional factsheet/KIM/FAQ pages)
└──────┬───────┘
       ▼
┌──────────────┐
│  2. LOADING  │  fetch HTML → strip boilerplate → clean text → persist raw/clean
└──────┬───────┘
       ▼
┌──────────────┐
│  3. CHUNKING │  strategy chosen post-inspection (recursive | semantic)
└──────┬───────┘  metadata: source_url, scheme, section, chunk_index
       ▼
┌──────────────┐
│ 4. EMBEDDING │  sentence-transformers/all-MiniLM-L6-v2
└──────┬───────┘
       ▼
┌──────────────┐
│ 5. VECTOR DB │  ChromaDB (persistent)
└──────┬───────┘
       ▼
┌────────────────────────────────────────────┐
│ 6. RETRIEVAL  │  embed query → top-k search   │
│                + similarity threshold        │
└──────┬─────────────────────────────────────┘
       ▼
┌────────────────────────────────────────────┐
│ 7. GUARDS     │  PII filter → advice-refusal   │
└──────┬─────────────────────────────────────┘
       ▼
┌────────────────────────────────────────────┐
│ 8. GENERATION │  facts-only prompt, ≤3 sent.  │
│                1 citation + freshness stamp   │
└──────┬─────────────────────────────────────┘
       ▼
┌──────────────┐
│   9. UI      │  chat thread + citation + disclaimer
└──────────────┘
```

**Data ingestion** = stages 1–5 (offline, batch).
**Data retrieval** = stages 6–8 (online, per query).

---

## 7. Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.10+ |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (persistent client) |
| Chunking | LangChain text splitters (recursive / semantic) |
| Retrieval | Chroma similarity search + relevance threshold |
| LLM | One hosted LLM API (temperature 0) |
| Orchestration | LangChain (LCEL) |
| UI | Streamlit (preferred) — single page |
| Fetching | `requests` + `BeautifulSoup4` (static HTML) |
| Config | `python-dotenv`, `pydantic` |
| Packaging | `requirements.txt`, `.env.example` |

---

## 8. UI / UX Specification

### 8.1 Layout

```
┌────────────────────────────────────────────┐
│  HDFC Mutual Fund Facts Assistant          │
│  Facts-only. No investment advice.         │
├────────────────────────────────────────────┤
│  💡 Try:                                   │
│  • What is the exit load of HDFC Large Cap?│
│  • What is the lock-in period for HDFC     │
│    ELSS Tax Saver Fund?                    │
│  • What is the minimum SIP for HDFC Small  │
│    Cap Fund?                               │
├────────────────────────────────────────────┤
│  You: <query>                              │
│  Assistant: <≤3 sentences>                 │
│  Source: <one link>                        │
│  Last updated from sources: YYYY-MM-DD     │
└────────────────────────────────────────────┘
```

### 8.2 Behaviours

- Streaming or immediate single response; spinner during retrieval.
- Citation link opens source page in a new tab.
- Refusal state renders in a distinct style with educational link.
- PII-block state renders a fixed notice; input cleared.
- "Sources" expander lists all 5 corpus URLs.

### 8.3 Disclaimer (verbatim, used in UI + deliverables)

> **Facts-only. No investment advice.** Answers are generated from public HDFC AMC scheme pages and may be incomplete or out of date. Verify all details with the official factsheet and your mutual fund distributor before acting. Mutual fund investments are subject to market risks; read all scheme-related documents carefully.

---

## 9. Example Queries (validation set)

**Must answer with citation (facts):**

1. What is the expense ratio of HDFC Large Cap Fund – Direct – Growth?
2. What is the exit load on HDFC Equity Fund (Flexi Cap)?
3. What is the minimum SIP amount for HDFC Small Cap Fund – Direct – Growth?
4. What is the lock-in period for HDFC ELSS Tax Saver Fund, and when can I exit?
5. What is the benchmark and riskometer category of HDFC Balanced Advantage Fund?
6. How do I download my capital-gains statement?
7. What is the NAV of HDFC Large Cap Fund? *(if present in source)*

**Must refuse (advice):**

8. Should I buy HDFC Small Cap Fund now?
9. Which of these 5 funds is the best for long-term wealth creation?
10. Is HDFC ELSS a good fit for me?

**Must not appear (PII):**

11. My PAN is ABCDE1234F and my phone is 9876543210 — update my details.

---

## 10. Evaluation & Acceptance Criteria

### 10.1 Metrics

| Metric | Target |
|---|---|
| Answer groundedness (claims traceable to a chunk) | 100% |
| Answers with exactly one valid citation | 100% |
| Answers ≤3 sentences | 100% |
| Advice/portfolio questions correctly refused | 100% (10/10 style) |
| PII questions blocked, nothing stored | 100% |
| Performance/return claims made | 0 |
| Retrieval hit rate on factual validation set | ≥ 8/10 with threshold not misfiring |
| Answer latency (warm) | < 5 s |

### 10.2 Acceptance checklist

- [ ] All 7 RAG stages implemented and visible in code/README diagram
- [ ] Chunking strategy chosen from real corpus inspection, rationale documented
- [ ] Corpus = 5 HDFC schemes, source CSV/MD committed
- [ ] Every factual answer has one working citation link
- [ ] Refusal path works with educational link
- [ ] PII filter tested and query log verified clean
- [ ] ≤3-sentence limit and freshness stamp present
- [ ] Disclaimer text in UI
- [ ] README with setup + scope + known limits
- [ ] Sample Q&A file with 5–10 entries
- [ ] Demo video ≤3 min recorded as backup
- [ ] `requirements.txt` pinned; clean-machine setup verified

---

## 11. Risks & Mitigations

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| 1 | Source pages are JS-rendered / blocked; scraper gets empty shell | High | Fall back to official factsheet PDFs (public) and manual text export; verify fetch returns >1 KB of scheme text before ingesting |
| 2 | Hallucinated fee/lock-in figures | High | Strict context-only prompt, temperature 0, single-citation rule, answer-level "insufficient context" path |
| 3 | Semantic chunking over-splits fee tables | Medium | Compare recursive vs semantic on the real corpus; keep tables intact via section-aware metadata |
| 4 | Retrieval returns irrelevant chunks confidently | Medium | Similarity threshold + "no answer" branch; show retrieved sources so failures are visible |
| 5 | Bot drifts into advice | High | Advice/refusal classifier ahead of generation + hard prompt rules + refusal test set in CI-ish script |
| 6 | Rate limiting / ToS on crawling during demo | Medium | Cache corpus to disk once; demo reads from disk, no live fetch |
| 7 | Scope creep beyond 5 schemes | Medium | Fixed corpus; extra sources require a source-list update |
| 8 | PII accidentally logged | High | Redaction before logging; log store disabled by default in demo |

---

## 12. Milestones

| # | Milestone | Output | Est. |
|---|---|---|---|
| M1 | Source collection & cleaning | `data/raw/*.html`, `data/clean/*.txt`, `sources.csv` | 0.5 day |
| M2 | Chunking strategy study | `notes/chunking.md` with chosen strategy + params | 0.5 day |
| M3 | Embedding + Chroma index | `chroma/` persisted index, embeddings cached | 0.5 day |
| M4 | Retrieval + prompt + generation | `src/chain.py`, answer/citation logic | 1 day |
| M5 | Guards (PII + advice refusal) | `src/guards.py`, test script | 0.5 day |
| M6 | UI (Streamlit) | Single-page app with disclaimer + 3 examples | 0.5 day |
| M7 | Evaluation on validation set | `samples/sample_qa.md`, metrics table | 0.5 day |
| M8 | Docs, README, demo video | README, source list, ≤3 min video | 0.5 day |

**Total:** ~4.5 working days.

---

## 13. Open Questions

1. Which LLM API will the team use for generation, and is a key available for the demo (offline fallback needed)?
2. Is live crawling permitted by the demo hosts, or should we commit the cleaned corpus to the repo?
3. Do we add official HDFC factsheet PDFs as supplementary sources, or keep strictly to the 5 Groww pages?
4. Should the corpus be rebuilt per run (slow) or cached with a version stamp?
5. Hosting target for the prototype link — Streamlit Community Cloud, or local-only + video?

---

## 14. Appendix — RAG Pipeline Checklist

| Stage | Component | Status |
|---|---|---|
| 1 | Source list (5 public URLs) | Specified |
| 2 | Loading / cleaning | Specified |
| 3 | Chunking (strategy decided on data) | Specified |
| 4 | Embedding (`all-MiniLM-L6-v2`) | Specified |
| 5 | Vector store (ChromaDB) | Specified |
| 6 | Retrieval (top-k + threshold) | Specified |
| 7 | Guards (PII, advice refusal) | Specified |
| 8 | Generation (facts-only, 1 citation) | Specified |
| 9 | UI + disclaimer | Specified |
| 10 | Docs, samples, demo | Specified |
