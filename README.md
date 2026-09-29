# HDFC Mutual Fund Facts Assistant

A facts-only RAG chatbot for five HDFC AMC mutual fund schemes. It answers
questions about exit load, lock-in period, minimum SIP and lump-sum amounts,
expense ratio, benchmark and riskometer category, and AUM, citing the specific
page and factsheet each fact came from. It refuses to recommend or compare
schemes, blocks personal identifiers before they reach the model, and answers
"I don't have that" when the corpus does not support an answer — because a
mutual-fund facts interface that guesses is worse than one that admits ignorance.

**Table of contents**

1. [What it is](#1-what-it-is)
2. [Scope](#2-scope)
3. [Architecture](#3-architecture)
4. [Setup](#4-setup)
5. [Rebuilding the index](#5-rebuilding-the-index)
6. [Tests and evaluation](#6-tests-and-evaluation)
7. [Known limits](#7-known-limits)
8. [Deliverables](#8-deliverables)
9. [Disclaimer](#9-disclaimer)
10. [Source List](#10-source-list)
11. [Sample Questions](#10-sample-questions)

---

## 1. What it is

Ask it "What is the lock-in period for HDFC ELSS Tax Saver Fund?" and it answers
in at most three sentences, cites the URL the fact came from, and stamps the
source date. Ask it "Should I buy HDFC Small Cap Fund now?" and it declines,
because SEBI's position is that a facts interface may state what a scheme
discloses but must not recommend one. The design goal throughout is that every
guarantee is *enforced in code and testable offline*, not promised in a prompt:
the citation allow-list, the refusal paths and the ignorance path can all be
verified with no API key, which is why the test suite runs in 0.4 seconds.

## 2. Scope

Five HDFC AMC schemes, direct-growth variants, from two source types: the public
Groww scheme page and HDFC's official monthly factsheet.

| Scheme | Source page |
|---|---|
| HDFC Large Cap Fund - Direct - Growth | [groww.in](https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth) |
| HDFC Flexi Cap Fund - Direct - Growth | [groww.in](https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth) |
| HDFC ELSS Tax Saver Fund - Direct - Plan - Growth | [groww.in](https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth) |
| HDFC Small Cap Fund - Direct - Growth | [groww.in](https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth) |
| HDFC Balanced Advantage Fund - Direct - Growth | [groww.in](https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth) |

All five factsheets are one combined HDFC document, so 10 source files resolve to
**6 distinct permitted URLs**. The registry is `config/sources.csv`, and it is the
only place a URL appears in the codebase — which is what makes the citation
allow-list enforceable rather than aspirational.

**Not in scope:** NAV and live prices, performance figures, portfolio holdings by
scheme, account servicing, and anything outside these five schemes.

## 3. Architecture

```
                      ┌──────────────────────────────────────────┐
   user query  ──────▶│ S0  PII guard          → PII_BLOCKED     │
                      │ S1  Advice guard       → REFUSAL         │  no LLM call
                      │ S2  Retrieval + scheme/section filter     │  no LLM call
                      │ S3  Sufficiency (τ = 0.35) → INSUFFICIENT  │  no LLM call
                      ├──────────────────────────────────────────┤
                      │ S4  Generation  (OpenAI-compatible API)   │  ← the only
                      │ S5  Post-validation (deterministic)       │    network call
                      │ S6  Query log (sanitized text only)       │
                      └──────────────────────────────────────────┘
                                     │
                                     ▼
                      intent · ≤3 sentences · one allow-listed citation
                      · freshness line · top similarity for the UI
```

The order is the design. Four of the five ways this product must behave
correctly — ignore blank input, refuse PII, refuse advice, admit ignorance — are
decided **before** a token is spent. A guard that ran after generation would be a
guard that had already paid for the answer it was going to throw away.

`S5` is the layer that makes the output contract true. It is deterministic and
takes `(raw_string, hits)`, so the safety guarantees are testable with no key and
no network: it enforces at most three sentences, rejects performance claims and
advice language, strips any URL not present in the retrieved context, substitutes
a real citation, and attaches the freshness line.

| Layer | Module | Notes |
|---|---|---|
| Ingest | `src/ingest/` | fetch → clean → chunk → embed → index |
| Retrieval | `src/retrieval/search.py` | exact cosine over a numpy matrix, top-k, scheme + section pre-filter |
| Guards | `src/guards/` | PII (outermost), advice |
| Generation | `src/generation/` | prompt contract, provider client, post-validation |
| Orchestrator | `src/pipeline.py` | the stage list above |
| UI | `app.py` | Streamlit; contains no backend logic |

Full detail, including the threshold evidence, is in
[`architecture.md`](architecture.md); the chunking study, including the bugs
measurement caught that reading the code did not, is in
[`notes/chunking.md`](notes/chunking.md).

## 4. Setup

```bash
git clone <repo-url>
cd <repo>

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then put your key in LLM_API_KEY
# LLM_API_KEY=sk-...
# LLM_MODEL=gpt-4o-mini           # optional
# LLM_BASE_URL=                    # optional: Groq, Together, OpenRouter, ...

python -m src.ingest.run_all       # build the index (~7 s on CPU)
streamlit run app.py
```

`LLM_API_KEY` is optional in the sense that the app starts without it: the
sidebar, the guards and the retrieval all work, and factual questions show a
"Generation unavailable" banner instead of a stack trace. You cannot evaluate
answer quality without one.

The provider is reached over plain HTTP through `requests`, so any
OpenAI-compatible endpoint works. `LLM_MODEL` and `LLM_BASE_URL` select the
provider; `langchain-openai` is deliberately not a dependency.

> **Note on paths.** A project directory containing a colon (`27:9:26`) breaks
> `python -m venv` on macOS, because the venv shebang line is not escaped. Create
> the venv somewhere without a colon, or use an absolute interpreter path.

## 5. Rebuilding the index

```bash
python -m src.ingest.run_all --rebuild
```

`--rebuild` deletes the existing index first. Use it after any change to
`config/sources.csv` or the chunking code — otherwise the old vectors survive and
results look stale. The run prints a stage table and writes `data/manifest.json`
with the chunk count, embedding model, dimension, chunk strategy and a corpus
hash, so a reviewer can tell whether the index matches the source files.

**A fresh clone needs no network access for the corpus.** The cleaned documents in
`data/clean/` are committed, and the loader returns cached text when it is
present, so the command above rebuilds all 349 vectors offline in about 13
seconds. Only the *first* embedding call on a new machine reaches HuggingFace, to
download `all-MiniLM-L6-v2` (~90 MB) into the local model cache. The 11 MB source
PDF and the generated `index/` (0.8 MB) are gitignored and are not needed to rebuild
either.

## 6. Tests and evaluation

```bash
pip install -r eval/requirements.txt
pytest tests -q                    # 189 tests, keyless, no network, ~3.5 s
python eval/run_eval.py            # writes samples/sample_qa.md
```

The tests need no API key and no built index, and they cover the PII and advice
guards (including the negatives that matter — "SIP of 500", "expense ratio 1.5%"
and "Nifty 50 TRI" must not be flagged as personal data), the post-validator
against raw model strings, and the chunking invariants.

`run_eval.py` runs the 11 PRD §9 validation queries through the real
`answer_query`, prints a table of expected vs actual intent, scheme match,
citation grounding and latency, writes `samples/sample_qa.md`, and **exits
non-zero if any case fails** so it works as a CI gate.

Several of these tests failed against code that earlier phases had signed off,
which is the argument for having written them: ten index chunks contained no fact at
all, a single adjective defeated an advice pattern, a lower-case PAN escaped
detection, refusals carried a source date they had no right to, and the corpus
turned out to contain the Groww returns tables. See `notes/chunking.md` §9 and
the Phase 11 section of `implementation.md`.

## 7. Known limits

Stated plainly, because the limits are more informative than the feature list.

**The corpus is five public scheme pages, and nothing else.** Any question outside
them returns "not found". There is no HDFC AMC-wide knowledge, no competitor
data, no folio context, and no way to ask a question the pages do not already
answer in text.

**Figures are as of `fetched_at`, and they change.** Expense ratios, exit loads
and minimum amounts are amended by official notifications. The freshness stamp on
each answer reports the source date, but the bot cannot know about an amendment
it has not re-fetched. Re-run the ingest to refresh.

**"No performance data" is enforced by the validator, not by the corpus.** This is
the limit most likely to be misread from the outside. The Groww pages do carry a
"Returns and rankings" table and a "Return calculator", and both are chunked into
the index. A section boundary even runs that table into the `exit_load` section,
so *"What is the exit load on HDFC Equity Fund (Flexi Cap)?"* retrieves the
returns table as its top hit at 0.535 similarity. What prevents those numbers
reaching a user is the output guard refusing performance language — and
`tests/test_validate.py` feeds the real chunk through that guard so it cannot
regress quietly. Filtering those chunks at ingest would be the better fix and is
not done.

**Retrieval is similarity-based, and similarity is not a scope test.** τ = 0.35 was
tuned on an 11-question validation set, not on a benchmark. An unfiltered query
always has a nearest neighbour somewhere, so three general-knowledge probes that
share a token with the corpus ("Who is the CEO of HDFC Bank?" at 0.654) still
clear the threshold. Raising τ was rejected because it would reject the in-scope
ELSS lock-in question at exactly 0.350. This is the main thing I would fix next.

**Chunk boundaries straddle section headings.** A chunk that runs to the next
heading picks up one incidental mention of it, so `infer_section` can mislabel a
chunk. That is how the returns table ended up labelled `exit_load`. Section
labels are a filter hint, not a guarantee — which is why the citation allow-list
and the output validator, not the section filter, are what enforce correctness.

**The guards are pattern-based, not exhaustive.** "Can you build me a balanced
portfolio?" reached the model until Phase 11 caught it — one adjective defeated
the pattern. Expect more of these. A pattern-based PII guard will also miss
unusual formats; it is a speed bump and a demonstration, not a compliance
guarantee.

**No return or performance computation, no account servicing, and no PII storage.**
There is no arithmetic over figures, no login, no transaction. The query log
records sanitized text and an intent, and is a no-op unless `LOG_QUERIES=true`.

**Verified against a real model, and it found two defects.** With a key
configured, `run_eval.py` reports **11/11, 100% grounding, 1.3 s mean latency**
(Groq, `qwen/qwen3.8-27b`). That run was not a formality — it exposed two bugs
that the keyless tests could not, because both needed a model that writes like a
model rather than like a stub:

* **The validator let the model state the freshness date.** `FRESHNESS_RE`
  matched `Last updated from sources: <full date>`, so a model writing
  `2024-01-01` *satisfied* the pattern and suppressed the real date computed from
  the retrieved chunks — a fabricated date displayed under a line that reads like
  a system guarantee. The label is now matched instead of the date, and the line
  is always rewritten from the sources. The prompt no longer asks the model for a
  date at all.
* **The performance guard refused a correct answer.** Naming a benchmark means
  writing "NIFTY 50 Hybrid Composite Debt 50:50 Index (**Total Returns** Index)",
  and the bare `\breturn(s|ed|ing)?\b` pattern called that a performance claim.
  Index names are now masked before the sweep, while three `%`-figure proximity
  patterns run against the original text so the exemption cannot be used to
  smuggle in "Total Returns Index of 32%".

Neither was findable without a key. The lesson generalises: the guards and the
validator are unit-tested, but only a real model exercises the *interaction*
between what the model volunteers and what the validator assumes it will not.

Two eval expectations were also wrong rather than the code. Case 4 and case 10
named the ELSS fund `Direct - Growth` when the canonical registry name is
`Direct - Plan - Growth`, and case 6 expected an answer about downloading a
capital-gains statement on the stated premise that the factsheet has a "How to
download" section — `capital gain`, `How to download` and `tax statement` occur
**zero** times across all ten documents, and servicing is out of scope, so
`INSUFFICIENT_CONTEXT` is the correct outcome. Both are recorded in
`eval/queries.json`.

**One scheme naming is inconsistent.** The corpus holds the Flexi Cap fund under
its Groww name, `HDFC Equity Fund`, labelled
`HDFC Flexi Cap Fund - Direct - Growth`. A user asking about
"HDFC Equity Fund (Flexi Cap)" gets an answer about the direct-growth variant
only; the other plans are not in the corpus.

## 8. Deliverables

| Deliverable | Path |
|---|---|
| Working prototype | `app.py`, `src/` |
| Source list | `config/sources.csv` |
| Architecture and threshold evidence | `architecture.md` |
| Chunking study | `notes/chunking.md` |
| Implementation log, phase by phase | `implementation.md` |
| Sample Q&A (generated) | `samples/sample_qa.md` |
| Disclaimer (plain text) | `samples/disclaimer.txt` |
| Demo walkthrough | `DEMO_SCRIPT.md` |
| Test suite | `tests/` |
| Evaluation harness | `eval/run_eval.py`, `eval/queries.json` |
| Demo video | record from `DEMO_SCRIPT.md` — **not yet recorded** |

## 9. Disclaimer

Verbatim from PRD §8.3, and identical in `app.py` and `samples/disclaimer.txt`:

> **Facts-only. No investment advice.** Answers are generated from public HDFC AMC
> scheme pages and may be incomplete or out of date. Verify all details with the
> official factsheet and your mutual fund distributor before acting. Mutual fund
> investments are subject to market risks; read all scheme-related documents
> carefully.

## 10. Source List

Large Cap: https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth

Flexi Cap: https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth

ELSS: https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth

Small Cap: https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth

Balanced Advantage (Hybrid): https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth

## 11. Sample Questions

What is the expense ratio of HDFC Large Cap Fund – Direct Plan – Growth?
What is the minimum SIP amount for HDFC Large Cap Fund – Direct Plan – Growth?
Is there an exit load for HDFC Flexi Cap Fund – Direct Plan – Growth?
What is the benchmark for HDFC Flexi Cap Fund – Direct Plan – Growth?
What is the lock-in period for HDFC ELSS Tax Saver Fund – Direct Plan – Growth?
What is the riskometer level for HDFC ELSS Tax Saver Fund – Direct Plan – Growth?
What is the minimum SIP amount for HDFC Small Cap Fund – Direct Plan – Growth?
What is the exit load for HDFC Small Cap Fund – Direct Plan – Growth?
What is the benchmark for HDFC Balanced Advantage Fund – Direct Plan – Growth?
What is the riskometer level for HDFC Balanced Advantage Fund – Direct Plan – Growth?
How can I download an HDFC Mutual Fund account statement?
How can I download an HDFC Mutual Fund capital-gains statement?
