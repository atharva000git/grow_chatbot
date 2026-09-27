# Phase 3 — Chunking study (`FR-2`)

## 1. Observed structure of the real corpus

10 documents across 6 URLs: 5 Groww scheme pages (HTML) and 5 page-sliced
copies of the official August 2026 HDFC factsheet (PDF).

| property | Groww (5 docs) | Factsheet (5 docs) |
| --- | --- | --- |
| cleaned chars | 8.4k – 33.4k | 6.9k – 20.9k |
| `## ` headings | 8 – 11 per doc | 0 |
| dominant structure | heading-delimited prose + labelled stat rows | narrow-column fact blocks, portfolio tables |
| citable metrics | expense ratio, benchmark, exit load, min amount, AUM, NAV | same + ELSS lock-in |

Two structural facts drove the implementation:

1. **The factsheet hard-wraps single facts across physical lines.** Its columns
   are narrow, so pypdf emits
   `LOCK-IN PERIOD\n3 years from the date of allotment of the\nrespective Units`.
   Left as-is, the label and its value are never a contiguous span, so neither
   the encoder nor the answer generator can treat them as one fact.
2. **One table per scheme is unrecoverable, not merely garbled.** On 4 of the
   factsheet's 144 pages, the *Industry Allocation of Equity Holding of Net
   Assets* summary table extracts as
   `Industry  llocation of E uity  oldin  of Net  ssets` /
   ` harmarmaceuticals    iotechnolo y` — leading characters dropped and spaces
   expanded. This is **not a library defect and is not fixable by extraction
   settings**: pypdf `plain` and `layout` and an independent PyMuPDF install all
   return the same corruption, so the PDF's text layer for that table carries no
   recoverable mapping. The region also contains **no digits at all**, so the
   allocation percentages are absent, not just misaligned — there is nothing to
   repair. The correct mitigation is to keep sector-allocation questions out of
   the demo. The *holdings* table on the same pages (company, industry, % to
   NAV) extracts cleanly, so "what are the top holdings" is unaffected.

## 2. Strategies compared

Both were run over the full 10-document corpus, not a sample.

| metric | `recursive` | `semantic` |
| --- | --- | --- |
| chunks | **201** | 217 |
| chars min / mean / max | 175 / 704 / **799** | 110 / 640 / 959 |
| chunks over `CHUNK_SIZE` (800) | **0** | 4 |
| chunks covering `LOCK-IN PERIOD` | 1 | 1 |
| chunks covering `EXIT LOAD` | **20** (10 docs) | 14 (10 docs) |
| chunks covering expense ratio | 19 (10 docs) | 18 (10 docs) |
| chunks covering benchmark | 25 (10 docs) | **26** (10 docs) |
| chunks covering min-amount language | **22** (10 docs) | 18 (**8** docs) |
| `lock_in` section labelled | 1 | 1 |
| wall clock | **~0.0 s** | ~8.4 s |

Per-document chunk counts, recursive: `balanced-advantage` (Groww) is the
largest at 52 — it alone is 26% of the corpus.

## 3. Parameters

`CHUNK_SIZE=800`, `CHUNK_OVERLAP=120`, `MIN_CHUNK_CHARS=80`, separators
`["\n## ", "\n\n", "\n", ". ", " "]`, merge ceiling `CHUNK_SIZE * 1.3 = 1040`,
semantic breakpoint threshold percentile 85. `CHUNK_SIZE` was **not** tuned.

## 4. Merge rule: 0 merges, 13 flagged

`_merge_mid_line_starts` fired **zero** times across all 10 documents. Every
candidate was rejected because two adjacent chunks (~700 chars each) always sum
past the 1040-char ceiling. The protection in step 4 of the spec is therefore
**inert at these parameters** — it can only trigger when the splitter happens to
emit a very small chunk.

13 chunks start mid-sentence, all in the factsheets, and all left in place
because the merge would breach the ceiling — the behaviour step 4 specifies
("otherwise keep and flag"):

```
hdfc-fs-flexi-cap::c2            hdfc-fs-elss::c2
hdfc-fs-flexi-cap::c10           hdfc-fs-elss::c10
hdfc-fs-flexi-cap::c11           hdfc-fs-elss::c11
hdfc-fs-flexi-cap::c13           hdfc-fs-small-cap::c10
hdfc-fs-balanced-advantage::c1   hdfc-fs-balanced-advantage::c3
hdfc-fs-balanced-advantage::c11  hdfc-fs-balanced-advantage::c12
hdfc-fs-balanced-advantage::c31
```

Each was inspected: **none separates a citable fund fact from its label.**
They are performance-table footnotes (`computed after accounting for the cash
flow by using XIRR method…`), fund-manager bio tables, and the garbled sector
column. Raising the ceiling to catch them would push chunks toward ~1400 chars,
trading a cosmetic gain for worse embedding resolution, so the ceiling was left
at the specified 1.3× and the cases are recorded here instead.

## 5. Decision: `recursive` (default, unchanged)

`CHUNK_STRATEGY` stays `"recursive"`.

> **Superseded after Phase 5.** This section originally compared the two
> strategies and found recursive ahead. That comparison was invalid: both
> strategies inherited the same granularity defect described in §7, so it ranked
> two flawed options against each other. The finding below is kept because the
> per-strategy numbers are still accurate; the decision it supports is not.
> Granularity is now fixed in the chunker, and the two strategies have not been
> re-compared under the fix — recursive remains the default on the grounds that
> it is deterministic, respects `CHUNK_SIZE` exactly, and needs no model at
> chunking time.

The spec framed semantic as an escalation available only if the data demanded it. Recursive matched
semantic on the fact that matters (both recover the ELSS lock-in as a contiguous
span in exactly one chunk), and then beat it on every other axis: it covered
**more** exit-load chunks (20 vs 14) and more min-amount chunks spanning **10**
source documents rather than 8, produced no chunk over the 800-char ceiling
against semantic's 4 (max 959), and ran in effectively zero time against 8.4 s.
The extra 16 chunks semantic produced bought no additional fact coverage. The
semantic path is retained behind the same config switch and remains reproducible
via `python -m src.ingest.chunk`.

## 6. Deviations from the Phase 3 spec, and open items

1. **`SemanticChunker` is not importable from `langchain_text_splitters==0.2.0`**
   as the spec assumed. It lives in `langchain_experimental`, which has **no
   0.2.x release** to match `langchain==0.2.2`; installing `0.3.1` forces
   `langchain` 0.3.30, `langchain-core` 0.3.86 and `numpy` 2.5.3, breaking
   every pin. `chunk_semantic` therefore implements the same algorithm
   (percentile-85 cosine-distance breakpoints over sentence embeddings from the
   shared MiniLM embedder) directly on `sentence-transformers`. The shared
   `src/retrieval/embedder.py` was pulled forward from Phase 4 for this.
2. **Section inference scans the whole chunk**, not its first 200 characters.
   The 200-char window labelled the ELSS chunk `fees` — the pattern
   `statutory levies on expenses` appears at index <200 while `LOCK-IN` sits at
   index 368 — so the corpus extension's whole purpose was invisible in the
   metadata. `overview` fell 156 → 127 and `lock_in` appeared once.
3. **Whitespace is normalised per chunk** (new step, §1.1) so a label and its
   value form one span. This is what makes acceptance criterion 4 pass.
4. **Chunk-count expectation revised** from 25–60 to 150–260: the original
   figure assumed 5 documents and the corpus is now 10.
5. **Closed, no action available:** the per-scheme industry-allocation table
   (§1.2) is unrecoverable at the PDF text-layer level on 4 of 144 pages. It
   affects no citable metric and no holding name. Recorded as a known corpus
   limit; the only remedy would be OCR, which is not worth the dependency and
   the risk of misreading financial figures. Keep sector-allocation questions
   out of the demo question set.

## 7. Addendum — fact-level granularity (added after Phase 5)

Smoke-testing the Phase 5 index with live queries exposed a defect that
supersedes the strategy comparison in §5.

**Symptom.** The ELSS lock-in — the fact the factsheet was added to the corpus
to capture — was unretrievable. Across six phrasings the chunk holding it
scored 0.128–0.253 and never both ranked in the top 4 and cleared
`SIMILARITY_THRESHOLD` (0.35), so retrieval would have answered
`INSUFFICIENT_CONTEXT` on the demo's headline question. Scheme filtering did not
help (rank 5–11 even within ELSS-only chunks), so this was granularity, not
ranking or threshold tuning.

**Cause.** §1 recorded that the factsheet packs expense ratio, benchmark,
lock-in and exit load into one dense block. At 744 characters that chunk also
absorbed the start of the holdings table, so its embedding averaged over four
unrelated facts plus a table of bond tickers. A lock-in query matched weakly
against that average. The single `section` label (`lock_in`) was correct while
the chunk it labelled was not — a labelling fix could not have helped.

**Fix.** `segment_facts` splits a factsheet at its all-caps metric headers, so
each metric becomes its own chunk:

```
LOCK-IN PERIOD 3 years from the date of allotment of the respective Units   (73 chars)
EXIT LOAD$$ Nil                                                             (15 chars)
#BENCHMARK INDEX NIFTY 500 Index (TRI)                                       (38 chars)
```

Consecutive all-caps lines are absorbed into one header, because the holdings
table wraps company names that way (`CORONA REMEDIES` / `LIMITED` /
`Pharmaceuticals` / `& Biotechnology 0.42 0.00`). Without that, each fragment
looked like its own header, fell under the minimum length and was dropped,
leaving a `LIMITED Pharmaceuticals…` chunk with the holding's name gone. This
was caught by a holdings-row integrity check and fixed before commit.

Fact blocks use `FACT_MIN_CHARS = 25` rather than `MIN_CHUNK_CHARS = 80`:
the lock-in fact is 73 characters, so the prose floor would have deleted a real
fact. `MIN_CHUNK_CHARS` exists to strip nav crumbs from prose; a block
introduced by a recognised metric header is not a crumb.

Applied to PDF sources only. Groww pages already use `## ` headings and their
all-caps lines are bond tickers inside holdings tables, so segmenting them would
fragment prose; their chunk counts are unchanged (13/16/13/16/52 before and
after).

**Result.** 201 → 273 chunks, 96.4% of the previous corpus text retained, zero
nav/boilerplate leakage, and all holdings rows intact.

| query | before | after |
| --- | --- | --- |
| "lock-in period for ELSS" | rank 17, 0.168 | **rank 1, 0.604** |
| "how long is the lock-in period" | rank 3, 0.179 | **rank 1, 0.685** |
| "ELSS lock in period" | rank 17, 0.162 | **rank 1, 0.555** |
| "lock in period of tax saver fund" | not retrieved | **rank 1, 0.537** |
| "3 year lock in" | rank 1, 0.236 | **rank 1, 0.596** |
| "can I redeem ELSS before 3 years" | not retrieved | rank 3, 0.349 |

Five of six phrasings now resolve at rank 1. The sixth retrieves at rank 3 and
0.349, 0.001 below the threshold — genuinely borderline, and worth re-checking
once Phase 6 adds reranking.

**Follow-on defect found while verifying the above:** Chroma's HNSW search is
approximate and its default `hnsw:search_ef` is 10, barely above `TOP_K`. At
`n_results=8` it silently dropped the lock-in chunk even though a full scan of
all 273 records put it at rank 1 with 0.555. Pinning `hnsw:search_ef` and
`hnsw:construction_ef` to 256 in the collection metadata fixed it. Without
this, Phase 6 would have looked like it was working while missing the best
match on roughly half of these queries.
