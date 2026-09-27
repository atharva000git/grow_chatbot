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
2. **The factsheet's portfolio/sector table extracts with dropped leading
   capitals and doubled spaces** — 95 garbled lines, e.g.
   `ood  roducts` for `Food Products`, `erospace    efense` for
   `Aerospace Defense`. This is a **Phase 2 PDF-extraction defect, not a
   chunking bug**: the splitter reproduces its input faithfully
   (`RecursiveCharacterTextSplitter` was bisected with `keep_separator` both
   ways and preserves every character). It is confined to the industry /
   sector classification columns of the five factsheets — **0 garbled lines in
   all five Groww documents** — and holding names, expense ratios, exit loads,
   benchmarks and the lock-in are all clean. Follow-up: see §6.

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

`CHUNK_STRATEGY` stays `"recursive"`. The spec framed semantic as an escalation
available only if the data demanded it, and the data did not. Recursive matched
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
5. **Open (Phase 2 follow-up, not blocking):** 95 garbled lines in the
   factsheets' sector/industry columns (§1.2). Worth a targeted repair before
   the demo, since "what are the top sectors?" would surface them. Holding
   names and every citable metric are unaffected.
