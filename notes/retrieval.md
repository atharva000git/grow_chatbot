# Retrieval window

Why `TOP_K` is 10, what it costs, and the two bugs that had to be fixed before
the number meant anything.

## The change

`TOP_K` went from 4 to 10. That is the number of retrieved chunks placed in the
prompt as context.

## The number was cosmetic until the search was fixed

Raising the constant alone changed almost nothing. Measured hit counts for real
questions, `k=10` requested:

| question | hits before the fix |
| --- | --- |
| expense ratio, HDFC Large Cap | 2 |
| lock-in period, HDFC ELSS | 1 |
| fund manager, HDFC Flexi Cap | 3 |
| minimum SIP, HDFC Small Cap | 5 |

Mean 3.8 of 10. The prompt saw 2 or 3 chunks, not 10.

The cause was in the filter relaxation loop. It tried the narrowest filter
first and `return`ed as soon as that attempt produced any hit at all:

```python
for index, (filter_scheme, filter_section) in enumerate(attempts):
    hits = run(filter_scheme, filter_section, top_k)
    if hits:
        return hits          # <- one matching chunk ends the search
```

A section filter that matched two chunks ended the search right there. The
remaining attempts, and `top_k` itself, never mattered.

The fix accumulates instead of short-circuiting: every attempt contributes new
chunks, deduplicated by `chunk_id`, until the window is full. Order is preserved
narrowest-first rather than re-sorted by similarity, because `hits[0]` is load
bearing - the sufficiency gate, the citation and the answer's anchor all read it.
After the fix the same questions return 8-10 chunks, mostly from one scheme.

Relaxing to the unfiltered `(None, None)` step is a last resort, not the default
filler. It is the only step that admits other schemes' chunks, and a fact from
the wrong fund is worse than a narrow answer. A single scheme holds 40+ chunks,
so the scheme-anchored steps fill the window on their own.

## What it costs

Same question, prompt measured against the live API:

| `k` | chunks | prompt chars | prompt tokens |
| --- | --- | --- | --- |
| 4 | 4 | 1,679 | 546 |
| 10 | 9 | 3,749 | 1,156 |

**2.12x the prompt tokens.** Per-call latency barely moved (0.33-0.55 s), but
the eval makes six model calls in a row and crosses Groq's per-minute token
budget partway through. Observed as HTTP 429 on cases 5 and 7.

That is now retried with backoff (`RATE_LIMIT_ATTEMPTS`, in
`src/generation/llm.py`) instead of failing the answer - the question was fine,
the quota was not. A 429 is also reported as `ERROR:PROVIDER` rather than
`ERROR:NO_KEY`, which is what it used to say, sending the hunt for a key that
was present the whole time.

If a different provider has a tighter budget, `TOP_K` is the dial.

## Why widening could not make it answer out-of-scope questions

`is_sufficient` is an any-hit gate on the **best** similarity, and hits come
back ranked descending:

```python
def is_sufficient(hits):
    return best_similarity(hits) >= settings.SIMILARITY_THRESHOLD
```

Candidates 5-10 are by construction lower-scoring than candidate 1, so
appending them cannot lift the best score past the threshold. Widening the
window cannot turn a refused question into an answered one. Verified across
`k` in `{1, 4, 10, 25}` in `tests/test_retrieval.py`, because this is the
property the whole change rests on.

Out-of-scope probes, re-measured after the switch to exact numpy search:

| query | best sim | k=4 | k=10 |
| --- | --- | --- | --- |
| Who is the prime minister of India? | 0.368 | refused | refused |
| What is the weather in Mumbai today? | 0.300 | refused | refused |
| What is the population of Japan? | 0.309 | refused | refused |
| Explain the difference between a mutual fund and a stock | 0.440 | **answers** | **answers** |
| Who is the CEO of HDFC Bank? | 0.654 | **answers** | **answers** |

The last two are a real gap and are **not** caused by the wider window - they
score above 0.35 at `k=4` too, because a "what is" question about a fund
structurally resembles the fund-overview chunks. A threshold cannot fix that;
it needs intent classification. Left as-is and recorded here rather than
quietly widened.

## Exact search exposed a threshold weakness that Chroma was hiding

The prime-minister row changed from **0.009 to 0.368** when Chroma was replaced
with an exact scan, which is worth being precise about, because 0.009 was the
comforting number.

"What is the prime minister of India?" scores 0.368 against a chunk containing
**zero** shared content words — a table of debt-maturity dates. There is no
semantic relationship there. 0.009 was not the embedding model being right about
relevance; it was Chroma's approximate HNSW happening to return a poor candidate
set for that query, so a weak chunk was compared and a strong one never was. The
exact scan compares everything, finds the genuinely nearest neighbour, and that
neighbour is a numeric table that scores 0.368 purely on embedding geometry.

So the retrieval gate now *passes* this query: `is_sufficient()` returns true at
0.368, and the model is called. In all 11 live eval cases the model still emits
`INSUFFICIENT_CONTEXT` on its own, and `src/generation/validate.py:190` converts
that into a clean refusal with `retrieval_hits=0`. The user-visible behaviour is
correct.

The correctness is now the *model's* doing rather than the retriever's, though,
and that is a weaker guarantee than it looks. A threshold on cosine similarity
cannot separate "a question about a fund" from "a question that is not about a
fund" once both land near 0.35-0.65, because the corpus is entirely fund-shaped
and a 384-dim MiniLM space has no reason to push unrelated text to zero. A
cheap, deterministic backstop is available and is *not* implemented here: reject
a hit that shares no content word with the query, which would restore a
retrieval-level refusal for this case. It is left as a decision rather than
slipped in, because the false-negative cost is real — "How much does it cost to
invest?" shares no literal token with a TER table but is a perfectly answerable
question here. Intent classification is the principled fix.

## The bug the wider window was hiding

Section detection matched tokens as plain substrings:

```python
("fees", ("expense ratio", "expense", "ter")),
```

`ter` is the common abbreviation for total expense ratio. It is also a substring
of `riskome`**`ter`**. Since `fees` is listed before `riskometer`, **every
riskometer question was filtered to the fees section** - the one section
guaranteed not to contain the riskometer.

It stayed invisible because the old early-return then relaxed and found the
right chunk anyway, so the answer came out correct for the wrong reason. The
new fill logic would have hidden it further. Fixed with word-boundary matching
and an optional plural, so `SIPs` and `expenses` still match:

```python
re.compile(rf"\b{re.escape(token)}s?\b", re.IGNORECASE)
```

Regression test: `test_ter_does_not_match_inside_riskometer`.

## A second bug the new fill logic introduced, caught by its own tests

Dedupe by `chunk_id` is required, because a chunk reachable through both
`(scheme, section)` and `(scheme, None)` would otherwise be counted twice. The
first implementation of it fetched `top_k - len(collected)` candidates from
each relaxation step and then dropped the duplicates — which cannot backfill.
With two filters sharing a chunk, the narrow step returned 2, the next step
returned 8 of which 2 were duplicates, and the window finished at **8 of 10**
with the unfiltered step unable to help.

The fix is to let each step *skip* ids an earlier step already contributed and
keep scanning deeper, so `limit` counts only new material:

```python
picked = [
    index
    for index in candidates[order]
    if str(rows[int(index)]["chunk_id"]) not in exclude
][:limit]
```

`test_window_fills_beyond_a_narrow_filter` fails at 8 without this.

## Net effect

| | before | after |
| --- | --- | --- |
| chunks reaching the prompt | 3.8 mean | 8-10 |
| prompt tokens | 546 | 1,156 |
| eval | 11/11 | 11/11 |
| grounding | 6/6 | 6/6 |
| riskometer questions | filtered to `fees` | filtered to `riskometer` |
| tests | 163 | 189 |

## Memory, and why this file exists next to the vector store

`TOP_K=10` puts 10 chunks in a prompt, which costs tokens but no memory. The
deployment blocker was elsewhere: the service peaked at 615-644 MB against
Render's 512 MiB free-tier cap. Replacing Chroma with the numpy index cut that
to a sampled 550 MB / `ru_maxrss` 567 MB — a real ~65 MB, and it made search
exact, but **still over the limit**. The remaining ~423 MB is `import torch`
(155 MB) plus model construction (268 MB). Full measurements, including three
approaches that did not work, are in `architecture.md` section 15. The next step
is removing torch from the query-time path, not further tuning.

