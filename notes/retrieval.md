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

Out-of-scope probes give identical verdicts at `k=4` and `k=10`:

| query | best sim | k=4 | k=10 |
| --- | --- | --- | --- |
| Who is the prime minister of India? | 0.009 | refused | refused |
| What is the weather in Mumbai today? | 0.300 | refused | refused |
| What is the population of Japan? | 0.309 | refused | refused |
| Explain the difference between a mutual fund and a stock | 0.440 | **answers** | **answers** |
| Who is the CEO of HDFC Bank? | 0.654 | **answers** | **answers** |

The last two are a real gap and are **not** caused by the wider window - they
score above 0.35 at `k=4` too, because a "what is" question about a fund
structurally resembles the fund-overview chunks. A threshold cannot fix that;
it needs intent classification. Left as-is and recorded here rather than
quietly widened.

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

## Net effect

| | before | after |
| --- | --- | --- |
| chunks reaching the prompt | 3.8 mean | 8-10 |
| prompt tokens | 546 | 1,156 |
| eval | 11/11 | 11/11 |
| grounding | 6/6 | 6/6 |
| riskometer questions | filtered to `fees` | filtered to `riskometer` |
| tests | 163 | 184 |
