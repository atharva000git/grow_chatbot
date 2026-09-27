# Demo script — HDFC Mutual Fund Facts Assistant

A 3-minute walkthrough. Follow it in order; it is written so someone who has not
seen the code can record it. Every step gives the exact command to run or the
exact text to type, and what the screen should show.

**Before you start**

```bash
python -m src.ingest.run_all --rebuild     # ~7 s, prints a stage table ending "chunks 349"
streamlit run app.py                      # opens http://localhost:8501
```

Put the terminal and the browser side by side. The sidebar should show
**349 chunks** and a threshold of **0.35**; if it does not, stop and re-run the
ingest command, because every later step depends on it.

---

## 1. Corpus — what the bot is allowed to know (0:00–0:15)

Open `config/sources.csv` in the editor.

> "Five HDFC AMC schemes, two documents each — the Groww scheme page and the
> official monthly factsheet. Ten files, six permitted URLs. This file is the
> only place a URL appears in the codebase, which is what makes the citation
> allow-list enforceable."

Point at the `source_url` column. Do not read the rows aloud.

## 2. Loading — what the text looks like (0:15–0:30)

Open `data/clean/hdfc-large-cap.txt`, then `data/clean/hdfc-fs-elss.txt`.

> "Cleaned once at ingest: navigation and boilerplate stripped. The factsheet is
> a PDF, so its text layer is messy — 'EXIT LOAD$$', '2026@@'. We keep that noise
> rather than hand-editing, so the corpus is reproducible from the source."

## 3. Chunking — 349 chunks, one fact each (0:30–0:50)

```bash
~/.venvs/hdfc-rag/bin/python -c "
import json
rows=[json.loads(l) for l in open('data/chunks/chunks.jsonl')]
for r in rows[:2]+[x for x in rows if x['section']=='lock_in'][:1]:
    print(f\"{r['chunk_id']}  [{r['section']}]\"); print('   ',r['text'][:90]); print()
"
```

> "One chunk per fact, with the section it came from. The ELSS lock-in is its own
> 73-character chunk — bundled with the expense ratio and two benchmarks it
> scored 0.128 and was never retrievable. `notes/chunking.md` has the full study,
> including the two bugs measurement caught that reading the code did not."

## 4. Embedding (0:50–0:58)

```bash
~/.venvs/hdfc-rag/bin/python -c "
from config import settings
print(settings.EMBEDDING_MODEL, settings.EMBEDDING_DIM)"
```

> "all-MiniLM-L6-v2, 384 dimensions, run locally on CPU. No embedding API, so no
> per-query cost and no data leaving the machine."

## 5. Vector store (0:58–1:05)

```bash
~/.venvs/hdfc-rag/bin/python -c "
import json; m=json.load(open('data/manifest.json'))
print('chunks', m['n_chunks'], '| docs', m['n_docs'], '| hash', m['corpus_hash'])"
```

> "Chroma, 349 vectors, and the manifest carries a corpus hash so a reviewer can
> tell whether the index matches the source files."

## 6. Retrieval — the threshold doing visible work (1:05–1:20)

```bash
~/.venvs/hdfc-rag/bin/python -c "
import sys; sys.path.insert(0,'.')
from src.retrieval import search
for h in search('What is the lock-in period for HDFC ELSS Tax Saver Fund?',k=3):
    print(f'{h.similarity:.3f}  [{h.chunk.section}] {h.chunk.text[:60]}')"
```

> "Cosine similarity, threshold 0.35. This one scores well above it. Now watch
> what happens to a question that has nothing to do with mutual funds."

```bash
~/.venvs/hdfc-rag/bin/python -c "
import sys; sys.path.insert(0,'.')
from src.retrieval import search
for q in ['Who is the prime minister of India?','What is the weather in Mumbai?']:
    print(f'{search(q,k=1)[0].similarity:.3f}  {q}')"
```

> "0.009. There is always a nearest neighbour somewhere, so similarity alone
> cannot police scope — which is exactly why the bot admits ignorance rather than
> answering from its least-bad match."

## 7. A grounded answer (1:20–1:40)

In the app, click **"What is the expense ratio of HDFC Large Cap Fund - Direct -
Growth?"**.

Expected: `✅ Answer`, at most three sentences, one citation, the
`Last updated from sources:` line, and a caption reading
`Answer · 4 sources · top match 0.5xx`.

> "The top-match number in that caption is the grounding evidence. Every number in
> that sentence came from the cited chunk, and the citation can only be a URL from
> `sources.csv` — the validator strips anything else the model invents."

## 8. Guard: advice is refused (1:40–1:55)

> "Should I buy HDFC Small Cap Fund now?"

Expected: `⚠️ Refused`, styled as a warning, citing a real scheme page. **No
sentence telling the user to buy or hold.** Say aloud that this is refused
*before* any model call, so it costs nothing and cannot leak.

## 9. Guard: PII is blocked (1:55–2:10)

> "My PAN is ABCDE1234F and my phone is 9876543210 — update my details."

Expected: red error styling, no answer, no citation. Say that the identifier
never reaches retrieval, the model, or the query log, and that nothing is stored.

## 10. Out of scope — and the honest limit (2:10–2:30)

> "What is the 5-year return of HDFC Large Cap Fund?"

Expected: `ℹ️ Not in sources`, no number.

> "The pages do contain a returns table, so the corpus is not free of performance
> data — what stops it being quoted is the output validator, not the chunker.
> `tests/test_validate.py` feeds the real chunk through it so that cannot
> regress quietly. And there is no NAV here, because a static page cannot carry a
> live figure: I would rather say 'not found' than quote a stale number."

## 11. Closing — deliverables (2:30–3:00)

> "Five things to hand over: the working prototype, the source list in
> `config/sources.csv`, the README with an honest known-limits section, the
> sample Q&A generated by `eval/run_eval.py`, and this script. The test suite is
> 130 tests, runs keyless in under half a second, and five of them caught real
> bugs in code every earlier phase had already signed off — including ten chunks
> in the index that contained no fact at all."

---

## If a step misbehaves

| Symptom | Cause | Do this |
|---|---|---|
| Sidebar says 0 chunks | index not built | `python -m src.ingest.run_all --rebuild` |
| "Generation unavailable" banner | no `LLM_API_KEY` | `cp .env.example .env`, add the key, restart |
| `EmbedderUnavailable` | model not in the HF cache | run once with network |
| A guard step answers anyway | guards not wired | `~/.venvs/hdfc-rag/bin/python -m pytest tests -q` |

**One thing to avoid claiming.** Do not say the bot "refuses out-of-scope
questions reliably". It refuses by threshold, and `architecture.md` §5.3 records
three general-knowledge probes that still clear 0.35 because they share a token
with the corpus. The honest claim is that it refuses the probes it was measured
on and that the guard layers are what make the failure visible.
