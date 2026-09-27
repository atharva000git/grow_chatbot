"""Stage 3 - Chunking. Satisfies PRD FR-2.

The strategy is a config value, not a code fork: `settings.CHUNK_STRATEGY`
selects between the recursive splitter and the semantic splitter. Both were run
over the real corpus; the comparison is recorded in notes/chunking.md.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from src.models import REPO_ROOT, Chunk, SourceDoc

log = logging.getLogger(__name__)

CHUNKS_PATH = REPO_ROOT / "data" / "chunks" / "chunks.jsonl"
SEPARATORS = ["\n## ", "\n\n", "\n", ". ", " "]
MERGE_CEILING = 1.3
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n{2,}")

SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("lock_in", re.compile(r"lock[-\s]?in", re.I)),
    ("exit_load", re.compile(r"exit[-\s]?load", re.I)),
    # Manager biographies are entity-heavy and mention "benchmark" while
    # describing performance responsibility, which pulled them into the
    # benchmark bucket and let a bio outrank the real index at 0.620 vs 0.424.
    # They are frequent enough to need their own label, not just a higher
    # priority, so a section filter can exclude them outright.
    ("management", re.compile(r"view details|\beducation\b|\bexperience\b|also manages|\bmanages these schemes\b", re.I)),
    ("riskometer", re.compile(r"riskometer|risk\s+category|rated\s+\w+\s+risk|risk\s+profile", re.I)),
    ("fees", re.compile(r"expense\s+ratio|statutory\s+levies\s+on\s+expenses|additional\s+expense", re.I)),
    ("benchmark", re.compile(r"benchmark", re.I)),
    ("aum", re.compile(r"fund\s+size|assets?\s+under\s+management|\bAUM\b", re.I)),
    ("statements", re.compile(r"capital\s+gains?\s+statement|tax\s+report|account\s+statement|download\s+statement", re.I)),
    ("sip", re.compile(r"\bSIP\b|systematic\s+investment|monthly\s+installment|\bmin\.?\s+for\s+sip", re.I)),
    ("nav", re.compile(r"\bNAV\b|net\s+asset\s+value", re.I)),
    ("faq", re.compile(r"\bFAQ\b|frequently\s+asked", re.I)),
)

MID_LINE_START = re.compile(r"^[a-z,.;:%)\]]")
PAGE_MARKER = re.compile(r"\.{2,}\s*Contd on next page", re.I)

# A factsheet metric is an all-caps header line followed by its hard-wrapped
# value: "LOCK-IN PERIOD" / "3 years from the date of allotment of the" /
# "respective Units". FACT_MIN_CHARS is the floor for those blocks, and it is
# deliberately far below MIN_CHUNK_CHARS: "LOCK-IN PERIOD 3 years from the date
# of allotment of the respective Units" is 73 characters, so the prose floor
# would silently delete a real fact. MIN_CHUNK_CHARS exists to strip nav crumbs
# out of prose; a block introduced by a recognised metric header is not a crumb.
FACT_MIN_CHARS = 25
FACT_HEADER_MAX = 60
# Groww renders each summary metric as a short label line over a short value
# line, so these two ceilings are what distinguish a metric from prose.
FIELD_LABEL_MAX = 45
FIELD_VALUE_MAX = 30


class ChunkingError(RuntimeError):
    pass


def infer_section(text: str) -> str:
    """Label a chunk by the fact it *mostly* carries.

    Scored by how many times each fact is mentioned rather than by which
    pattern comes first in SECTION_PATTERNS. First-match-by-priority was wrong
    for the long descriptive chunks: a 583-character "About this fund" passage
    that mentions exit load once was labelled `exit_load` outright, so the
    minimum-SIP, AUM and benchmark chunks all inherited a label describing a
    fact they did not contain. Counting makes one incidental mention lose to a
    chunk that is genuinely about that fact, and ties still fall back to the
    documented priority order.
    """
    best_name, best_count = "overview", 0
    for priority, (name, pattern) in enumerate(SECTION_PATTERNS):
        count = len(pattern.findall(text))
        if count > best_count:
            best_name, best_count = name, count
    return best_name


def _mid_line_starts(chunks: list[str]) -> int:
    return sum(1 for c in chunks if MID_LINE_START.match(c.lstrip()))


def _merge_mid_line_starts(chunks: list[str]) -> tuple[list[str], int, int]:
    """Fold a chunk that begins mid-sentence into its predecessor so a figure is
    never separated from the label that introduces it."""
    merged: list[str] = []
    merges = 0
    ceiling = int(settings.CHUNK_SIZE * MERGE_CEILING)
    for chunk in chunks:
        if merged and MID_LINE_START.match(chunk.lstrip()):
            combined = f"{merged[-1]}\n{chunk}"
            if len(combined) <= ceiling:
                merged[-1] = combined
                merges += 1
                continue
        merged.append(chunk)
    return merged, merges, _mid_line_starts(merged)


def chunk_recursive(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=SEPARATORS,
        keep_separator=True,
    )
    return splitter.split_text(text)


def chunk_semantic(text: str, breakpoint_threshold: float = 85.0) -> list[str]:
    """Group sentences by embedding-distance breakpoints.

    Implemented directly on sentence-transformers rather than
    langchain_experimental.SemanticChunker: no 0.2.x release of that package
    exists to match langchain==0.2.2, and 0.3.x would force a langchain and
    numpy upgrade that breaks the rest of the pins.
    """
    from src.retrieval.embedder import embed_texts

    sentences = [s for s in SENTENCE_BREAK.split(text) if s and s.strip()]
    if len(sentences) < 2:
        return [text] if text.strip() else []

    vectors = embed_texts(sentences)
    distances = [1.0 - float(sum(a * b for a, b in zip(vectors[i], vectors[i + 1])))
                 for i in range(len(vectors) - 1)]
    if not distances:
        return [text]

    ordered = sorted(distances)
    index = min(int(len(ordered) * breakpoint_threshold / 100.0), len(ordered) - 1)
    threshold = ordered[index]

    groups: list[list[str]] = [[sentences[0]]]
    for offset, sentence in enumerate(sentences[1:]):
        if distances[offset] > threshold:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)

    ceiling = int(settings.CHUNK_SIZE * MERGE_CEILING)
    chunks: list[str] = []
    for group in groups:
        joined = " ".join(group)
        if len(joined) <= ceiling:
            chunks.append(joined)
        else:
            chunks.extend(chunk_recursive(joined))
    return chunks


def is_fact_header(line: str) -> bool:
    """True for a short, mostly-uppercase metric header such as `LOCK-IN PERIOD`
    or `#BENCHMARK INDEX`."""
    text = line.strip()
    if not text or len(text) > FACT_HEADER_MAX:
        return False
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3:
        return False
    return sum(1 for c in letters if c.isupper()) / len(letters) >= 0.85


PAGE_HEADER = re.compile(
    r"^## PDF page \d+ of \d+\n(?:For Product label[^\n]*\n)?"
    r"\d+\s*\|\s*[A-Za-z]+ \d{4}\n[^\n]*\n?",
    re.M,
)


def strip_page_headers(text: str) -> str:
    """Drop the repeated factsheet page banner.

    `## PDF page 64 of 144` / `64 | August 2026` / `HDFC ELSS - Tax Saver Fund`
    is 293 characters of pure boilerplate that names the fund on every single
    page. It is not merely useless: because the query also names the fund, that
    banner outranked the 73-character lock-in fact for "What is the lock-in
    period for HDFC ELSS Tax Saver Fund?" (0.694 vs 0.482) and pushed the
    answer to rank 6. The banner appears on only 2 of the ELSS factsheet's
    pages, so this is a narrow removal, not a filter on legitimate content.
    """
    return PAGE_HEADER.sub("", text)


# Groww summary metrics, matched as a label prefix. An explicit allow-list rather
# than a shape heuristic: "any short line over a short line containing a digit"
# also matches the return-calculator and peer-comparison tables, which are grids
# of repeated figures and must stay as grid chunks.
GROWW_METRICS: tuple[str, ...] = (
    "NAV:",
    "Min. for SIP",
    "Min. for 1st investment",
    "Min. for 2nd investment",
    "Min. for Additional",
    "Fund size (AUM)",
    "Expense ratio",
    "Exit load",
    "Stamp duty on investment",
    "Lock-in period",
)


def metric_prefix(line: str) -> str | None:
    """Return the Groww metric this line introduces, if any."""
    text = line.strip().lstrip("#").strip()
    for metric in GROWW_METRICS:
        if text == metric or text.startswith(f"{metric} ") or text.startswith(f"{metric}:"):
            return metric
    return None


def is_field_value(line: str) -> bool:
    """True for a short value line such as `₹100` or `0.78%`."""
    text = line.strip()
    if not text or len(text) > FIELD_VALUE_MAX:
        return False
    return any(ch.isdigit() or ch in "₹%" for ch in text)


def segment_fields(text: str) -> list[tuple[str, str]]:
    """Split Groww pages into (metric, value) blocks and prose blocks.

    The page opens with a run of bare metric pairs:

        NAV: 25 Sep '26
        ₹159.82
        Min. for SIP
        ₹100
        Fund size (AUM)
        ₹41,890.86 Cr
        Expense ratio
        0.78%

    Left bundled into one 438-character chunk, four citable facts shared a single
    embedding, so a query for one of them matched all four weakly. A metric also
    absorbs following prose, so `## Exit load` plus "Exit load of 1% if redeemed
    within 1 year" stays a single atomic fact rather than being split apart.
    """
    lines = text.split("\n")
    blocks: list[tuple[str, str]] = []
    prose: list[str] = []
    index = 0

    def flush() -> None:
        joined = "\n".join(prose).strip()
        if joined:
            blocks.append(("", joined))
        prose.clear()

    while index < len(lines):
        line = lines[index]
        metric = metric_prefix(line)
        if metric is None:
            prose.append(line)
            index += 1
            continue
        flush()
        head = line.strip()
        index += 1
        body: list[str] = []
        # A metric owns its own value line plus at most two wrapped value lines.
        while index < len(lines) and len(body) < 3:
            nxt = lines[index]
            if metric_prefix(nxt) is not None or nxt.strip().startswith("##"):
                break
            if is_field_value(nxt) or (body and len(body[0]) <= FIELD_VALUE_MAX and is_field_value(body[0])):
                body.append(nxt)
                index += 1
                continue
            break
        blocks.append((head, "\n".join(body).strip()))
    flush()
    return blocks


ABOUT_HEADING = re.compile(r"^## About\b.*$", re.M)
NEXT_HEADING = re.compile(r"^## ", re.M)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def about_sentences(text: str) -> list[tuple[int, int, str]]:
    """Sentence spans inside each "## About" passage, as (start, end, sentence).

    The passage ends at the next `## ` heading, not at end of file. Bounding it
    at EOF made the last About section swallow the entire remainder of the page,
    and the sentence splitter then dropped every fragment below FACT_MIN_CHARS,
    silently deleting 30% of the corpus - the return calculator, the peer table
    and the manager biography.
    """
    out: list[tuple[int, int, str]] = []
    for match in ABOUT_HEADING.finditer(text):
        start = match.end()
        nxt = NEXT_HEADING.search(text, start)
        end = nxt.start() if nxt else len(text)
        body = text[start:end].strip()
        offset = start + (len(text[start:end]) - len(text[start:end].lstrip()))
        for sentence in SENTENCE_SPLIT.split(body):
            cleaned = " ".join(sentence.split())
            if len(cleaned) >= FACT_MIN_CHARS:
                found = text.find(cleaned[:60], offset)
                if found != -1:
                    out.append((found, found + len(cleaned), cleaned))
                    offset = found + len(cleaned)
    return out


def split_about(text: str) -> list[str]:
    """Break the "About this fund" passage into one sentence per chunk.

    This passage repeats the fund's full name in nearly every sentence:

        HDFC Small Cap Fund Direct Growth is a Equity Mutual Fund Scheme
        launched by HDFC Mutual Fund. This scheme was made available to
        investors on 10 Dec 1999. Dhruv Muchhal is the Current Fund Manager of
        HDFC Small Cap Fund Direct Growth fund.

    Bundled, the name appears five times in 583 characters, and cosine similarity
    scores that repetition against any query naming the fund. It outranked the
    real minimum-SIP answer (0.736 vs 0.563) and the real expense-ratio answer
    (0.765 vs 0.518). One sentence per chunk caps the name at one or two
    mentions, so the sentence that actually carries the fact competes on the
    fact rather than on the fund's name. The heading is dropped because it is
    itself a fifth repetition; scheme and category stay in the chunk metadata.
    """
    return [sentence for _, _, sentence in about_sentences(text)]


def _without_about(text: str) -> str:
    """`text` with every About sentence removed, so the rest is chunked once."""
    spans = about_sentences(text)
    if not spans:
        return text
    out: list[str] = []
    cursor = 0
    for start, end, _ in spans:
        out.append(text[cursor:start])
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def segment_facts(text: str) -> list[tuple[str, str]]:
    """Split a factsheet into (header, body) blocks at its metric headers.

    Consecutive all-caps lines are absorbed into a single header, because the
    holdings table wraps company names that way:

        CORONA REMEDIES
        LIMITED
        Pharmaceuticals
        & Biotechnology 0.42 0.00

    Treated line-by-line, "CORONA REMEDIES" looks like a metric header with an
    empty body, falls under FACT_MIN_CHARS and is dropped, leaving a useless
    "LIMITED Pharmaceuticals & Biotechnology 0.42 0.00" chunk with the holding's
    name gone. Absorbing keeps the name, sector and weight together.

    Header "" means a run of prose before the first metric, which is chunked as
    ordinary text rather than treated as a fact.
    """
    lines = text.split("\n")
    blocks: list[tuple[str, str]] = []
    header: str | None = None
    body: list[str] = []
    for line in lines:
        if is_fact_header(line):
            if header is not None and any(part.strip() for part in body):
                # previous header already has its value: close that block
                blocks.append((header, "\n".join(body)))
                header = line.strip()
            elif header is not None:
                # consecutive all-caps line: a wrapped name, absorb it
                header = f"{header}\n{line.strip()}"
            else:
                header = line.strip()
            body = []
        else:
            body.append(line)
    if header is not None or any(part.strip() for part in body):
        blocks.append((header or "", "\n".join(body)))
    return blocks


def chunk_facts(text: str, strategy: str) -> list[str]:
    """One chunk per metric, so a fact is not averaged together with its
    neighbours in the embedding.

    Bundling was the cause of the ELSS lock-in being unretrievable: the fact
    shared a 744-character chunk with the expense ratio, two benchmarks, the
    exit load and the start of the holdings table, so a lock-in query scored
    0.128-0.253 against it and never cleared SIMILARITY_THRESHOLD.
    """
    out: list[str] = []
    for header, body in segment_facts(text):
        block = "\n".join(part for part in (header, body) if part.strip()).strip()
        if not block:
            continue
        if not header:
            out.extend(c for c in chunk_recursive(block) if len(c.strip()) >= settings.MIN_CHUNK_CHARS)
        elif len(block) <= settings.CHUNK_SIZE:
            if len(block) >= FACT_MIN_CHARS:
                out.append(block)
        else:
            out.extend(c for c in chunk_recursive(block) if len(c.strip()) >= settings.MIN_CHUNK_CHARS)
    return out


def _normalize(chunk: str) -> str:
    """Collapse the factsheet's hard line breaks into single spaces.

    The August 2026 factsheet is laid out in narrow columns, so pypdf emits one
    fact across several physical lines:

        LOCK-IN PERIOD\\n3 years from the date of allotment of the\\nrespective Units

    Left alone that label and value are never a contiguous span, which is what
    the encoder needs to embed and what the answer generator needs to quote. Run
    after _merge_mid_line_starts, which needs the real line structure to detect
    a mid-sentence start.
    """
    return " ".join(PAGE_MARKER.sub(" ", chunk).split())


def chunk_html(text: str, strategy: str) -> list[str]:
    """Groww pages: atomic metric pairs, sentence-split About, prose as-is."""
    out: list[str] = []
    about = split_about(text)
    body = _without_about(text)

    for header, block in segment_fields(body):
        joined = "\n".join(part for part in (header, block) if part.strip()).strip()
        if not joined:
            continue
        # A recognised metric is exempt from FACT_MIN_CHARS. "Min. for SIP ₹100"
        # is 17 characters and "NAV: 25 Sep '26 ₹1,189.08" is 22, so the floor
        # deleted the atomic chunks it was meant to protect - the same mistake
        # that buried the 73-character ELSS lock-in. The floor exists to strip
        # nav crumbs, and a metric introduced by an allow-listed label is not a
        # crumb; prose blocks still go through it.
        floor = 1 if metric_prefix(header) else FACT_MIN_CHARS
        if header and len(joined) <= settings.CHUNK_SIZE and len(joined) >= floor:
            out.append(joined)
        elif len(joined) <= settings.CHUNK_SIZE:
            out.extend(c for c in chunk_recursive(joined) if len(c.strip()) >= settings.MIN_CHUNK_CHARS)
        elif strategy == "semantic":
            out.extend(chunk_semantic(joined))
        else:
            out.extend(chunk_recursive(joined))
    return out + about


def chunk_document(doc: SourceDoc, strategy: str | None = None) -> list[Chunk]:
    text = Path(doc.clean_path).read_text(encoding="utf-8")
    chosen = (strategy or settings.CHUNK_STRATEGY).lower()
    if chosen not in ("semantic", "recursive"):
        raise ChunkingError(f"unknown CHUNK_STRATEGY {chosen!r}; use 'recursive' or 'semantic'")

    if doc.source_type == "pdf":
        # Factsheet: strip the repeated page banner, then split metrics apart at
        # their all-caps headers before the strategy runs.
        raw = chunk_facts(strip_page_headers(text), chosen)
    else:
        # Groww: same intent, different markers - label/value pairs instead of
        # all-caps headers. Its all-caps lines are bond tickers inside holdings
        # tables, so the all-caps rule would fragment prose here.
        raw = chunk_html(text, chosen)
    merged, merges, remaining = _merge_mid_line_starts(raw)
    log.info(
        "%s (%s): %d chunks; %d mid-line start(s) merged, %d kept because merging "
        "would exceed %d chars",
        doc.source_id, chosen, len(merged), merges, remaining,
        int(settings.CHUNK_SIZE * MERGE_CEILING),
    )

    return [
        Chunk(
            chunk_id=f"{doc.source_id}::c{index}",
            text=_normalize(chunk),
            source_url=doc.source_url,
            scheme=doc.scheme,
            category=doc.category,
            section=infer_section(chunk),
            chunk_index=index,
            fetched_at=doc.fetched_at,
        )
        for index, chunk in enumerate(merged)
    ]


def chunk_all(docs: list[SourceDoc], strategy: str | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, strategy=strategy))
    return chunks


def save_chunks(chunks: list[Chunk], path: Path = CHUNKS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")
    return path


def load_chunks(path: Path = CHUNKS_PATH) -> list[Chunk]:
    with path.open(encoding="utf-8") as handle:
        return [Chunk(**json.loads(line)) for line in handle if line.strip()]


def _stats(chunks: list[Chunk]) -> dict:
    lengths = [len(c.text) for c in chunks]
    return {
        "chunks": len(chunks),
        "min": min(lengths) if lengths else 0,
        "mean": round(sum(lengths) / len(lengths)) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "sections": dict(Counter(c.section for c in chunks)),
    }


def study(docs: list[SourceDoc]) -> None:
    for strategy in ("recursive", "semantic"):
        started = time.perf_counter()
        chunks = chunk_all(docs, strategy=strategy)
        elapsed = time.perf_counter() - started
        s = _stats(chunks)
        print(f"\n=== {strategy} ===  {elapsed:.1f}s")
        print(f"  chunks={s['chunks']}  chars min/mean/max={s['min']}/{s['mean']}/{s['max']}")
        print(f"  per doc: {dict(Counter(c.chunk_id.split('::')[0] for c in chunks))}")
        print(f"  sections: {s['sections']}")
        empty = [k for k, v in s["sections"].items() if v == 0]
        print(f"  empty sections: {empty or 'none'}")


if __name__ == "__main__":
    from src.ingest.load import load_all

    study(load_all())
