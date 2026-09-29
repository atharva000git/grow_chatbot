"""Stage 2 - Loading and cleaning of the 5 HDFC scheme pages. Satisfies PRD FR-1.

A source that yields too little text is a hard failure, not a warning: a
JS-rendered page returns an empty shell and would otherwise poison retrieval.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from pypdf import PdfReader

from config import settings
from src.models import REPO_ROOT, SourceDoc, build_source_docs, load_sources

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
FETCH_TIMEOUT = 30
BOILERPLATE_TAGS = ("script", "style", "nav", "header", "footer", "aside", "noscript", "form", "iframe")
BOILERPLATE_HINTS = (
    "footer", "nav", "menu", "breadcrumb", "cookie", "newsletter", "social",
    "sidebar", "promo", "advert", "banner", "hamburger", "modal", "popup", "sitemap",
)
MAIN_SELECTORS = ("main", "[role=main]", "#root", "body")
HEADING_TAGS = ("h1", "h2", "h3", "h4")
BLOCK_TAGS = ("p", "li", "tr", "td", "th", "div", "section", "article", "br")
RAW_DIR = REPO_ROOT / "data" / "raw"
CLEAN_DIR = REPO_ROOT / "data" / "clean"

REMEDY = (
    "Remedies: use the official HDFC factsheet PDF for this scheme, or export the "
    "page text by hand into data/clean/<source_id>.txt and re-run."
)


class IngestError(RuntimeError):
    pass


def _require_permitted(url: str) -> None:
    """Host allow-list: the fetcher refuses any URL absent from config/sources.csv."""
    permitted = {row["source_url"] for row in load_sources()}
    if url not in permitted:
        raise IngestError(f"URL not in config/sources.csv allow-list: {url}")


def fetch_html(url: str) -> str:
    _require_permitted(url)
    response = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response.text


def fetch_bytes(url: str) -> bytes:
    _require_permitted(url)
    response = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise IngestError(f"{url} did not return a PDF (content-type={response.headers.get('Content-Type')})")
    return response.content


def extract_pdf_text(payload: bytes, page_start: int, page_end: int) -> str:
    """Text of a page slice, 1-indexed and inclusive. The official factsheet is
    144 pages covering every HDFC scheme, so each scheme gets its own slice."""
    reader = PdfReader(io.BytesIO(payload))
    total = len(reader.pages)
    start = max(1, page_start or 1)
    end = min(total, page_end or total)
    if start > total:
        raise IngestError(f"page_start {start} exceeds PDF page count {total}")
    pages = []
    for number in range(start, end + 1):
        text = reader.pages[number - 1].extract_text() or ""
        pages.append(f"## PDF page {number} of {total}\n{text}")
    return "\n\n".join(pages)


def _emit(node: Tag, parts: list[str]) -> None:
    if node.name in HEADING_TAGS:
        heading = " ".join(node.get_text(" ", strip=True).split())
        if heading:
            parts.append(f"\n## {heading}\n")
        return
    if node.name in BLOCK_TAGS:
        parts.append("\n")
    for child in node.children:
        if isinstance(child, NavigableString):
            text = " ".join(str(child).split())
            if text:
                parts.append(text + " ")
        elif isinstance(child, Tag):
            _emit(child, parts)
    if node.name in BLOCK_TAGS:
        parts.append("\n")


def _decompose_boilerplate(soup: BeautifulSoup) -> None:
    """Nav/footer blocks on these pages are plain divs, not <footer> elements."""
    for tag in soup.find_all(True):
        if tag.decomposed:
            continue
        attrs = " ".join(
            filter(None, [*(tag.get("class") or []), tag.get("id"), tag.get("data-testid")])
        ).lower()
        if any(hint in attrs for hint in BOILERPLATE_HINTS):
            tag.decompose()


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(BOILERPLATE_TAGS):
        tag.decompose()
    _decompose_boilerplate(soup)
    root = next((soup.select_one(sel) for sel in MAIN_SELECTORS if soup.select_one(sel)), None)
    if root is None:
        return ""
    parts: list[str] = []
    _emit(root, parts)
    return "".join(parts)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    lines = [line.strip() for line in text.splitlines()]
    kept = [line for line in lines if len(line) >= 2]
    collapsed: list[str] = []
    for line in kept:
        if not line and collapsed and not collapsed[-1]:
            continue
        collapsed.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(collapsed)).strip()


def _paths(source_id: str, source_type: str) -> tuple[Path, Path]:
    suffix = "pdf" if source_type == "pdf" else "html"
    return RAW_DIR / f"{source_id}.{suffix}", CLEAN_DIR / f"{source_id}.txt"


def _pdf_cache_path(url: str) -> Path:
    return RAW_DIR / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]}.pdf"


def _load_pdf_payload(url: str) -> bytes:
    cached = _pdf_cache_path(url)
    if cached.exists() and cached.stat().st_size > 0:
        return cached.read_bytes()
    payload = fetch_bytes(url)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(payload)
    return payload


def load_source(doc: SourceDoc, fetched_at: str | None = None) -> SourceDoc:
    """Clean one source, preferring the committed cleaned copy over the network.

    `fetched_at` overrides the freshness stamp for a *cached* document. A
    cleaned file on disk was fetched on the date recorded by whoever built the
    index, not today, and the stamp is quoted to the user on every answer, so a
    rebuild that stamps today silently moves a 3-week-old figure forward. The
    override is ignored on the network path, where today genuinely is the
    fetch date.
    """
    raw_path, clean_path = _paths(doc.source_id, doc.source_type)
    today = datetime.now(timezone.utc).date().isoformat()

    if clean_path.exists() and clean_path.stat().st_size > 0:
        cleaned = clean_path.read_text(encoding="utf-8")
        if doc.source_type == "pdf":
            raw_path = _pdf_cache_path(doc.source_url)
        if not raw_path.exists():
            raw_path = Path("")
        return dataclasses.replace(
            doc,
            fetched_at=fetched_at or today,
            content_hash=_hash(cleaned),
            raw_path=str(raw_path),
            clean_path=str(clean_path),
        )

    if doc.source_type == "pdf":
        payload = _load_pdf_payload(doc.source_url)
        cleaned = clean_text(extract_pdf_text(payload, doc.page_start, doc.page_end))
        raw_path = _pdf_cache_path(doc.source_url)
    else:
        html = fetch_html(doc.source_url)
        cleaned = clean_text(extract_main_text(html))
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(html, encoding="utf-8")

    if len(cleaned) < settings.MIN_DOC_CHARS:
        raise IngestError(
            f"{doc.source_id}: extracted {len(cleaned)} chars from {doc.source_url}, "
            f"below MIN_DOC_CHARS={settings.MIN_DOC_CHARS}. {REMEDY}"
        )

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.write_text(cleaned, encoding="utf-8")

    return dataclasses.replace(
        doc,
        fetched_at=today,
        content_hash=_hash(cleaned),
        raw_path=str(raw_path),
        clean_path=str(clean_path),
    )


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_fetched_at(path: Path) -> dict[str, str]:
    """Read a source_id -> fetch-date map from JSON or from a chunks.jsonl.

    Accepts the chunker's own output because that file is the corpus of record
    committed to the repo: it is the only place the original fetch date of a
    cached document still exists once `data/raw/` is gone. Chunk ids are
    "<source_id>::cN", so the prefix is the source id and every chunk of a
    document carries the same date.
    """
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        dates: dict[str, str] = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = str(row.get("chunk_id", "")).split("::", 1)[0]
            stamp = str(row.get("fetched_at", ""))
            if source_id and stamp:
                dates.setdefault(source_id, stamp)
        return dates
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise IngestError(f"{path}: expected a source_id -> date object")
    return {str(key): str(value) for key, value in payload.items() if value}


def load_all(fetched_at: dict[str, str] | None = None) -> list[SourceDoc]:
    results: list[SourceDoc] = []
    for doc in build_source_docs():
        try:
            results.append(load_source(doc, (fetched_at or {}).get(doc.source_id)))
        except IngestError as exc:
            print(f"FAILED {doc.source_id}: {exc}")
            raise
        except requests.RequestException as exc:
            raise IngestError(f"{doc.source_id}: request to {doc.source_url} failed: {exc}. {REMEDY}") from exc
    return results
