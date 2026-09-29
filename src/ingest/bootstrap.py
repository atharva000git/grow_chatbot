"""Deploy-time index build - the step a hosting platform does not run for you.

Locally the index is a build artifact you create once. On Streamlit Community
Cloud, Render, or any container that clones the repo, it is *absent*: `index/`
and `data/embeddings/` are gitignored, and the platform starts `streamlit run`
without ever invoking `python -m src.ingest.run_all`. The clone is therefore
correct but unbuildable, and the first question raises `IndexNotBuilt` before a
single guard or generator runs - which is exactly what the deployed app did.

`ensure_index` is the missing build step, called once at app start. It is a
no-op when the index already exists, so local behaviour and the test suite are
untouched, and on a fresh deploy it runs the ordinary ingest pipeline.

Three properties make it safe to call from a web app:

1. **Offline.** `data/clean/` is committed and `load_all()` returns the cached
   text, so the build needs no source URLs. The only network call is the
   one-time download of the embedding model, which the app needs anyway to
   encode the *query*.
2. **Atomic.** It delegates to `run_all`, whose store swap means readers see
   either the old index or the new one, never a half-written `vectors.npy`.
3. **Single-builder.** A host may run several server processes against one
   checkout. A lock file outside the index directory keeps two of them from
   embedding the same 349 chunks at once; the losers wait for the winner and
   then load what it wrote.

The lock lives beside `index/` rather than inside it because `build_index`
deletes and replaces that directory wholesale.
"""

from __future__ import annotations

import logging
import os
import time

from src.ingest.store import INDEX_PATH, collection_count
from src.models import REPO_ROOT

log = logging.getLogger(__name__)

LOCK_PATH = INDEX_PATH.with_name(INDEX_PATH.name + ".build.lock")
BUILD_WAIT_SECONDS = 300
POLL_SECONDS = 2.0

# The committed chunk file is the corpus of record: it is the only surviving
# record of when these documents were fetched, once `data/raw/` is gitignored.
# Without it a deploy-time build stamps every figure with the build date, and
# the freshness line the app shows users would be wrong by however long ago the
# ingest actually ran.
CORPUS_OF_RECORD = REPO_ROOT / "data" / "chunks" / "chunks.jsonl"

# Resolved once per process. Streamlit re-executes the whole script on every
# rerun, and re-statting the index each time buys nothing: a process that has
# already built or found one will not find a different answer on the next rerun.
_RESOLVED: bool | None = None


def index_is_built() -> bool:
    """True when a usable, non-empty index is on disk."""
    return collection_count() > 0


def _acquire_lock() -> bool:
    """Create the lock file exclusively. False if another process holds it."""
    try:
        handle = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    os.close(handle)
    return True


def _release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def _build() -> bool:
    from src.ingest.run_all import main

    started = time.perf_counter()
    log.info("no index at %s; building from the committed corpus", INDEX_PATH)
    # argv is passed explicitly: `main()` would otherwise parse Streamlit's own
    # command line and fail on arguments it does not recognise.
    main(["--rebuild", "--fetched-at-from", str(CORPUS_OF_RECORD)])
    log.info("index build finished in %.1fs", time.perf_counter() - started)
    return index_is_built()


def ensure_index(wait_seconds: int = BUILD_WAIT_SECONDS) -> bool:
    """Return True once a usable index exists, building it if necessary."""
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED

    if not index_is_built():
        if not _acquire_lock():
            deadline = time.monotonic() + wait_seconds
            while not index_is_built() and time.monotonic() < deadline:
                time.sleep(POLL_SECONDS)
            if not index_is_built():
                # The holder outlived the wait - a crashed build, most likely,
                # which leaves the lock file behind. Take it over rather than
                # serve a dead app; the store swap keeps our write atomic.
                log.warning("index build lock held for over %ss; taking it over", wait_seconds)
                _RESOLVED = _build()
                return _RESOLVED
        else:
            try:
                # Re-check under the lock: the previous holder may have
                # finished between our failed stat and the lock acquisition.
                _RESOLVED = _build() if not index_is_built() else True
            finally:
                _release_lock()
            return _RESOLVED

    _RESOLVED = True
    return _RESOLVED
