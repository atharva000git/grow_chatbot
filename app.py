"""HDFC Mutual Fund Facts Assistant - Streamlit UI (FR-9).

Single page, and its job is to make the guarantees visible rather than to look
polished. Every answer shows which chunks grounded it, which intent produced it
and which source was cited, so a reviewer can see the guards and the retrieval
threshold working instead of taking the interface's word for it.

No backend logic lives here. Anything the UI needs that `src.pipeline` does not
provide is a change to Phase 9, not to this file.
"""

from __future__ import annotations

import streamlit as st

from src.generation.llm import LLMUnavailable
from src.generation.prompt import newest_source_date
from src.ingest.bootstrap import ensure_index, index_is_built
from src.models import load_sources
from src.pipeline import answer_query, corpus_status

st.set_page_config(
    page_title="HDFC Mutual Fund Facts Assistant",
    page_icon="📊",
    layout="wide",
)

# Verbatim from PRD.md section 8.3. Kept as a module-level constant so the same
# string can be copied into the deliverables without retyping it.
DISCLAIMER = (
    "**Facts-only. No investment advice.** Answers are generated from public HDFC "
    "AMC scheme pages and may be incomplete or out of date. Verify all details with "
    "the official factsheet and your mutual fund distributor before acting. Mutual "
    "fund investments are subject to market risks; read all scheme-related documents "
    "carefully."
)

# PRD.md section 9 items 1, 4 and 6.
EXAMPLE_QUESTIONS = [
    "What is the expense ratio of HDFC Large Cap Fund - Direct - Growth?",
    "What is the lock-in period for HDFC ELSS Tax Saver Fund, and when can I exit?",
    "How do I download my capital-gains statement?",
]

INTENT_STYLE = {
    "ANSWER": ("✅", "Answer", None),
    "REFUSAL": ("⚠️", "Refused - advice request", "warning"),
    "INSUFFICIENT_CONTEXT": ("ℹ️", "Not in sources", "info"),
    "PII_BLOCKED": ("⛔", "Blocked - personal identifier", "error"),
}

st.title("HDFC Mutual Fund Facts Assistant")
st.info(DISCLAIMER)

# The index is a gitignored build artifact, so a fresh clone on Streamlit Cloud
# or Render has none. Build it here rather than leaving the first question to
# raise IndexNotBuilt. No-op once the index exists.
if not index_is_built():
    with st.spinner("Building the vector index from the committed corpus (first run only)..."):
        ensure_index()

status = corpus_status()
with st.sidebar:
    st.subheader("Corpus")
    if status["ready"]:
        st.metric("Chunks indexed", status["n_chunks"])
        st.write(f"**Embedding model:** `{status['embedding_model']}`")
        st.write(f"**Similarity threshold:** `{status['similarity_threshold']}`")
        if status["built_at"]:
            st.caption(f"Index built {status['built_at']}")
    else:
        st.error("Index not built - run `python -m src.ingest.run_all`")
    with st.expander("Sources"):
        for row in load_sources():
            st.markdown(f"[{row['scheme']}]({row['source_url']})")

st.write("Ask about exit load, lock-in, benchmark, minimum SIP, expense ratio or NAV "
         "for the five HDFC schemes. I answer only from the official pages and factsheets, "
         "and I will say so when a fact is not there.")

for example in EXAMPLE_QUESTIONS:
    if st.button(example, key=f"ex_{example[:24]}"):
        st.session_state.pending = example

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if message["role"] == "assistant":
            if message.get("citation"):
                st.markdown(f"Source: [{message['citation']}]({message['citation']})")
            if message.get("last_updated"):
                st.caption(f"Last updated from sources: {message['last_updated']}")
            if message.get("intent"):
                icon, label, level = INTENT_STYLE.get(
                    message["intent"], ("•", message["intent"], None)
                )
                caption = f"{icon} {label} · {message.get('hits', 0)} sources"
                if message.get("similarity") is not None:
                    caption += f" · top match {message['similarity']:.3f}"
                if level:
                    getattr(st, level)(caption)
                else:
                    st.caption(caption)

typed = st.chat_input("Ask a factual question about the 5 HDFC schemes")
query = typed or st.session_state.pop("pending", None)

if query:
    st.session_state.messages.append({"role": "user", "text": query})
    try:
        response = answer_query(query)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": response.answer_text,
                "citation": response.citation_url,
                "last_updated": response.last_updated or newest_source_date([]),
                "intent": response.intent,
                "hits": response.retrieval_hits,
                "similarity": response.top_similarity,
            }
        )
    except LLMUnavailable as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": "Generation unavailable - set `LLM_API_KEY` in `.env` and restart.",
                "citation": None,
                "last_updated": "",
                "intent": "INSUFFICIENT_CONTEXT",
                "hits": 0,
                "similarity": None,
            }
        )
        st.session_state.last_error = str(exc)
    st.rerun()

if st.session_state.get("last_error"):
    st.error(f"Generation unavailable - set `LLM_API_KEY` in `.env` and restart. "
             f"({st.session_state.last_error})")
