"""
ui/app.py

Streamlit chat interface for the Safaricom Financial Intelligence RAG.
Routes each question to either the BigQuery SQL path or the RAG path,
displays the answer with source citations (for RAG) or the generated SQL
(for SQL questions), and records thumbs up/down feedback.

Usage:
    uv run streamlit run ui/app.py
"""

import sys
import uuid
from pathlib import Path

# Streamlit's script runner adds this file's own directory (ui/) to sys.path,
# not the project root -- so `import config` and the package imports below
# would fail without this. Adding the project root explicitly makes this
# work regardless of Streamlit version or how it's invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from ingestion.embed import OnnxEmbedder
from monitoring.feedback import record_feedback
from monitoring.tracer import get_tracer
from retrieval.rag import answer_question
from retrieval.router import classify_question
from retrieval.search import build_minsearch_index, build_qdrant_client, hybrid_search, load_chunks
from retrieval.sql_query import format_results, run_query

CHUNKS_GLOB = "embeddings/*.jsonl"

st.set_page_config(page_title="Safaricom Financial Intelligence")
st.title("Safaricom Financial Intelligence")
st.caption("Ask about Safaricom's financials, M-PESA, or the Kenya/Ethiopia trajectory (FY08-FY26).")


@st.cache_resource
def load_retrieval_stack():
    records = load_chunks(CHUNKS_GLOB)
    minsearch_index = build_minsearch_index(records)
    qdrant_client = build_qdrant_client(records)
    embedder = OnnxEmbedder()
    return records, minsearch_index, qdrant_client, embedder


records, minsearch_index, qdrant_client, embedder = load_retrieval_stack()
tracer = get_tracer()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    if not sources:
        return
    with st.expander("Sources"):
        for source in sources:
            preview = source["text"][:150]
            st.markdown(f"- **{source['fiscal_year']}, p.{source['page_number']}**: {preview}...")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources"))

question = st.chat_input("Ask a question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        trace_id = str(uuid.uuid4())
        sources = []

        with tracer.start_as_current_span("answer_question") as span:
            span.set_attribute("question", question)

            route = classify_question(question)
            span.set_attribute("route", route)

            if route == "SQL":
                with st.spinner("Querying BigQuery..."):
                    try:
                        rows = run_query(question)
                        answer = format_results(rows)
                    except Exception as e:
                        answer = f"Sorry, I couldn't run a valid query for that question. ({e})"
            else:
                with st.spinner("Searching annual reports..."):
                    sources = hybrid_search(
                        question, records, minsearch_index, qdrant_client, embedder, num_results=5
                    )
                    answer = answer_question(question, records, minsearch_index, qdrant_client, embedder)

        st.markdown(answer)
        render_sources(sources)

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

        col1, col2, _ = st.columns([1, 1, 8])
        with col1:
            if st.button("Helpful", key=f"up_{trace_id}"):
                record_feedback(trace_id, question, answer, 1)
                st.success("Thanks for the feedback.")
        with col2:
            if st.button("Not helpful", key=f"down_{trace_id}"):
                record_feedback(trace_id, question, answer, -1)
                st.info("Thanks -- noted.")