"""
ui/app.py

Streamlit chat interface for the Safaricom Financial Intelligence RAG.
Routes each question to either the BigQuery SQL path or the RAG path,
displays the answer with source citations (for RAG) or the generated SQL
(for SQL questions), and records thumbs up/down feedback.

If the SQL path comes up empty (BigQuery mart tables only cover a subset of
FY14-26, with gaps) or fails outright (hallucinated column, invalid SQL),
falls back to the RAG path over the full FY08-26 PDF corpus instead of
leaving the user with a bare "no data" message.

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
from retrieval.rag import answer_question, is_refusal
from retrieval.router import classify_question
from retrieval.search import build_minsearch_index, build_qdrant_client, hybrid_search, load_chunks
from retrieval.sql_query import SQLGenerationError, format_results, run_query
from retrieval.web_fallback import web_search_answer

CHUNKS_GLOB = "embeddings/*.jsonl"

MART_COVERAGE_NOTE = (
    "the mart tables cover a subset of FY14-FY26, with some years/columns incomplete"
)

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
if "editing_id" not in st.session_state:
    st.session_state.editing_id = None


def render_sources(sources):
    if not sources:
        return
    with st.expander("Sources"):
        for source in sources:
            preview = source["text"][:150]
            st.markdown(f"- **{source['fiscal_year']}, p.{source['page_number']}**: {preview}...")


def render_web_sources(web_sources):
    if not web_sources:
        return
    with st.expander("Web sources (not Safaricom's own filings)"):
        for r in web_sources:
            title = r.get("title", "untitled")
            href = r.get("href", "")
            st.markdown(f"- [{title}]({href})")


def render_generated_sql(sql):
    if not sql:
        return
    with st.expander("Generated SQL"):
        st.code(sql, language="sql")


def render_feedback_buttons(message):
    trace_id = message.get("trace_id")
    if not trace_id:
        return
    col1, col2, _ = st.columns([1, 1, 8])
    with col1:
        if st.button("Helpful", key=f"up_{trace_id}"):
            record_feedback(trace_id, message.get("question", ""), message["content"], 1)
            st.success("Thanks for the feedback.")
    with col2:
        if st.button("Not helpful", key=f"down_{trace_id}"):
            record_feedback(trace_id, message.get("question", ""), message["content"], -1)
            st.info("Thanks -- noted.")


def run_rag_fallback(question):
    """
    Search the full FY08-26 PDF corpus (jsonl chunks) directly. If that
    comes back as a genuine refusal (not just a partial answer -- see
    retrieval.rag.is_refusal), escalate once to a last-resort web search
    and append it, clearly labeled as web content rather than Safaricom's
    own filings. Returns (answer, sources, web_sources).
    """
    sources = hybrid_search(question, records, minsearch_index, qdrant_client, embedder, num_results=5)
    rag_answer = answer_question(question, records, minsearch_index, qdrant_client, embedder)

    web_sources = []
    if is_refusal(rag_answer):
        web_answer, web_sources = web_search_answer(question)
        rag_answer = (
            f"{rag_answer}\n\n---\n\n**From the web** (not Safaricom's own filings -- "
            f"verify independently):\n\n{web_answer}"
        )

    return rag_answer, sources, web_sources


def process_question(question):
    """
    Runs the full classify -> SQL/RAG/OTHER -> answer pipeline and appends
    the user question plus the assistant answer to st.session_state.messages.

    Deliberately does NOT render the chat bubbles itself -- the render loop
    below handles display uniformly whether this is a brand new question or
    a resubmitted edit, so there's only one rendering path to keep in sync
    rather than two copies of the same routing logic.
    """
    st.session_state.messages.append({
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": question,
    })

    trace_id = str(uuid.uuid4())
    sources = []
    web_sources = []
    generated_sql = None

    with st.spinner("Thinking..."):
        with tracer.start_as_current_span("answer_question") as span:
            span.set_attribute("question", question)

            route = classify_question(question)
            span.set_attribute("route", route)

            if route == "OTHER":
                # Not actually a Safaricom financial question -- a greeting,
                # small talk, or a question about the app/system itself.
                # Skip both the BigQuery and RAG pipelines entirely rather
                # than burning a query on something like "how are you".
                answer = (
                    "I'm built to answer questions about Safaricom's financial history, "
                    "M-PESA, or the Kenya/Ethiopia trajectory (FY08-FY26) -- I don't have "
                    "a good answer for that one here. Try asking about a specific fiscal "
                    "year, metric, or trend, e.g. \"What was M-PESA revenue in FY2025?\" or "
                    "\"What factors drove M-PESA growth?\""
                )
            elif route == "SQL":
                try:
                    rows, generated_sql = run_query(question, return_sql=True)
                except SQLGenerationError as e:
                    generated_sql = e.sql
                    rows = None  # signal: SQL failed outright, not just empty
                except Exception as e:
                    generated_sql = None
                    answer = f"Sorry, I couldn't run a valid query for that question. ({e})"
                    rows = "error_handled"  # sentinel so we skip the block below

                if rows == "error_handled":
                    pass
                elif rows:
                    answer = format_results(rows)
                else:
                    # Empty result set OR SQL generation/execution failed outright.
                    # Either way, the mart tables didn't answer this -- fall back
                    # to searching the PDF corpus directly (FY08-26, wider and
                    # more complete than the mart tables), which itself escalates
                    # to web search if it also comes back as a refusal.
                    rag_answer, sources, web_sources = run_rag_fallback(question)
                    reason = (
                        "didn't have this"
                        if rows == []
                        else "couldn't answer this (query generation failed)"
                    )
                    answer = (
                        f"The structured financial tables {reason} -- {MART_COVERAGE_NOTE}. "
                        f"Here's what the annual reports say instead:\n\n{rag_answer}"
                    )
            else:
                rag_answer, sources, web_sources = run_rag_fallback(question)
                answer = rag_answer

    st.session_state.messages.append({
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "web_sources": web_sources,
        "generated_sql": generated_sql,
        "trace_id": trace_id,
        "question": question,
    })


# Single render pass over all messages. User messages get an Edit affordance;
# the message currently being edited renders as a text_area with Save/Cancel
# instead of plain markdown. Editing and saving truncates everything after
# that point (the old answer depended on the old question) and regenerates
# via the same process_question() used for brand new questions.
messages = st.session_state.messages
i = 0
while i < len(messages):
    message = messages[i]

    if message["role"] == "user":
        if message["id"] == st.session_state.editing_id:
            with st.chat_message("user"):
                new_text = st.text_area(
                    "Edit your question",
                    value=message["content"],
                    key=f"edit_box_{message['id']}",
                    label_visibility="collapsed",
                )
                save_col, cancel_col, _ = st.columns([1, 1, 6])
                with save_col:
                    if st.button("Save & resubmit", key=f"save_{message['id']}"):
                        st.session_state.messages = st.session_state.messages[:i]
                        st.session_state.editing_id = None
                        process_question(new_text)
                        st.rerun()
                with cancel_col:
                    if st.button("Cancel", key=f"cancel_{message['id']}"):
                        st.session_state.editing_id = None
                        st.rerun()
        else:
            with st.chat_message("user"):
                st.markdown(message["content"])
                if st.button("Edit", key=f"edit_{message['id']}"):
                    st.session_state.editing_id = message["id"]
                    st.rerun()
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])
            render_generated_sql(message.get("generated_sql"))
            render_sources(message.get("sources"))
            render_web_sources(message.get("web_sources"))
            render_feedback_buttons(message)

    i += 1

question = st.chat_input("Ask a question...")
if question:
    process_question(question)
    st.rerun()