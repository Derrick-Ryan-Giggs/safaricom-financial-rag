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
from openai import RateLimitError  # sql_query.py's client is OpenAI's SDK pointed at Groq's
                                    # OpenAI-compatible endpoint, so this is the correct
                                    # exception class -- not groq.RateLimitError.

from ingestion.embed import OnnxEmbedder
from monitoring.conversation_store import clear_all, load_messages, save_message, truncate_from
from monitoring.feedback import record_feedback
from monitoring.tracer import get_tracer
from retrieval.rag import answer_from_chunks, is_refusal, verify_no_answer
from retrieval.rerank import rerank_chunks
from retrieval.router import classify_question
from retrieval.search import build_minsearch_index, build_qdrant_client, hybrid_search, load_chunks
from retrieval.sql_query import SQLGenerationError, format_results, run_query
from retrieval.web_fallback import web_search_answer

CHUNKS_GLOB = "embeddings/*.jsonl"

MART_COVERAGE_NOTE = (
    "the mart tables cover a subset of FY08-FY26, with some years/columns incomplete"
)

EXAMPLE_QUESTIONS = [
    "What was M-PESA revenue in FY2025?",
    "What factors drove M-PESA growth?",
    "Compare Ethiopia EBIT across FY23 and FY24.",
    "What was Safaricom's overall equity score?",
]

st.set_page_config(page_title="Safaricom Financial Intelligence")
st.title("Safaricom Financial Intelligence")
st.caption("Ask about Safaricom's financials, M-PESA, or the Kenya/Ethiopia trajectory (FY08-FY26).")

# Reserves this screen position now (top of page), but its content is filled in
# further down -- AFTER process_question() is actually defined. Streamlit renders
# a container at the position it was CREATED, not where it's filled, so this is
# what lets the starter-question buttons appear at the top while still safely
# calling a function that doesn't exist yet at this point in the script.
starter_questions_container = st.container()


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
    st.session_state.messages = load_messages()
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
    # Icon-only labels (not "Helpful" / "Not helpful") so both buttons are
    # exactly one character wide -- guarantees identical size and padding
    # regardless of container width, instead of relying on a column ratio
    # wide enough to fit the longer label without wrapping.
    col1, col2, _ = st.columns([1, 1, 10])
    with col1:
        if st.button("👍", key=f"up_{trace_id}", help="Helpful"):
            record_feedback(trace_id, message.get("question", ""), message["content"], 1)
            st.toast("Thanks for the feedback!")
    with col2:
        if st.button("👎", key=f"down_{trace_id}", help="Not helpful"):
            record_feedback(trace_id, message.get("question", ""), message["content"], -1)
            st.toast("Thanks -- noted.")


def run_rag_fallback(question):
    """
    Search the full FY08-26 PDF corpus (jsonl chunks) directly. Retrieves a
    wider top-20 RRF-fused candidate set via ONE hybrid_search call, then
    reranks down to the top-10 most relevant via a cross-encoder before
    generation -- RRF fusion rank and true query-relevance aren't the same
    thing, so this narrows on relevance specifically, right before the
    chunks are used for both the Sources display and generation (see
    answer_from_chunks' docstring in retrieval/rag.py).

    Returns pieces separately (rag_answer, rag_was_refusal, web_answer,
    sources, web_sources) rather than one pre-composed string, so each
    caller can phrase its own final message without producing
    contradictory back-to-back sentences like "here's what the reports
    say" immediately followed by the model's own "the reports don't have
    this" refusal text.
    """
    candidates = hybrid_search(question, records, minsearch_index, qdrant_client, embedder, num_results=20)
    sources = rerank_chunks(question, candidates, top_n=10)
    rag_answer = answer_from_chunks(question, sources)
    rag_was_refusal = is_refusal(rag_answer)

    if rag_was_refusal:
        # Confirmed via live testing (three separate instances now): the
        # first pass sometimes refuses even when a chunk directly states
        # the answer. One stricter, narrower re-check over the SAME chunks
        # before accepting the refusal and escalating to web search.
        rechecked = verify_no_answer(question, sources)
        if rechecked is not None:
            rag_answer = rechecked
            rag_was_refusal = False

    web_answer = None
    web_sources = []
    if rag_was_refusal:
        web_answer, web_sources = web_search_answer(question)

    return rag_answer, rag_was_refusal, web_answer, sources, web_sources


def process_question(question):
    """
    Runs the full classify -> SQL/RAG/OTHER -> answer pipeline and appends
    the user question plus the assistant answer to st.session_state.messages.

    Deliberately does NOT render the chat bubbles itself -- the render loop
    below handles display uniformly whether this is a brand new question or
    a resubmitted edit, so there's only one rendering path to keep in sync
    rather than two copies of the same routing logic.
    """
    user_message = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": question,
    }
    user_message["seq"] = save_message(user_message)
    st.session_state.messages.append(user_message)

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
                except RateLimitError:
                    # Confirmed live Aug 17: the old generic except below
                    # embedded str(e) directly in the user-facing answer,
                    # which for a 429 is the provider's raw JSON error body.
                    # This catch runs first and gives a plain message instead.
                    generated_sql = None
                    answer = "I'm getting a lot of questions right now -- please try again in a few seconds."
                    rows = "error_handled"  # sentinel so we skip the block below
                except Exception:
                    generated_sql = None
                    answer = "Sorry, I couldn't run a valid query for that question."
                    rows = "error_handled"  # sentinel so we skip the block below

                if rows == "error_handled":
                    pass
                elif rows:
                    answer = format_results(rows)
                else:
                    # Empty result set OR SQL generation/execution failed outright.
                    # Either way, the mart tables didn't answer this -- fall back
                    # to searching the PDF corpus directly (FY08-26, wider and
                    # more complete than the mart tables).
                    rag_answer, rag_was_refusal, web_answer, sources, web_sources = run_rag_fallback(question)
                    reason = (
                        "didn't have this"
                        if rows == []
                        else "couldn't answer this (query generation failed)"
                    )
                    if not rag_was_refusal:
                        answer = (
                            f"The structured financial tables {reason} -- {MART_COVERAGE_NOTE}. "
                            f"Here's what the annual reports say instead:\n\n{rag_answer}"
                        )
                    elif web_sources:
                        answer = (
                            f"The structured financial tables {reason} -- {MART_COVERAGE_NOTE} -- "
                            f"and the annual report excerpts don't cover this either. Here's what a "
                            f"web search found instead (not Safaricom's own filings -- verify "
                            f"independently):\n\n{web_answer}"
                        )
                    else:
                        answer = (
                            f"The structured financial tables {reason} -- {MART_COVERAGE_NOTE} -- "
                            f"and neither the annual report excerpts nor a web search turned up an "
                            f"answer to this."
                        )
            else:
                rag_answer, rag_was_refusal, web_answer, sources, web_sources = run_rag_fallback(question)
                if not rag_was_refusal:
                    answer = rag_answer
                elif web_sources:
                    answer = (
                        f"Safaricom's own annual reports don't cover this. Here's what a web "
                        f"search found instead (not Safaricom's own filings -- verify "
                        f"independently):\n\n{web_answer}"
                    )
                else:
                    answer = "Neither the annual report excerpts nor a web search turned up an answer to this."

    assistant_message = {
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "web_sources": web_sources,
        "generated_sql": generated_sql,
        "trace_id": trace_id,
        "question": question,
    }
    assistant_message["seq"] = save_message(assistant_message)
    st.session_state.messages.append(assistant_message)


with starter_questions_container:
    st.markdown("**Try asking:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, example in zip(cols, EXAMPLE_QUESTIONS):
        with col:
            if st.button(example, key=f"example_{example}"):
                # Starter questions start a FRESH conversation, not append to
                # whatever's already there. clear_all() wipes the persisted
                # store directly, so the clicked question becomes message #1
                # and stays that way across reloads, rather than only looking
                # that way until load_messages() pulls the old history back in.
                clear_all()
                st.session_state.messages = []
                process_question(example)
                st.rerun()

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
                save_col, cancel_col, _ = st.columns([2, 2, 6])
                with save_col:
                    if st.button("Save", key=f"save_{message['id']}"):
                        if message.get("seq") is not None:
                            truncate_from(message["seq"])
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
                if st.button(" Edit", key=f"edit_{message['id']}", help="Edit this question"):
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