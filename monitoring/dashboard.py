"""
monitoring/dashboard.py

Feedback + observability dashboard (LLM Zoomcamp Module 5 requirement).
Reads directly from the SQLite database written by monitoring/tracer.py
(`spans` table) and monitoring/feedback.py (`feedback` table) -- same DB
the live app already writes to, no separate data pipeline needed.

Usage:
    uv run streamlit run monitoring/dashboard.py
"""

import sys
from pathlib import Path

# Same fix as ui/app.py: Streamlit's script runner adds this file's own
# directory (monitoring/) to sys.path, not the project root -- so
# `from monitoring.tracer import ...` below would fail without this,
# since the `monitoring` package itself needs the project root on the
# path to resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import sqlite3

import pandas as pd
import streamlit as st

from monitoring.tracer import DB_PATH

st.set_page_config(page_title="Safaricom RAG -- Monitoring", layout="wide")
st.title("Monitoring Dashboard")
st.caption("Feedback and pipeline observability, read from monitoring/traces.db")


@st.cache_data(ttl=30)
def load_feedback() -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM feedback", conn)
        conn.close()
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        # Table doesn't exist yet -- nobody has given feedback in the main
        # app, so monitoring/feedback.py's CREATE TABLE has never run.
        return pd.DataFrame(columns=["trace_id", "question", "answer", "rating", "created_at"])
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], unit="s")
    return df


@st.cache_data(ttl=30)
def load_spans() -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM spans", conn)
        conn.close()
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        return pd.DataFrame(columns=["trace_id", "span_id", "name", "start_time", "end_time", "duration_ms", "attributes"])
    if not df.empty:
        df["start_time"] = pd.to_datetime(df["start_time"], unit="s")

        def extract_route(attrs_json):
            try:
                return json.loads(attrs_json).get("route", "unknown")
            except (TypeError, ValueError):
                return "unknown"

        df["route"] = df["attributes"].apply(extract_route)
    return df


feedback_df = load_feedback()
spans_df = load_spans()

# ui/app.py wraps each full question in one "answer_question" span (see
# tracer.start_as_current_span("answer_question") in process_question) --
# retrieval/rag.py and retrieval/sql_query.py don't emit their own child
# spans, so this is the whole trace per question, not just a sub-step.
question_spans = spans_df[spans_df["name"] == "answer_question"] if not spans_df.empty else spans_df

if feedback_df.empty and question_spans.empty:
    st.info("No data yet -- ask some questions and leave feedback in the main app first.")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Total questions", len(question_spans))
m2.metric("Thumbs up", int((feedback_df["rating"] == 1).sum()) if not feedback_df.empty else 0)
m3.metric("Thumbs down", int((feedback_df["rating"] == -1).sum()) if not feedback_df.empty else 0)

st.divider()

st.subheader("1. Feedback ratio")
if not feedback_df.empty:
    ratio = feedback_df["rating"].map({1: "Helpful", -1: "Not helpful"}).value_counts()
    st.bar_chart(ratio)
else:
    st.caption("No feedback recorded yet.")

st.subheader("2. Feedback over time")
if not feedback_df.empty:
    labeled = feedback_df.assign(
        date=feedback_df["created_at"].dt.date,
        label=feedback_df["rating"].map({1: "Helpful", -1: "Not helpful"}),
    )
    daily = labeled.groupby(["date", "label"]).size().unstack(fill_value=0)
    st.line_chart(daily)
else:
    st.caption("No feedback recorded yet.")

st.subheader("3. Question volume over time")
if not question_spans.empty:
    daily_volume = question_spans.assign(date=question_spans["start_time"].dt.date).groupby("date").size()
    st.line_chart(daily_volume)
else:
    st.caption("No questions recorded yet.")

st.subheader("4. Route distribution (SQL / RAG / OTHER)")
if not question_spans.empty:
    st.bar_chart(question_spans["route"].value_counts())
else:
    st.caption("No questions recorded yet.")

st.subheader("5. Response latency")
if not question_spans.empty:
    bins = pd.cut(question_spans["duration_ms"], bins=10)
    hist = bins.value_counts().sort_index()
    hist.index = hist.index.astype(str)
    st.bar_chart(hist)
    st.caption(
        f"Median: {question_spans['duration_ms'].median():.0f} ms · "
        f"p95: {question_spans['duration_ms'].quantile(0.95):.0f} ms"
    )
else:
    st.caption("No questions recorded yet.")