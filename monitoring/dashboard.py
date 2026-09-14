"""
monitoring/dashboard.py

Feedback + observability dashboard (LLM Zoomcamp Module 5 requirement).
Reads from the same Firestore collections monitoring/tracer.py ("spans")
and monitoring/feedback.py ("feedback") write to. Migrated off local
SQLite -- this dashboard runs as its own Cloud Run service, with its own
disk, separate from rag-app's, so the old traces.db file it read was
never the file rag-app was writing to.

Usage:
    uv run streamlit run monitoring/dashboard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from google.cloud import firestore

import config

st.set_page_config(page_title="Safaricom RAG -- Monitoring", layout="wide")
st.title("Monitoring Dashboard")
st.caption("Feedback and pipeline observability, read from Firestore")

_db = firestore.Client(project=config.GCP_PROJECT_ID)


@st.cache_data(ttl=30)
def load_feedback() -> pd.DataFrame:
    rows = [doc.to_dict() for doc in _db.collection("feedback").stream()]
    df = pd.DataFrame(rows, columns=["trace_id", "question", "answer", "rating", "created_at"])
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], unit="s")
    return df


@st.cache_data(ttl=30)
def load_spans() -> pd.DataFrame:
    rows = [doc.to_dict() for doc in _db.collection("spans").stream()]
    df = pd.DataFrame(rows, columns=["trace_id", "span_id", "name", "start_time", "end_time", "duration_ms", "attributes"])
    if not df.empty:
        df["start_time"] = pd.to_datetime(df["start_time"], unit="s")
        df["route"] = df["attributes"].apply(lambda a: (a or {}).get("route", "unknown"))
    return df


feedback_df = load_feedback()
spans_df = load_spans()

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