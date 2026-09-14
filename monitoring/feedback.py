"""
monitoring/feedback.py

Stores thumbs up/down feedback in Firestore, keyed by trace_id so it can
be joined back to the trace that produced the answer. Migrated off local
SQLite for the same reason as monitoring/tracer.py -- rag-app and the
dashboard's Cloud Run service don't share a disk.
"""

import time

from google.cloud import firestore

import config

COLLECTION = "feedback"

_db: firestore.Client | None = None


def _get_client() -> firestore.Client:
    global _db
    if _db is None:
        _db = firestore.Client(project=config.GCP_PROJECT_ID)
    return _db


def record_feedback(trace_id: str, question: str, answer: str, rating: int) -> None:
    """rating: 1 for thumbs up, -1 for thumbs down."""
    _get_client().collection(COLLECTION).add({
        "trace_id": trace_id,
        "question": question,
        "answer": answer,
        "rating": rating,
        "created_at": time.time(),
    })


def get_feedback_summary() -> dict:
    counts = {1: 0, -1: 0}
    for doc in _get_client().collection(COLLECTION).stream():
        rating = doc.to_dict().get("rating")
        if rating in counts:
            counts[rating] += 1
    return {"thumbs_up": counts[1], "thumbs_down": counts[-1], "total": counts[1] + counts[-1]}