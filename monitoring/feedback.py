"""
monitoring/feedback.py

Stores user thumbs up/down feedback on generated answers in the same
SQLite database used by tracer.py, keyed by trace_id so feedback can be
joined back to the retrieval/generation trace that produced the answer.
"""

import sqlite3
import time
from pathlib import Path

from monitoring.tracer import DB_PATH


def _get_connection() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            trace_id TEXT,
            question TEXT,
            answer TEXT,
            rating INTEGER,
            created_at REAL
        )
        """
    )
    conn.commit()
    return conn


def record_feedback(trace_id: str, question: str, answer: str, rating: int) -> None:
    """
    rating: 1 for thumbs up, -1 for thumbs down.
    """
    conn = _get_connection()
    conn.execute(
        "INSERT INTO feedback VALUES (?, ?, ?, ?, ?)",
        (trace_id, question, answer, rating, time.time()),
    )
    conn.commit()
    conn.close()


def get_feedback_summary() -> dict:
    conn = _get_connection()
    cursor = conn.execute("SELECT rating, COUNT(*) FROM feedback GROUP BY rating")
    counts = dict(cursor.fetchall())
    conn.close()

    return {
        "thumbs_up": counts.get(1, 0),
        "thumbs_down": counts.get(-1, 0),
        "total": sum(counts.values()),
    }