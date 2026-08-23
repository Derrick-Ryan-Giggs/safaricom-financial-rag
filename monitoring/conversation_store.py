"""
monitoring/conversation_store.py

Persists the chat conversation to the same SQLite database used by
monitoring/tracer.py and monitoring/feedback.py, so refreshing the browser
(which resets Streamlit's in-memory session_state -- each page refresh is a
new WebSocket connection to Streamlit, hence a fresh session, even though
the server process itself never stopped) doesn't lose the conversation.
ui/app.py loads from here on startup instead of always starting empty.

Single-user, single-conversation store -- no multi-user/session keying.
Appropriate for a personal tool, not a multi-tenant product.
"""

import json
import sqlite3
import time
from pathlib import Path

from monitoring.tracer import DB_PATH


def _get_connection() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_messages (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT,
            role TEXT,
            content TEXT,
            sources_json TEXT,
            web_sources_json TEXT,
            generated_sql TEXT,
            trace_id TEXT,
            question TEXT,
            created_at REAL
        )
        """
    )
    conn.commit()
    return conn


def _slim_sources(sources: list[dict] | None) -> list[dict]:
    """
    Keep only what render_sources() actually displays (fiscal_year,
    page_number, text, source_file) -- drop the 384-dim embedding vector
    and other retrieval-internal fields so the persisted conversation
    doesn't balloon in size for no display benefit.

    source_file is kept (unlike before) so build_source_url() in ui/app.py
    can still link directly to the source PDF after a page reload -- it
    was previously dropped here, which meant the link only worked for a
    message's first render and silently vanished (no source_file to build
    from) the moment load_messages() reconstructed it from persisted
    storage.
    """
    if not sources:
        return []
    return [
        {
            "fiscal_year": s.get("fiscal_year"),
            "page_number": s.get("page_number"),
            "text": s.get("text"),
            "source_file": s.get("source_file"),
        }
        for s in sources
    ]


def save_message(message: dict) -> int:
    """Insert one message (same dict shape used in st.session_state.messages) and return its seq."""
    conn = _get_connection()
    cursor = conn.execute(
        """
        INSERT INTO conversation_messages
            (message_id, role, content, sources_json, web_sources_json, generated_sql, trace_id, question, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message["id"],
            message["role"],
            message["content"],
            json.dumps(_slim_sources(message.get("sources"))),
            json.dumps(message.get("web_sources") or []),
            message.get("generated_sql"),
            message.get("trace_id"),
            message.get("question"),
            time.time(),
        ),
    )
    conn.commit()
    seq = cursor.lastrowid
    conn.close()
    return seq


def load_messages() -> list[dict]:
    """Reconstruct st.session_state.messages' shape, in original order."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT seq, message_id, role, content, sources_json, web_sources_json, generated_sql, trace_id, question "
        "FROM conversation_messages ORDER BY seq ASC"
    ).fetchall()
    conn.close()

    messages = []
    for seq, message_id, role, content, sources_json, web_sources_json, generated_sql, trace_id, question in rows:
        messages.append({
            "seq": seq,
            "id": message_id,
            "role": role,
            "content": content,
            "sources": json.loads(sources_json) if sources_json else [],
            "web_sources": json.loads(web_sources_json) if web_sources_json else [],
            "generated_sql": generated_sql,
            "trace_id": trace_id,
            "question": question,
        })
    return messages


def truncate_from(seq: int) -> None:
    """Delete this message and everything inserted after it (used when an edited question is resubmitted)."""
    conn = _get_connection()
    conn.execute("DELETE FROM conversation_messages WHERE seq >= ?", (seq,))
    conn.commit()
    conn.close()


def clear_all() -> None:
    conn = _get_connection()
    conn.execute("DELETE FROM conversation_messages")
    conn.commit()
    conn.close()