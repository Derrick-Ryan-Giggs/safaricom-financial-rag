"""
monitoring/conversation_store.py

Persists the chat conversation to Firestore. Migrated from local SQLite --
Cloud Run instances are stateless and ephemeral with no shared disk between
instances, and everything is wiped on scale-to-zero or a new revision, so a
local sqlite3 file only ever survived within one instance's lifetime and
was invisible to any other instance. Firestore is reachable identically
from every instance via the same attached service account already used for
BigQuery and Secret Manager elsewhere in this project -- no new auth
pattern, just an added `roles/datastore.user` IAM grant.

Same message dict shape and function signatures as the prior SQLite
version (save_message returns an int seq, load_messages returns messages
ordered by seq ascending, truncate_from(seq) deletes that message and
everything after it) -- ui/app.py needed NO changes for this migration.

Single-user, single-conversation store -- no multi-user/session keying.
Appropriate for a personal tool, not a multi-tenant product. At real
multi-instance concurrency this would need per-session keying to avoid
different users sharing one conversation; not a concern at this project's
actual traffic level.

Requires a Firestore database to exist first (Native mode, one-time):
    gcloud firestore databases create --database="(default)" \\
        --location=africa-south1 --project=safaricom-intelligence \\
        --type=firestore-native

And the service account needs read/write access:
    gcloud projects add-iam-policy-binding safaricom-intelligence \\
        --member="serviceAccount:safaricom-intel-sa@safaricom-intelligence.iam.gserviceaccount.com" \\
        --role="roles/datastore.user"
"""

import time

from google.cloud import firestore

import config

COLLECTION = "conversation_messages"
COUNTER_DOC_ID = "conversation_seq"

_db: firestore.Client | None = None


def _get_client() -> firestore.Client:
    global _db
    if _db is None:
        _db = firestore.Client(project=config.GCP_PROJECT_ID)
    return _db


@firestore.transactional
def _increment_counter(transaction: firestore.Transaction, counter_ref) -> int:
    """
    Firestore has no native autoincrement, unlike SQLite's AUTOINCREMENT
    this was built on. A transaction on a single counter document is what
    guarantees the same strictly-increasing, collision-free seq values --
    a plain read-then-write (without a transaction) would risk two
    concurrent writers both reading the same current value and producing
    a duplicate seq; a timestamp-based approach would avoid the
    transaction but isn't a guaranteed match for SQLite's exact semantics.
    Given this is confirmed single-user, contention is very unlikely in
    practice, but the transaction costs nothing extra to keep.
    """
    snapshot = counter_ref.get(transaction=transaction)
    current = snapshot.get("value") if snapshot.exists else 0
    new_value = current + 1
    transaction.set(counter_ref, {"value": new_value})
    return new_value


def _next_seq() -> int:
    db = _get_client()
    counter_ref = db.collection("counters").document(COUNTER_DOC_ID)
    transaction = db.transaction()
    return _increment_counter(transaction, counter_ref)


def _slim_sources(sources: list[dict] | None) -> list[dict]:
    """
    Keep only what render_sources() actually displays (fiscal_year,
    page_number, text, source_file) -- drop the 384-dim embedding vector
    and other retrieval-internal fields so persisted messages stay well
    within Firestore's 1MiB per-document limit, not just for size hygiene.
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
    seq = _next_seq()
    db = _get_client()
    db.collection(COLLECTION).document(str(seq)).set({
        "seq": seq,
        "message_id": message["id"],
        "role": message["role"],
        "content": message["content"],
        "sources": _slim_sources(message.get("sources")),
        "web_sources": message.get("web_sources") or [],
        "generated_sql": message.get("generated_sql"),
        "trace_id": message.get("trace_id"),
        "question": message.get("question"),
        "created_at": time.time(),
    })
    return seq


def load_messages() -> list[dict]:
    """Reconstruct st.session_state.messages' shape, in original order."""
    db = _get_client()
    docs = db.collection(COLLECTION).order_by("seq", direction=firestore.Query.ASCENDING).stream()

    messages = []
    for doc in docs:
        data = doc.to_dict()
        messages.append({
            "seq": data.get("seq"),
            "id": data.get("message_id"),
            "role": data.get("role"),
            "content": data.get("content"),
            "sources": data.get("sources") or [],
            "web_sources": data.get("web_sources") or [],
            "generated_sql": data.get("generated_sql"),
            "trace_id": data.get("trace_id"),
            "question": data.get("question"),
        })
    return messages


def _delete_matching(query) -> None:
    """
    Shared batch-delete helper for truncate_from/clear_all. Firestore
    batched writes cap at 500 operations -- chunks into multiple batches
    when there's more history than that, rather than assuming it always
    fits in one (a real risk for clear_all on a long-lived conversation).
    """
    db = _get_client()
    batch = db.batch()
    count = 0
    for doc in query.stream():
        batch.delete(doc.reference)
        count += 1
        if count >= 500:
            batch.commit()
            batch = db.batch()
            count = 0
    if count > 0:
        batch.commit()


def truncate_from(seq: int) -> None:
    """Delete this message and everything inserted after it (used when an edited question is resubmitted)."""
    db = _get_client()
    _delete_matching(db.collection(COLLECTION).where("seq", ">=", seq))


def clear_all() -> None:
    db = _get_client()
    _delete_matching(db.collection(COLLECTION))