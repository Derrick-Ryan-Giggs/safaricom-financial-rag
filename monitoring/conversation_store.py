"""
monitoring/conversation_store.py

Persists the chat conversation to Firestore, partitioned by session_id so
each browser session gets its own isolated conversation thread instead of
every visitor to the deployed URL sharing one. Migrated from local SQLite
-- Cloud Run instances are stateless and ephemeral with no shared disk
between instances, and everything is wiped on scale-to-zero or a new
revision, so a local sqlite3 file only ever survived within one instance's
lifetime and was invisible to any other instance.

session_id is generated client-side in ui/app.py (a UUID stored in the
URL's query params, so it survives a page refresh but differs per visitor)
and threaded through every function here -- this is a real signature
change from the pre-session-isolation version, so ui/app.py's call sites
needed updating too, unlike the earlier SQLite -> Firestore migration
which kept signatures identical.

REQUIRES a composite index (session_id ASC, seq ASC) on the
conversation_messages collection -- both load_messages (equality filter +
order_by on a different field) and truncate_from (equality filter + range
filter on a different field) need one. Without it, these queries throw
FailedPrecondition at runtime rather than degrading gracefully:
    gcloud firestore indexes composite create \\
        --collection-group=conversation_messages \\
        --field-config=field-path=session_id,order=ascending \\
        --field-config=field-path=seq,order=ascending \\
        --project=safaricom-intelligence

Also requires (same as before this file's session-isolation update):
    gcloud firestore databases create --database="(default)" \\
        --location=africa-south1 --project=safaricom-intelligence \\
        --type=firestore-native

    gcloud projects add-iam-policy-binding safaricom-intelligence \\
        --member="serviceAccount:safaricom-intel-sa@safaricom-intelligence.iam.gserviceaccount.com" \\
        --role="roles/datastore.user"

NOT a real multi-tenant auth system -- session_id has no identity behind
it, isn't validated, and isn't tied to a user account. It only prevents
different browser sessions from accidentally sharing/colliding on one
conversation. Appropriate for a personal demo tool, not a product with
real users to protect from each other.
"""

import time

from google.cloud import firestore

import config

COLLECTION = "conversation_messages"
COUNTER_DOC_ID = "conversation_seq"

# Crude per-session cap per roadmap item 3 -- rag-app is public and
# unauthenticated with nothing else stopping repeated Groq/BigQuery usage
# from one session. Not tuned against real traffic, just a starting point.
MAX_QUESTIONS_PER_HOUR = 30

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
    this was built on. A transaction on a single counter document
    guarantees strictly-increasing, collision-free seq values even across
    different sessions writing concurrently. The counter is intentionally
    GLOBAL (shared across all sessions), not per-session -- that's simpler
    than sharded per-session counters and still gives every message a
    unique seq, since uniqueness (not per-session sequential numbering
    starting at 1) is all truncate_from/document-ID-as-seq actually needs.
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


def save_message(message: dict, session_id: str) -> int:
    """Insert one message (same dict shape used in st.session_state.messages) and return its seq."""
    seq = _next_seq()
    db = _get_client()
    db.collection(COLLECTION).document(str(seq)).set({
        "seq": seq,
        "session_id": session_id,
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


def load_messages(session_id: str) -> list[dict]:
    """Reconstruct st.session_state.messages' shape for this session only, in original order."""
    db = _get_client()
    docs = (
        db.collection(COLLECTION)
        .where("session_id", "==", session_id)
        .order_by("seq", direction=firestore.Query.ASCENDING)
        .stream()
    )

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
    fits in one.
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


def truncate_from(seq: int, session_id: str) -> None:
    """
    Delete this message and everything inserted after it, WITHIN this
    session only (used when an edited question is resubmitted). The
    session_id filter matters here specifically -- without it, editing a
    message would delete every session's messages from that global seq
    onward, not just the editing session's own history.
    """
    db = _get_client()
    _delete_matching(
        db.collection(COLLECTION)
        .where("session_id", "==", session_id)
        .where("seq", ">=", seq)
    )


def clear_all(session_id: str) -> None:
    """Wipe this session's conversation only -- other sessions are untouched."""
    db = _get_client()
    _delete_matching(db.collection(COLLECTION).where("session_id", "==", session_id))


def check_rate_limit(session_id: str) -> bool:
    """
    Returns True if this session can ask another question this hour, False
    if it's already at MAX_QUESTIONS_PER_HOUR. Increments the counter as a
    side effect -- call this once per incoming question, BEFORE running
    classify/SQL/RAG, so a rejected question never reaches the LLM or
    BigQuery.

    Keyed by session_id + the current UTC hour, so the cap resets on its
    own every hour with no cleanup job needed -- old hour-bucket documents
    are just abandoned, not deleted. Fine at this traffic level; if that
    ever matters, a Firestore TTL policy on `expires_at` handles it:
        gcloud firestore fields ttls update expires_at \\
            --collection-group=rate_limits --enable-ttl \\
            --project=safaricom-intelligence
    """
    hour_bucket = int(time.time() // 3600)
    doc_ref = _get_client().collection("rate_limits").document(f"{session_id}_{hour_bucket}")

    @firestore.transactional
    def _check_and_increment(transaction, ref):
        snapshot = ref.get(transaction=transaction)
        current = snapshot.get("count") if snapshot.exists else 0
        if current >= MAX_QUESTIONS_PER_HOUR:
            return False
        transaction.set(ref, {
            "count": current + 1,
            "session_id": session_id,
            "hour_bucket": hour_bucket,
            "expires_at": time.time() + 7200,  # 2hr buffer past the hour it belongs to
        })
        return True

    return _check_and_increment(_get_client().transaction(), doc_ref)