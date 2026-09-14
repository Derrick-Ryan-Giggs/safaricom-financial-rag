"""
monitoring/tracer.py

Sets up OpenTelemetry tracing for the RAG pipeline, exporting spans to
Firestore instead of local SQLite -- rag-app and monitoring/dashboard.py
(a separate Cloud Run service) have no shared disk, so a local traces.db
was invisible to the dashboard no matter how much traffic rag-app saw.
Same reasoning as monitoring/conversation_store.py's earlier migration.

Call get_tracer() to get a ready-to-use tracer -- it handles one-time
setup internally. trace.set_tracer_provider() can only be called once per
Python process, so setup_tracing() is a no-op on repeated calls.
"""

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from google.cloud import firestore

import config

SPANS_COLLECTION = "spans"

_initialized = False
_db: firestore.Client | None = None


def _get_client() -> firestore.Client:
    global _db
    if _db is None:
        _db = firestore.Client(project=config.GCP_PROJECT_ID)
    return _db


def _firestore_safe(attrs: dict) -> dict:
    """Firestore maps support scalars and lists but not tuples -- OTel
    attributes can be tuples, so convert those before writing."""
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in attrs.items()}


class FirestoreSpanExporter(SpanExporter):
    """Writes finished spans to Firestore so the dashboard -- a separate
    Cloud Run service -- can actually read them. See module docstring."""

    def export(self, spans) -> SpanExportResult:
        db = _get_client()
        batch = db.batch()
        for span in spans:
            trace_id = format(span.context.trace_id, "032x")
            span_id = format(span.context.span_id, "016x")
            doc_ref = db.collection(SPANS_COLLECTION).document(f"{trace_id}_{span_id}")
            batch.set(doc_ref, {
                "trace_id": trace_id,
                "span_id": span_id,
                "name": span.name,
                "start_time": span.start_time / 1e9,
                "end_time": span.end_time / 1e9,
                "duration_ms": (span.end_time - span.start_time) / 1e6,
                "attributes": _firestore_safe(dict(span.attributes or {})),
            })
        batch.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def setup_tracing(service_name: str = "safaricom-rag") -> None:
    global _initialized
    if _initialized:
        return

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(SimpleSpanProcessor(FirestoreSpanExporter()))
    trace.set_tracer_provider(provider)
    _initialized = True


def get_tracer(name: str = "safaricom-rag"):
    setup_tracing()
    return trace.get_tracer(name)