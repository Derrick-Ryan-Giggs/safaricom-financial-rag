"""
monitoring/tracer.py

Sets up OpenTelemetry tracing for the RAG pipeline, exporting spans to a
local SQLite database (no OTel collector needed for this project's scale).

Call get_tracer() to get a ready-to-use tracer -- it handles one-time setup
internally. trace.set_tracer_provider() can only be called once per Python
process, so setup_tracing() is a no-op on repeated calls.
"""

import json
import sqlite3
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

import os

# Overridable so a Docker named volume can persist this file at a path
# outside the source tree (e.g. /data/traces.db) without shadowing
# tracer.py/feedback.py themselves -- mounting a volume directly at
# monitoring/ would wipe out this module's own source code from the
# container's view. Defaults to the original relative path for local,
# non-Docker development.
DB_PATH = os.environ.get("TRACES_DB_PATH", "monitoring/traces.db")

_initialized = False


class SQLiteSpanExporter(SpanExporter):
    """Writes finished spans to a local SQLite table for the feedback dashboard."""

    def __init__(self, db_path: str = DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spans (
                trace_id TEXT,
                span_id TEXT,
                name TEXT,
                start_time REAL,
                end_time REAL,
                duration_ms REAL,
                attributes TEXT
            )
            """
        )
        self.conn.commit()

    def export(self, spans) -> SpanExportResult:
        for span in spans:
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    format(span.context.trace_id, "032x"),
                    format(span.context.span_id, "016x"),
                    span.name,
                    span.start_time / 1e9,
                    span.end_time / 1e9,
                    (span.end_time - span.start_time) / 1e6,
                    json.dumps(dict(span.attributes or {})),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self.conn.close()


def setup_tracing(service_name: str = "safaricom-rag") -> None:
    global _initialized
    if _initialized:
        return

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter()))
    trace.set_tracer_provider(provider)
    _initialized = True


def get_tracer(name: str = "safaricom-rag"):
    setup_tracing()
    return trace.get_tracer(name)