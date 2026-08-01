"""
airflow/dags/ingestion_dag.py

Orchestrates ingestion for whatever PDFs are sitting in raw/: for each one,
extract -> chunk -> embed. ingestion/extract.py, chunk.py, and embed.py each
only ever process ONE file via --pdf/--input/--output -- this DAG is what
does the "for every PDF" looping, using Airflow's dynamic task mapping so
each file's chain shows up as its own set of tasks in the UI (individually
retriable) rather than one opaque loop hidden inside a single task. Once
every file is embedded, refreshes the GCS backup and the BigQuery chunks
table.

Idempotent: discover_pdfs() skips any PDF that already has a matching
embeddings/*.jsonl output, so re-running only processes newly-added PDFs --
matching the skip-if-exists convention ingestion/upload_gcs.py already uses,
rather than reprocessing the whole corpus (expensive: hi_res PDF extraction
does layout detection + OCR) every run.

All steps shell out to `uv run python -m folder.filename`, matching this
project's own established convention (config.py only resolves relative to
the project root, never by direct file path).

Requires the accompanying docker-compose.yml + Dockerfile in this same
airflow/ directory.

Usage:
    From the airflow/ directory:
        docker compose up airflow-init
        docker compose up
    Airflow UI: http://localhost:8080 (admin/admin) -- trigger
    "safaricom_ingestion" manually (schedule=None; PDFs are added by hand
    when a new results announcement drops, not on a fixed cadence).
"""

from __future__ import annotations

import glob
import subprocess
from pathlib import Path

import pendulum
from airflow.decorators import dag, task

PROJECT_ROOT = "/opt/airflow/project"


def run(*args: str) -> None:
    """Run a project script the same way you would by hand: `uv run python -m ...`."""
    subprocess.run(["uv", "run", "python", "-m", *args], cwd=PROJECT_ROOT, check=True)


@dag(
    dag_id="safaricom_ingestion",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=["safaricom", "ingestion", "rag"],
)
def safaricom_ingestion():

    @task
    def discover_pdfs() -> list[dict]:
        """
        One dict per raw PDF still needing processing, giving each
        downstream task the exact --input/--output paths it needs.
        Skips PDFs that already have a matching embeddings/*.jsonl file.
        """
        pairs = []
        for pdf_path in sorted(glob.glob(f"{PROJECT_ROOT}/raw/*.pdf")):
            stem = Path(pdf_path).stem
            embedded_path = f"{PROJECT_ROOT}/embeddings/{stem}.jsonl"
            if Path(embedded_path).exists():
                continue  # already fully processed in a previous run
            pairs.append({
                "pdf": pdf_path,
                "processed": f"{PROJECT_ROOT}/processed/{stem}.jsonl",
                "chunked": f"{PROJECT_ROOT}/chunks/{stem}.jsonl",
                "embedded": embedded_path,
            })
        return pairs

    @task
    def extract_one(pair: dict) -> dict:
        run("ingestion.extract", "--pdf", pair["pdf"], "--output", pair["processed"])
        return pair

    @task
    def chunk_one(pair: dict) -> dict:
        run("ingestion.chunk", "--input", pair["processed"], "--output", pair["chunked"])
        return pair

    @task
    def embed_one(pair: dict) -> dict:
        run("ingestion.embed", "--input", pair["chunked"], "--output", pair["embedded"])
        return pair

    @task
    def upload_to_gcs(pairs: list[dict]) -> None:
        run("ingestion.upload_gcs", "--raw-dir", "raw", "--processed-dir", "processed")

    @task
    def refresh_bigquery_chunks(pairs: list[dict]) -> None:
        run("ingestion.upload_to_bigquery", "--chunks", "embeddings/*.jsonl")

    pairs = discover_pdfs()
    extracted = extract_one.expand(pair=pairs)
    chunked = chunk_one.expand(pair=extracted)
    embedded = embed_one.expand(pair=chunked)

    upload_to_gcs(embedded)
    refresh_bigquery_chunks(embedded)


safaricom_ingestion()