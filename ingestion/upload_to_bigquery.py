"""
ingestion/upload_to_bigquery.py

Creates the `safaricom_rag` BigQuery dataset and a `chunks` table (schema
matching the embedded chunk records in embeddings/*.jsonl), then loads all
chunks into it via a batch load job.

Deliberately uses a LOAD job (client.load_table_from_file), not streaming
inserts -- BigQuery load jobs are free; streaming inserts are billed.
Given this project's size (~2,100 chunks including 384-dim embedding
vectors, a few MB total), cost is negligible against BigQuery's 1TB/month
free tier either way, but a load job is the correct default regardless.

Re-running this replaces the table's contents (WRITE_TRUNCATE) rather than
appending, so it's safe to run again after re-embedding.

Usage:
    uv run python -m ingestion.upload_to_bigquery --chunks "embeddings/*.jsonl"
"""

import argparse
import glob
import json
import tempfile
from pathlib import Path

from google.cloud import bigquery

import config

DATASET_ID = "safaricom_rag"
TABLE_ID = "chunks"

SCHEMA = [
    bigquery.SchemaField("chunk_id", "STRING"),
    bigquery.SchemaField("text", "STRING"),
    bigquery.SchemaField("fiscal_year", "STRING"),
    bigquery.SchemaField("page_number", "INTEGER"),
    bigquery.SchemaField("source_file", "STRING"),
    bigquery.SchemaField("gcs_uri", "STRING"),
    bigquery.SchemaField("source_id", "STRING"),
    bigquery.SchemaField("title", "STRING"),
    bigquery.SchemaField("author", "STRING"),
    bigquery.SchemaField("created_date", "TIMESTAMP"),
    bigquery.SchemaField("document_type", "STRING"),
    bigquery.SchemaField("topic", "STRING"),
    bigquery.SchemaField("category", "STRING"),
    bigquery.SchemaField("chunk_index", "INTEGER"),
    bigquery.SchemaField("chunk_count", "INTEGER"),
    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
]


def ensure_dataset(client: bigquery.Client) -> None:
    dataset_ref = f"{config.GCP_PROJECT_ID}.{DATASET_ID}"
    try:
        client.get_dataset(dataset_ref)
        print(f"Dataset {dataset_ref} already exists.")
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset)
        print(f"Created dataset {dataset_ref}.")


def ensure_table(client: bigquery.Client) -> str:
    table_ref = f"{config.GCP_PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    try:
        client.get_table(table_ref)
        print(f"Table {table_ref} already exists -- will be replaced by the load job below.")
    except Exception:
        table = bigquery.Table(table_ref, schema=SCHEMA)
        client.create_table(table)
        print(f"Created table {table_ref}.")
    return table_ref


def load_chunk_records(chunk_glob: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(chunk_glob)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def upload_chunks(client: bigquery.Client, table_ref: str, records: list[dict]) -> None:
    allowed_fields = {field.name for field in SCHEMA}
    cleaned = [{k: v for k, v in r.items() if k in allowed_fields} for r in records]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for record in cleaned:
            tmp.write(json.dumps(record, default=str) + "\n")
        tmp_path = tmp.name

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        schema=SCHEMA,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    try:
        with open(tmp_path, "rb") as source_file:
            load_job = client.load_table_from_file(source_file, table_ref, job_config=job_config)
        load_job.result()
    finally:
        Path(tmp_path).unlink()

    print(f"Loaded {len(cleaned)} chunks into {table_ref}.")


def main():
    parser = argparse.ArgumentParser(description="Load embedded chunks into BigQuery.")
    parser.add_argument("--chunks", default="embeddings/*.jsonl", help="Glob pattern for embedded JSONL files.")
    args = parser.parse_args()

    client = bigquery.Client(project=config.GCP_PROJECT_ID)
    ensure_dataset(client)
    table_ref = ensure_table(client)

    records = load_chunk_records(args.chunks)
    print(f"Found {len(records)} chunks to upload.")
    upload_chunks(client, table_ref, records)


if __name__ == "__main__":
    main()
