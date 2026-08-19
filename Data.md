# Data Acquisition

This project depends on 19 years of Safaricom PLC annual report PDFs (FY08
through FY26) that are not committed to this repository. They are excluded
via `.gitignore` because committing ~19 PDFs directly into git history
would permanently bloat every future clone, regardless of whether the
PDFs are later removed.

## Pull the PDFs from GCS

The full set is mirrored in the same GCS bucket the Airflow ingestion DAG
reads from, so this is the actual source of truth, not a stale copy.

    mkdir -p raw
    gsutil -m cp "gs://safaricom-rag/raw/*.pdf" raw/

Requires the `gsutil` CLI (part of the Google Cloud SDK:
https://cloud.google.com/sdk/docs/install). No GCP credentials or project
are required for this command, since the `raw/` prefix is public-read.

## Original source

All 19 documents are Safaricom PLC's own public investor relations
materials. If the GCS bucket above is ever unavailable, the originals can
be found at Safaricom's investor relations page:
https://www.safaricom.co.ke/investor-relations

## Already-processed data (no download needed)

`embeddings/*.jsonl` (19 files, one per PDF, ~2,100 chunks total with
384-dim vectors) is committed directly in this repo. Retrieval, evaluation
(`evaluation/metrics.py`, `answer_quality.py`), and inspection all work
straight from these files without needing the raw PDFs or GCS access at
all. Pulling the PDFs is only necessary to re-run extraction/chunking/
embedding from scratch (e.g. after fixing a fiscal-year mislabeling, or
adding a new year's report).

## Note on BigQuery mart tables

The SQL path (`retrieval/sql_query.py`) reads from mart tables
(`mart_ke_et_trajectory`, `mart_mpesa_growth_trends`, `mart_revenue_mix`)
built by a separate dbt project, not by this repository. Cloning this repo
alone does not populate those tables. The RAG and web-fallback paths work
independently of this dependency; only the SQL path requires it.