# Safaricom Financial Intelligence Assistant

SQL + RAG over Safaricom's annual reports (FY08-FY26) and BigQuery mart
tables. A capstone project for LLM Zoomcamp 2026.

## Problem Description

Safaricom PLC has published financial results for every fiscal year since
2008, but that history lives in two disconnected forms. Structured metrics
(revenue, EBITDA, M-PESA growth) sit in BigQuery tables built from
hand-curated data, useful for trend analysis but blind to the *why* behind
the numbers. The *why*, strategic commentary, competitive context, one-off
explanations, is locked inside 19 years of PDF annual reports and results
presentations, effectively unsearchable beyond manually opening each file.

Answering a question like "How did M-PESA revenue grow in FY19, and what
drove it?" today means separately querying a table for the number and
skimming a 100+ page PDF for the explanation, with no way to ask both in
one place, and no way to search across 19 years of history at once.

This project builds a single conversational assistant that answers both
kinds of questions from one interface:

- **Structured questions** ("What was M-PESA revenue in FY2025?") are
  answered by generating and running SQL against BigQuery mart tables
  built from Safaricom's own reported figures.
- **Narrative questions** ("What factors drove M-PESA growth?") are
  answered via hybrid retrieval (keyword + vector search) over ~2,100
  chunks extracted from 19 annual report PDFs spanning FY08 to FY26, with
  source citations (fiscal year and page number) attached to every answer.

An LLM router classifies each incoming question and sends it down the
appropriate path automatically, so the person asking never needs to know
which system holds the answer.

## Architecture

```
Safaricom Annual Report PDFs (FY08-FY26, 19 documents)
        |
unstructured[pdf] extraction (hi_res + table structure inference)
        |
Chunking (500 tokens, 50 overlap, sentence-boundary-aware)
        |
ONNX embeddings (all-MiniLM-L6-v2, 384-dim)
        |
Hybrid search: minsearch (keyword) + Qdrant (vector) + RRF
        |
LLM Router (SQL vs RAG decision)
    /                              \
BigQuery SQL                    RAG (hybrid search + LLM generation)
(structured questions)          (narrative questions, cited by
                                  fiscal year + page number)
    \                              /
        Streamlit chat interface
        + OpenTelemetry tracing
        + thumbs up/down feedback
```

Ingestion (extract -> chunk -> embed -> upload) runs as an Airflow DAG
(`safaricom_ingestion`) with dynamic task mapping over discovered PDFs,
containerized via Docker Compose (Postgres + LocalExecutor, deliberately
not Celery, since this is a batch pipeline run manually when a new results
announcement drops, not a high-throughput production system).

## Dataset

19 unique fiscal years, FY08 through FY26, sourced directly from
Safaricom's investor relations materials (press commentaries, results
booklets, earnings booklets, and results presentations, whose naming
conventions were inconsistent across the 18-year span and required
content-based verification, not just filename parsing, to assign correct
fiscal years). ~2,100 chunks after extraction and chunking.

BigQuery mart tables (`dbt_rgiggs_mart`, built in a separate dbt project):
`mart_ke_et_trajectory`, `mart_mpesa_growth_trends`, `mart_revenue_mix`.

## Tech Stack

- **LLM**: Groq (`openai/gpt-oss-120b`)
- **Embeddings**: ONNX Runtime, `Xenova/all-MiniLM-L6-v2`, CPU-only
  (deliberately avoiding a ~2GB CUDA PyTorch pull for a project with no
  GPU)
- **Vector search**: Qdrant (in-memory, HNSW, cosine distance)
- **Keyword search**: minsearch
- **PDF extraction**: `unstructured[pdf]`, `hi_res` strategy + table
  structure inference, Tesseract OCR
- **Structured data**: BigQuery
- **Orchestration**: Apache Airflow (LocalExecutor, Docker Compose)
- **Interface**: Streamlit
- **Monitoring**: OpenTelemetry, custom SQLite span exporter
- **Secrets**: GCP Secret Manager
- **Package management**: uv

## Evaluation Results

**Retrieval** (Hit Rate / MRR against a 500-question ground truth
benchmark, curated from 6,222 candidates and stratified across all 19
fiscal years so no single year dominates):

| k | Hit Rate | MRR |
|---|---|---|
| 2 | 0.47 | 0.421 |
| 5 | 0.61 | 0.4625 |
| 10 | 0.696 | 0.4763 |

k=5 matches what the production RAG path actually retrieves.

**Answer quality** (LLM-as-judge against reference answers, over the same
500 questions, run through the actual production RAG pipeline): 157
questions (31.4%) are honest refusals, the system declining rather than
guessing when retrieval doesn't surface enough evidence. Of the 343
questions where the system attempted an answer, 97.1% were judged at
least partially correct and 48.7% fully correct, with only 2.9% genuinely
wrong. Reporting refusal rate and attempted-answer quality separately
matters here: the combined "relevant" rate across all 500 (33.6%) looks
much weaker on its own, because it conflates retrieval misses with actual
generation errors, which are very different failure modes.

## Project Status Against Grading Criteria

| Criterion | Target | Status |
|---|---|---|
| Problem description | 2 | Done |
| Retrieval flow | 2 | Done |
| Retrieval evaluation | 2 | Done |
| LLM evaluation | 2 | Done |
| Interface | 2 | Done (Streamlit) |
| Ingestion pipeline | 2 | Done (Airflow, dynamic task mapping) |
| Monitoring | 2 | Partial (OTel + feedback capture done; 5-chart dashboard pending) |
| Containerization | 2 | Partial (Airflow fully dockerized; main app not yet) |
| Reproducibility | 2 | Partial (dependencies pinned via `uv.lock`; data not yet packaged for a fresh clone) |
| Hybrid search bonus | 1 | Done |
| Document re-ranking bonus | 1 | Not done |
| Query rewriting bonus | 1 | Not done |
| Cloud deployment bonus | 2 | Not done |

## Setup

```bash
git clone https://github.com/Derrick-Ryan-Giggs/safaricom-financial-rag.git
cd safaricom-financial-rag
uv sync
```

Requires a `.env` file with `GOOGLE_APPLICATION_CREDENTIALS` and
`GCP_PROJECT_ID`; all other configuration (API keys, bucket names, dataset
names) loads from GCP Secret Manager at runtime.

**Ingestion** (from `airflow/`):
```bash
docker compose up airflow-init
docker compose up
```
Airflow UI at `http://localhost:8080`.

**Chat interface**:
```bash
uv run streamlit run ui/app.py
```

## Known Limitations

- BigQuery mart tables only cover whatever years exist in the underlying
  seed data; SQL questions about years not represented there (2008-2013 in
  particular) return no results even though the PDF corpus covers those
  years narratively.
- SQL generation occasionally hallucinates column names or produces
  invalid SQL for complex questions; this path has not yet received the
  same level of formal evaluation as the RAG path.
- A fresh clone of this repo has the code but not the processed data
  (chunks, embeddings); reproducing the full pipeline currently means
  re-running ingestion from source PDFs.

## Reference

RAG pipeline design notes: https://medium.com/@derrickryangiggs/rag-pipeline-deep-dive-ingestion-chunking-embedding-and-vector-search-abd3c8bfc177