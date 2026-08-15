# Safaricom Financial Intelligence Assistant

A conversational assistant over Safaricom PLC's financial history: SQL
against BigQuery mart tables for structured questions, hybrid RAG over 19
years of annual report PDFs for narrative ones, and a live web search
fallback for anything neither source covers. A capstone project for LLM
Zoomcamp 2026.

## Table of Contents

- [Problem Description](#problem-description)
- [Solution Overview](#solution-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Data Quality: What Broke and How It Was Found](#data-quality-what-broke-and-how-it-was-found)
- [Evaluation](#evaluation)
- [Known Limitations](#known-limitations)
- [Reproducibility: How to Run](#reproducibility-how-to-run)

## Problem Description

Safaricom PLC has published financial results for every fiscal year since
2008, but that history lives in two disconnected forms. Structured metrics
(revenue, EBITDA, M-PESA growth) sit in BigQuery tables built from
hand-curated data, useful for trend analysis but blind to the *why* behind
the numbers. The *why*, strategic commentary, competitive context, one-off
explanations, is locked inside 19 years of PDF annual reports and results
presentations, effectively unsearchable beyond manually opening each file.

Answering "How did M-PESA revenue grow in FY19, and what drove it?" today
means separately querying a table for the number and skimming a 100+ page
PDF for the explanation, with no way to ask both in one place, and no way
to search across 19 years of history at once. And for anything genuinely
outside both sources, the honest answer has always been "go look it up
somewhere else."

This project builds a single conversational assistant that closes all
three gaps from one interface.

## Solution Overview

| Dimension | Detail |
|---|---|
| Scope | FY08 through FY26, 19 fiscal years of Safaricom annual reports, press commentaries, and results/earnings booklets |
| Structured data | BigQuery mart tables (`mart_ke_et_trajectory`, `mart_mpesa_growth_trends`, `mart_revenue_mix`), built in a separate dbt project |
| Unstructured data | ~2,100 chunks extracted from 19 PDFs, embedded and indexed for hybrid search |
| Fallback | Live web search when neither internal source has enough to answer, clearly labeled as unverified against a primary source |
| Interface | Streamlit chat, source citations, thumbs up/down feedback |
| Orchestration | Airflow DAG with dynamic task mapping, one extract/chunk/embed cycle per discovered PDF |

Key questions this answers:

- What was M-PESA revenue in a specific fiscal year, and how did it compare
  to the year before? (SQL path)
- What factors drove a specific metric's growth or decline, based on
  Safaricom's own stated commentary? (RAG path, cited by fiscal year and
  page number)
- Anything current-events-adjacent or outside the PDF corpus entirely,
  answered via live search, explicitly flagged as not verified against a
  primary source.

## Architecture

```
Safaricom Annual Report PDFs (19 documents, FY08-FY26)
        |
unstructured[pdf] extraction (hi_res + table structure inference,
Tesseract OCR)
        |
Chunking (500 tokens, 50 overlap, sentence-boundary-aware)
        |
ONNX embeddings (all-MiniLM-L6-v2, 384-dim, CPU-only)
        |
Hybrid search: minsearch (keyword) + Qdrant (vector) + Reciprocal Rank
Fusion
        |
        +----------------------------------------------------------+
        |                                                          |
LLM Router                                                          |
    |                                                               |
    +-- SQL questions --> BigQuery, generated + validated SQL       |
    |                     against mart_ke_et_trajectory,            |
    |                     mart_mpesa_growth_trends,                 |
    |                     mart_revenue_mix                          |
    |                                                               |
    +-- Narrative questions --> hybrid search --> LLM generation,   |
    |                           cited by fiscal year + page number  |
    |                                                               |
    +-- Neither has enough --> live web search fallback, answer     |
                                explicitly labeled as unverified     |
                                against a primary source             |
        |
Streamlit chat interface, OpenTelemetry tracing, thumbs up/down
feedback, feedback dashboard
```

Ingestion (extract -> chunk -> embed -> refresh BigQuery -> upload to GCS)
runs as an Airflow DAG (`safaricom_ingestion`) with dynamic task mapping
over discovered PDFs, containerized via Docker Compose (Postgres +
LocalExecutor, deliberately not Celery, since this is a batch pipeline run
manually when a new results announcement drops, not a high-throughput
production system).

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| LLM | Groq (`openai/gpt-oss-120b`) | Router classification, SQL generation, RAG generation, web fallback synthesis |
| Embeddings | ONNX Runtime, `Xenova/all-MiniLM-L6-v2` | 384-dim, CPU-only, avoids a ~2GB CUDA PyTorch pull for a project with no GPU |
| Vector search | Qdrant | In-memory, HNSW, cosine distance |
| Keyword search | minsearch | Combined with vector search via Reciprocal Rank Fusion |
| PDF extraction | `unstructured[pdf]` | `hi_res` strategy, table structure inference, Tesseract OCR |
| Structured data | BigQuery | Mart tables (separate dbt project) plus a refreshed chunks/embeddings table for the RAG side |
| Web fallback | `duckduckgo_search` (DDGS) | No API key required |
| Orchestration | Apache Airflow | LocalExecutor, Docker Compose, dynamic task mapping |
| Interface | Streamlit | Chat UI plus a separate feedback dashboard |
| Monitoring | OpenTelemetry | Custom SQLite span exporter (no collector needed at this scale) |
| Secrets | GCP Secret Manager | Only `GOOGLE_APPLICATION_CREDENTIALS` and `GCP_PROJECT_ID` in `.env` |
| Package management | uv | All dependency versions pinned via `uv.lock` |

## Data Sources

19 unique fiscal years, sourced directly from Safaricom's investor
relations materials. Naming conventions changed several times across the
18-year span, which mattered more than expected, see the data quality
section below.

| Era | Format | Fiscal Years |
|---|---|---|
| Earliest history | Investor presentations, narrative/slide-deck format | FY08, FY10, FY11, FY12, FY13, FY14 |
| Press commentaries | Narrative prose, headline figures embedded in text | FY09, FY15 through FY19 |
| Results/earnings booklets | Structured template, tables and segment detail | FY20 through FY26 |

## Project Structure

```
safaricom-financial-rag/
├── config.py                      # loads secrets from GCP Secret Manager
├── main.py
├── ingestion/
│   ├── extract.py                 # unstructured PDF extraction
│   ├── verify_fiscal_years.py     # cross-checks filename vs document content
│   ├── fix_fiscal_years.py        # corrects mislabeled fiscal_year/title fields
│   ├── chunk.py                   # 500 tokens, 50 overlap, sentence-aware
│   ├── embed.py                   # ONNX embeddings
│   ├── upload_gcs.py              # raw PDFs + processed JSONL to GCS
│   └── upload_to_bigquery.py      # chunks + embeddings to BigQuery
├── retrieval/
│   ├── search.py                  # hybrid search: minsearch + Qdrant + RRF
│   ├── router.py                  # LLM classification: SQL vs RAG
│   ├── sql_query.py               # BigQuery SQL generation + execution
│   ├── rag.py                     # hybrid search + LLM generation
│   ├── web_fallback.py            # live web search when internal sources
│   │                                 come up short
│   └── check_rank.py              # retrieval debugging: where does a known
│                                     chunk actually rank for a given query
├── evaluation/
│   ├── ground_truth.py            # resumable Q&A pair generation with
│   │                                 reference answers
│   ├── curate_ground_truth.py     # dedup + year-stratified sampling
│   ├── metrics.py                 # Hit Rate, MRR at configurable k
│   ├── answer_quality.py          # LLM-as-judge against reference answers
│   ├── check_refusals.py          # separates honest refusals from actual
│   │                                 wrong answers
│   ├── compare_verdicts.py        # diffs judge verdicts across iterations
│   ├── sql_ground_truth.py        # ground truth for the SQL path
│   ├── sql_eval.py                # SQL path accuracy evaluation
│   └── mart_completeness_check.py # confirms mart table schema/coverage
│                                     assumptions before trusting them
├── monitoring/
│   ├── tracer.py                  # OpenTelemetry + SQLite span exporter
│   ├── feedback.py                # thumbs up/down capture
│   ├── conversation_store.py      # persisted chat history
│   └── dashboard.py               # Streamlit feedback dashboard
├── ui/
│   └── app.py                     # Streamlit chat interface
├── airflow/
│   ├── dags/ingestion_dag.py      # discover_pdfs -> extract/chunk/embed
│   │                                 (dynamic task mapping) -> refresh
│   │                                 BigQuery -> upload to GCS
│   ├── Dockerfile
│   └── docker-compose.yml         # Postgres + LocalExecutor
├── docker/
│   └── Dockerfile
├── models/Xenova/all-MiniLM-L6-v2/  # cached ONNX model + tokenizer
├── config.py, pyproject.toml, uv.lock
└── .env.example
```

## Data Quality: What Broke and How It Was Found

**Fiscal years were mislabeled for five documents, and the pattern was
inconsistent enough to need content-based verification, not just filename
parsing.** Safaricom's own filenames use different conventions across
eras: some use the starting calendar year of a fiscal year span
(`FY14-15Press_Commentary.pdf`), some use the ending year, some use
2-digit, some 4-digit, and at least one 2008-era file used a single digit
(`FY_8Results...`). Trusting the filename alone got it wrong in five
cases, most notably `FY14-15Press_Commentary.pdf`, `FY15-16Press_
Commentary.pdf`, and `FY16-17Press_Commentary.pdf`, which are actually
FY15, FY16, and FY17 respectively: each document's own "YEAR ENDED"
statement contradicts what its filename implies. `verify_fiscal_years.py`
checks every document's content against its filename-derived label;
`fix_fiscal_years.py` corrects the ones that disagree, leaving anything
unverifiable untouched rather than guessing.

Because `chunk_id` is a freshly generated UUID on every chunking run,
correcting an already-processed document's fiscal year meant more than a
metadata patch: it meant re-chunking (new IDs), re-embedding, purging the
now-orphaned old chunk IDs from the ground truth set, and regenerating
ground truth for just those documents. The `evaluation/ground_truth.py`
pipeline was built resumable specifically because of this: any interrupted
or partially-invalidated run can restart and pick up only what's actually
missing, tracked by which `chunk_id`s already appear in the output file.

**Reciprocal Rank Fusion needed a real BigQuery client library fix, not a
prompt tweak.** `qdrant-client` 1.18+ removed `.search()` entirely in favor
of `.query_points()`, with a renamed `query=` parameter and a response
object wrapping results in `.points` instead of returning them directly. A
version mismatch, not a logic bug: this only showed up the first time
hybrid search actually ran against real data.

**Generated SQL failed with "must be qualified with a dataset" on the very
first real query**, because the LLM was given bare table names in its
schema context but BigQuery's client needs either a fully-qualified
`project.dataset.table` reference or a default dataset set on the query
job itself. Fixed by setting `default_dataset` on the `QueryJobConfig`
rather than hoping the model always writes fully-qualified names. Separately,
four of the seven mart tables named in the original project plan
(`stg_company_overview`, `stg_kenya_ethiopia`, `stg_mpesa_metrics`,
`stg_revenue_segments`) turned out not to exist under those names in
`dbt_rgiggs_mart` at all, confirmed via `bq ls` rather than assumed;
`sql_query.py`'s schema introspection was trimmed to the three tables that
actually exist.

**The headline answer-quality number looked much worse than the system
actually performed, until refusals were separated from wrong answers.**
Early LLM-as-judge runs reported roughly a third of answers as "relevant,"
which sounds mediocre until you notice that most of the "not relevant"
verdicts were the model honestly declining to answer when retrieval hadn't
surfaced enough evidence, not the model getting something wrong. A
regex-based refusal detector (`check_refusals.py`) initially undercounted
these too: phrasings like "do not contain information about" (missing
"enough"), "no information...about" with extra words in between, and "do
not mention" (plural, vs. the singular "does not mention" the first
version matched) all slipped through as if they were substantive wrong
answers. Broadening the detector and re-splitting the results (no new LLM
calls needed, since the underlying answers were already generated) showed
the real picture: the model refuses roughly a third of the time when it
genuinely lacks evidence, and is right or partly right about 97% of the
time when it actually attempts an answer. See [Evaluation](#evaluation)
for the numbers.

**A handful of ground-truth question/answer pairs were simply wrong**,
caught by manually reading through failure cases rather than trusting
verdicts blindly. One question, "Who are the owners of Safaricom?",
carried a generated reference answer of "ESSAR Communications", a phrase
that actually describes a *competitor* in a "Competitive Landscape"
section of the source document, not Safaricom's own ownership. The
question-generation LLM had misattributed a fact from the same chunk to
the wrong entity. A recurring, lower-severity pattern: several fiscal
years share near-identical boilerplate phrases (e.g. "Strong financial and
commercial performance" appears in FY15 through FY18 alike), so questions
generated from one year's occurrence of common phrasing have no single
correct year to retrieve, a structural ambiguity in the ground truth
itself, not a retrieval failure.

## Evaluation

**Retrieval** (Hit Rate / MRR against a 500-question benchmark, curated
from 6,222 candidates and stratified across all 19 fiscal years so no
single year, particularly the more document-heavy later ones, dominates
the evaluation set):

| k | Hit Rate | MRR |
|---|---|---|
| 2 | 0.47 | 0.421 |
| 5 | 0.61 | 0.4625 |
| 10 | 0.696 | 0.4763 |

k=5 matches what the production RAG path actually retrieves.

**Answer quality**, run through the actual production RAG pipeline over
the same 500 questions, judged against reference answers: 157 questions
(31.4%) are honest refusals, the system declining rather than guessing
when retrieval doesn't surface enough evidence. Of the 343 questions where
the system attempted an answer, 97.1% were judged at least partially
correct and 48.7% fully correct, with only 2.9% genuinely wrong.

**SQL path evaluation and further answer-quality iterations** (a dedicated
ground truth and accuracy harness for the SQL path, plus several
subsequent refinements to the answer-quality methodology) exist in
`evaluation/sql_eval.py`, `sql_ground_truth.py`, and later `answer_
quality_v2` through `v4` result files, built to close the gap where SQL
generation had received no formal evaluation despite real bugs (hallucinated
column names, invalid aggregation syntax) surfacing under live use.
Numbers from these runs aren't reflected above yet, add them here once
finalized.

## Known Limitations

- BigQuery mart tables only cover whatever years exist in the underlying
  seed data from the separate dbt project; SQL questions about years
  outside that range return no results even though the PDF corpus covers
  those years narratively.
- SQL generation has occasionally hallucinated column names or produced
  invalid SQL for complex questions (aggregate/window function mismatches
  in particular). The dedicated SQL evaluation harness exists to
  systematically find and prioritize these rather than fixing them one at
  a time as spotted.
- The web search fallback is intentionally unverified against a primary
  source and says so explicitly in its own answers; treat it as a last
  resort, not a citation-quality source the way the RAG path is.
- `ingestion/upload_to_bigquery.py` loads chunks and embeddings into
  `safaricom_rag.chunks` for archival/inspection, but retrieval itself
  still reads from local `embeddings/*.jsonl`, not from that table. The
  BigQuery chunks table is not yet in the runtime retrieval path.

## Reproducibility: How to Run

```bash
git clone https://github.com/Derrick-Ryan-Giggs/safaricom-financial-rag.git
cd safaricom-financial-rag
uv sync
```

Requires a `.env` file with `GOOGLE_APPLICATION_CREDENTIALS` and
`GCP_PROJECT_ID`; all other configuration (API keys, bucket names, dataset
names) loads from GCP Secret Manager at runtime, never hardcoded and never
committed.

**Full stack (recommended)** — `docker-compose.yml` lives at the repo
root and brings up Postgres, Airflow, Qdrant, the chat app, and the
feedback dashboard together:
```bash
docker compose up airflow-init
docker compose up
```
- Airflow UI: `http://localhost:8081` — unpause and trigger
  `safaricom_ingestion`; it discovers PDFs and dynamically maps
  extract/chunk/embed tasks per document, then refreshes the BigQuery
  chunks table and uploads to GCS.
- Chat app: `http://localhost:8601`
- Feedback dashboard: `http://localhost:8602`

The `app` and `dashboard` services expect a GCP service account key at
`~/.gcp/safaricom-intelligence-sa.json` on the host (bind-mounted
read-only into the container); adjust that path in `docker-compose.yml`
if your key lives elsewhere.

**Running the chat app or dashboard outside Docker** (e.g. for faster
iteration during development):
```bash
docker compose up qdrant     # search.py needs a reachable Qdrant server
uv run streamlit run ui/app.py
uv run streamlit run monitoring/dashboard.py
```

**Re-running evaluation** (both are resumable; safe to interrupt and
rerun the same command):
```bash
uv run python -m evaluation.ground_truth --chunks "embeddings/*.jsonl" --output evaluation/ground_truth_v1.jsonl
uv run python -m evaluation.curate_ground_truth --input evaluation/ground_truth_v1.jsonl --output evaluation/ground_truth_curated_v1.jsonl --target-size 500
uv run python -m evaluation.metrics --ground-truth evaluation/ground_truth_curated_v1.jsonl --chunks "embeddings/*.jsonl" --k 2 5 10
uv run python -m evaluation.answer_quality --ground-truth evaluation/ground_truth_curated_v1.jsonl --chunks "embeddings/*.jsonl" --output evaluation/answer_quality_v1.jsonl
```