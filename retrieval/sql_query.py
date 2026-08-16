"""
retrieval/sql_query.py

Generates and runs BigQuery SQL for structured questions ("What was M-PESA
revenue in FY2025?") against the mart tables in BIGQUERY_MART_DATASET.

Schema is introspected live from BigQuery rather than hardcoded, so this
stays correct as mart tables evolve without needing to update this file.

Usage:
    uv run python retrieval/sql_query.py --question "What was M-PESA revenue in FY2025?"
"""

import argparse
import json

from google.cloud import bigquery
from openai import OpenAI

import config

MART_TABLES = [
    "mart_ke_et_trajectory",
    "mart_mpesa_growth_trends",
    "mart_revenue_mix",
]
# The original HANDOFF.md also listed stg_company_overview, stg_kenya_ethiopia,
# stg_mpesa_metrics, and stg_revenue_segments -- confirmed via `bq ls` that
# these don't exist as physical tables in dbt_rgiggs_mart. If they live in a
# different dataset (e.g. a separate staging schema), add them back here with
# their correct dataset qualification once located.

SQL_SYSTEM_PROMPT = """You are a BigQuery SQL generator for a Safaricom financial data warehouse.

You may ONLY generate SELECT statements. Never generate INSERT, UPDATE, DELETE, DROP, or DDL of any kind.
Only query tables and columns from the schema provided below -- do not invent table or column names.

If nothing in the schema -- no table, no column -- can answer the question, do NOT invent a
plausible-sounding name, and do NOT write a placeholder query like "SELECT NULL" just to produce
something that runs without error. A query that runs but answers nothing is worse than admitting
there's no match: it looks like a real (empty) result instead of an honest "not available here."
Instead, respond with exactly this and nothing else: NO_MATCHING_COLUMN

fiscal_year is stored as a full 4-digit integer (see the example row below) -- NOT a 2-digit short form.
A question naming "FY26" means fiscal_year = 2026, not fiscal_year = 26.

Return ONLY the SQL query, no explanation, no markdown code fences.

Schema (each table includes one real example row so you can see actual value formats):
{schema}

Worked examples:
Q: What was M-PESA revenue in FY2025?
A: SELECT mpesa_revenue_kes_bn FROM mart_mpesa_growth_trends WHERE fiscal_year = 2025

Q: Compare Ethiopia EBIT across FY23 and FY24.
A: SELECT fiscal_year, et_ebit_kes_bn FROM mart_ke_et_trajectory WHERE fiscal_year IN (2023, 2024)

Q: What was Safaricom's cash and cash equivalents as of 31-Mar-24?
A: NO_MATCHING_COLUMN
"""

MAX_SQL_RETRIES = 2


class SQLGenerationError(Exception):
    """
    Raised when SQL generation or execution fails, but carries the SQL that
    was attempted (or generated) so callers can still show it to the user --
    e.g. for debugging hallucinated columns or invalid queries in the UI,
    rather than only surfacing the raw BigQuery error text.
    """

    def __init__(self, message: str, sql: str | None = None):
        super().__init__(message)
        self.sql = sql


def get_bq_client() -> bigquery.Client:
    return bigquery.Client(project=config.GCP_PROJECT_ID)


def get_mart_schema(client: bigquery.Client) -> str:
    """
    Introspect column names and types for each mart table, plus one real
    sample row per table, formatted as a compact text block for the LLM
    prompt. The sample row exists specifically so the model sees actual
    value formats (e.g. fiscal_year=2026, not "26") instead of guessing --
    confirmed via evaluation/sql_eval.py that without this, the generator
    is inconsistent about whether fiscal_year is 2-digit or 4-digit for the
    exact same question pattern.
    """
    lines = []
    for table_name in MART_TABLES:
        table_ref = f"{config.GCP_PROJECT_ID}.{config.BIGQUERY_MART_DATASET}.{table_name}"
        try:
            table = client.get_table(table_ref)
        except Exception as e:
            print(f"Warning: could not read schema for {table_name}: {e}")
            continue

        columns = ", ".join(f"{field.name} ({field.field_type})" for field in table.schema)
        lines.append(f"Table `{table_name}`: {columns}")

        try:
            sample_rows = list(client.query(f"SELECT * FROM `{table_ref}` LIMIT 1").result())
            if sample_rows:
                sample = dict(sample_rows[0])
                sample_str = ", ".join(f"{k}={v}" for k, v in sample.items())
                lines.append(f"  Example row: {sample_str}")
        except Exception as e:
            print(f"Warning: could not fetch sample row for {table_name}: {e}")

    return "\n".join(lines)


def generate_sql(
    question: str,
    schema: str,
    previous_attempt: str | None = None,
    previous_error: str | None = None,
    model: str = config.LLM_MODEL,
) -> str:
    client = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

    messages = [
        {"role": "system", "content": SQL_SYSTEM_PROMPT.format(schema=schema)},
        {"role": "user", "content": question},
    ]
    if previous_attempt and previous_error:
        messages.append({"role": "assistant", "content": previous_attempt})
        messages.append({
            "role": "user",
            "content": (
                f"That query failed with this error:\n{previous_error}\n\n"
                "Correct it using only real tables and columns from the schema above."
            ),
        })

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    sql = response.choices[0].message.content.strip()
    return sql.removeprefix("```sql").removeprefix("```").removesuffix("```").strip()


def is_safe_select(sql: str) -> bool:
    """Reject anything that isn't a single SELECT statement."""
    normalized = sql.strip().lower()
    if not normalized.startswith("select"):
        return False
    forbidden = ("insert", "update", "delete", "drop", "alter", "create", "truncate", "merge")
    return not any(keyword in normalized for keyword in forbidden)


def run_query(question: str, return_sql: bool = False, model: str = config.LLM_MODEL):
    """
    Generate and run SQL for `question`.

    By default returns just the result rows (list[dict]), matching the
    original behavior so the CLI entry point below doesn't change.

    On a BigQuery execution failure (hallucinated column, invalid GROUP BY,
    etc.), retries up to MAX_SQL_RETRIES times, feeding the actual error
    back to the model so it can self-correct, before giving up.

    If return_sql=True, returns (rows, sql) on success. On final failure
    (unsafe SQL rejected, or all retries exhausted), raises
    SQLGenerationError with the last attempted SQL attached as `.sql`, so a
    caller (like the Streamlit UI) can still show what was generated even
    though it didn't run.
    """
    client = get_bq_client()
    schema = get_mart_schema(client)
    sql = generate_sql(question, schema, model=model)

    job_config = bigquery.QueryJobConfig(
        default_dataset=f"{config.GCP_PROJECT_ID}.{config.BIGQUERY_MART_DATASET}"
    )

    last_error = None
    for attempt in range(MAX_SQL_RETRIES + 1):
        print(f"Generated SQL (attempt {attempt + 1}/{MAX_SQL_RETRIES + 1}):\n{sql}\n")

        if sql.strip() == "NO_MATCHING_COLUMN":
            # The model explicitly determined nothing in the schema answers
            # this -- treat it the same as a query that ran and found zero
            # rows, rather than an error. This is what stops attempt 2 from
            # degenerating into a placeholder like "SELECT NULL" (confirmed
            # live: a cash-and-equivalents question, once no real column
            # existed, produced a syntactically valid but meaningless query
            # instead of admitting the mismatch) -- callers already have
            # correct handling for "SQL succeeded, empty result" (fall back
            # to RAG), so mapping this to the same outcome is the correct
            # semantics, not a special case to add elsewhere.
            print("Model determined no schema column matches this question.")
            if return_sql:
                return [], sql
            return []

        if not is_safe_select(sql):
            message = f"Refusing to run non-SELECT or unsafe query: {sql}"
            if return_sql:
                raise SQLGenerationError(message, sql)
            raise ValueError(message)

        try:
            query_job = client.query(sql, job_config=job_config)
            rows = [dict(row) for row in query_job.result()]
            if return_sql:
                return rows, sql
            return rows
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_SQL_RETRIES:
                print(f"Query failed, retrying with error feedback: {last_error}")
                sql = generate_sql(question, schema, previous_attempt=sql, previous_error=last_error, model=model)

    if return_sql:
        raise SQLGenerationError(last_error, sql)
    raise RuntimeError(last_error)


def prettify_column(col: str) -> str:
    return col.replace("_", " ").title()


def format_results(rows: list[dict]) -> str:
    """
    Format BigQuery result rows as readable text instead of a raw Python
    repr (e.g. "[{'mpesa_revenue_kes_bn': Decimal('161.12')}]"). Single-row
    results render as labeled lines; multi-row results render as a markdown
    table, which Streamlit's st.markdown displays natively.

    Column labels are a simple underscore-to-title-case conversion (e.g.
    "mpesa_revenue_kes_bn" -> "Mpesa Revenue Kes Bn") -- not perfect (ideally
    "M-PESA Revenue (KES Bn)"), but far more readable than raw column names
    without needing a hardcoded per-column label map.
    """
    if not rows:
        return "No matching data found in the mart tables."

    if len(rows) == 1:
        lines = [f"**{prettify_column(k)}**: {v}" for k, v in rows[0].items()]
        return "\n\n".join(lines)

    columns = list(rows[0].keys())
    header = "| " + " | ".join(prettify_column(c) for c in columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    data_rows = ["| " + " | ".join(str(row.get(c, "")) for c in columns) + " |" for row in rows]
    return "\n".join([header, separator] + data_rows)


def main():
    parser = argparse.ArgumentParser(description="Answer a structured question via generated BigQuery SQL.")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    rows = run_query(args.question)
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()