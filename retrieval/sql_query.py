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
Only query tables from the schema provided below -- do not invent table or column names.
Return ONLY the SQL query, no explanation, no markdown code fences.

Schema:
{schema}
"""


def get_bq_client() -> bigquery.Client:
    return bigquery.Client(project=config.GCP_PROJECT_ID)


def get_mart_schema(client: bigquery.Client) -> str:
    """
    Introspect column names and types for each mart table, formatted as a
    compact text block for the LLM prompt.
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

    return "\n".join(lines)


def generate_sql(question: str, schema: str) -> str:
    client = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SQL_SYSTEM_PROMPT.format(schema=schema)},
            {"role": "user", "content": question},
        ],
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


def run_query(question: str) -> list[dict]:
    client = get_bq_client()
    schema = get_mart_schema(client)
    sql = generate_sql(question, schema)

    print(f"Generated SQL:\n{sql}\n")

    if not is_safe_select(sql):
        raise ValueError(f"Refusing to run non-SELECT or unsafe query: {sql}")

    job_config = bigquery.QueryJobConfig(
        default_dataset=f"{config.GCP_PROJECT_ID}.{config.BIGQUERY_MART_DATASET}"
    )
    query_job = client.query(sql, job_config=job_config)
    rows = [dict(row) for row in query_job.result()]
    return rows


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

    def prettify_column(col: str) -> str:
        return col.replace("_", " ").title()

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