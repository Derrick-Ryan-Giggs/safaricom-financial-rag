"""
evaluation/mart_completeness_check.py

Schema-driven completeness check across all mart tables: for every numeric
column, reports which fiscal_year rows have a NULL value. Produces one
clean, complete punch-list of every incomplete column -- meant to be
handed off to the Safaricom Intelligence project (where the actual
seed-CSV fixes live), rather than checking columns one at a time by hand
in the BigQuery console.

Scope: numeric columns only (the actual financial metrics). Descriptive/
dimension columns (period_label, source_file, etc.) aren't checked --
those being present or not isn't the "missing data" concern this exists
to catch.

Usage:
    uv run python -m evaluation.mart_completeness_check
"""

import config
from retrieval.sql_query import MART_TABLES, get_bq_client

NUMERIC_TYPES = {"INTEGER", "INT64", "FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC"}


def check_table_completeness(client, table_name: str) -> None:
    table_ref = f"{config.GCP_PROJECT_ID}.{config.BIGQUERY_MART_DATASET}.{table_name}"
    try:
        table = client.get_table(table_ref)
    except Exception as e:
        print(f"Skipping {table_name}: could not read schema ({e}).")
        return

    field_names = [f.name for f in table.schema]
    if "fiscal_year" not in field_names:
        print(f"Skipping {table_name}: no fiscal_year column to anchor completeness checks to.")
        return

    numeric_columns = [
        f.name for f in table.schema
        if f.field_type in NUMERIC_TYPES and f.name != "fiscal_year"
    ]
    if not numeric_columns:
        print(f"Skipping {table_name}: no numeric columns found.")
        return

    print(f"\n=== {table_name} ({len(numeric_columns)} numeric columns) ===")
    any_gaps = False

    for col in numeric_columns:
        query = f"""
            SELECT fiscal_year
            FROM `{table_ref}`
            WHERE {col} IS NULL
            ORDER BY fiscal_year
        """
        rows = list(client.query(query).result())
        if rows:
            any_gaps = True
            years = ", ".join(str(r["fiscal_year"]) for r in rows)
            print(f"  INCOMPLETE  {col}: missing for {years}")

    if not any_gaps:
        print("  (no missing values in any numeric column)")


def main():
    client = get_bq_client()
    for table_name in MART_TABLES:
        check_table_completeness(client, table_name)
    print(
        "\nDone. Each 'INCOMPLETE' line above is a specific column + year-list "
        "to fix in the Safaricom Intelligence project's seed CSVs -- not "
        "something fixable in this repo."
    )


if __name__ == "__main__":
    main()