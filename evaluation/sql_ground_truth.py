"""
evaluation/sql_ground_truth.py

Generates ground truth for the SQL retrieval path directly from BigQuery:
for every numeric column in each mart table, queries the real value per
fiscal year and pairs it with a template-generated natural-language
question. Schema-driven (introspected live at runtime, same approach as
retrieval/sql_query.py's get_mart_schema) rather than hardcoded, so this
stays correct if the mart tables' columns change.

Note on question phrasing: questions are generated as "What was {column
name} in {fiscal year}?" using the column name directly (prettified). This
is a floor-level test -- it checks whether the pipeline can correctly
answer a question that names a column close to verbatim. It does NOT test
robustness to natural paraphrasing (e.g. "cash and cash equivalents" for a
column that doesn't exist under that name, which is the kind of case that's
been hallucinating). That's a real limitation of this v1 harness, not
something it claims to cover.

Usage:
    uv run python -m evaluation.sql_ground_truth --output evaluation/sql_ground_truth_v1.jsonl
"""

import argparse
import json

import config
from retrieval.sql_query import MART_TABLES, get_bq_client, prettify_column

NUMERIC_TYPES = {"INTEGER", "INT64", "FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC"}


def fiscal_year_label(value) -> str | None:
    """Normalize a fiscal_year column value (e.g. 2022 or '2022') to 'FY22'."""
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return f"FY{year % 100:02d}"


def generate_ground_truth(client, table_name: str) -> list[dict]:
    table_ref = f"{config.GCP_PROJECT_ID}.{config.BIGQUERY_MART_DATASET}.{table_name}"

    try:
        table = client.get_table(table_ref)
    except Exception as e:
        print(f"Skipping {table_name}: could not read schema ({e}).")
        return []

    field_names = [f.name for f in table.schema]
    if "fiscal_year" not in field_names:
        print(f"Skipping {table_name}: no fiscal_year column to anchor questions to.")
        return []

    numeric_columns = [
        f.name for f in table.schema
        if f.field_type in NUMERIC_TYPES and f.name != "fiscal_year"
    ]
    if not numeric_columns:
        print(f"Skipping {table_name}: no numeric columns found.")
        return []

    select_cols = ", ".join(["fiscal_year"] + numeric_columns)
    rows = [dict(row) for row in client.query(f"SELECT {select_cols} FROM `{table_ref}`").result()]

    # Some mart tables have more than one row for the same fiscal_year (an
    # actual data quality issue, not a hypothetical -- confirmed here for
    # mart_ke_et_trajectory). Without deduping, every question for that
    # year gets generated twice, doubling LLM calls and BigQuery queries
    # for no benefit. Keep the first row seen per year and surface the
    # duplication loudly rather than silently absorbing it, matching this
    # project's established pattern of flagging real seed data issues.
    seen_years = set()
    deduped_rows = []
    duplicate_years = []
    for row in rows:
        fy = row.get("fiscal_year")
        if fy in seen_years:
            duplicate_years.append(fy)
            continue
        seen_years.add(fy)
        deduped_rows.append(row)

    if duplicate_years:
        print(
            f"  WARNING: {table_name} has duplicate rows for fiscal_year(s) "
            f"{sorted(set(duplicate_years))} -- using first row found for each. "
            f"This may indicate a real data quality issue worth checking directly."
        )
    rows = deduped_rows

    records = []
    for row in rows:
        fy_label = fiscal_year_label(row.get("fiscal_year"))
        if not fy_label:
            continue
        for col in numeric_columns:
            value = row.get(col)
            if value is None:
                continue
            records.append({
                "question": f"What was {prettify_column(col)} in {fy_label}?",
                "table": table_name,
                "column": col,
                "fiscal_year": fy_label,
                "expected_value": str(value),
            })

    print(f"{table_name}: {len(records)} ground-truth questions generated.")
    return records


def main():
    parser = argparse.ArgumentParser(description="Generate SQL-path ground truth from live BigQuery data.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    client = get_bq_client()
    all_records = []
    for table_name in MART_TABLES:
        all_records.extend(generate_ground_truth(client, table_name))

    with open(args.output, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_records)} total ground-truth questions to {args.output}")


if __name__ == "__main__":
    main()