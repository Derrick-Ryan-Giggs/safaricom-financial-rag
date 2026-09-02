"""
tests/test_sql_query.py

Pure-function tests for retrieval/sql_query.py's safety and
hallucination guards. validate_sql_columns() and is_safe_select() are
exactly the kind of deterministic, no-LLM-needed logic where a real bug
(the "must be qualified with a dataset" failure, a real-but-wrong-scope
column being trusted) should have been caught here first.
"""

import pytest

from retrieval.sql_query import is_safe_select, validate_sql_columns

SCHEMA = {
    "mart_mpesa_growth_trends": {"fiscal_year", "mpesa_revenue_kes_bn", "mpesa_txn_value_kes_bn"},
    "mart_ke_et_trajectory": {"fiscal_year", "et_ebit_kes_bn", "ke_ebit_kes_bn"},
}


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT mpesa_revenue_kes_bn FROM mart_mpesa_growth_trends WHERE fiscal_year = 2025", True),
        ("SELECT fiscal_year, et_ebit_kes_bn FROM mart_ke_et_trajectory WHERE fiscal_year IN (2023, 2024)", True),
        ("SELECT * FROM mart_mpesa_growth_trends", True),
        ("SELECT SUM(mpesa_revenue_kes_bn) FROM mart_mpesa_growth_trends", True),
        ("SELECT made_up_column FROM mart_mpesa_growth_trends", False),
        ("SELECT mpesa_revenue_kes_bn FROM made_up_table", False),
        ("NOT EVEN SQL", False),
        # Documents a known, accepted limitation (see validate_sql_columns'
        # own docstring): an AS alias name isn't a real column and isn't in
        # SQL_SELECT_KEYWORDS either, so it gets treated as a hallucinated
        # reference even though the query is actually fine. Not something
        # to "fix" here -- this test exists so a change to that behavior
        # (intentional or not) shows up as a diff, not a surprise.
        ("SELECT SUM(mpesa_revenue_kes_bn) AS total FROM mart_mpesa_growth_trends", False),
    ],
)
def test_validate_sql_columns(sql, expected):
    assert validate_sql_columns(sql, SCHEMA) is expected


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT * FROM mart_mpesa_growth_trends", True),
        ("select fiscal_year from mart_ke_et_trajectory", True),
        ("DROP TABLE mart_mpesa_growth_trends", False),
        ("SELECT * FROM t; DELETE FROM t", False),
        ("UPDATE mart_mpesa_growth_trends SET x = 1", False),
    ],
)
def test_is_safe_select(sql, expected):
    assert is_safe_select(sql) is expected