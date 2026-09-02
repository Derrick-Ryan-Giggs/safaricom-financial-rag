"""
tests/test_router.py

Golden-set test for classify_question()'s SQL/RAG/OTHER routing, including
the exact schema-implausible case (net taxation payable) that caused a
confirmed live bug: taxation sounds financial and tabular, but isn't one
of the real mart columns, so it must route to RAG rather than wasting a
SQL query.

Two tiers, deliberately:
- The trivial-OTHER shortcut needs no LLM or BigQuery call at all -- always
  runs, catches a regression in that exact-match list for free.
- The full golden set calls the real Groq model, on purpose: mocking the
  LLM response would only prove a mock echoes itself back, not catch an
  actual router PROMPT regression, which is the whole point per the
  roadmap. This needs a real GROQ_API_KEY and network access, so it's
  skipped automatically when GROQ_API_KEY isn't set -- add it as a GitHub
  Actions repo secret (Settings -> Secrets and variables -> Actions) for
  this tier to run in CI; everything else in the suite runs without it.

BigQuery schema introspection is mocked either way (fixed_schema fixture
below) -- that's a separate live dependency from the LLM call, and mocking
it doesn't weaken what this test is actually checking (routing given a
known schema), it just removes a second live dependency this test doesn't
need in order to do its job.
"""

import os

import pytest

import config
from retrieval import router

_FAKE_SCHEMA = {
    "mart_mpesa_growth_trends": {"fiscal_year", "mpesa_revenue_kes_bn", "mpesa_txn_value_kes_bn"},
    "mart_ke_et_trajectory": {"fiscal_year", "et_ebit_kes_bn", "ke_ebit_kes_bn"},
    "mart_revenue_mix": {"fiscal_year", "voice_revenue_kes_bn", "data_revenue_kes_bn"},
}

GOLDEN_SET = [
    ("What was M-PESA revenue in FY2025?", "SQL"),
    ("Compare Ethiopia EBIT across FY23 and FY24.", "SQL"),
    ("What was the voice revenue in FY22?", "SQL"),
    ("What factors drove M-PESA growth?", "RAG"),
    ("Why did Ethiopia losses narrow?", "RAG"),
    ("Who is Safaricom's CEO?", "RAG"),
    ("What was the net taxation payable in FY23?", "RAG"),  # the confirmed regression case
    ("What was Safaricom's cash and cash equivalents as of 31-Mar-24?", "RAG"),
    ("What type of financial information does the Safaricom annual report include?", "RAG"),
    ("hi", "OTHER"),
    ("how are you", "OTHER"),
    ("thanks", "OTHER"),
    ("what can you do", "OTHER"),
    ("how do we get BigQuery to cover FY08-13", "OTHER"),
]


@pytest.fixture(autouse=True)
def fixed_schema(monkeypatch):
    monkeypatch.setattr(router, "_get_schema_columns", lambda: _FAKE_SCHEMA)


def test_trivial_other_shortcut_needs_no_llm_call():
    assert router.classify_question("hello") == "OTHER"
    assert router.classify_question("Testing!") == "OTHER"


@pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="golden-set routing needs a real GROQ_API_KEY (set as a repo secret in CI)",
)
@pytest.mark.parametrize("question, expected_route", GOLDEN_SET)
def test_classify_question_golden_set(monkeypatch, question, expected_route):
    monkeypatch.setattr(config, "GROQ_API_KEY", os.environ["GROQ_API_KEY"])
    assert router.classify_question(question) == expected_route