"""
tests/test_search.py

Pure-function tests for retrieval/search.py's fiscal-year and source-file
normalization. Both directly caused real bugs -- fiscal year spelling
inconsistencies breaking retrieval filtering, and a contaminated absolute
source_file path breaking the PDF deep-link -- so these test the exact
spellings/shapes already confirmed to exist in the corpus, not just an
arbitrary happy path.
"""

import pytest

from retrieval.search import extract_fiscal_year, normalize_fiscal_year, normalize_source_file


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("FY_8", "FY8"),
        ("FY08", "FY8"),
        ("FY8", "FY8"),
        ("FY_20", "FY20"),
        ("FY20", "FY20"),
        ("FY2020", "FY20"),
        ("FY2025", "FY25"),
        (2025, "FY25"),
        ("FY_2019", "FY19"),
    ],
)
def test_normalize_fiscal_year(raw, expected):
    assert normalize_fiscal_year(raw) == expected


def test_normalize_fiscal_year_no_digits_returns_input_unchanged():
    assert normalize_fiscal_year("unknown") == "unknown"


@pytest.mark.parametrize(
    "question, expected",
    [
        ("What was M-PESA revenue in FY2025?", "FY25"),
        ("Compare Ethiopia EBIT across FY_23 and FY24.", "FY23"),  # first match wins
        ("What is Safaricom's goal by 2030?", None),
    ],
)
def test_extract_fiscal_year(question, expected):
    assert extract_fiscal_year(question) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("raw/FY_10ResultspresentationAnnualResults.pdf", "raw/FY_10ResultspresentationAnnualResults.pdf"),
        (
            "/mnt/storage/Desktop/safaricom-financial-rag/raw/FY2019_Press_Commentary.pdf",
            "raw/FY2019_Press_Commentary.pdf",
        ),
    ],
)
def test_normalize_source_file(raw, expected):
    assert normalize_source_file(raw) == expected