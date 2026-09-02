"""
tests/test_extract.py

Unit tests for table_element_to_text() -- the fix for a confirmed bug
where unstructured's flattened Table `element.text` separates row labels
from their numeric values (full story in table_element_to_text's own
docstring, ingestion/extract.py).

These build a plain HTML string rather than actually partitioning a PDF or
importing unstructured at all -- the whole reason
`from unstructured.partition.pdf import partition_pdf` was moved to a
deferred import inside extract_pdf_elements() is so this file runs without
the heavy, optional `extract` dependency group (torch, unstructured)
installed.
"""

from ingestion.extract import clean_text, table_element_to_text


def test_table_element_to_text_pairs_rows_from_html():
    html = (
        "<table>"
        "<tr><td>Operating free cash flow</td><td>45,017.6</td></tr>"
        "<tr><td>Net Interest paid/received</td><td>-1,203.4</td></tr>"
        "<tr><td>Net taxation payable</td><td>-8,650.2</td></tr>"
        "</table>"
    )
    # This is exactly the failure shape confirmed live: unstructured's own
    # flattened element.text for this table ran every label together, then
    # every value together, with no pairing -- the input table_element_to_text
    # exists to route around, not to reproduce.
    raw_text = (
        "Operating free cash flow Net Interest paid/received Net taxation payable "
        "45,017.6 -1,203.4 -8,650.2"
    )

    result = table_element_to_text(raw_text, html)

    assert result.split("\n") == [
        "Operating free cash flow | 45,017.6",
        "Net Interest paid/received | -1,203.4",
        "Net taxation payable | -8,650.2",
    ]


def test_table_element_to_text_handles_multi_column_rows():
    html = (
        "<table>"
        "<tr><th>Metric</th><th>FY23</th><th>FY24</th></tr>"
        "<tr><td>Revenue</td><td>100</td><td>120</td></tr>"
        "</table>"
    )
    result = table_element_to_text("ignored raw text", html)
    assert result.split("\n") == [
        "Metric | FY23 | FY24",
        "Revenue | 100 | 120",
    ]


def test_table_element_to_text_drops_empty_cells_within_a_row():
    html = "<table><tr><td>Label</td><td></td><td>42</td></tr></table>"
    assert table_element_to_text("ignored", html) == "Label | 42"


def test_table_element_to_text_falls_back_without_html():
    raw_text = "Some table text with no html available"
    assert table_element_to_text(raw_text, None) == clean_text(raw_text)


def test_table_element_to_text_falls_back_when_html_has_no_complete_rows():
    # No closing </tr>, so the parser never completes a row -- degrades to
    # the old flattened-text behavior rather than raising.
    raw_text = "Fallback text"
    assert table_element_to_text(raw_text, "<table><tr><td>unterminated") == clean_text(raw_text)