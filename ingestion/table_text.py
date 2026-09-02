"""
ingestion/table_text.py

Pure text-transformation helpers used by ingestion/extract.py, split out
into their own module specifically so they can be unit tested (see
tests/test_table_text.py) without needing `unstructured` installed --
extract.py's own `from unstructured.partition.pdf import partition_pdf`
pulls in the optional, heavy `[extract]` dependency group (torch included),
which the base test suite deliberately doesn't install.
"""

import re
import unicodedata

from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    """
    NFKC-normalize and collapse whitespace. unstructured already strips a lot
    of layout noise during hi_res partitioning, so this is a light pass, not
    a full cleaning pipeline.
    """
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", normalized).strip()


def table_element_to_text(element) -> str:
    """
    Convert a Table element to text that keeps each row's label paired with
    its own value(s), instead of unstructured's default `element.text` --
    which concatenates every cell's text in raw reading order and silently
    separates a row's label from its value whenever the table's visual
    layout isn't a strict left-to-right, top-to-bottom read.

    CONFIRMED bug this fixes (retrieval/rag.py's RAG_SYSTEM_PROMPT third
    paragraph; evaluation/answer_quality_v4.jsonl, chunk_id
    580c2810-ca27-400d-8e50-1878b52db81e): a cash-flow-statement table's
    line-item labels ("Operating free cash flow", "Net taxation payable",
    ...) came out of element.text as one run of labels followed by a
    separate run of numbers, with no reliable pairing between them -- the
    model picked whichever number sat nearest a label, which was often the
    wrong one, and the true "Net taxation payable" figure wasn't even
    identifiable in the excerpt.

    unstructured's hi_res strategy (extract_pdf_elements's
    infer_table_structure=True) attaches an HTML rendering of the detected
    table structure at element.metadata.text_as_html, which DOES preserve
    real row/column structure -- this parses that and re-emits one line per
    row as "label: value, value, ...", so every value stays next to the
    label it actually belongs to.

    Falls back to clean_text(element.text) if text_as_html is missing
    (older unstructured versions, or a table-structure-inference failure on
    one specific table) -- degraded for that table (the original bug can
    resurface just for it) but not broken.
    """
    metadata = getattr(element, "metadata", None)
    html = getattr(metadata, "text_as_html", None) if metadata is not None else None

    fallback = clean_text(getattr(element, "text", "") or "")
    if not html:
        return fallback

    try:
        soup = BeautifulSoup(html, "html.parser")
        rows = []
        for tr in soup.find_all("tr"):
            cells = [clean_text(cell.get_text(" ", strip=True)) for cell in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if not cells:
                continue
            rows.append(cells[0] if len(cells) == 1 else f"{cells[0]}: {', '.join(cells[1:])}")
    except Exception:
        return fallback

    return "\n".join(rows) if rows else fallback