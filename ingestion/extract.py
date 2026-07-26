"""
ingestion/extract.py

Extracts structured elements (narrative text, titles, tables, list items) from
Safaricom annual report PDFs using unstructured's hi_res strategy (YOLOX layout
detection + Tesseract OCR + table structure inference).

Usage:
    uv run python ingestion/extract.py --pdf raw/FY2024.pdf --output processed/FY2024.jsonl

Each output JSONL line corresponds to one extracted element and carries the
metadata schema defined in HANDOFF.md, minus chunk_id (assigned later in
chunk.py once elements are split into fixed-size chunks).
"""

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from unstructured.partition.pdf import partition_pdf

# Categories to keep -- everything else (Header, Footer, PageBreak, etc.) is
# treated as noise and dropped.
KEEP_CATEGORIES = {"NarrativeText", "Title", "Table", "ListItem"}

# Elements shorter than this (in characters, after cleaning) are dropped as
# noise -- stray page numbers, running heads that slipped through, etc.
MIN_CHARS = 20

# Matches FY-prefixed years in any of the real filename conventions seen in
# the source PDFs: FY14, FY18, FY_20, FY2019. Order matters -- 4-digit is
# tried before 2-digit so "FY2019" isn't truncated to "FY20".
FY_PATTERN = re.compile(r"FY[_-]?(\d{4}|\d{2}|\d{1})", re.IGNORECASE)

# Fallback for filenames with no "FY" prefix at all (e.g. the 2014-2015
# press commentary, which predates Safaricom's FY-prefixed naming).
FALLBACK_YEAR_PATTERN = re.compile(r"(20\d{2})")


def compute_source_id(pdf_path: str) -> str:
    """MD5 hash of the PDF path, used as a stable source_id across runs."""
    return hashlib.md5(pdf_path.encode("utf-8")).hexdigest()


def extract_fiscal_year(pdf_path: str) -> str:
    """
    Pull fiscal year from filename and normalize to canonical 2-digit form
    (e.g. FY2019 -> FY19), matching the convention already used in the
    BigQuery mart tables (dbt_rgiggs_mart) so RAG chunks and structured
    query results can be joined/filtered on the same fiscal_year value.

    Handles the real naming conventions seen across the source PDFs:
    FY14, FY18, FY_20 (underscore), FY2019 (4-digit).

    Falls back to scanning for a bare 4-digit year (e.g. "2014" in
    "Full_Year_2014-2015_...") for the one filename with no FY prefix.
    This fallback is a best guess -- it prints a warning so it can be
    manually verified rather than silently trusted.
    """
    name = Path(pdf_path).name
    match = FY_PATTERN.search(name)
    if match:
        digits = match.group(1)
        if len(digits) == 4:
            short = digits[-2:]
        elif len(digits) == 1:
            short = f"0{digits}"
        else:
            short = digits
        return f"FY{short}"

    # No FY-prefixed match; try fallback 4-digit year pattern.
    fallback = FALLBACK_YEAR_PATTERN.search(name)
    if fallback:
        year = fallback.group(1)
        short = year[-2:]
        print(
            f"WARNING: no FY-prefixed year found in '{name}'. "
            f"Inferred FY{short} from raw year '{year}' -- verify this manually."
        )
        return f"FY{short}"

    return "unknown"


def infer_topic(text: str) -> str:
    """
    Rough keyword-based topic tag, used as a coarse filter signal downstream.
    This is a first-pass heuristic, not ground truth -- the RAG router relies
    on hybrid search relevance, not on this label being perfect.
    """
    lowered = text.lower()
    if "m-pesa" in lowered or "mpesa" in lowered:
        return "mpesa"
    if "ethiopia" in lowered:
        return "ethiopia"
    if "customer" in lowered or "subscriber" in lowered:
        return "customers"
    if any(kw in lowered for kw in ("revenue", "ebit", "profit", "capex", "financial")):
        return "financials"
    return "general"


def clean_text(text: str) -> str:
    """
    NFKC-normalize and collapse whitespace. unstructured already strips a lot
    of layout noise during hi_res partitioning, so this is a light pass, not
    a full cleaning pipeline.
    """
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", normalized).strip()


def extract_pdf_elements(pdf_path: str, gcs_raw_uri: str | None = None) -> list[dict]:
    """
    Run unstructured's hi_res partition on a single PDF and return a list of
    metadata dicts, one per kept element, ready to be written as JSONL.
    """
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    source_id = compute_source_id(pdf_path)
    fiscal_year = extract_fiscal_year(pdf_path)
    created_date = datetime.now(timezone.utc).isoformat()
    gcs_uri = gcs_raw_uri or f"gs://safaricom-rag/raw/{pdf_path_obj.name}"

    print(f"Partitioning {pdf_path_obj.name} (hi_res, table structure inference)...")
    elements = partition_pdf(
        filename=str(pdf_path_obj),
        strategy="hi_res",
        infer_table_structure=True,
    )
    print(f"Raw elements from unstructured: {len(elements)}")

    records = []
    for element in elements:
        category = element.category
        if category not in KEEP_CATEGORIES:
            continue

        raw_text = element.text or ""
        text = clean_text(raw_text)
        if len(text) < MIN_CHARS:
            continue

        page_number = None
        if element.metadata and element.metadata.page_number is not None:
            page_number = element.metadata.page_number

        record = {
            "source_id": source_id,
            "title": f"Safaricom Annual Report {fiscal_year}",
            "created_date": created_date,
            "author": "Safaricom PLC",
            "document_type": "PDF",
            "topic": infer_topic(text),
            "page_number": page_number,
            "text": text,
            "category": category,
            "fiscal_year": fiscal_year,
            "source_file": str(pdf_path_obj),
            "gcs_uri": gcs_uri,
        }
        records.append(record)

    print(f"Kept elements after category + length filtering: {len(records)}")
    return records


def write_jsonl(records: list[dict], output_path: str) -> None:
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {output_path_obj}")


def main():
    parser = argparse.ArgumentParser(description="Extract elements from a Safaricom annual report PDF.")
    parser.add_argument("--pdf", required=True, help="Path to the local PDF file.")
    parser.add_argument("--output", required=True, help="Path to write the output JSONL file.")
    parser.add_argument(
        "--gcs-raw-uri",
        default=None,
        help="Override the gs:// URI recorded for this PDF's raw source.",
    )
    args = parser.parse_args()

    records = extract_pdf_elements(args.pdf, gcs_raw_uri=args.gcs_raw_uri)
    write_jsonl(records, args.output)


if __name__ == "__main__":
    main()