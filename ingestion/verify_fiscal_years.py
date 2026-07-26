"""
ingestion/verify_fiscal_years.py

Cross-checks each processed JSONL file's filename-derived fiscal_year
against what the document's own text actually says (e.g. "YEAR ENDED
31 MARCH 2009" or "Fiscal Year Ended March 31, 2008"). Filenames across
2008-2026 use inconsistent conventions -- some starting-year, some
ending-year, some spelled out, some with no FY label at all -- so content
is the real source of truth; the filename regex is only a heuristic.

Does not auto-correct anything -- just reports mismatches so they can be
reviewed before deciding whether to fix and re-chunk/re-embed/re-generate
ground truth for the affected files.

Usage:
    uv run python -m ingestion.verify_fiscal_years --processed-dir processed
"""

import argparse
import glob
import json
import re

# Handles both date orderings seen in the source PDFs:
#   "YEAR ENDED 31 MARCH 2019"        (day month year)
#   "Fiscal Year Ended March 31, 2008" (month day, year)
YEAR_ENDED_PATTERN = re.compile(
    r"YEAR ENDED\s+(?:\d{1,2}\s+\w+\s+(\d{4})|\w+\s+\d{1,2},?\s+(\d{4}))",
    re.IGNORECASE,
)


def find_content_fiscal_year(records: list[dict]) -> str | None:
    """The 'year ended' statement consistently appears in the first few
    title/heading elements of each document, so only the first 10 records
    need checking."""
    for record in records[:10]:
        match = YEAR_ENDED_PATTERN.search(record["text"])
        if match:
            year = match.group(1) or match.group(2)
            return f"FY{year[-2:]}"
    return None


def main():
    parser = argparse.ArgumentParser(description="Verify filename-derived fiscal years against document content.")
    parser.add_argument("--processed-dir", default="processed")
    args = parser.parse_args()

    mismatches = []
    for path in sorted(glob.glob(f"{args.processed_dir}/*.jsonl")):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            continue

        filename_fy = records[0]["fiscal_year"]
        content_fy = find_content_fiscal_year(records)

        if content_fy is None:
            print(f"{path}: filename says {filename_fy}, no 'YEAR ENDED' phrase found -- unverified.")
        elif content_fy != filename_fy:
            print(f"{path}: MISMATCH -- filename says {filename_fy}, content says {content_fy}")
            mismatches.append((path, filename_fy, content_fy))
        else:
            print(f"{path}: OK -- filename and content agree ({filename_fy})")

    if mismatches:
        print(f"\n{len(mismatches)} mismatch(es) found. Review before chunking/embedding these files.")
    else:
        print("\nNo mismatches found among files where content could be verified.")


if __name__ == "__main__":
    main()