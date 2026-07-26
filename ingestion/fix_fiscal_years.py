"""
ingestion/fix_fiscal_years.py

Corrects fiscal_year (and the dependent title field) in processed/*.jsonl
files where the document's own content disagrees with the filename-derived
value, using the same detection logic as verify_fiscal_years.py. Only
rewrites files where a genuine mismatch is found -- files where no
"YEAR ENDED" phrase could be found in the content are left untouched
rather than guessed at.

Usage:
    uv run python -m ingestion.fix_fiscal_years --processed-dir processed
"""

import argparse
import glob
import json
import re

YEAR_ENDED_PATTERN = re.compile(
    r"YEAR ENDED\s+(?:\d{1,2}\s+\w+\s+(\d{4})|\w+\s+\d{1,2},?\s+(\d{4}))",
    re.IGNORECASE,
)


def find_content_fiscal_year(records: list[dict]) -> str | None:
    for record in records[:10]:
        match = YEAR_ENDED_PATTERN.search(record["text"])
        if match:
            year = match.group(1) or match.group(2)
            return f"FY{year[-2:]}"
    return None


def main():
    parser = argparse.ArgumentParser(description="Fix fiscal_year/title fields based on document content.")
    parser.add_argument("--processed-dir", default="processed")
    args = parser.parse_args()

    fixed_files = []
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

        if content_fy is None or content_fy == filename_fy:
            continue

        print(f"Fixing {path}: {filename_fy} -> {content_fy}")
        for record in records:
            record["fiscal_year"] = content_fy
            record["title"] = record["title"].replace(filename_fy, content_fy)

        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        fixed_files.append((path, filename_fy, content_fy))

    if fixed_files:
        print(f"\nFixed {len(fixed_files)} file(s):")
        for path, old, new in fixed_files:
            print(f"  {path}: {old} -> {new}")
    else:
        print("No fixes needed.")


if __name__ == "__main__":
    main()