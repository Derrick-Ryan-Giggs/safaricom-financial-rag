"""
scripts/reextract_all.py

Re-runs extract -> chunk -> embed for every PDF in raw/, deliberately NOT
using airflow/dags/ingestion_dag.py's discover_pdfs() -- that task skips
any PDF that already has a matching embeddings/*.jsonl output, which is
exactly wrong here: the whole point of this script is to force
reprocessing of files that already have (pre-table-fix) embeddings, now
that ingestion/extract.py's table_element_to_text() changes what Table
elements produce.

Runs each file's extract/chunk/embed as a subprocess (matching the DAG's
own `uv run python -m ...` convention), one file fully through the chain
before moving to the next -- easy to stop and resume partway (already-done
files can be skipped with --skip), and any one file's failure doesn't
lose progress on the others.

Usage:
    uv run python scripts/reextract_all.py
    uv run python scripts/reextract_all.py --skip FY23-Results-Booklet-11-May-2023  # already verified separately
    uv run python scripts/reextract_all.py --only FY24-Results-Booklet-9-May-2024   # just one file

After this finishes, force a Qdrant reseed (both local and Cloud) so the
updated embeddings actually get used -- see load_chunks()'s docstring in
retrieval/search.py: build_qdrant_client()'s idempotent check only compares
point COUNT, so an existing collection with the same number of points as
before won't auto-refresh with the new table text just because this ran.
    uv run python -c "from retrieval.search import build_qdrant_client; import os; os.environ['QDRANT_HOST']='localhost'; build_qdrant_client.__globals__['QdrantClient'](host='localhost', port=6333).delete_collection('safaricom_chunks')"
(or more simply: delete the collection via the Qdrant dashboard / Cloud
console, then just run the app / evaluation once so build_qdrant_client
reseeds it from the refreshed embeddings/*.jsonl files.)
"""

import argparse
import glob
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(module: str, *args: str) -> None:
    subprocess.run([sys.executable, "-m", module, *args], cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Re-run extract/chunk/embed for raw PDFs, forcing reprocessing even where embeddings exist."
    )
    parser.add_argument(
        "--skip", action="append", default=[], metavar="STEM",
        help="PDF filename stem to skip (e.g. a file already verified separately). Repeatable.",
    )
    parser.add_argument(
        "--only", action="append", default=[], metavar="STEM",
        help="Process only this PDF filename stem. Repeatable. If omitted, processes every PDF in raw/.",
    )
    args = parser.parse_args()

    pdf_paths = sorted(glob.glob(str(PROJECT_ROOT / "raw" / "*.pdf")))
    if not pdf_paths:
        print(f"No PDFs found under {PROJECT_ROOT / 'raw'}. Nothing to do.")
        return

    skip = set(args.skip)
    only = set(args.only)

    targets = []
    for pdf_path in pdf_paths:
        stem = Path(pdf_path).stem
        if stem in skip:
            print(f"Skipping {stem} (--skip)")
            continue
        if only and stem not in only:
            continue
        targets.append((pdf_path, stem))

    if not targets:
        print("No PDFs matched after applying --skip/--only. Nothing to do.")
        return

    print(f"Reprocessing {len(targets)} of {len(pdf_paths)} PDFs found in raw/:")
    for _, stem in targets:
        print(f"  - {stem}")
    print()

    failures = []
    for i, (pdf_path, stem) in enumerate(targets, start=1):
        processed_path = PROJECT_ROOT / "processed" / f"{stem}.jsonl"
        chunked_path = PROJECT_ROOT / "chunks" / f"{stem}.jsonl"
        embedded_path = PROJECT_ROOT / "embeddings" / f"{stem}.jsonl"

        print(f"[{i}/{len(targets)}] {stem}")
        try:
            run("ingestion.extract", "--pdf", pdf_path, "--output", str(processed_path))
            run("ingestion.chunk", "--input", str(processed_path), "--output", str(chunked_path))
            run("ingestion.embed", "--input", str(chunked_path), "--output", str(embedded_path))
        except subprocess.CalledProcessError as e:
            print(f"  FAILED at step for {stem}: {e}", file=sys.stderr)
            failures.append(stem)
            continue

    print()
    if failures:
        print(f"Done with {len(failures)} failure(s): {', '.join(failures)}")
        sys.exit(1)
    else:
        print(f"All {len(targets)} PDFs reprocessed successfully.")
        print(
            "Next: force a Qdrant reseed (local AND Cloud) -- delete the "
            "safaricom_chunks collection, then run the app once so "
            "build_qdrant_client() rebuilds it from the refreshed embeddings."
        )


if __name__ == "__main__":
    main()