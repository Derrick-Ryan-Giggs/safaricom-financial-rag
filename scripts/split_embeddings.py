# scripts/split_embeddings.py
"""
One-off: splits embeddings/recovered.jsonl (consolidated, all 19 PDFs
mixed together) into per-PDF embeddings/{stem}.jsonl files, matching the
naming discover_pdfs() in airflow/dags/ingestion_dag.py expects.

No re-extraction, re-chunking, or re-embedding -- this is a pure
partition of already-correct data (the same 2,109 vectors already live
in Qdrant Cloud). Run once from the project root.
"""
import json
from collections import defaultdict
from pathlib import Path

INPUT = Path("embeddings/recovered.jsonl")
OUTPUT_DIR = Path("embeddings")

def stem_from_source_file(source_file: str) -> str:
    # source_file looks like "raw/2008-2009_results_announcement_and_investor_update.pdf"
    return Path(source_file).stem

def main():
    by_stem = defaultdict(list)
    total = 0
    with INPUT.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            stem = stem_from_source_file(record["source_file"])
            by_stem[stem].append(line)
            total += 1

    print(f"Read {total} chunks across {len(by_stem)} source PDFs")

    for stem, lines in sorted(by_stem.items()):
        out_path = OUTPUT_DIR / f"{stem}.jsonl"
        out_path.write_text("\n".join(lines) + "\n")
        print(f"  {out_path}  ({len(lines)} chunks)")

    print(f"\nDone. {len(by_stem)} files written to {OUTPUT_DIR}/")
    print("recovered.jsonl left untouched -- verify output, then decide "
          "whether to keep or remove it.")

if __name__ == "__main__":
    main()