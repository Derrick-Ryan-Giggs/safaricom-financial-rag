"""
evaluation/curate_ground_truth.py

Takes the raw candidate pool from ground_truth.py and produces a smaller,
curated benchmark: exact-duplicate questions removed, then stratified by
fiscal_year so years with fewer chunks (e.g. early Press Commentary docs)
aren't drowned out by years with more chunks (e.g. later Results Booklets).

Usage:
    uv run python -m evaluation.curate_ground_truth --input evaluation/ground_truth_v1.jsonl --output evaluation/ground_truth_curated_v1.jsonl --target-size 500
"""

import argparse
import json
import random
from collections import defaultdict


def load_pairs(path: str) -> list[dict]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def deduplicate(pairs: list[dict]) -> list[dict]:
    """Remove exact-duplicate questions (case-insensitive, whitespace-normalized)."""
    seen = set()
    deduped = []
    for pair in pairs:
        key = " ".join(pair["question"].lower().split())
        if key not in seen:
            seen.add(key)
            deduped.append(pair)
    return deduped


def stratified_sample(pairs: list[dict], target_size: int, seed: int = 42) -> list[dict]:
    """
    Group pairs by fiscal_year and sample proportionally, capping any single
    year's share so it can't dominate the curated set -- without this, years
    with more chunks (e.g. FY26 at 249 chunks vs FY19 at 48) would flood the
    benchmark and skew Hit Rate/MRR toward how well retrieval performs on
    those years specifically.
    """
    random.seed(seed)
    by_year = defaultdict(list)
    for pair in pairs:
        by_year[pair["fiscal_year"]].append(pair)

    num_years = len(by_year)
    per_year_target = max(1, target_size // num_years)

    sampled = []
    for year_pairs in by_year.values():
        random.shuffle(year_pairs)
        sampled.extend(year_pairs[:per_year_target])

    # If under target after equal-per-year sampling (some years had fewer
    # pairs than per_year_target), top up randomly from the remaining pool.
    if len(sampled) < target_size:
        sampled_keys = {p["chunk_id"] + p["question"] for p in sampled}
        remaining = [p for p in pairs if p["chunk_id"] + p["question"] not in sampled_keys]
        random.shuffle(remaining)
        sampled.extend(remaining[: target_size - len(sampled)])

    random.shuffle(sampled)
    return sampled[:target_size]


def main():
    parser = argparse.ArgumentParser(description="Deduplicate and curate a ground-truth benchmark.")
    parser.add_argument("--input", required=True, help="Path to raw candidate pool from ground_truth.py.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-size", type=int, default=500)
    args = parser.parse_args()

    pairs = load_pairs(args.input)
    print(f"Loaded {len(pairs)} candidate pairs.")

    deduped = deduplicate(pairs)
    print(f"After deduplication: {len(deduped)} pairs.")

    curated = stratified_sample(deduped, args.target_size)
    years_represented = len(set(p["fiscal_year"] for p in curated))
    print(f"Curated benchmark: {len(curated)} pairs, stratified across {years_represented} fiscal years.")

    with open(args.output, "w", encoding="utf-8") as f:
        for pair in curated:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Wrote curated benchmark to {args.output}")


if __name__ == "__main__":
    main()