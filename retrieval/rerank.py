"""
evaluation/tune_alpha.py

Grid-search retrieval/search.py's RRF alpha weighting against a ground-truth
file (Part 2, Retrieval #1). Builds the retrieval stack (minsearch index,
Qdrant client, embedder) ONCE, and retrieves ONCE per question -- alpha only
affects RRF fusion (cheap local math), not the embedding computation or the
Qdrant round-trip, so every alpha in the sweep is scored from the same
single retrieval pass rather than repeating it once per alpha value.

Usage (must run as a module, not a bare script path -- otherwise
`import evaluation.metrics` can't resolve):
    uv run python -m evaluation.tune_alpha --chunks "embeddings/*.jsonl"

Defaults to evaluation/ground_truth_v1.jsonl. That file is the ~6,219-row
raw candidate set (not a curated subset) -- pass --sample for a fast first
pass on a random subset before running the full file, or --ground-truth to
point at a smaller curated file instead.
"""

import argparse
import json
import random

from evaluation.metrics import hit_rate_and_mrr_multi_alpha, load_ground_truth
from retrieval.search import build_minsearch_index, build_qdrant_client, load_chunks
from ingestion.embed import OnnxEmbedder

DEFAULT_ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.7]
DEFAULT_GROUND_TRUTH = "evaluation/ground_truth_v1.jsonl"
SAMPLE_SEED = 42  # fixed so repeated --sample runs are comparable, not a fresh random subset each time


def main():
    parser = argparse.ArgumentParser(description="Grid-search RRF alpha against ground truth.")
    parser.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH, help=f"Path to ground-truth file (default: {DEFAULT_GROUND_TRUTH}).")
    parser.add_argument("--chunks", required=True, help="Glob pattern for embedded JSONL files.")
    parser.add_argument("--k", type=int, default=5, help="Top-k cutoff for Hit Rate/MRR (default 5).")
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS,
        help=f"Alpha values to grid-search (default: {DEFAULT_ALPHAS}).",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Randomly sample this many ground-truth pairs (fixed seed) instead of using all of them -- "
             "useful for a fast first pass on the ~6,219-row raw file before running the full set.",
    )
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)
    if args.sample is not None and args.sample < len(ground_truth):
        ground_truth = random.Random(SAMPLE_SEED).sample(ground_truth, args.sample)

    records = load_chunks(args.chunks)

    minsearch_index = build_minsearch_index(records)
    qdrant_client = build_qdrant_client(records)
    embedder = OnnxEmbedder()

    print(f"Scoring {len(ground_truth)} ground-truth pairs across alphas {args.alphas} (single retrieval pass each)...")

    results = hit_rate_and_mrr_multi_alpha(
        ground_truth, records, minsearch_index, qdrant_client, embedder,
        num_results=args.k, alphas=tuple(args.alphas),
    )

    results.sort(key=lambda r: (r["hit_rate"], r["mrr"]), reverse=True)

    print(json.dumps(results, indent=2))
    best = results[0]
    print(f"\nBest alpha: {best['alpha']} (Hit Rate@{best['k']}={best['hit_rate']:.4f}, MRR={best['mrr']:.4f})")


if __name__ == "__main__":
    main()