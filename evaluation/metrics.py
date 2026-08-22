"""
evaluation/metrics.py

Computes retrieval metrics (Hit Rate, MRR) against a ground-truth file, and
an LLM-as-judge score for generated answers, following the LLM Zoomcamp
Module 4 evaluation methodology.

Usage:
    uv run python -m evaluation.metrics --ground-truth evaluation/ground_truth.jsonl --chunks "embeddings/*.jsonl"
"""

import argparse
import json

from openai import OpenAI

import config
from ingestion.embed import OnnxEmbedder
from retrieval.search import (
    build_minsearch_index,
    build_qdrant_client,
    hybrid_search,
    load_chunks,
    reciprocal_rank_fusion,
    retrieve_rankings,
)

JUDGE_PROMPT = """You are evaluating whether a generated answer is relevant and correct given the
question and the excerpts it was based on.

Question: {question}
Answer: {answer}

Respond with exactly one word: RELEVANT, PARTLY_RELEVANT, or NOT_RELEVANT.
"""


def load_ground_truth(path: str) -> list[dict]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def hit_rate_and_mrr(
    ground_truth: list[dict],
    records: list[dict],
    minsearch_index,
    qdrant_client,
    embedder: OnnxEmbedder,
    num_results: int = 5,
    alpha: float = 0.6,
) -> dict:
    """
    Single-alpha version -- runs a full hybrid_search per question. Kept
    as-is for any caller scoring one fixed alpha. For grid-searching
    several alpha values against the SAME ground truth, use
    hit_rate_and_mrr_multi_alpha instead -- it retrieves once per question
    and reuses that across every alpha, rather than repeating the
    embedding + Qdrant round-trip once per alpha value.
    """
    hits = 0
    reciprocal_ranks = []

    for pair in ground_truth:
        results = hybrid_search(
            pair["question"], records, minsearch_index, qdrant_client, embedder,
            num_results=num_results, alpha=alpha,
        )
        result_ids = [r["chunk_id"] for r in results]

        if pair["chunk_id"] in result_ids:
            hits += 1
            rank = result_ids.index(pair["chunk_id"]) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(ground_truth)
    return {
        "k": num_results,
        "alpha": alpha,
        "hit_rate": hits / n if n else 0.0,
        "mrr": sum(reciprocal_ranks) / n if n else 0.0,
        "total_questions": n,
    }


def hit_rate_and_mrr_multi_alpha(
    ground_truth: list[dict],
    records: list[dict],
    minsearch_index,
    qdrant_client,
    embedder: OnnxEmbedder,
    num_results: int = 5,
    alphas: tuple[float, ...] = (0.5,),
) -> list[dict]:
    """
    Scores every alpha in `alphas` from ONE retrieval pass per question.
    alpha only affects RRF fusion (cheap local arithmetic) -- the
    embedding computation and the Qdrant network round-trip don't depend
    on it at all, so a 5-value sweep via repeated hit_rate_and_mrr() calls
    was doing 5x the retrieval work it needed to for the same result.
    This does the retrieval once and re-fuses per alpha instead.
    """
    hits = {alpha: 0 for alpha in alphas}
    reciprocal_ranks: dict[float, list[float]] = {alpha: [] for alpha in alphas}

    for pair in ground_truth:
        keyword_ranking, vector_ranking = retrieve_rankings(
            pair["question"], minsearch_index, qdrant_client, embedder,
            num_candidates=num_results * 2,
        )

        for alpha in alphas:
            fused_ids = reciprocal_rank_fusion(
                [keyword_ranking, vector_ranking], weights=[alpha, 1 - alpha]
            )[:num_results]

            if pair["chunk_id"] in fused_ids:
                hits[alpha] += 1
                rank = fused_ids.index(pair["chunk_id"]) + 1
                reciprocal_ranks[alpha].append(1.0 / rank)
            else:
                reciprocal_ranks[alpha].append(0.0)

    n = len(ground_truth)
    return [
        {
            "k": num_results,
            "alpha": alpha,
            "hit_rate": hits[alpha] / n if n else 0.0,
            "mrr": sum(reciprocal_ranks[alpha]) / n if n else 0.0,
            "total_questions": n,
        }
        for alpha in alphas
    ]


def llm_judge(question: str, answer: str) -> str:
    client = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=question, answer=answer)}],
        temperature=0,
        max_tokens=5,
    )
    return response.choices[0].message.content.strip().upper()


def main():
    parser = argparse.ArgumentParser(description="Compute Hit Rate and MRR against ground truth.")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--chunks", required=True, help="Glob pattern for embedded JSONL files.")
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5],
        help="One or more top-k cutoffs to evaluate, e.g. --k 2 5 10. Note: rag.py's production path uses k=5.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Keyword-ranking weight passed to hybrid_search, 0-1 (default 0.5, equal weighting).",
    )
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)
    records = load_chunks(args.chunks)

    minsearch_index = build_minsearch_index(records)
    qdrant_client = build_qdrant_client(records)
    embedder = OnnxEmbedder()

    results = [
        hit_rate_and_mrr(
            ground_truth, records, minsearch_index, qdrant_client, embedder,
            num_results=k, alpha=args.alpha,
        )
        for k in args.k
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()