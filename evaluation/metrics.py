"""
evaluation/metrics.py

Computes retrieval metrics (Hit Rate, MRR) against a ground-truth file, and
an LLM-as-judge score for generated answers, following the LLM Zoomcamp
Module 4 evaluation methodology.

Usage:
    uv run python evaluation/metrics.py --ground-truth evaluation/ground_truth.jsonl --chunks "embeddings/*.jsonl"
"""

import argparse
import json

from openai import OpenAI

import config
from ingestion.embed import OnnxEmbedder
from retrieval.search import build_minsearch_index, build_qdrant_client, hybrid_search, load_chunks

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
) -> dict:
    hits = 0
    reciprocal_ranks = []

    for pair in ground_truth:
        results = hybrid_search(pair["question"], records, minsearch_index, qdrant_client, embedder)
        result_ids = [r["chunk_id"] for r in results]

        if pair["chunk_id"] in result_ids:
            hits += 1
            rank = result_ids.index(pair["chunk_id"]) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(ground_truth)
    return {
        "hit_rate": hits / n if n else 0.0,
        "mrr": sum(reciprocal_ranks) / n if n else 0.0,
        "total_questions": n,
    }


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
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)
    records = load_chunks(args.chunks)

    minsearch_index = build_minsearch_index(records)
    qdrant_client = build_qdrant_client(records)
    embedder = OnnxEmbedder()

    metrics = hit_rate_and_mrr(ground_truth, records, minsearch_index, qdrant_client, embedder)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()