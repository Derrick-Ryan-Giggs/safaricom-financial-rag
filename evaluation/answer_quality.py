"""
evaluation/answer_quality.py

Runs the actual RAG path (retrieval.rag.answer_question) against every
question in the curated ground truth benchmark, then judges each generated
answer against its reference_answer using the LLM. This measures answer
quality end-to-end -- not just whether retrieval found the right chunk
(that's evaluation/metrics.py's hit_rate_and_mrr), but whether the full
production pipeline produces a good answer from what it retrieves.

Resumable: if --output already exists, questions already present in it are
skipped, same pattern as evaluation/ground_truth.py. Each question costs
two Groq calls (one to generate the answer, one to judge it), so a 500-
question run is ~1000 calls -- expect a runtime on the same order as the
ground truth generation step.

Usage:
    uv run python -m evaluation.answer_quality --ground-truth evaluation/ground_truth_curated_v1.jsonl --chunks "embeddings/*.jsonl" --output evaluation/answer_quality_v1.jsonl
"""

import argparse
import json
import time

from openai import OpenAI, RateLimitError

import config
from evaluation.metrics import load_ground_truth
from ingestion.embed import OnnxEmbedder
from retrieval.rag import answer_question, is_refusal
from retrieval.search import build_minsearch_index, build_qdrant_client, load_chunks

JUDGE_PROMPT = """You are evaluating a generated answer against a known-correct reference answer.

Question: {question}
Reference answer: {reference_answer}
Generated answer: {generated_answer}

Does the generated answer convey the same key facts as the reference answer, even if worded
differently? Respond with exactly one word: RELEVANT, PARTLY_RELEVANT, or NOT_RELEVANT.
"""

MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 2


def call_with_backoff(fn, *args, **kwargs):
    """Generic retry wrapper for any Groq/OpenAI-client call in this module."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except RateLimitError:
            wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
            print(f"Rate limited, waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)
    print("Warning: giving up after max retries.")
    return None


def judge_answer(client: OpenAI, question: str, reference_answer: str, generated_answer: str) -> str:
    response = call_with_backoff(
    client.chat.completions.create,
    model=config.LLM_MODEL,
    messages=[{
        "role": "user",
        "content": JUDGE_PROMPT.format(
            question=question,
            reference_answer=reference_answer,
            generated_answer=generated_answer,
        ),
    }],
    temperature=0,
    max_tokens=300,
)
    
    if response is None:
        return "UNKNOWN"

    verdict = response.choices[0].message.content.strip().upper()
    return verdict if verdict in ("RELEVANT", "PARTLY_RELEVANT", "NOT_RELEVANT") else "UNKNOWN"


def load_processed_questions(output_path: str) -> set:
    processed = set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    processed.add(json.loads(line)["question"])
    except FileNotFoundError:
        pass
    return processed


def run_evaluation(ground_truth: list[dict], records: list[dict], output_path: str) -> None:
    minsearch_index = build_minsearch_index(records)
    qdrant_client = build_qdrant_client(records)
    embedder = OnnxEmbedder()
    client = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

    already_done = load_processed_questions(output_path)
    remaining = [pair for pair in ground_truth if pair["question"] not in already_done]

    if already_done:
        print(f"Resuming: {len(already_done)} questions already processed, {len(remaining)} remaining.")

    verdict_counts = {"RELEVANT": 0, "PARTLY_RELEVANT": 0, "NOT_RELEVANT": 0, "UNKNOWN": 0, "REFUSED": 0}

    with open(output_path, "a", encoding="utf-8") as outfile:
        for i, pair in enumerate(remaining):
            generated = call_with_backoff(
                answer_question,
                pair["question"], records, minsearch_index, qdrant_client, embedder,
            )
            if generated is None:
                print(f"Warning: skipping question after failed generation: {pair['question'][:80]}")
                continue

            if is_refusal(generated):
                verdict = "REFUSED"
            else:
                verdict = judge_answer(client, pair["question"], pair["reference_answer"], generated)

            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

            record = {
                "question": pair["question"],
                "reference_answer": pair["reference_answer"],
                "generated_answer": generated,
                "verdict": verdict,
                "chunk_id": pair["chunk_id"],
                "fiscal_year": pair["fiscal_year"],
            }
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            outfile.flush()

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(remaining)} remaining questions...")

            # Base pacing between question iterations (each iteration is 2
            # Groq calls: generation + judging). Backoff above absorbs the
            # rest if this isn't enough.
            time.sleep(0.5)

    print(f"Done. New verdicts this run: {verdict_counts}")


def summarize(output_path: str) -> dict:
    records = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    # is_refusal now lives in retrieval/rag.py -- it's used live by
    # ui/app.py to decide when to escalate to the web search fallback, so
    # this evaluation script imports the same detector instead of keeping
    # its own copy that could drift out of sync with production behavior.
    refusals = [r for r in records if is_refusal(r["generated_answer"])]
    attempted = [r for r in records if not is_refusal(r["generated_answer"])]

    def count_verdicts(subset):
        counts = {}
        for r in subset:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        return counts

    verdicts_all = count_verdicts(records)
    verdicts_attempted = count_verdicts(attempted)
    n_attempted = len(attempted)

    return {
        "total_questions": total,
        "refusal_count": len(refusals),
        "refusal_rate": len(refusals) / total if total else 0.0,
        "attempted_count": n_attempted,
        "verdict_counts_all_questions": verdicts_all,
        "verdict_counts_among_attempted_only": verdicts_attempted,
        "relevant_rate_among_attempted": verdicts_attempted.get("RELEVANT", 0) / n_attempted if n_attempted else 0.0,
        "relevant_or_partly_rate_among_attempted": (
            (verdicts_attempted.get("RELEVANT", 0) + verdicts_attempted.get("PARTLY_RELEVANT", 0)) / n_attempted
            if n_attempted else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG answer generation + LLM-as-judge over the curated ground truth benchmark."
    )
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--chunks", required=True, help="Glob pattern for embedded JSONL files.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)
    records = load_chunks(args.chunks)
    print(f"Loaded {len(ground_truth)} ground-truth questions, {len(records)} chunks.")

    run_evaluation(ground_truth, records, args.output)

    summary = summarize(args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()