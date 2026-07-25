"""
evaluation/ground_truth.py

Generates ground-truth (question, reference_answer, chunk_id) triples for
retrieval and answer-quality evaluation, following the LLM Zoomcamp Module 4
approach: for each chunk, ask the LLM to generate a handful of questions
that chunk would answer, along with a reference answer grounded in that
chunk's text.

Resumable: if --output already exists, chunks whose chunk_id already appears
in it are skipped, so an interrupted run can continue without regenerating
pairs (and re-spending Groq quota) for chunks already done.

Treat this file's output as a large CANDIDATE pool, not the final
evaluation set -- run evaluation/curate_ground_truth.py afterward to
deduplicate and produce a smaller, year-balanced benchmark.

Usage:
    uv run python -m evaluation.ground_truth --chunks "embeddings/*.jsonl" --output evaluation/ground_truth_v1.jsonl --questions-per-chunk 3
"""

import argparse
import json
import time

from openai import OpenAI, RateLimitError

import config
from retrieval.search import load_chunks

QUESTION_GEN_PROMPT = """Based on this excerpt from a Safaricom annual report, generate {n} distinct
questions that this excerpt -- and only this excerpt -- would be the best source to answer. For each
question, also give a short reference answer grounded ONLY in this excerpt's text.

Return ONLY a JSON array of objects with keys "question" and "reference_answer". No explanation,
no markdown fences.

Excerpt ({fiscal_year}, page {page_number}):
{text}
"""

MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 2


def generate_questions_for_chunk(client: OpenAI, chunk: dict, n: int) -> list[dict]:
    """
    Returns a list of {"question": ..., "reference_answer": ...} dicts.
    Retries on rate limit errors with exponential backoff; gives up and
    returns an empty list after MAX_RETRIES so one stubborn chunk doesn't
    block the whole batch -- a resumable rerun will retry it later.
    """
    prompt = QUESTION_GEN_PROMPT.format(
        n=n,
        fiscal_year=chunk["fiscal_year"],
        page_number=chunk["page_number"],
        text=chunk["text"],
    )

    response = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            break
        except RateLimitError:
            wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
            print(f"Rate limited, waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)

    if response is None:
        print(f"Warning: giving up on chunk {chunk.get('chunk_id')} after {MAX_RETRIES} rate limit retries.")
        return []

    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [
                {
                    "question": str(item.get("question", "")),
                    "reference_answer": str(item.get("reference_answer", "")),
                }
                for item in parsed
                if item.get("question")
            ]
    except json.JSONDecodeError:
        pass

    print(f"Warning: could not parse questions for chunk {chunk.get('chunk_id')}: {raw[:200]}")
    return []


def load_processed_chunk_ids(output_path: str) -> set:
    """For resumability: read chunk_ids already present in an existing output file."""
    processed = set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    processed.add(json.loads(line)["chunk_id"])
    except FileNotFoundError:
        pass
    return processed


def generate_ground_truth(chunks: list[dict], questions_per_chunk: int, output_path: str) -> None:
    client = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

    already_done = load_processed_chunk_ids(output_path)
    remaining = [c for c in chunks if c["chunk_id"] not in already_done]

    if already_done:
        print(f"Resuming: {len(already_done)} chunks already processed, {len(remaining)} remaining.")

    total_pairs = 0
    with open(output_path, "a", encoding="utf-8") as outfile:
        for i, chunk in enumerate(remaining):
            pairs = generate_questions_for_chunk(client, chunk, questions_per_chunk)

            for pair in pairs:
                record = {
                    "question": pair["question"],
                    "reference_answer": pair["reference_answer"],
                    "chunk_id": chunk["chunk_id"],
                    "fiscal_year": chunk["fiscal_year"],
                    "page_number": chunk["page_number"],
                    "source_file": chunk["source_file"],
                }
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                outfile.flush()
                total_pairs += 1

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(remaining)} remaining chunks, {total_pairs} new pairs so far...")

            # Base pacing -- keeps well under Groq's 6,000 TPM even without
            # hitting a 429. The backoff above handles the case where this
            # isn't enough on its own.
            time.sleep(0.5)

    print(f"Wrote {total_pairs} new ground-truth pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ground-truth Q&A pairs for retrieval evaluation.")
    parser.add_argument("--chunks", required=True, help="Glob pattern for embedded JSONL files.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--questions-per-chunk", type=int, default=3)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    print(f"Loaded {len(chunks)} chunks.")
    generate_ground_truth(chunks, args.questions_per_chunk, args.output)


if __name__ == "__main__":
    main()