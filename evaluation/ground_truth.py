"""
evaluation/ground_truth.py

Generates ground-truth question/chunk_id pairs for retrieval evaluation
(Hit Rate, MRR), following the LLM Zoomcamp Module 4 approach: for each
chunk, ask the LLM to generate a handful of questions that chunk would
answer, then record (question, chunk_id) as ground truth.

Usage:
    uv run python evaluation/ground_truth.py --chunks "embeddings/*.jsonl" --output evaluation/ground_truth.jsonl --questions-per-chunk 5

Note: the target of ~360 pairs mentioned in HANDOFF.md depends on the final
chunk count after chunk.py runs on all 12 PDFs -- tune --questions-per-chunk
(or sample a subset of chunks) once that count is known, rather than
assuming a fixed number here.
"""

import argparse
import json
import time

from openai import OpenAI

import config
from retrieval.search import load_chunks

QUESTION_GEN_PROMPT = """Based on this excerpt from a Safaricom annual report, generate {n} distinct
questions that this excerpt -- and only this excerpt -- would be the best source to answer.
Return ONLY a JSON array of strings, no explanation, no markdown fences.

Excerpt ({fiscal_year}, page {page_number}):
{text}
"""


def generate_questions_for_chunk(client: OpenAI, chunk: dict, n: int) -> list[str]:
    prompt = QUESTION_GEN_PROMPT.format(
        n=n,
        fiscal_year=chunk["fiscal_year"],
        page_number=chunk["page_number"],
        text=chunk["text"],
    )
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        questions = json.loads(raw)
        if isinstance(questions, list):
            return [str(q) for q in questions]
    except json.JSONDecodeError:
        pass

    print(f"Warning: could not parse questions for chunk {chunk.get('chunk_id')}: {raw[:200]}")
    return []


def generate_ground_truth(chunks: list[dict], questions_per_chunk: int, output_path: str) -> None:
    client = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

    total_pairs = 0
    with open(output_path, "w", encoding="utf-8") as outfile:
        for i, chunk in enumerate(chunks):
            questions = generate_questions_for_chunk(client, chunk, questions_per_chunk)

            for question in questions:
                record = {"question": question, "chunk_id": chunk["chunk_id"]}
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_pairs += 1

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(chunks)} chunks, {total_pairs} pairs so far...")

            # Groq rate limit (6,000 TPM) -- a short pause keeps this well under
            # the limit for a batch job like this. Adjust if you hit 429s.
            time.sleep(0.5)

    print(f"Wrote {total_pairs} ground-truth pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ground-truth Q&A pairs for retrieval evaluation.")
    parser.add_argument("--chunks", required=True, help="Glob pattern for embedded JSONL files.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--questions-per-chunk", type=int, default=5)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    print(f"Loaded {len(chunks)} chunks.")
    generate_ground_truth(chunks, args.questions_per_chunk, args.output)


if __name__ == "__main__":
    main()