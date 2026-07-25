"""
retrieval/rag.py

The RAG answer path: hybrid search over embedded chunks, build a grounded
prompt with citations, and generate an answer via Groq.

Usage:
    uv run python retrieval/rag.py --chunks "embeddings/*.jsonl" --question "What factors drove M-PESA growth?"
"""

import argparse

from openai import OpenAI

import config
from ingestion.embed import OnnxEmbedder
from retrieval.search import build_minsearch_index, build_qdrant_client, hybrid_search, load_chunks

RAG_SYSTEM_PROMPT = """You answer questions about Safaricom's financial history using ONLY the
provided excerpts from annual reports. Cite the fiscal year and page number for each claim,
e.g. (FY19, p.3). If the excerpts don't contain enough information to answer, say so directly
rather than guessing.
"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = [
        f"[{chunk['fiscal_year']}, p.{chunk['page_number']}] {chunk['text']}" for chunk in chunks
    ]
    context = "\n\n".join(context_blocks)
    return f"Excerpts:\n{context}\n\nQuestion: {question}"


def answer_question(
    question: str,
    records: list[dict],
    minsearch_index,
    qdrant_client,
    embedder: OnnxEmbedder,
    num_results: int = 5,
) -> str:
    # num_results defaults higher here than config.NUM_RESULTS (2) -- "why"
    # questions need more supporting context across fiscal years, and since
    # chunks here top out around 128 tokens (confirmed empirically), even
    # 5 chunks stays well within Groq's 6,000 TPM limit.
    chunks = hybrid_search(question, records, minsearch_index, qdrant_client, embedder, num_results=num_results)
    prompt = build_prompt(question, chunks)

    client = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Answer a question via RAG over embedded chunks.")
    parser.add_argument("--chunks", required=True, help="Glob pattern for embedded JSONL files.")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    records = load_chunks(args.chunks)
    minsearch_index = build_minsearch_index(records)
    qdrant_client = build_qdrant_client(records)
    embedder = OnnxEmbedder()

    answer = answer_question(args.question, records, minsearch_index, qdrant_client, embedder)
    print(answer)


if __name__ == "__main__":
    main()