"""
retrieval/router.py

Uses the LLM to classify an incoming question as either a structured (SQL)
question or an unstructured (RAG) question, before running the more
expensive retrieval path.

Usage:
    uv run python retrieval/router.py --question "What was M-PESA revenue in FY2025?"
"""

import argparse

from openai import OpenAI

import config

ROUTER_SYSTEM_PROMPT = """You classify questions about Safaricom's financial history into one of two categories:

SQL: questions asking for specific numbers, metrics, or trends that live in structured tables
     (e.g. "What was M-PESA revenue in FY2025?", "Compare Ethiopia EBIT across FY23 and FY24").

RAG: questions asking "why" or for narrative explanation, context, or qualitative detail found in
     annual report text (e.g. "What factors drove M-PESA growth?", "Why did Ethiopia losses narrow?").

Respond with exactly one word: SQL or RAG. Nothing else.
"""


def classify_question(question: str) -> str:
    client = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=5,
    )
    label = response.choices[0].message.content.strip().upper()
    return label if label in ("SQL", "RAG") else "RAG"  # default to RAG if the model returns anything unexpected


def main():
    parser = argparse.ArgumentParser(description="Classify a question as SQL or RAG.")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    label = classify_question(args.question)
    print(label)


if __name__ == "__main__":
    main()