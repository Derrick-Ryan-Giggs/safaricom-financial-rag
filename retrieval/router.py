"""
retrieval/router.py

Uses the LLM to classify an incoming question as SQL, RAG, or OTHER, before
running the more expensive retrieval path.

Usage:
    uv run python retrieval/router.py --question "What was M-PESA revenue in FY2025?"
"""

import argparse

from openai import OpenAI

import config

# Obvious non-financial inputs get classified for free, without spending a
# Groq call or depending on the model correctly following the OTHER
# instruction below every time. This is a narrow, exact-match list on
# purpose -- anything even slightly ambiguous still goes to the LLM.
_TRIVIAL_OTHER = {
    "hi", "hello", "hey", "hiya", "yo",
    "how are you", "hows it going", "how's it going",
    "thanks", "thank you", "ok", "okay", "test", "testing",
}

ROUTER_SYSTEM_PROMPT = """You classify incoming questions into exactly one of three categories:

SQL: questions asking for specific numbers, metrics, or trends that live in structured tables
     (e.g. "What was M-PESA revenue in FY2025?", "Compare Ethiopia EBIT across FY23 and FY24").

RAG: questions asking "why" or for narrative explanation, context, or qualitative detail found in
     annual report text (e.g. "What factors drove M-PESA growth?", "Why did Ethiopia losses narrow?").

OTHER: anything that is NOT actually asking about Safaricom's financial data -- greetings ("hi",
       "how are you"), small talk, or questions about this application/tool itself rather than
       about Safaricom (e.g. "how do we get BigQuery to cover FY08-13", "what can you do").

If a question is not clearly and specifically about Safaricom's financial history, M-PESA, or the
Kenya/Ethiopia business, choose OTHER. Do not pick SQL or RAG just because a word like "data" or
"BigQuery" appears in the sentence -- a question *about* the system is not a question *to* it.

Respond with exactly one word: SQL, RAG, or OTHER. Nothing else.
"""


def classify_question(question: str) -> str:
    normalized = question.strip().lower().rstrip("!.?")
    if normalized in _TRIVIAL_OTHER:
        return "OTHER"

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
    return label if label in ("SQL", "RAG", "OTHER") else "RAG"  # default to RAG if the model returns anything unexpected


def main():
    parser = argparse.ArgumentParser(description="Classify a question as SQL, RAG, or OTHER.")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    label = classify_question(args.question)
    print(label)


if __name__ == "__main__":
    main()