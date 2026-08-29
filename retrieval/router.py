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
from retrieval.sql_query import get_bq_client, get_mart_schema

# Obvious non-financial inputs get classified for free, without spending a
# Groq call or depending on the model correctly following the OTHER
# instruction below every time. This is a narrow, exact-match list on
# purpose -- anything even slightly ambiguous still goes to the LLM.
_TRIVIAL_OTHER = {
    "hi", "hello", "hey", "hiya", "yo",
    "how are you", "hows it going", "how's it going",
    "thanks", "thank you", "ok", "okay", "test", "testing",
}

_schema_columns_cache: dict[str, set[str]] | None = None


def _get_schema_columns() -> dict[str, set[str]]:
    """
    Lazily introspects and caches the mart schema's column names for the
    process lifetime -- same singleton-cache pattern as rerank.py's
    get_reranker(). This is a cheap upfront relevance check only;
    sql_query.py still does its own full introspection (with sample rows)
    when actually generating SQL, since that needs richer context than a
    yes/no check does. Cache is stale until next cold start if the mart
    schema changes mid-process -- acceptable for a scales-to-zero Cloud
    Run service that gets a fresh process on most invocations anyway.
    """
    global _schema_columns_cache
    if _schema_columns_cache is None:
        _, _schema_columns_cache = get_mart_schema(get_bq_client())
    return _schema_columns_cache


def _schema_summary(schema_columns: dict[str, set[str]]) -> str:
    lines = []
    for table, columns in schema_columns.items():
        cols = ", ".join(sorted(c for c in columns if c != "fiscal_year"))
        lines.append(f"- `{table}`: {cols}")
    return "\n".join(lines)


ROUTER_SYSTEM_PROMPT_TEMPLATE = """You classify incoming questions into exactly one of three categories:

SQL: questions asking for specific numbers, metrics, or trends -- and ONLY if what's being asked
     is a plausible match for one of these actual columns (don't assume a column exists just
     because the question sounds like it should have one; if the specific metric or category
     isn't a reasonable match for anything listed, that's RAG, not SQL):

{schema_summary}

     e.g. "What was M-PESA revenue in FY2025?", "Compare Ethiopia EBIT across FY23 and FY24".

RAG: questions asking "why" or for narrative explanation, context, or qualitative detail found in
     annual report text (e.g. "What factors drove M-PESA growth?", "Why did Ethiopia losses narrow?"),
     OR any other genuine question specifically about Safaricom -- its business, leadership, history,
     products, competitors -- that isn't a plausible match for the schema above, even if you're not
     sure the annual reports actually cover it either (e.g. "Who is Safaricom's CEO?", "What was the
     net taxation payable in FY23?" -- taxation isn't one of the columns listed above). When in doubt
     between RAG and SQL, prefer RAG -- RAG will correctly say so if the reports don't cover it,
     while a wrong SQL guess wastes a query.

OTHER: only for input that is NOT a real question about Safaricom at all -- greetings ("hi", "how
       are you"), small talk, or questions about this application/tool itself rather than about
       Safaricom (e.g. "how do we get BigQuery to cover FY08-13", "what can you do").

Respond with exactly one word: SQL, RAG, or OTHER. Nothing else.
"""


def classify_question(question: str) -> str:
    normalized = question.strip().lower().rstrip("!.?")
    if normalized in _TRIVIAL_OTHER:
        return "OTHER"

    try:
        prompt = ROUTER_SYSTEM_PROMPT_TEMPLATE.format(
            schema_summary=_schema_summary(_get_schema_columns())
        )
    except Exception as e:
        # BigQuery introspection failing here (network, permissions) shouldn't
        # break routing -- fall back to a schema-blind prompt. sql_query.py's
        # validate_sql_columns() guard still protects correctness downstream
        # even without this upfront filter.
        print(f"Warning: could not fetch schema for router prompt: {e}")
        prompt = ROUTER_SYSTEM_PROMPT_TEMPLATE.format(schema_summary="(schema unavailable)")

    client = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",  # not config.LLM_MODEL -- classification doesn't need reasoning
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=300,   # fine again, this model doesn't burn tokens on hidden reasoning
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