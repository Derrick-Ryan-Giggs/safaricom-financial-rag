"""
retrieval/rag.py

The RAG answer path: hybrid search over embedded chunks, build a grounded
prompt with citations, and generate an answer via Groq.

Usage:
    uv run python retrieval/rag.py --chunks "embeddings/*.jsonl" --question "What factors drove M-PESA growth?"
"""

import argparse
import re

from openai import OpenAI

import config
from ingestion.embed import OnnxEmbedder
from retrieval.search import build_minsearch_index, build_qdrant_client, hybrid_search, load_chunks

# Moved here from evaluation/answer_quality.py: originally a diagnostic-only
# regex for scoring the 500-question benchmark, now also used live by
# ui/app.py to decide when to escalate to the web search fallback (only on
# a genuine refusal, never on a partial-but-real answer -- see
# retrieval/web_fallback.py for why that line is drawn deliberately strict).
REFUSAL_PATTERN = re.compile(
    r"do(?:es)? ?n['o]t (?:contain|mention|provide|specify|state)"
    r"|no information (?:is )?(?:provided |given )?.{0,30}(?:about|on|regarding)"
    r"|could ?n['o]t find"
    r"|cannot answer|can'?t answer"
    r"|unable to (?:answer|determine|find)"
    r"|don'?t have enough information"
    r"|cannot (?:determine|verify|find)"
    r"|not (?:directly )?(?:mentioned|stated|specified) in",
    re.IGNORECASE,
)


def is_refusal(generated_answer: str) -> bool:
    return bool(REFUSAL_PATTERN.search(generated_answer))


# Confirmed bug (see evaluation/answer_quality_v1.jsonl, e.g. chunk_id
# b13467f2 and 8af6bdc6): the model sometimes retrieves a chunk that
# directly answers the question, then refuses anyway -- either because it
# wants a raw baseline number to compute a delta itself when the excerpt
# already states the change directly ("increased by 24%" IS the answer to
# "how did it change"), or because it trusts a chunk's source-report label
# over what the chunk's text actually says (an FY22 report excerpt can
# state an FY20 number directly, e.g. in a YoY comparison table). The two
# new paragraphs below address those two patterns specifically.
RAG_SYSTEM_PROMPT = """You answer questions about Safaricom's financial history using ONLY the
provided excerpts from annual reports. Cite the fiscal year and page number for each claim,
e.g. (FY19, p.3). If the excerpts don't contain enough information to answer, say so directly
rather than guessing.

Each excerpt is labeled with the fiscal year of the report it came from -- but annual reports
routinely restate prior-year and multi-year figures for comparison. An excerpt labeled FY22 may
directly state an FY20 number (e.g. in a year-over-year table). If an excerpt states a specific
figure for the year being asked about, use it and cite the year the number itself refers to --
not just the excerpt's source-report label.

If an excerpt directly states a change, growth rate, or percentage (e.g. "increased by 24%"),
treat that statement itself as the answer to a "how did X change" question. Do not withhold it
while waiting for two raw baseline figures to compute the delta yourself -- a stated relative
change is a complete answer, not an incomplete one.

Only say the excerpts don't contain enough information if none of them state a figure, change,
or fact that actually answers the question being asked.
"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = [
        f"[{chunk['fiscal_year']}, p.{chunk['page_number']}] {chunk['text']}" for chunk in chunks
    ]
    context = "\n\n".join(context_blocks)
    return f"Excerpts:\n{context}\n\nQuestion: {question}"


def answer_from_chunks(question: str, chunks: list[dict]) -> str:
    """
    Generate an answer from an already-retrieved set of chunks.

    Split out from answer_question() specifically so a caller that needs to
    both DISPLAY the retrieved chunks (ui/app.py's Sources expander) and use
    them for generation can do ONE hybrid_search call and guarantee both use
    the identical chunks. Previously, ui/app.py called hybrid_search()
    separately (num_results=5, for the Sources panel) from what
    answer_question() used internally (num_results=10, for generation) --
    two independent searches with different result-set sizes, not
    guaranteed to return the same top chunks. That meant the "Sources"
    shown to the user weren't necessarily what the model actually saw,
    which could look exactly like an unexplained refusal despite a source
    panel that clearly contains the answer (confirmed live: a "which two
    cities" question refused, while the displayed Sources included the
    exact chunk stating "available in Nairobi and Mombasa").
    """
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


def answer_question(
    question: str,
    records: list[dict],
    minsearch_index,
    qdrant_client,
    embedder: OnnxEmbedder,
    num_results: int = 10,
) -> str:
    """
    Convenience wrapper for CLI/standalone use (retrieves then answers in
    one call). ui/app.py should prefer calling hybrid_search() once and
    passing the result to answer_from_chunks() directly, so the same
    chunks are used for both the Sources display and generation -- see
    answer_from_chunks()'s docstring for why that distinction matters.

    num_results defaults higher here than config.NUM_RESULTS (2) -- "why"
    questions need more supporting context across fiscal years, and since
    chunks here top out around 128 tokens (confirmed empirically), even
    10 chunks stays well within Groq's 6,000 TPM limit.
    """
    chunks = hybrid_search(question, records, minsearch_index, qdrant_client, embedder, num_results=num_results)
    return answer_from_chunks(question, chunks)


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