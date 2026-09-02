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
from retrieval.rerank import rerank_chunks
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
# state an FY20 number directly, e.g. in a year-over-year table). The two
# new paragraphs below address those two patterns specifically.
# Confirmed bug (see evaluation/answer_quality_v1.jsonl, e.g. chunk_id
# b13467f2 and 8af6bdc6): the model sometimes retrieves a chunk that
# directly answers the question, then refuses anyway -- either because it
# wants a raw baseline number to compute a delta itself when the excerpt
# already states the change directly ("increased by 24%" IS the answer to
# "how did it change"), or because it trusts a chunk's source-report label
# over what the chunk's text actually says (an FY22 report excerpt can
# state an FY20 number directly, e.g. in a year-over-year table). The two
# paragraphs below address those two patterns specifically.
#
# Fourth paragraph added after evaluation/answer_quality_v4.jsonl (chunk_id
# 580c2810-ca27-400d-8e50-1878b52db81e, "What was the net taxation payable
# in FY23?"): the opposite failure mode from the other three -- instead of
# refusing despite evidence, the model answered confidently from a
# cash-flow-statement excerpt where PDF extraction had separated line-item
# labels from their numeric values, and picked a number (160,352.0, most
# likely "Operating cash flow") that was never actually paired with the
# label being asked about. The true "Net taxation payable" value wasn't
# even present in the excerpt. Placed after the "only refuse if nothing
# answers it" line, as an explicit exception to it -- otherwise that line's
# push toward finding an answer could override this one.
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

One exception to the above: some excerpts are financial tables where PDF extraction has
separated row labels from their numeric values -- a run of line-item names (e.g. "Operating
free cash flow, Net Interest paid/received, Net taxation payable") followed by a separate run
of numbers, with no label restated next to each one. Do not assume the number positioned
nearest a label, or the first number after the label list, is that label's value -- extraction
order does not reliably preserve which number belongs to which row, especially when the count
of numbers doesn't clearly match the count of labels. Only state a figure for a specific line
item when the excerpt makes the pairing explicit (e.g. "Net taxation payable was 45,017.6", or
the label sits immediately next to its number with nothing else between them). If a table
excerpt separates labels from values like this and you cannot confidently tell which number
belongs to the item being asked about, that counts as the excerpts not containing enough
information -- say so rather than citing the nearest number.
"""

# This is a THIRD confirmed instance of the refusal-despite-evidence bug
# (see evaluation/answer_quality_v1.jsonl for the first two: chunk_id
# b13467f2, 8af6bdc6). This one is more basic than either -- no cross-year
# label confusion, no computed delta needed. A chunk stated "Our 4G network
# is now available in Nairobi and Mombasa" verbatim, in response to "which
# two cities are currently available," and the model still refused. Three
# recurrences of the same failure mode in three different shapes means
# another prompt clause is unlikely to be the fix -- RAG_SYSTEM_PROMPT
# already has two targeted paragraphs for the first two patterns and this
# is a new one anyway. Instead: force one stricter, narrower re-read of the
# SAME chunks before accepting a refusal, rather than trying to prevent
# every possible refusal shape via prompt engineering alone.
VERIFY_REFUSAL_PROMPT = """You previously said the excerpts below don't contain enough information to
answer the question. Before accepting that, re-read every excerpt again, one at a time.

If ANY excerpt states an answer directly -- even a single short sentence, a bare fact, or a number
-- state that answer now, plainly, citing the fiscal year and page number, e.g. (FY19, p.3).

Only if, after this careful re-check, truly none of the excerpts state anything that answers the
question, respond with exactly this and nothing else: NO_ANSWER_FOUND
"""


def verify_no_answer(question: str, chunks: list[dict]) -> str | None:
    """
    Run only when the first generation pass looked like a refusal (see
    is_refusal). Forces a second, narrower-purpose pass over the exact
    same chunks -- "just check if any excerpt states this, quote it" is a
    meaningfully different (and often more reliable) task for a small
    model than the general "answer this question, cite sources,
    distinguish years" instruction used the first time. Returns the
    corrected answer if the re-check finds one, or None if the refusal
    holds even on careful re-reading.
    """
    prompt = build_prompt(question, chunks)
    client = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": VERIFY_REFUSAL_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    result = response.choices[0].message.content.strip()
    if result == "NO_ANSWER_FOUND" or is_refusal(result):
        return None
    return result


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


def answer_from_chunks_stream(question: str, chunks: list[dict]):
    """
    Streaming counterpart to answer_from_chunks(), for ui/app.py's chat
    rendering via st.write_stream() -- generation is the slowest-feeling
    part of a RAG answer, especially on the 120b model, and this lets the
    UI show tokens as they arrive instead of a blank spinner until the
    whole response is ready.

    Not a replacement for answer_from_chunks(): evaluation/answer_quality.py
    and any caller that needs the complete string back in one call (rather
    than rendering it live) should keep using answer_from_chunks(). Both
    build the identical prompt via build_prompt(), so this isn't a second
    diverging code path -- just a different way of consuming the same call.

    Yields text deltas (str), not full-text-so-far snapshots -- matches
    what st.write_stream() expects, since it concatenates each piece itself.
    """
    prompt = build_prompt(question, chunks)
    client = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    stream = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta


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

    num_results is the FINAL count after reranking, matching ui/app.py's
    run_rag_fallback pattern: retrieves a wider 2x candidate pool via
    hybrid_search, then reranks down to num_results via
    retrieval/rerank.py's cross-encoder. Kept in sync deliberately -- if
    this wrapper is what evaluation/answer_quality.py calls to generate
    answers for scoring, it needs to go through the same retrieve+rerank
    steps as production, or the eval isn't measuring what's actually live.
    """
    candidates = hybrid_search(
        question, records, minsearch_index, qdrant_client, embedder, num_results=num_results * 2
    )
    chunks = rerank_chunks(question, candidates, top_n=num_results)
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