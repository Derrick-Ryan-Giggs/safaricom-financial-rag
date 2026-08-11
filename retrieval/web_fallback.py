"""
retrieval/web_fallback.py

Last-resort web search fallback. Only meant to be invoked when the RAG path
over Safaricom's own annual reports produces a genuine refusal (see
retrieval.rag.is_refusal) -- i.e. the primary-source corpus doesn't cover
this at all. Deliberately NOT triggered on a partial-but-real RAG answer:
this project's value is being grounded in Safaricom's own filings, and
blending in unverified web content whenever an answer is merely "thin"
would quietly erode that.

Uses DuckDuckGo via the `ddgs` package -- no API key, no cost, matching
this project's cost-conscious setup. Caveat: ddgs is an unofficial wrapper
around DuckDuckGo's results, not a real API, so it can be rate-limited or
change behavior without notice. Acceptable for an occasional last-resort
fallback; not something to depend on as a primary path. If that becomes a
problem, Tavily (free tier, built for LLM/RAG use, needs a free signup) is
the natural upgrade -- swap search_web()'s implementation, everything else
here stays the same.

Usage:
    uv run python -m retrieval.web_fallback --question "Who is Safaricom's current CEO?"
"""

import argparse

from ddgs import DDGS
from openai import OpenAI

import config

WEB_SYSTEM_PROMPT = """You just performed a live web search for this question. Answer using
ONLY the search results below, which you yourself retrieved just now -- do not refer to them
as having been provided to you by someone else. Cite the source for each claim using its
number, e.g. (Source 2). If the results don't contain enough information to answer, say so
directly rather than guessing. Make clear to the reader that this information comes from a
general web search, not from Safaricom's own annual reports or regulatory filings -- it has
not been verified against a primary source.
"""

MAX_RESULTS = 5


def search_web(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        print(f"Warning: web search failed ({e}).")
        return []


def build_prompt(question: str, results: list[dict]) -> str:
    blocks = [
        f"[Source {i + 1}: {r.get('title', 'untitled')} -- {r.get('href', '')}]\n{r.get('body', '')}"
        for i, r in enumerate(results)
    ]
    context = "\n\n".join(blocks)
    return f"Web search results:\n{context}\n\nQuestion: {question}"


def web_search_answer(question: str) -> tuple[str, list[dict]]:
    """
    Returns (answer_text, results). Results are returned alongside the
    answer so the caller can display them distinctly from RAG's chunk
    Sources -- this is web content, not a primary-source filing, and that
    distinction should stay visible to whoever's reading the answer.
    """
    results = search_web(question)
    if not results:
        return "A web search didn't turn up anything for this either.", []

    prompt = build_prompt(question, results)
    client = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": WEB_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content, results


def main():
    parser = argparse.ArgumentParser(description="Answer a question via last-resort web search.")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    answer, results = web_search_answer(args.question)
    print(answer)
    print(f"\n({len(results)} web results used)")


if __name__ == "__main__":
    main()