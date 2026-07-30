"""
retrieval/search.py

Hybrid search over the embedded chunks: minsearch for keyword matching,
Qdrant (in-memory HNSW) for vector similarity, combined via Reciprocal Rank
Fusion (RRF), matching the approach from LLM Zoomcamp Module 4.

Usage:
    uv run python retrieval/search.py --chunks "embeddings/*.jsonl" --query "What drove M-PESA revenue growth?"
"""

import argparse
import glob
import json
import re

from minsearch import Index
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import config
from ingestion.embed import EMBEDDING_DIM, OnnxEmbedder

RRF_K = 1  # matches the k value used in Module 4's evaluation
COLLECTION_NAME = "safaricom_chunks"

# Matches "FY09", "FY 09" -- the corpus's own canonical 2-digit format.
_FY_SHORT = re.compile(r"\bFY\s?(\d{2})\b", re.IGNORECASE)
# Matches "FY2009", "FY 2009".
_FY_LONG = re.compile(r"\bFY\s?(20\d{2})\b", re.IGNORECASE)
# Matches a bare 4-digit year, e.g. "in 2009", "fiscal year 2009".
_YEAR_BARE = re.compile(r"\b(20\d{2})\b")


def load_chunks(chunk_glob: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(chunk_glob)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def extract_fiscal_year(query: str) -> str | None:
    """
    Pull a fiscal year reference out of a question and normalize it to the
    corpus's canonical 2-digit format (e.g. "FY09"), so it can be used to
    steer keyword search toward the right year's chunks.

    This exists because annual report body text almost never contains the
    literal string "FY09" -- it says "year ended 31 March 2009" -- so a
    question naming "FY09" has no strong keyword or semantic signal pointing
    at the right chunks on its own; it's just competing against 18 other
    years' worth of similarly-worded text (e.g. "revenue... increased...").
    """
    match = _FY_SHORT.search(query)
    if match:
        return f"FY{match.group(1)}"

    match = _FY_LONG.search(query)
    if match:
        return f"FY{match.group(1)[-2:]}"

    match = _YEAR_BARE.search(query)
    if match:
        return f"FY{match.group(1)[-2:]}"

    return None


def build_minsearch_index(records: list[dict]) -> Index:
    index = Index(
        text_fields=["text", "title", "topic"],
        keyword_fields=["fiscal_year", "category", "source_id"],
    )
    index.fit(records)
    return index


def build_qdrant_client(records: list[dict]) -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    points = [
        PointStruct(id=i, vector=record["embedding"], payload=record)
        for i, record in enumerate(records)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return client


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """
    Combine multiple ranked lists of chunk_ids into one, scoring each id by
    sum(1 / (k + rank)) across all lists it appears in.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def hybrid_search(
    query: str,
    records: list[dict],
    minsearch_index: Index,
    qdrant_client: QdrantClient,
    embedder: OnnxEmbedder,
    num_results: int = config.NUM_RESULTS,
) -> list[dict]:
    by_id = {r["chunk_id"]: r for r in records}

    target_year = extract_fiscal_year(query)

    # If the question names a specific fiscal year, run a year-filtered
    # keyword pass so those chunks aren't drowned out by ~18 other years'
    # worth of similarly-worded text. This is a HARD filter, but only on
    # the keyword side, and only as long as it finds something -- if no
    # chunk is tagged with that year at all, fall back to the unfiltered
    # pass rather than returning nothing from this ranking entirely.
    if target_year:
        keyword_hits = minsearch_index.search(
            query, filter_dict={"fiscal_year": target_year}, num_results=num_results * 2
        )
        if not keyword_hits:
            keyword_hits = minsearch_index.search(query, num_results=num_results * 2)
    else:
        keyword_hits = minsearch_index.search(query, num_results=num_results * 2)

    keyword_ranking = [hit["chunk_id"] for hit in keyword_hits]

    # Vector search stays UNFILTERED deliberately. A report labeled FY22 can
    # directly state an FY20 number (e.g. in a year-over-year comparison
    # table) -- that's exactly the cross-year reference retrieval/rag.py's
    # system prompt was updated to make use of. Hard-filtering vector search
    # by the asked-about year would silently break that fix: RRF fusion lets
    # a strong same-year keyword match and a strong cross-year semantic
    # match both surface, rather than picking one strategy globally.
    query_vector = embedder.embed([query])[0]
    vector_response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=num_results * 2,
    )
    vector_ranking = [hit.payload["chunk_id"] for hit in vector_response.points]

    fused_ids = reciprocal_rank_fusion([keyword_ranking, vector_ranking])[:num_results]
    return [by_id[cid] for cid in fused_ids if cid in by_id]


def main():
    parser = argparse.ArgumentParser(description="Hybrid search over embedded chunks.")
    parser.add_argument("--chunks", required=True, help="Glob pattern for embedded JSONL files, e.g. 'embeddings/*.jsonl'")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    records = load_chunks(args.chunks)
    print(f"Loaded {len(records)} chunks.")

    minsearch_index = build_minsearch_index(records)
    qdrant_client = build_qdrant_client(records)
    embedder = OnnxEmbedder()

    results = hybrid_search(args.query, records, minsearch_index, qdrant_client, embedder)
    for r in results:
        print(f"[{r['fiscal_year']} p{r['page_number']}] {r['text'][:150]}")


if __name__ == "__main__":
    main()