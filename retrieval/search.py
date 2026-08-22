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
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

import config
from ingestion.embed import EMBEDDING_DIM, OnnxEmbedder

RRF_K = 1  # matches the k value used in Module 4's evaluation
COLLECTION_NAME = "safaricom_chunks"
UPSERT_BATCH_SIZE = 256  # keeps each request under Qdrant's default 32MB payload limit

# Broadened to catch every confirmed raw variant: "FY8", "FY_8", "FY08",
# "FY20", "FY_20", "FY2020", "FY 25", etc. -- the underscore and the digit
# count are both optional/variable on purpose, since the real corpus mixes
# them for the same year (confirmed: FY_8 and FY8 both exist, same for
# FY_20 and FY20). normalize_fiscal_year() below is what actually collapses
# all of these to one value; this pattern's job is just to find candidates.
FISCAL_YEAR_PATTERN = re.compile(r"\bFY[_\s]?(\d{1,4})\b", re.IGNORECASE)


def normalize_fiscal_year(raw) -> str:
    """
    Canonicalizes any fiscal-year spelling to "FY" + the year's last two
    digits, no leading zero, no underscore -- e.g. "FY_8", "FY08", "FY8"
    all become "FY8"; "FY_20", "FY20", "FY2020" all become "FY20".

    CONFIRMED the raw corpus is inconsistent this way across source files
    (both "FY_8" and "FY8" exist for the same year; same for "FY_20" and
    "FY20"). `% 100` is what makes this handle both 4-digit and short forms
    uniformly: 2008 % 100 == 8, 8 % 100 == 8, 20 % 100 == 20 -- same
    canonical result regardless of how many digits the source used.
    """
    digits = re.sub(r"\D", "", str(raw))
    if not digits:
        return str(raw)
    return f"FY{int(digits) % 100}"


def extract_fiscal_year(question: str) -> str | None:
    """
    Best-effort extraction of an explicit fiscal year from the question
    text, e.g. "FY25", "FY2025", "FY_25" -> "FY25". Returns None on no
    confident match, so callers fall back to unfiltered search rather than
    guessing. Reuses normalize_fiscal_year so the question side and the
    chunk-data side always produce identical canonical strings.
    """
    match = FISCAL_YEAR_PATTERN.search(question)
    if not match:
        return None
    return normalize_fiscal_year(match.group(0))


def load_chunks(chunk_glob: str) -> list[dict]:
    """
    Normalizes each record's fiscal_year in place to the canonical "FYn"
    form as it's loaded -- see normalize_fiscal_year's docstring for why.

    OPERATIONAL NOTE: build_qdrant_client()'s idempotent check only
    compares point COUNT, not payload content -- an existing collection
    won't auto-refresh with normalized values just because this code
    changed. Delete the collection once after deploying this fix, then
    the next build_qdrant_client(records) call re-seeds it fresh.
    """
    records = []
    for path in sorted(glob.glob(chunk_glob)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if "fiscal_year" in record:
                        record["fiscal_year"] = normalize_fiscal_year(record["fiscal_year"])
                    records.append(record)
    return records


def build_minsearch_index(records: list[dict]) -> Index:
    index = Index(
        text_fields=["text", "title", "topic"],
        keyword_fields=["fiscal_year", "category", "source_id"],
    )
    index.fit(records)
    return index


import os

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_HTTPS = os.environ.get("QDRANT_HTTPS", "false").lower() == "true"


def build_qdrant_client(records: list[dict]) -> QdrantClient:
    """
    Connects to a persistent Qdrant server (QDRANT_HOST/QDRANT_PORT env
    vars, defaulting to localhost:6333). Indexes idempotently: if the
    collection already exists and its point count matches len(records),
    skips re-uploading entirely.
    """
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, https=QDRANT_HTTPS, timeout=60)

    already_indexed = False
    if client.collection_exists(COLLECTION_NAME):
        info = client.get_collection(COLLECTION_NAME)
        already_indexed = info.points_count == len(records)

    if not already_indexed:
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=i, vector=record["embedding"], payload=record)
            for i, record in enumerate(records)
        ]
        for i in range(0, len(points), UPSERT_BATCH_SIZE):
            batch = points[i : i + UPSERT_BATCH_SIZE]
            client.upsert(collection_name=COLLECTION_NAME, points=batch)

    return client


def retrieve_rankings(
    query: str,
    minsearch_index: Index,
    qdrant_client: QdrantClient,
    embedder: OnnxEmbedder,
    num_candidates: int,
    fiscal_year: str | None = None,
) -> tuple[list[str], list[str]]:
    """
    Runs ONLY the keyword and vector retrieval steps -- no RRF fusion.
    Fusion (and its alpha weighting) is cheap local arithmetic; embedding
    the query and the Qdrant network round-trip are the expensive,
    alpha-independent parts. Exists so a caller grid-searching alpha
    across many alpha values for the SAME question (e.g.
    evaluation/tune_alpha.py) can retrieve once and re-fuse many times,
    instead of repeating the embedding computation and network call once
    per alpha value -- a 5-value alpha sweep was otherwise doing 5x the
    retrieval work it needed to.
    """
    if fiscal_year is None:
        fiscal_year = extract_fiscal_year(query)

    keyword_filter = {"fiscal_year": fiscal_year} if fiscal_year is not None else None
    keyword_hits = minsearch_index.search(
        query, filter_dict=keyword_filter, num_results=num_candidates
    )
    keyword_ranking = [hit["chunk_id"] for hit in keyword_hits]

    query_vector = embedder.embed([query])[0]
    qdrant_filter = None
    if fiscal_year is not None:
        qdrant_filter = Filter(must=[FieldCondition(key="fiscal_year", match=MatchValue(value=fiscal_year))])

    vector_response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        query_filter=qdrant_filter,
        limit=num_candidates,
    )
    vector_ranking = [hit.payload["chunk_id"] for hit in vector_response.points]

    return keyword_ranking, vector_ranking


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = RRF_K, weights: list[float] | None = None
) -> list[str]:
    """
    Combine multiple ranked lists of chunk_ids into one, scoring each id by
    sum(weight * 1 / (k + rank)) across all lists it appears in.

    weights defaults to equal weighting (1.0 per ranking) if omitted, which
    reproduces the exact prior behavior and output order.
    """
    if weights is None:
        weights = [1.0] * len(rankings)

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank + 1)

    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def hybrid_search(
    query: str,
    records: list[dict],
    minsearch_index: Index,
    qdrant_client: QdrantClient,
    embedder: OnnxEmbedder,
    num_results: int = config.NUM_RESULTS,
    alpha: float = 0.6,
    fiscal_year: str | None = None,
) -> list[dict]:
    """
    alpha weights the keyword (minsearch) ranking; (1 - alpha) weights the
    vector (Qdrant) ranking. Default 0.6 is the confirmed best value from
    evaluation/tune_alpha.py's full run against all 6,219 rows of
    ground_truth_v1.jsonl at k=20 (matching this project's actual
    post-rerank candidate width): Hit Rate@20=0.7086, MRR=0.4354, both the
    best of {0.3, 0.4, 0.5, 0.6, 0.7} -- though only marginally ahead of
    0.5 (the old unweighted-equivalent default), so don't over-read this
    as a large tuning win. fiscal_year, if given (as a canonical "FYn"
    string), restricts both rankings to that year; if omitted,
    auto-extracts one from `query`.

    Delegates the actual retrieval to retrieve_rankings() -- this function
    now just adds fusion on top, so there's one source of truth for the
    keyword/vector retrieval logic shared with anything (like
    evaluation/tune_alpha.py) that needs to retrieve once and fuse many
    times with different alpha values.
    """
    by_id = {r["chunk_id"]: r for r in records}

    keyword_ranking, vector_ranking = retrieve_rankings(
        query, minsearch_index, qdrant_client, embedder,
        num_candidates=num_results * 2, fiscal_year=fiscal_year,
    )

    fused_ids = reciprocal_rank_fusion(
        [keyword_ranking, vector_ranking], weights=[alpha, 1 - alpha]
    )[:num_results]
    return [by_id[cid] for cid in fused_ids if cid in by_id]


def main():
    parser = argparse.ArgumentParser(description="Hybrid search over embedded chunks.")
    parser.add_argument("--chunks", required=True, help="Glob pattern for embedded JSONL files, e.g. 'embeddings/*.jsonl'")
    parser.add_argument("--query", required=True)
    parser.add_argument("--alpha", type=float, default=0.6, help="Keyword-ranking weight, 0-1 (default 0.6, confirmed best via tune_alpha.py)")
    parser.add_argument("--fiscal-year", type=str, default=None, help="Override auto-extracted fiscal year filter, e.g. FY25")
    args = parser.parse_args()

    records = load_chunks(args.chunks)
    print(f"Loaded {len(records)} chunks.")

    minsearch_index = build_minsearch_index(records)
    qdrant_client = build_qdrant_client(records)
    embedder = OnnxEmbedder()

    results = hybrid_search(
        args.query, records, minsearch_index, qdrant_client, embedder,
        alpha=args.alpha, fiscal_year=args.fiscal_year,
    )
    for r in results:
        print(f"[{r['fiscal_year']} p{r['page_number']}] {r['text'][:150]}")


if __name__ == "__main__":
    main()