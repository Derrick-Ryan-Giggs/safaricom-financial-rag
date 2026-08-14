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

from minsearch import Index
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import config
from ingestion.embed import EMBEDDING_DIM, OnnxEmbedder

RRF_K = 1  # matches the k value used in Module 4's evaluation
COLLECTION_NAME = "safaricom_chunks"


def load_chunks(chunk_glob: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(chunk_glob)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
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


def build_qdrant_client(records: list[dict]) -> QdrantClient:
    """
    Connects to a persistent Qdrant server (QDRANT_HOST/QDRANT_PORT env
    vars, defaulting to localhost:6333 -- a local Qdrant instance must be
    reachable there for this to work, e.g. via `docker compose up qdrant`).

    Indexes idempotently: if the collection already exists and its point
    count matches len(records), skips re-uploading entirely. This matters
    because unlike the old in-memory client (fresh every process, so
    always re-indexed), a persistent server means every app restart would
    otherwise re-upload all ~2,100 embeddings for no reason.
    """
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

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

    keyword_hits = minsearch_index.search(query, num_results=num_results * 2)
    keyword_ranking = [hit["chunk_id"] for hit in keyword_hits]

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