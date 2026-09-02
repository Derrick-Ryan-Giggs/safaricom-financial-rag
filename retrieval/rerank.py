"""
retrieval/rerank.py

Cross-encoder reranking step, applied after RRF fusion and before
generation. hybrid_search's fused rank reflects keyword/vector agreement,
not necessarily true query relevance -- reranking a wider candidate set
down to the final top-N is confirmed to be one of the highest
cost-to-benefit changes to a RAG pipeline's retrieval quality, and it runs
on CPU for free since fastembed (already a project dependency for
embeddings) ships ONNX cross-encoder rerankers directly.

Usage pattern (see ui/app.py's run_rag_fallback):
    candidates = hybrid_search(question, ..., num_results=20)
    sources = rerank_chunks(question, candidates, top_n=10)
"""

from fastembed.rerank.cross_encoder import TextCrossEncoder

import os
from pathlib import Path

RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"  # ~80MB, confirmed available in fastembed's reranker list

# fastembed defaults to caching downloaded models under the SYSTEM TEMP
# DIRECTORY (/tmp/fastembed_cache on Linux) unless told otherwise --
# confirmed via fastembed's own GitHub issues (#569, #681) this is a known,
# reported problem specifically because temp directories aren't persistent.
# Inside Docker, /tmp is part of the container's writable layer and gets
# wiped on every rebuild -- meaning every `docker compose up --build` was
# very likely re-triggering a fresh ~80-90MB download from Hugging Face on
# the next reranked question. Pointing this at a stable path (env-overridable,
# same pattern as QDRANT_HOST etc. elsewhere in this project) fixes it for
# local dev IF that path is also volume-mounted in docker-compose.yml so it
# survives rebuilds -- see the deployment note below for why this alone
# isn't enough for Cloud Run.
#
# Default changed from the hardcoded "/app/.fastembed_cache" (Docker-only --
# nothing at /app on a bare local `uv run` outside a container) to a
# repo-relative path, so this works out of the box in both places without
# needing FASTEMBED_CACHE_DIR set at all. The env var override still wins
# when set (e.g. Cloud Run, where a mounted volume or a different path is
# needed instead).
RERANKER_CACHE_DIR = os.environ.get(
    "FASTEMBED_CACHE_DIR", str(Path(__file__).parent.parent / ".fastembed_cache")
)

_encoder: TextCrossEncoder | None = None


def get_reranker() -> TextCrossEncoder:
    """
    Lazy singleton -- loading the ONNX model has real cost (comparable to
    OnnxEmbedder's own load), so this avoids repeating it per question
    within one process. Does NOT by itself prevent a re-download across
    process restarts -- that's what RERANKER_CACHE_DIR being a persistent,
    non-temp path is for. Call once at app startup (e.g. inside
    load_retrieval_stack) if you want the load cost paid upfront rather
    than on the first question.
    """
    global _encoder
    if _encoder is None:
        os.makedirs(RERANKER_CACHE_DIR, exist_ok=True)
        already_cached = os.path.isdir(RERANKER_CACHE_DIR) and len(os.listdir(RERANKER_CACHE_DIR)) > 0
        _encoder = TextCrossEncoder(
            model_name=RERANKER_MODEL,
            cache_dir=RERANKER_CACHE_DIR,
            # Confirmed separate fastembed bug (qdrant/fastembed#218): even
            # with a populated cache_dir, fastembed still makes a network
            # call to Hugging Face to check the repo before using cached
            # files, unless told not to. Once something's actually in the
            # cache dir, skip that check entirely -- no reason to touch the
            # network at all for a model that's already on disk.
            local_files_only=already_cached,
        )
    return _encoder


def rerank_chunks(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """
    Reranks `chunks` (as returned by hybrid_search) by cross-encoder
    relevance to `query`; returns the top_n most relevant.

    Pass in a wider candidate set than top_n (e.g. hybrid_search(...,
    num_results=20) then rerank_chunks(..., top_n=10)) -- reranking narrows
    a candidate pool down, it doesn't widen one, so top_n should be smaller
    than len(chunks) for this step to actually change anything.
    """
    if not chunks:
        return chunks

    documents = [c["text"] for c in chunks]
    scores = list(get_reranker().rerank(query, documents))

    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_n]]