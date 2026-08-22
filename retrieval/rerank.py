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

RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"  # ~80MB, confirmed available in fastembed's reranker list

_encoder: TextCrossEncoder | None = None


def get_reranker() -> TextCrossEncoder:
    """
    Lazy singleton -- loading the ONNX model has real cost (comparable to
    OnnxEmbedder's own load), so this avoids repeating it per question.
    Call once at app startup (e.g. inside load_retrieval_stack) if you want
    the load cost paid upfront rather than on the first question.
    """
    global _encoder
    if _encoder is None:
        _encoder = TextCrossEncoder(model_name=RERANKER_MODEL)
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