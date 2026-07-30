from retrieval.search import build_minsearch_index, build_qdrant_client, load_chunks, hybrid_search
from ingestion.embed import OnnxEmbedder

records = load_chunks("embeddings/*.jsonl")
idx = build_minsearch_index(records)
qc = build_qdrant_client(records)
embedder = OnnxEmbedder()

query = "What was Safaricom's revenue in FY09?"
results = hybrid_search(query, records, idx, qc, embedder, num_results=20)

for i, r in enumerate(results):
    tag = " <== TARGET" if "Revenue 70,480" in r["text"] else ""
    print(i, f"[{r['fiscal_year']} p{r['page_number']}]", r["text"][:70].replace("\n", " "), tag)
