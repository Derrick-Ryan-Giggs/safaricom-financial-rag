"""
ingestion/embed.py

Generates embeddings for chunked text using ONNX Runtime with the
Xenova/all-MiniLM-L6-v2 model (384-dim), matching the embedder already used
in llm-zoomcamp-code (Module 2). Downloads the ONNX model and tokenizer on
first use and caches them under EMBEDDING_MODEL_PATH.

Usage:
    uv run python ingestion/embed.py --input chunks/FY2019_Press_Commentary.jsonl --output embeddings/FY2019_Press_Commentary.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

import config

MODEL_REPO = "Xenova/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def ensure_model_files() -> tuple[str, str]:
    """
    Download the ONNX model and tokenizer files from the Xenova repo if not
    already cached under EMBEDDING_MODEL_PATH, and return their local paths.
    """
    model_dir = Path(config.EMBEDDING_MODEL_PATH)
    model_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = model_dir / "model.onnx"
    tokenizer_path = model_dir / "tokenizer.json"

    if not onnx_path.exists():
        print(f"Downloading ONNX model from {MODEL_REPO}...")
        downloaded = hf_hub_download(repo_id=MODEL_REPO, filename="onnx/model.onnx")
        onnx_path.write_bytes(Path(downloaded).read_bytes())

    if not tokenizer_path.exists():
        print(f"Downloading tokenizer from {MODEL_REPO}...")
        downloaded = hf_hub_download(repo_id=MODEL_REPO, filename="tokenizer.json")
        tokenizer_path.write_bytes(Path(downloaded).read_bytes())

    return str(onnx_path), str(tokenizer_path)


class OnnxEmbedder:
    """Thin wrapper around the ONNX Runtime session + tokenizer for embedding text."""

    def __init__(self):
        onnx_path, tokenizer_path = ensure_model_files()
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=256)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), EMBEDDING_DIM) array of mean-pooled, L2-normalized embeddings."""
        encodings = self.tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        last_hidden_state = outputs[0]  # (batch, seq_len, hidden_dim)

        mask = attention_mask[:, :, None].astype(np.float32)
        summed = (last_hidden_state * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        mean_pooled = summed / counts

        norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        return mean_pooled / np.clip(norms, a_min=1e-9, a_max=None)


def embed_file(input_path: str, output_path: str, batch_size: int = 32) -> None:
    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(input_path_obj, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    embedder = OnnxEmbedder()

    with open(output_path_obj, "w", encoding="utf-8") as outfile:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            texts = [r["text"] for r in batch]
            vectors = embedder.embed(texts)

            for record, vector in zip(batch, vectors):
                record["embedding"] = vector.tolist()
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{input_path_obj.name}: embedded {len(records)} chunks -> {output_path_obj}")


def main():
    parser = argparse.ArgumentParser(description="Generate ONNX embeddings for chunked text.")
    parser.add_argument("--input", required=True, help="Path to input chunked JSONL.")
    parser.add_argument("--output", required=True, help="Path to write JSONL with embeddings added.")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    embed_file(args.input, args.output, batch_size=args.batch_size)


if __name__ == "__main__":
    main()