"""
ingestion/chunk.py

Splits extracted elements (from extract.py's JSONL output) into chunks sized
for embedding: up to CHUNK_SIZE tokens with CHUNK_OVERLAP tokens of overlap
between adjacent chunks, using the same tokenizer as the embedding model so
chunk sizes are accurate rather than approximated by word count.

Most extracted elements (titles, bullets, short paragraphs) are already well
under CHUNK_SIZE and pass through as a single chunk. Only long elements --
typically Table or dense NarrativeText blocks -- get split, using a
sentence-boundary-respecting sliding window.

Usage:
    uv run python ingestion/chunk.py --input processed/FY2019_Press_Commentary.jsonl --output chunks/FY2019_Press_Commentary.jsonl
"""

import argparse
import json
import re
import uuid
from pathlib import Path

from tokenizers import Tokenizer

import config

# Sentence-ish boundary: split after '.', '!', '?' followed by whitespace and
# a capital letter or open paren. Doesn't need to be perfect -- it only
# affects where a chunk boundary falls, not what stays in.
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

_tokenizer = None


def get_tokenizer() -> Tokenizer:
    """
    Lazily load the tokenizer matching the embedding model, so this module
    can be imported without immediately hitting the network.
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return _tokenizer


def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text).ids)


def split_sentences(text: str) -> list[str]:
    sentences = SENTENCE_SPLIT_PATTERN.split(text)
    return [s.strip() for s in sentences if s.strip()]


def pack_sentences_into_chunks(sentences: list[str], chunk_size: int, overlap: int) -> list[str]:
    """
    Greedily pack sentences into chunks up to `chunk_size` tokens. When a
    chunk is full, start the next one by carrying over the trailing
    sentences that fit within `overlap` tokens, so context isn't lost at
    the boundary.
    """
    chunks = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        if current and current_tokens + sentence_tokens > chunk_size:
            chunks.append(" ".join(current))

            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current):
                s_tokens = count_tokens(s)
                if overlap_tokens + s_tokens > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += s_tokens

            current = overlap_sentences
            current_tokens = overlap_tokens

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_element(record: dict, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[dict]:
    """
    Turn one extracted element record into one or more chunk records. Each
    output record is a copy of the input metadata with `text` replaced by
    the chunk's text and a new `chunk_id` assigned.
    """
    text = record["text"]
    token_count = count_tokens(text)

    if token_count <= chunk_size:
        chunk_texts = [text]
    else:
        sentences = split_sentences(text)
        chunk_texts = pack_sentences_into_chunks(sentences, chunk_size, overlap)

    chunk_records = []
    for i, chunk_text in enumerate(chunk_texts):
        chunk_record = dict(record)
        chunk_record["chunk_id"] = str(uuid.uuid4())
        chunk_record["text"] = chunk_text
        chunk_record["chunk_index"] = i
        chunk_record["chunk_count"] = len(chunk_texts)
        chunk_records.append(chunk_record)

    return chunk_records


def chunk_file(input_path: str, output_path: str) -> None:
    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0

    with open(input_path_obj, "r", encoding="utf-8") as infile, open(output_path_obj, "w", encoding="utf-8") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total_in += 1

            for chunk_record in chunk_element(record):
                outfile.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
                total_out += 1

    print(f"{input_path_obj.name}: {total_in} elements -> {total_out} chunks")


def main():
    parser = argparse.ArgumentParser(description="Chunk extracted elements for embedding.")
    parser.add_argument("--input", required=True, help="Path to input JSONL from extract.py.")
    parser.add_argument("--output", required=True, help="Path to write chunked JSONL.")
    args = parser.parse_args()

    chunk_file(args.input, args.output)


if __name__ == "__main__":
    main()