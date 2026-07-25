"""Convert raw FinQA JSON into the processed corpus + answer-key files.

Usage:
    python scripts/build_corpus.py --splits train test

Writes to $DATA_PROCESSED_DIR (default data/processed/):
    finqa_chunks.jsonl   the corpus (embeddable content)
    finqa_qa.jsonl       the answer key (never embedded)
"""

import argparse
import json
import os
import sys
from collections import Counter

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.chunking import record_to_chunks, record_to_qa  # noqa: E402

load_dotenv()

RAW_DIR = os.getenv("DATA_RAW_DIR", "data/raw")
PROCESSED_DIR = os.getenv("DATA_PROCESSED_DIR", "data/processed")


def build(splits: list[str]) -> None:
    all_chunks: list[dict] = []
    all_qa: list[dict] = []

    for split in splits:
        path = os.path.join(RAW_DIR, "finqa", f"{split}.json")
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        print(f"{split}: {len(records)} records")
        for record in records:
            for chunk in record_to_chunks(record):
                chunk["split"] = split
                all_chunks.append(chunk)
            qa = record_to_qa(record)
            qa["split"] = split
            all_qa.append(qa)

    _assert_evidence_present(all_chunks, all_qa)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    _write_jsonl(os.path.join(PROCESSED_DIR, "finqa_chunks.jsonl"), all_chunks)
    _write_jsonl(os.path.join(PROCESSED_DIR, "finqa_qa.jsonl"), all_qa)

    types = Counter(c["chunk_type"] for c in all_chunks)
    print(
        f"\nWrote {len(all_chunks)} chunks "
        f"({types['text']} text, {types['table_row']} table_row) "
        f"and {len(all_qa)} questions to {PROCESSED_DIR}/"
    )


def _assert_evidence_present(chunks: list[dict], qa: list[dict]) -> None:
    """Every gold evidence id must have a matching chunk in the same document."""
    by_doc: dict[str, set[str]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], set()).add(c["local_id"])
    missing = []
    for row in qa:
        available = by_doc.get(row["doc_id"], set())
        for ev in row["gold_evidence_ids"]:
            if ev not in available:
                missing.append((row["id"], ev))
    if missing:
        raise AssertionError(
            f"{len(missing)} gold evidence ids have no matching chunk. "
            f"Examples: {missing[:5]}"
        )
    print(f"Integrity OK: all gold evidence ids resolve to a chunk ({len(qa)} questions).")


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["dev", "test"],
        help="FinQA splits to include (default: dev test -- the queried splits). "
        "Add 'train' when a stage needs it (few-shot examples, detector training).",
    )
    build(parser.parse_args().splits)


if __name__ == "__main__":
    main()
