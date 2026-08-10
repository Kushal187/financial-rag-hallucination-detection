"""Load the processed dataset, so every stage reads the schema from one place."""

import json
import os

from dotenv import load_dotenv

load_dotenv()

_PROCESSED_DIR = os.getenv("DATA_PROCESSED_DIR", "data/processed")
_CHUNKS_FILE = "finqa_chunks.jsonl"
_QA_FILE = "finqa_qa.jsonl"


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run scripts/build_corpus.py to generate it."
        )
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_chunks(path: str | None = None) -> list[dict]:
    """Load the corpus chunks (the searchable, embeddable content)."""
    return _read_jsonl(path or os.path.join(_PROCESSED_DIR, _CHUNKS_FILE))


def load_qa(path: str | None = None) -> list[dict]:
    """Load the answer-key rows (question + gold answer + gold evidence)."""
    return _read_jsonl(path or os.path.join(_PROCESSED_DIR, _QA_FILE))
