"""BM25 retrieval over the shared FinQA chunks corpus."""
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """Lowercase + whitespace tokenize. Punctuation kept."""
    return str(text).lower().split()


def load_chunks_by_doc(
    jsonl_path: Union[str, Path],
    split: Optional[str] = None,
) -> Dict[str, List[dict]]:
    """Load chunks from JSONL and group them by doc_id."""
    grouped: Dict[str, List[dict]] = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            chunk = json.loads(line)
            if split is not None and chunk.get("split") != split:
                continue
            grouped[chunk["doc_id"]].append(chunk)
    for doc_id in grouped:
        grouped[doc_id].sort(key=lambda c: (c["chunk_type"], c["position"]))
    return dict(grouped)


class BM25Retriever:
    """BM25 retriever over the chunks of a single document."""

    def __init__(self, chunks: List[dict]):
        if not chunks:
            raise ValueError("BM25Retriever needs at least one chunk")
        self.chunks = chunks
        self.ids = [c["local_id"] for c in chunks]
        self.texts = [c["content"] for c in chunks]
        tokenized_corpus = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def rank(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Return the top-k (local_id, score) pairs."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]


_CHUNKS_BY_DOC: Optional[Dict[str, List[dict]]] = None
_RETRIEVER_CACHE: Dict[str, BM25Retriever] = {}
_CHUNKS_PATH = "data/processed/finqa_chunks.jsonl"


def _get_chunks_by_doc() -> Dict[str, List[dict]]:
    """Lazily load all chunks once, shared across calls."""
    global _CHUNKS_BY_DOC
    if _CHUNKS_BY_DOC is None:
        _CHUNKS_BY_DOC = load_chunks_by_doc(_CHUNKS_PATH)
    return _CHUNKS_BY_DOC


def retrieve(question: str, doc_id: str, k: int = 5) -> List[str]:
    """Return the top-k chunk local_ids for `question`, scoped to `doc_id`."""
    if doc_id not in _RETRIEVER_CACHE:
        chunks_by_doc = _get_chunks_by_doc()
        chunks = chunks_by_doc.get(doc_id)
        if not chunks:
            return []
        _RETRIEVER_CACHE[doc_id] = BM25Retriever(chunks)
    retriever = _RETRIEVER_CACHE[doc_id]
    return [local_id for local_id, _ in retriever.rank(question, k)]


def chunks_for_doc(doc_id: str) -> List[dict]:
    """Return the cached chunk dicts for `doc_id` (empty if unknown)."""
    return _get_chunks_by_doc().get(doc_id, [])
