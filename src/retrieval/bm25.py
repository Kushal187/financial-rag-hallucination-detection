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
    """Return the top-k chunk local_ids for `question`, scoped to `doc_id`.

    Matches the shared retriever contract used by dense.retrieve and hybrid.retrieve.
    """
    if doc_id not in _RETRIEVER_CACHE:
        chunks_by_doc = _get_chunks_by_doc()
        chunks = chunks_by_doc.get(doc_id)
        if not chunks:
            return []
        _RETRIEVER_CACHE[doc_id] = BM25Retriever(chunks)
    retriever = _RETRIEVER_CACHE[doc_id]
    return [local_id for local_id, _ in retriever.rank(question, k)]


def chunks_for_doc(doc_id: str) -> List[dict]:
    """Return the cached chunk dicts for `doc_id` (empty if unknown).

    Exposes the corpus this module already holds in memory, so callers that need
    `content`/`position` alongside a ranking (e.g. src.retrieval.rerank) don't have to
    build a second index over the same 86k rows.
    """
    return _get_chunks_by_doc().get(doc_id, [])

# # src/retrieval/bm25.py
# """BM25 retrieval for FinQA examples.

# Each example gets its own local BM25 index over its text sentences + table rows.
# Candidate IDs follow the FinQA convention (text_N, table_N) so retrieval output
# can be evaluated directly against `gold_inds`.
# """
# from typing import List, Tuple
# from rank_bm25 import BM25Okapi


# def tokenize(text: str) -> List[str]:
#     """Lowercase + whitespace tokenize. Punctuation kept (matters for $, %, numbers)."""
#     return str(text).lower().split()


# def build_candidates(example: dict) -> List[Tuple[str, str]]:
#     """Return [(candidate_id, candidate_text), ...] for one FinQA example.

#     Text: pre_text + post_text with continuous numbering (text_0, text_1, ...).
#     Table: every row of table_ori (table_0 = header, table_1+ = data rows).
#     Table data rows attach column headers to each cell so year/column context
#     is preserved when BM25 tokenizes.
#     """
#     candidates: List[Tuple[str, str]] = []

#     # --- Text ---
#     all_text = example.get("pre_text", []) + example.get("post_text", [])
#     for i, sent in enumerate(all_text):
#         candidates.append((f"text_{i}", sent))

#     # --- Table ---
#     table = example.get("table_ori", [])
#     if table:
#         headers = table[0]
#         # Header row indexed as table_0
#         candidates.append(("table_0", " | ".join(str(c) for c in headers)))
#         # Data rows: attach header name to each non-label cell
#         for i in range(1, len(table)):
#             row = table[i]
#             if not row:
#                 candidates.append((f"table_{i}", ""))
#                 continue
#             parts = [str(row[0])]  # first cell is the row label
#             for j in range(1, len(row)):
#                 header_name = str(headers[j]) if j < len(headers) else ""
#                 parts.append(f"{header_name}: {row[j]}")
#             candidates.append((f"table_{i}", " | ".join(parts)))

#     return candidates


# class BM25Retriever:
#     """BM25 retriever over the candidates of a single FinQA example.

#     Usage:
#         retriever = BM25Retriever(example)
#         top_k = retriever.retrieve(example["qa"]["question"], k=5)
#         # -> [("text_1", 3.42), ("table_2", 2.11), ...]
#     """

#     def __init__(self, example: dict):
#         self.candidates = build_candidates(example)
#         self.ids = [cid for cid, _ in self.candidates]
#         self.texts = [t for _, t in self.candidates]
#         tokenized_corpus = [tokenize(t) for t in self.texts]
#         self.bm25 = BM25Okapi(tokenized_corpus)

#     def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
#         scores = self.bm25.get_scores(tokenize(query))
#         ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
#         return ranked[:k]