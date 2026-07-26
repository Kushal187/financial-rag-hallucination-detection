# src/retrieval/bm25.py
"""BM25 retrieval for FinQA examples.

Each example gets its own local BM25 index over its text sentences + table rows.
Candidate IDs follow the FinQA convention (text_N, table_N) so retrieval output
can be evaluated directly against `gold_inds`.
"""
from typing import List, Tuple
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """Lowercase + whitespace tokenize. Punctuation kept (matters for $, %, numbers)."""
    return str(text).lower().split()


def build_candidates(example: dict) -> List[Tuple[str, str]]:
    """Return [(candidate_id, candidate_text), ...] for one FinQA example.

    Text: pre_text + post_text with continuous numbering (text_0, text_1, ...).
    Table: every row of table_ori (table_0 = header, table_1+ = data rows).
    Table data rows attach column headers to each cell so year/column context
    is preserved when BM25 tokenizes.
    """
    candidates: List[Tuple[str, str]] = []

    # --- Text ---
    all_text = example.get("pre_text", []) + example.get("post_text", [])
    for i, sent in enumerate(all_text):
        candidates.append((f"text_{i}", sent))

    # --- Table ---
    table = example.get("table_ori", [])
    if table:
        headers = table[0]
        # Header row indexed as table_0
        candidates.append(("table_0", " | ".join(str(c) for c in headers)))
        # Data rows: attach header name to each non-label cell
        for i in range(1, len(table)):
            row = table[i]
            if not row:
                candidates.append((f"table_{i}", ""))
                continue
            parts = [str(row[0])]  # first cell is the row label
            for j in range(1, len(row)):
                header_name = str(headers[j]) if j < len(headers) else ""
                parts.append(f"{header_name}: {row[j]}")
            candidates.append((f"table_{i}", " | ".join(parts)))

    return candidates


class BM25Retriever:
    """BM25 retriever over the candidates of a single FinQA example.

    Usage:
        retriever = BM25Retriever(example)
        top_k = retriever.retrieve(example["qa"]["question"], k=5)
        # -> [("text_1", 3.42), ("table_2", 2.11), ...]
    """

    def __init__(self, example: dict):
        self.candidates = build_candidates(example)
        self.ids = [cid for cid, _ in self.candidates]
        self.texts = [t for _, t in self.candidates]
        tokenized_corpus = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]