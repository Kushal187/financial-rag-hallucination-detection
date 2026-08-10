"""Cross-encoder reranking over a candidate pool from a first-stage retriever.

A cross-encoder reads (question, chunk) jointly, which is what BM25 and the bi-encoder
cannot do, but it is too slow to score a whole corpus. So a cheap retriever supplies
`pool` candidates and only those are re-scored. Recall@pool of the first stage is a hard
ceiling on recall after reranking.
"""

import os

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()

DEFAULT_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
DEFAULT_POOL = int(os.getenv("RERANK_POOL", "20"))
DEFAULT_BASE = os.getenv("RERANK_BASE", "hybrid")

_model: CrossEncoder | None = None


def get_model() -> CrossEncoder:
    """Load the cross-encoder once, shared across calls."""
    global _model
    if _model is None:
        _model = CrossEncoder(DEFAULT_MODEL)
    return _model


def _candidates(question: str, doc_id: str, n: int, base: str) -> list[tuple[str, str]]:
    """Fetch `n` first-stage candidates as ranked `(local_id, content)` pairs.

    Both branches reuse content the first stage already holds, so no second copy of the
    corpus is needed. Imports are inside the branch so `base="bm25"` needs no Weaviate.
    """
    if base == "hybrid":
        from src.retrieval.embed import embed_texts
        from src.retrieval.hybrid import DEFAULT_ALPHA
        from src.retrieval.weaviate_store import hybrid_search

        query_vector = embed_texts([question])[0]
        hits = hybrid_search(question, query_vector, doc_id, n, alpha=DEFAULT_ALPHA)
        return [(hit["local_id"], hit["content"]) for hit in hits]

    if base == "bm25":
        from src.retrieval import bm25

        by_id = {c["local_id"]: c["content"] for c in bm25.chunks_for_doc(doc_id)}
        return [(lid, by_id[lid]) for lid in bm25.retrieve(question, doc_id, n) if lid in by_id]

    raise ValueError(f"unknown rerank base {base!r} (expected 'hybrid' or 'bm25')")


def rank(
    question: str,
    doc_id: str,
    k: int = 5,
    pool: int = DEFAULT_POOL,
    base: str = DEFAULT_BASE,
) -> list[tuple[str, float]]:
    """Return the top-k `(local_id, score)` pairs, best first.

    The pool is scored in one batched call. Ties keep first-stage order, and `pool` is
    floored at `k` rather than raising partway through a run.
    """
    pool = max(pool, k)
    candidates = _candidates(question, doc_id, pool, base)
    if not candidates:
        return []

    scores = get_model().predict([(question, content) for _, content in candidates])
    ranked = sorted(
        ((local_id, float(score)) for (local_id, _), score in zip(candidates, scores)),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked[:k]


def retrieve(
    question: str,
    doc_id: str,
    k: int = 5,
    pool: int = DEFAULT_POOL,
    base: str = DEFAULT_BASE,
) -> list[str]:
    """Return the top-k chunk local_ids for `question`, reranked over a larger pool."""
    return [local_id for local_id, _ in rank(question, doc_id, k, pool, base)]
