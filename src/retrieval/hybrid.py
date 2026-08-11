"""Hybrid retriever: BM25 and vector search fused by Weaviate. alpha=0 is pure BM25,
alpha=1 is pure vector."""

import os

from src.retrieval.embed import embed_texts
from src.retrieval.weaviate_store import hybrid_search


DEFAULT_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))


def retrieve(question: str, doc_id: str, k: int = 5, alpha: float = DEFAULT_ALPHA) -> list[str]:
    """Return the top-k chunk `local_id`s for `question`, scoped to its filing page,
    ranked by hybrid BM25+vector fusion at the given `alpha`."""
    query_vector = embed_texts([question])[0]
    hits = hybrid_search(question, query_vector, doc_id, k, alpha=alpha)
    return [hit["local_id"] for hit in hits]
