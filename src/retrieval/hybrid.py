"""Hybrid retriever: BM25 keyword search + dense vector search, fused by Weaviate's
native hybrid query (relative score fusion). `alpha=0` is pure BM25, `alpha=1` is pure
vector; tune on the dev split (see scripts/sweep_hybrid_alpha.py) and set HYBRID_ALPHA.

Same shared retriever contract as src.retrieval.dense: retrieve(question, doc_id, k), so
it's a drop-in swap for eval.metrics.evaluate_retriever and the generation stage.
"""

import os

from src.retrieval.embed import embed_texts
from src.retrieval.weaviate_store import hybrid_search

# 0.6 chosen via scripts/sweep_hybrid_alpha.py on the dev split: it beat both pure BM25
# (alpha=0) and pure dense (alpha=1) at every k (see docs/data_schema.md).
DEFAULT_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))


def retrieve(question: str, doc_id: str, k: int = 5, alpha: float = DEFAULT_ALPHA) -> list[str]:
    """Return the top-k chunk `local_id`s for `question`, scoped to its filing page,
    ranked by hybrid BM25+vector fusion at the given `alpha`."""
    query_vector = embed_texts([question])[0]
    hits = hybrid_search(question, query_vector, doc_id, k, alpha=alpha)
    return [hit["local_id"] for hit in hits]
