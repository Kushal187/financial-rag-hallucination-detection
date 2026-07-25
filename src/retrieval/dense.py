"""Dense retriever: embed the question, search the vector store within its document.

This is the reference implementation of the shared retriever contract
`retrieve(question, doc_id, k) -> ranked chunk local_ids`. BM25 and hybrid retrievers
implement the same signature so they can be swapped into `eval.metrics.evaluate_retriever`
and the generation stage without changes.
"""

from src.retrieval.embed import embed_texts
from src.retrieval.weaviate_store import search


def retrieve(question: str, doc_id: str, k: int = 5) -> list[str]:
    """Return the top-k chunk `local_id`s for `question`, scoped to its filing page."""
    query_vector = embed_texts([question])[0]
    return [hit["local_id"] for hit in search(query_vector, doc_id, k)]
