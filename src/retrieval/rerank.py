"""Cross-encoder reranking over a candidate pool from a first-stage retriever.

BM25 and the dense bi-encoder both score a question and a chunk *independently* — the
chunk's representation is fixed before the question is seen. That is what makes them
cheap enough to run over a corpus, and also what makes them weak on FinQA, where the
distinguishing signal is usually an interaction ("which of these near-identical table
rows is the *2009* one?"). A cross-encoder reads (question, chunk) jointly in one
forward pass and models exactly that, but is far too slow to score a whole corpus.

The standard fix, used here: over-fetch `pool` candidates with the cheap hybrid
retriever, then re-score only those with the cross-encoder and keep the top `k`.
Recall@pool of the first stage is therefore a hard ceiling on Recall@k after reranking
— `pool` buys recall headroom at the cost of latency (swept in scripts/eval_rerank.py).

Same shared retriever contract as src.retrieval.dense: retrieve(question, doc_id, k), so
it's a drop-in swap for eval.metrics.evaluate_retriever and the generation stage.
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
    """Lazily load the cross-encoder once, shared across calls (mirrors embed.get_model)."""
    global _model
    if _model is None:
        _model = CrossEncoder(DEFAULT_MODEL)
    return _model


def _candidates(question: str, doc_id: str, n: int, base: str) -> list[tuple[str, str]]:
    """Fetch `n` first-stage candidates as ranked `(local_id, content)` pairs.

    Both branches read `content` straight out of what the first stage already holds —
    Weaviate returns it on the hit, BM25 has it in its cached chunk dicts — so reranking
    needs neither a second copy of the 86k-row corpus nor an import from src.generation
    (which would point the retrieval layer at the generation layer).

    Imported inside the branch so `base="bm25"` runs with no Weaviate credentials.
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
        # bm25.retrieve (not a fresh BM25Retriever) so the per-doc index cache is reused.
        return [(lid, by_id[lid]) for lid in bm25.retrieve(question, doc_id, n) if lid in by_id]

    raise ValueError(f"unknown rerank base {base!r} (expected 'hybrid' or 'bm25')")


def rank(
    question: str,
    doc_id: str,
    k: int = 5,
    pool: int = DEFAULT_POOL,
    base: str = DEFAULT_BASE,
) -> list[tuple[str, float]]:
    """Return the top-k `(local_id, cross-encoder score)` pairs, best first.

    The whole pool is scored in a single batched `predict` call — one forward pass per
    question, not one per candidate. Ties keep first-stage order (`sorted` is stable).

    `pool` is floored at `k`: asking for more results than candidates is a caller bug,
    but silently returning fewer is kinder than raising in the middle of an 883-question
    evaluation run.
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
    """Return the top-k chunk `local_id`s for `question`, scoped to its filing page,
    reranked by a cross-encoder over a larger first-stage pool."""
    return [local_id for local_id, _ in rank(question, doc_id, k, pool, base)]
