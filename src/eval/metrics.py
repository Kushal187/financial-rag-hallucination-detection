"""Evaluation metrics for the FinQA RAG + hallucination-detection pipeline.

Retrieval metrics live here. `evaluate_retriever` is retriever-agnostic: it takes a
`retrieve(question, doc_id, k) -> ranked local_ids` function, so dense, BM25, and hybrid
are all scored by the same harness (and their numbers stay comparable).

Still to implement in later stages:
  - generation:  numeric-tolerant exact match (FinQA answers are mostly numbers)
  - detection:   precision / recall / f1
"""

from collections.abc import Callable, Sequence


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Per-query recall@k: fraction of gold evidence ids found in the top-k retrieved ids.

    `|gold ∩ top_k| / |gold|`. Returns 1.0 when there is no gold evidence (vacuously
    complete), so such rows don't drag the average down.
    """
    gold = set(gold_ids)
    if not gold:
        return 1.0
    top_k = set(retrieved_ids[:k])
    return len(gold & top_k) / len(gold)


def evaluate_retriever(
    retrieve_fn: Callable[[str, str, int], list[str]],
    qa_rows: list[dict],
    ks: Sequence[int] = (1, 5, 10),
) -> dict[int, dict[str, float]]:
    """Score a retriever over QA rows, reporting recall@k for each k in `ks`.

    `retrieve_fn(question, doc_id, k)` must return ranked chunk local_ids. Retrieval runs
    once per question at the largest k, then each k is a prefix slice — so cost is one
    retrieval per question, not one per k.

    Returns `{k: {"recall": mean per-query recall@k,
                  "full":   fraction of questions with ALL gold evidence in the top-k}}`.
    "full" matters for FinQA: a question is only answerable if every gold fact is retrieved.
    """
    ks = tuple(sorted(ks))
    max_k = max(ks)
    per_k: dict[int, list[float]] = {k: [] for k in ks}
    for row in qa_rows:
        ranked = retrieve_fn(row["question"], row["doc_id"], max_k)
        gold = row["gold_evidence_ids"]
        for k in ks:
            per_k[k].append(recall_at_k(ranked, gold, k))
    return {
        k: {
            "recall": sum(vals) / len(vals) if vals else 0.0,
            "full": sum(v == 1.0 for v in vals) / len(vals) if vals else 0.0,
        }
        for k, vals in per_k.items()
    }
