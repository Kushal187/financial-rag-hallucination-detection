"""Unit tests for retrieval metrics (pure, no network / no Weaviate)."""

from src.eval.metrics import evaluate_retriever, recall_at_k


def test_recall_at_k_partial():
    # one of two gold ids is in the top-3
    assert recall_at_k(["a", "b", "c"], ["a", "x"], k=3) == 0.5


def test_recall_at_k_full():
    assert recall_at_k(["a", "b"], ["a", "b"], k=2) == 1.0


def test_recall_at_k_respects_k():
    # gold "b" sits at rank 2, so it's missed at k=1
    assert recall_at_k(["a", "b"], ["b"], k=1) == 0.0


def test_recall_at_k_empty_gold_is_vacuously_full():
    assert recall_at_k(["a"], [], k=1) == 1.0


def test_evaluate_retriever_aggregates():
    qa = [
        {"question": "q1", "doc_id": "d1", "gold_evidence_ids": ["a"]},
        {"question": "q2", "doc_id": "d2", "gold_evidence_ids": ["x", "y"]},
    ]
    ranked = {"d1": ["a", "b"], "d2": ["x", "z"]}
    res = evaluate_retriever(lambda q, doc, k: ranked[doc][:k], qa, ks=(1, 2))

    # q1: recall=1.0 at both k. q2: only "x" of {x,y} retrieved -> 0.5 at both k.
    assert res[1]["recall"] == 0.75
    assert res[2]["recall"] == 0.75
    # only q1 has ALL its gold retrieved
    assert res[2]["full"] == 0.5
