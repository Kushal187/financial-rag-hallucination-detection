"""Unit tests for retrieval metrics (pure, no network / no Weaviate)."""

from src.eval.metrics import (
    collect_rankings,
    compare_retrievers,
    evaluate_retriever,
    mean_latency_ms,
    per_type_recall,
    recall_at_k,
    summarize_rankings,
)


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


# --- ranking records: retrieve once, aggregate many ways -------------------------

_QA = [
    {"id": "q1", "question": "q1", "doc_id": "d1", "gold_evidence_ids": ["table_1"]},
    {"id": "q2", "question": "q2", "doc_id": "d2", "gold_evidence_ids": ["table_2", "text_9"]},
]
_RANKED = {"d1": ["table_1", "text_0"], "d2": ["table_2", "text_3"]}


def _fake_retrieve(question, doc_id, k):
    return _RANKED[doc_id][:k]


def test_collect_rankings_retrieves_once_at_max_k():
    calls = []

    def counting_retrieve(question, doc_id, k):
        calls.append((doc_id, k))
        return _RANKED[doc_id][:k]

    records = collect_rankings(counting_retrieve, _QA, k=2)

    assert calls == [("d1", 2), ("d2", 2)]
    assert [r["id"] for r in records] == ["q1", "q2"]
    assert records[0]["ranked"] == ["table_1", "text_0"]
    assert records[0]["gold_evidence_ids"] == ["table_1"]
    assert records[0]["latency_ms"] >= 0.0


def test_summarize_rankings_matches_evaluate_retriever():
    records = collect_rankings(_fake_retrieve, _QA, k=2)
    assert summarize_rankings(records, ks=(1, 2)) == evaluate_retriever(_fake_retrieve, _QA, ks=(1, 2))


def test_per_type_recall_micro_averages_over_gold_items():
    records = collect_rankings(_fake_retrieve, _QA, k=2)
    by_type = per_type_recall(records, ks=(2,))

    # gold table items: table_1 (found), table_2 (found) -> 2/2
    assert by_type["table"][2] == 1.0
    # gold text items: text_9 only, and d2's top-2 is [table_2, text_3] -> 0/1
    assert by_type["text"][2] == 0.0


def test_per_type_recall_respects_k():
    records = collect_rankings(_fake_retrieve, _QA, k=2)
    # at k=1, d2's ranking is just [table_2]; table gold is still 2/2, text still 0/1
    assert per_type_recall(records, ks=(1,))["table"][1] == 1.0


def test_mean_latency_ms_of_empty_records_is_zero():
    assert mean_latency_ms([]) == 0.0


def test_compare_retrievers_renders_a_markdown_table():
    results = {
        "hybrid": {1: {"recall": 0.5, "full": 0.4}, 5: {"recall": 0.83, "full": 0.7}},
        "rerank": {1: {"recall": 0.6, "full": 0.5}, 5: {"recall": 0.85, "full": 0.75}},
    }
    table = compare_retrievers(results, ks=(1, 5), latency_by_name={"hybrid": 40.0, "rerank": 95.0})
    lines = table.splitlines()

    assert lines[0] == "| retriever | Recall@1 | Recall@5 | Full@5 | ms/query |"
    assert lines[2] == "| hybrid | 50.0% | 83.0% | 70.0% | 40 |"
    assert lines[3] == "| rerank | 60.0% | 85.0% | 75.0% | 95 |"


def test_compare_retrievers_omits_latency_column_when_absent():
    results = {"bm25": {1: {"recall": 0.49, "full": 0.3}}}
    assert compare_retrievers(results, ks=(1,)).splitlines()[0] == "| retriever | Recall@1 | Full@1 |"
