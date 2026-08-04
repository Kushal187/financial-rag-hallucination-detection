"""Wiring tests for the cross-encoder reranker.

The cross-encoder itself is always mocked — loading the real model would download
~90MB and make the suite depend on the network. What's under test is the plumbing:
that the pool is over-fetched, scored in one batch, sorted, and truncated to k.
"""

from unittest.mock import patch

import numpy as np
import pytest

from src.retrieval import rerank


class _FakeCrossEncoder:
    """Scores each (question, content) pair by a lookup keyed on the content string."""

    def __init__(self, scores_by_content: dict[str, float]):
        self.scores_by_content = scores_by_content
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return np.array([self.scores_by_content[content] for _, content in pairs])


def _candidates(*local_ids: str) -> list[tuple[str, str]]:
    """First-stage output where each chunk's content is just `content-<local_id>`."""
    return [(lid, f"content-{lid}") for lid in local_ids]


def test_rank_reorders_by_cross_encoder_score_and_truncates_to_k():
    model = _FakeCrossEncoder({"content-a": 0.1, "content-b": 0.9, "content-c": 0.5})

    with patch("src.retrieval.rerank._candidates", return_value=_candidates("a", "b", "c")), \
         patch("src.retrieval.rerank.get_model", return_value=model):
        result = rerank.rank("q", "doc-1", k=2, pool=3)

    assert result == [("b", 0.9), ("c", 0.5)]


def test_rank_scores_the_whole_pool_in_one_batched_call():
    model = _FakeCrossEncoder({f"content-{c}": 0.0 for c in "abcd"})

    with patch("src.retrieval.rerank._candidates", return_value=_candidates("a", "b", "c", "d")), \
         patch("src.retrieval.rerank.get_model", return_value=model):
        rerank.rank("what is x?", "doc-1", k=2, pool=4)

    # One forward pass per question, not one per candidate.
    assert len(model.calls) == 1
    assert model.calls[0] == [
        ("what is x?", "content-a"),
        ("what is x?", "content-b"),
        ("what is x?", "content-c"),
        ("what is x?", "content-d"),
    ]


def test_rank_overfetches_pool_not_k():
    """The whole point: ask the first stage for `pool`, hand back `k`."""
    model = _FakeCrossEncoder({f"content-{c}": 0.0 for c in "abcde"})

    with patch("src.retrieval.rerank._candidates", return_value=_candidates(*"abcde")) as mock_cand, \
         patch("src.retrieval.rerank.get_model", return_value=model):
        result = rerank.rank("q", "doc-1", k=2, pool=5)

    assert mock_cand.call_args[0][2] == 5
    assert len(result) == 2


def test_rank_floors_pool_at_k():
    """pool < k is a caller bug; absorb it rather than raising mid-run."""
    model = _FakeCrossEncoder({f"content-{c}": 0.0 for c in "abc"})

    with patch("src.retrieval.rerank._candidates", return_value=_candidates(*"abc")) as mock_cand, \
         patch("src.retrieval.rerank.get_model", return_value=model):
        rerank.rank("q", "doc-1", k=5, pool=2)

    assert mock_cand.call_args[0][2] == 5


def test_rank_keeps_first_stage_order_on_ties():
    model = _FakeCrossEncoder({"content-a": 0.5, "content-b": 0.5, "content-c": 0.5})

    with patch("src.retrieval.rerank._candidates", return_value=_candidates("a", "b", "c")), \
         patch("src.retrieval.rerank.get_model", return_value=model):
        result = rerank.rank("q", "doc-1", k=3, pool=3)

    assert [lid for lid, _ in result] == ["a", "b", "c"]


def test_rank_returns_empty_for_unknown_doc_without_calling_the_model():
    model = _FakeCrossEncoder({})

    with patch("src.retrieval.rerank._candidates", return_value=[]), \
         patch("src.retrieval.rerank.get_model", return_value=model):
        assert rerank.rank("q", "nope", k=5) == []

    assert model.calls == []


def test_retrieve_strips_scores_and_matches_the_shared_contract():
    model = _FakeCrossEncoder({"content-a": 0.1, "content-b": 0.9})

    with patch("src.retrieval.rerank._candidates", return_value=_candidates("a", "b")), \
         patch("src.retrieval.rerank.get_model", return_value=model):
        assert rerank.retrieve("q", "doc-1", k=2, pool=2) == ["b", "a"]


def test_unknown_base_raises():
    with pytest.raises(ValueError, match="unknown rerank base"):
        rerank._candidates("q", "doc-1", 5, "faiss")


def test_candidates_hybrid_base_requests_pool_from_weaviate():
    fake_vector = np.array([0.1, 0.2], dtype="float32")
    fake_hits = [
        {"local_id": "table_3", "content": "row three", "score": 0.9},
        {"local_id": "text_5", "content": "sentence five", "score": 0.4},
    ]

    with patch("src.retrieval.embed.embed_texts", return_value=[fake_vector]) as mock_embed, \
         patch("src.retrieval.weaviate_store.hybrid_search", return_value=fake_hits) as mock_search:
        result = rerank._candidates("what is x?", "doc-1", 20, "hybrid")

    mock_embed.assert_called_once_with(["what is x?"])
    # pool (20), not k, is what the first stage is asked for
    assert mock_search.call_args[0][3] == 20
    # content comes back on the hit, so no second corpus index is needed
    assert result == [("table_3", "row three"), ("text_5", "sentence five")]


def test_candidates_bm25_base_needs_no_weaviate():
    chunks = [
        {"local_id": "text_1", "content": "sentence one"},
        {"local_id": "table_0", "content": "header row"},
    ]

    with patch("src.retrieval.bm25.chunks_for_doc", return_value=chunks), \
         patch("src.retrieval.bm25.retrieve", return_value=["table_0", "text_1"]) as mock_retrieve:
        result = rerank._candidates("q", "doc-1", 10, "bm25")

    mock_retrieve.assert_called_once_with("q", "doc-1", 10)
    # first-stage ranking order is preserved into the candidate list
    assert result == [("table_0", "header row"), ("text_1", "sentence one")]


def test_candidates_bm25_base_drops_unresolvable_ids():
    with patch("src.retrieval.bm25.chunks_for_doc", return_value=[]), \
         patch("src.retrieval.bm25.retrieve", return_value=["text_1"]):
        assert rerank._candidates("q", "doc-1", 10, "bm25") == []
