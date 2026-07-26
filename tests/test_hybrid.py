"""Wiring tests for the hybrid retriever, mocking out the network calls (embedding model
and Weaviate) so these run offline/without credentials."""

from unittest.mock import patch

import numpy as np

from src.retrieval import hybrid


def test_retrieve_wires_question_doc_id_k_alpha_and_extracts_local_ids():
    fake_vector = np.array([0.1, 0.2, 0.3], dtype="float32")
    fake_hits = [
        {"local_id": "table_3", "content": "...", "score": 0.9},
        {"local_id": "text_5", "content": "...", "score": 0.4},
    ]

    with patch("src.retrieval.hybrid.embed_texts", return_value=[fake_vector]) as mock_embed, \
         patch("src.retrieval.hybrid.hybrid_search", return_value=fake_hits) as mock_search:
        result = hybrid.retrieve("what is x?", "doc-1", k=2, alpha=0.7)

    mock_embed.assert_called_once_with(["what is x?"])
    mock_search.assert_called_once_with("what is x?", fake_vector, "doc-1", 2, alpha=0.7)
    assert result == ["table_3", "text_5"]


def test_retrieve_uses_default_alpha_when_not_specified():
    fake_vector = np.array([0.1], dtype="float32")
    with patch("src.retrieval.hybrid.embed_texts", return_value=[fake_vector]), \
         patch("src.retrieval.hybrid.hybrid_search", return_value=[]) as mock_search:
        hybrid.retrieve("q", "doc-1", k=5)

    _, kwargs = mock_search.call_args
    assert kwargs["alpha"] == hybrid.DEFAULT_ALPHA
