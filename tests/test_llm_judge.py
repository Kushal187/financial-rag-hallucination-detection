"""Tests for the LLM-as-a-judge hallucination verifier.

parse_verdict robustness is pure; build_judge_messages + verify are tested with the
Groq client mocked out (no network), mirroring tests/test_hybrid.py's patch style.
"""

import json
from unittest.mock import patch

import pytest

from src.detection import build_judge_messages, parse_verdict, verify
from src.detection import llm_judge


# --------------------------------------------------------------------------- #
# parse_verdict — robust to every malformed shape a judge might emit
# --------------------------------------------------------------------------- #
def test_parse_verdict_clean():
    raw = json.dumps(
        {
            "supported": False,
            "partial": True,
            "category": "numeric_error",
            "confidence": 0.9,
            "reasoning": "off by a unit",
            "cited_evidence": ["table_1"],
        }
    )
    v = parse_verdict(raw)
    assert v["supported"] is False
    assert v["category"] == "numeric_error"
    assert v["confidence"] == 0.9
    assert v["cited_evidence"] == ["table_1"]


def test_parse_verdict_fenced_json():
    assert parse_verdict("```json\n{\"supported\": true}\n```")["supported"] is True


def test_parse_verdict_trailing_prose():
    assert parse_verdict('Here you go: {"supported": false} thanks')["supported"] is False


def test_parse_verdict_malformed_defaults_to_unsupported():
    v = parse_verdict("the model rambled with no json")
    assert v["supported"] is False
    assert v["category"] == "unsupported_claim"
    assert v["confidence"] == 0.0
    assert v["cited_evidence"] == []


def test_parse_verdict_missing_category_gets_default():
    # supported=false with no category -> default unsupported_claim
    assert parse_verdict('{"supported": false}')["category"] == "unsupported_claim"
    # supported=true with no category -> "supported"
    assert parse_verdict('{"supported": true}')["category"] == "supported"


def test_parse_verdict_confidence_clamped():
    assert parse_verdict('{"supported": false, "confidence": 5}')["confidence"] == 1.0
    assert parse_verdict('{"supported": false, "confidence": -1}')["confidence"] == 0.0


def test_parse_verdict_cited_evidence_coerced():
    v = parse_verdict('{"supported": true, "cited_evidence": [1, 2]}')
    assert v["cited_evidence"] == ["1", "2"]


# --------------------------------------------------------------------------- #
# build_judge_messages — must be grounded, JSON-instructed, and gold-free
# --------------------------------------------------------------------------- #
def test_judge_messages_include_json_and_evidence():
    msgs = build_judge_messages("what is x?", [("text_1", "the value is 5")], "5")
    blob = " ".join(m["content"] for m in msgs).lower()
    assert "json" in blob
    assert "the value is 5" in blob
    assert "what is x?" in blob
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_judge_messages_never_expose_gold_answer():
    # The judge must decide from evidence only; leaking gold would make it circular.
    msgs = build_judge_messages("q?", [("text_1", "ev")], "5")
    blob = json.dumps(msgs).lower()
    assert "gold" not in blob


# --------------------------------------------------------------------------- #
# verify — mocked end-to-end + abstention short-circuit
# --------------------------------------------------------------------------- #
def test_verify_supported(mock_chat_json):
    mock_chat_json.return_value = json.dumps(
        {"supported": True, "category": "supported", "confidence": 0.95,
         "reasoning": "ok", "cited_evidence": ["text_1"]}
    )
    v = verify("what is x?", [("text_1", "value is 5")], "5")
    assert v["supported"] is True
    assert v["category"] == "supported"
    assert v["latency_ms"] >= 0.0


def test_verify_hallucination(mock_chat_json):
    mock_chat_json.return_value = json.dumps(
        {"supported": False, "category": "numeric_error", "confidence": 0.8,
         "reasoning": "wrong number", "cited_evidence": []}
    )
    v = verify("q?", [("text_1", "ev")], "999")
    assert v["supported"] is False
    assert v["category"] == "numeric_error"


def test_verify_abstention_short_circuits(mock_chat_json):
    # An abstention is supported=True with category abstention and NO llm call.
    v = verify("q?", [("text_1", "ev")], "I cannot determine the answer from the provided evidence.")
    assert v["supported"] is True
    assert v["category"] == "abstention"
    mock_chat_json.assert_not_called()


@pytest.fixture
def mock_chat_json():
    with patch.object(llm_judge, "chat_json") as m:
        yield m
