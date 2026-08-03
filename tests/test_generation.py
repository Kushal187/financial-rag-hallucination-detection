"""Tests for the generation stage.

Pure logic for answer.py (no network, no disk) and mocked-network tests for prompt
building + generate_answer wiring (mirrors tests/test_hybrid.py's patch style).
"""

import json
from unittest.mock import patch

import pytest

from src.generation.answer import (
    answers_match,
    extract_final_answer,
    normalize_answer,
    parse_json_object,
)
from src.generation import generate_answer
from src.generation.prompts import build_messages, load_few_shot_examples


# --------------------------------------------------------------------------- #
# normalize_answer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected_kind",
    [
        ("3.8", "number"),
        ("$3.8 million", "number"),
        ("1,234.56", "number"),
        ("53%", "number"),
        ("-12", "number"),
        ("yes", "yes_no"),
        ("No.", "yes_no"),
        ("YES it did", "yes_no"),
        ("", "none"),
        (None, "none"),
        ("cannot determine", "text"),
    ],
)
def test_normalize_answer_kind(raw, expected_kind):
    _value, kind = normalize_answer(raw)
    assert kind == expected_kind


def test_normalize_answer_strips_currency_and_separators():
    assert normalize_answer("$1,234.56")[0] == 1234.56


def test_normalize_answer_yes_canonicalized():
    assert normalize_answer("Yes.")[0] == "yes"
    assert normalize_answer("no")[0] == "no"


# --------------------------------------------------------------------------- #
# answers_match — the unit-ambiguity hazard is the load-bearing case
# --------------------------------------------------------------------------- #
def test_match_percent_x100_either_direction():
    # gold_answer="53%" (display), exe=0.53232 (fraction) -> both pred forms accepted
    assert answers_match("53", "53%", 0.53232)
    assert answers_match("0.532", "53%", 0.53232)


def test_match_adi_hundredx():
    # gold_answer="380", exe=3.8 -> both forms accepted
    assert answers_match("3.8", "380", 3.8)
    assert answers_match("380", "380", 3.8)


def test_match_wrong_number_rejected():
    assert not answers_match("100", "380", 3.8)


def test_match_sign_flip_rejected():
    assert not answers_match("-3.8", "380", 3.8)


def test_match_yes_no():
    assert answers_match("yes", "", "yes")
    assert not answers_match("no", "", "yes")


def test_match_abstention_is_wrong():
    assert not answers_match("cannot determine", "380", 3.8)


def test_match_tolerance():
    # exe 0.53232 -> variants include 53.232; 53.4 is within 1%, 60 is not.
    assert answers_match("53.4", "53%", 0.53232)
    assert not answers_match("60", "53%", 0.53232)


# --------------------------------------------------------------------------- #
# extract_final_answer / parse_json_object
# --------------------------------------------------------------------------- #
def test_extract_cot_final_answer():
    assert extract_final_answer("reason...\nFINAL ANSWER: 3.8", "cot") == "3.8"


def test_extract_structured_json_field():
    raw = json.dumps({"reasoning": "r", "answer": "yes", "answer_type": "yes_no"})
    assert extract_final_answer(raw, "structured") == "yes"


def test_extract_structured_fenced_with_prose():
    raw = "```json\n{\"answer\": \"53%\"}\n```\nDone."
    assert extract_final_answer(raw, "structured") == "53%"


def test_extract_structured_bad_json_falls_back():
    assert extract_final_answer("not json\nFINAL ANSWER: 12", "structured") == "12"


def test_extract_freeform_takes_last_number():
    # verbose reasoning: the first number ($100) is intermediate, the answer is last.
    text = "The initial investment is $100. The final value is 193.5. Return = 93.5%"
    assert extract_final_answer(text, "zero_shot") == "93.5%"


def test_extract_freeform_keeps_percent_suffix():
    assert extract_final_answer("result is 24.7%", "zero_shot") == "24.7%"


def test_extract_freeform_yesno_opening():
    assert extract_final_answer("Yes, it did exceed the threshold.", "zero_shot") == "yes"


def test_extract_freeform_plain_number():
    assert extract_final_answer("the answer is 127.4", "zero_shot") == "127.4"


def test_parse_json_object_variants():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert parse_json_object("prose {\"a\": 1} more") == {"a": 1}
    assert parse_json_object("not json at all") is None
    assert parse_json_object("[1, 2]") is None  # array, not object


# --------------------------------------------------------------------------- #
# prompt building (pure given a context; no Groq, no disk)
# --------------------------------------------------------------------------- #
def test_build_zero_shot_structure():
    msgs, needs_json = build_messages("zero_shot", "q?", [("text_1", "the value is 5")])
    assert needs_json is False
    assert msgs[0]["role"] == "system"
    blob = " ".join(m["content"] for m in msgs)
    assert "the value is 5" in blob and "q?" in blob


def test_build_structured_needs_json_and_token():
    msgs, needs_json = build_messages("structured", "q?", [("text_1", "x")])
    assert needs_json is True
    assert "json" in " ".join(m["content"] for m in msgs).lower()


def test_build_cot_requests_final_answer():
    msgs, needs_json = build_messages("cot", "q?", [("text_1", "x")])
    assert needs_json is False
    assert "final answer" in " ".join(m["content"] for m in msgs).lower()


def test_build_few_shot_alternating_turns():
    examples = [
        {"question": "a", "evidence": "e1", "answer": "1"},
        {"question": "b", "evidence": "e2", "answer": "2"},
    ]
    msgs, needs_json = build_messages("few_shot", "q?", [("text_1", "x")], few_shot_examples=examples)
    assert needs_json is False
    # system, then (user, assistant) per example, then final user
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]


def test_build_unknown_strategy_raises():
    with pytest.raises(ValueError):
        build_messages("bogus", "q?", [("t", "x")])


def test_load_few_shot_examples_stratified(monkeypatch):
    # Avoid hitting disk / the 86k-chunk index: stub the data + content lookups.
    import src.generation.prompts as prompts

    rows = [
        {"question": "yn", "doc_id": "d", "gold_answer": "", "gold_answer_exe": "yes",
         "gold_evidence_ids": ["text_1"], "split": "train"},
        {"question": "pct", "doc_id": "d", "gold_answer": "50%", "gold_answer_exe": 0.5,
         "gold_evidence_ids": ["text_1"], "split": "train"},
        {"question": "n1", "doc_id": "d", "gold_answer": "7", "gold_answer_exe": 7.0,
         "gold_evidence_ids": ["text_1"], "split": "train"},
        {"question": "n2", "doc_id": "d", "gold_answer": "8", "gold_answer_exe": 8.0,
         "gold_evidence_ids": ["text_1"], "split": "train"},
    ]
    monkeypatch.setattr(prompts, "load_qa", lambda: rows)
    monkeypatch.setattr(prompts, "get_chunk_contents", lambda doc, ids: [(i, "ev") for i in ids])
    examples = prompts.load_few_shot_examples(n=3, seed=1)
    answers = [e["answer"] for e in examples]
    assert len(examples) == 3
    # one of each shape is represented (round-robin), not three numeric rows
    assert "yes" in answers                       # yes_no bucket
    assert "50%" in answers                        # percent bucket
    assert any(a in ("7", "8") for a in answers)   # numeric bucket


# --------------------------------------------------------------------------- #
# generate_answer wiring (mocked client)
# --------------------------------------------------------------------------- #
def test_generate_answer_cot(mock_chat):
    mock_chat.return_value = "reason\nFINAL ANSWER: 3.8"
    result = generate_answer("q?", [("text_1", "the value is 3.8")], strategy="cot")
    assert result["answer"] == "3.8"
    assert result["answer_type"] == "number"
    assert result["strategy"] == "cot"
    assert "latency_ms" in result


def test_generate_answer_structured_parses_json(mock_chat_json):
    mock_chat_json.return_value = json.dumps(
        {"reasoning": "r", "answer": "yes", "answer_type": "yes_no"}
    )
    result = generate_answer("q?", [("text_1", "x")], strategy="structured")
    assert result["answer"] == "yes"
    assert result["answer_type"] == "yes_no"


def test_generate_answer_structured_fallback(mock_chat_json):
    mock_chat_json.return_value = "not json\nFINAL ANSWER: 12"
    result = generate_answer("q?", [("text_1", "x")], strategy="structured")
    assert result["answer"] == "12"


@pytest.fixture
def mock_chat():
    with patch("src.generation.client.chat") as m:
        yield m


@pytest.fixture
def mock_chat_json():
    with patch("src.generation.client.chat_json") as m:
        yield m
