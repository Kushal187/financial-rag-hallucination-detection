"""Tests for the labeling rule in scripts/build_detection_set.py.

This rule decides which answers count as hallucinations, so if it's wrong every number
in Stage 4 is wrong too — and it wouldn't crash, it would just quietly give bad results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_detection_set import declined, label_row  # noqa: E402
from score_detection import near_miss, score  # noqa: E402


def make_row(retrieved, correct, abstained=False, answer="42.0"):
    return {
        "retrieved_ids": retrieved,
        "correct": correct,
        "abstained": abstained,
        "answer": answer,
    }


def test_right_answer_with_the_evidence_is_supported():
    supported, reason = label_row(make_row(["table_1"], True), ["table_1"], k=5)
    assert supported is True
    assert reason == "correct"


def test_wrong_answer_with_the_evidence_is_a_hallucination():
    # The model had everything it needed and still got it wrong.
    supported, reason = label_row(make_row(["table_1"], False), ["table_1"], k=5)
    assert supported is False
    assert reason == "wrong_despite_evidence"


def test_wrong_answer_without_the_evidence_is_a_fabrication():
    # It had nothing to work from and still produced a number. This is the clearest
    # kind of hallucination, and the one a detector most needs to catch.
    supported, reason = label_row(make_row(["text_9"], False), ["table_1"], k=5)
    assert supported is False
    assert reason == "fabricated_without_evidence"


def test_partial_evidence_counts_as_missing():
    # Two gold chunks needed, only one retrieved -> the answer wasn't derivable.
    supported, reason = label_row(make_row(["table_1"], False), ["table_1", "table_2"], k=5)
    assert supported is False
    assert reason == "fabricated_without_evidence"


def test_refusing_without_the_evidence_is_supported():
    # Declining when you don't have the facts is exactly right, not a hallucination.
    supported, reason = label_row(
        make_row(["text_9"], False, answer="I cannot determine the answer from the provided evidence."),
        ["table_1"],
        k=5,
    )
    assert supported is True
    assert reason == "abstention_no_evidence"


def test_right_answer_without_the_evidence_is_skipped():
    # Lucky guess or reachable another way — we can't tell, so we don't label it.
    supported, reason = label_row(make_row(["text_9"], True), ["table_1"], k=5)
    assert supported is None
    assert reason == "correct_without_evidence"


def test_refusing_to_answer_counts_as_supported():
    supported, reason = label_row(make_row(["table_1"], False, abstained=True), ["table_1"], k=5)
    assert supported is True
    assert reason == "abstention"


def test_we_spot_a_refusal_even_when_the_run_file_says_otherwise():
    # Our older runs set abstained=false on answers that clearly refuse. Trusting that
    # flag labeled 5 refusals as hallucinations.
    row = make_row(["table_1"], False, answer="I cannot determine the answer from the provided evidence.")
    assert declined(row) is True
    assert label_row(row, ["table_1"], k=5)[0] is True


def test_a_hedged_answer_that_still_gives_a_number_is_not_a_refusal():
    row = make_row(["table_1"], False, answer="It cannot be determined exactly, but about 93.5")
    assert declined(row) is False


def test_near_miss_spots_a_rounding_error_but_not_a_made_up_number():
    close = {"gold_supported": False, "answer": "-2.0", "gold_answer_exe": -1.9}
    wild = {"gold_supported": False, "answer": "4200", "gold_answer_exe": -1.9}
    assert near_miss(close) is True
    assert near_miss(wild) is False


def test_near_miss_handles_percentages_stored_as_fractions():
    # FinQA writes 2.58% as 0.02581, so 2.5% is a 3% error, not a 9600% one.
    assert near_miss({"gold_supported": False, "answer": "2.5%", "gold_answer_exe": 0.02581}) is True


def test_score_counts_the_four_outcomes():
    def judged(gold_supported, judge_supported):
        return {"gold_supported": gold_supported, "verdict": {"supported": judge_supported}}

    rows = [
        judged(False, False),  # caught it
        judged(True, False),   # false alarm
        judged(False, True),   # missed it
        judged(True, True),    # correctly left alone
    ]
    assert score(rows) == (1, 1, 1, 1)
