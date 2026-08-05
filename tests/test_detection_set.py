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


def test_we_skip_rows_where_retrieval_missed_the_evidence():
    # The model might have read the wrong chunk correctly, so we can't call this a
    # hallucination. This skip is the whole reason the rule works.
    supported, reason = label_row(make_row(["text_9"], False), ["table_1"], k=5)
    assert supported is None
    assert reason == "incomplete_retrieval"


def test_we_skip_rows_where_only_some_evidence_was_found():
    supported, _ = label_row(make_row(["table_1"], False), ["table_1", "table_2"], k=5)
    assert supported is None


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
