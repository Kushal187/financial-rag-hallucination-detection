"""Tests for demo.py.

The demo is the one thing someone runs before reading anything else, so it fails in the
worst possible place — in front of an audience, or on a grader's fresh clone. Nothing
here calls an LLM or Weaviate; these pin the parts that break silently:

  * the two questions it is built around still exist in the corpus
  * every recorded run it falls back to is still on disk, with a row for those questions
  * the scoreboard arithmetic in part 3 is right (it prints numbers the report also
    quotes, so a bug here would contradict the write-up)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import demo  # noqa: E402
from src.data.load_data import load_qa  # noqa: E402


def test_the_demo_questions_are_in_the_corpus():
    ids = {row["id"] for row in load_qa()}
    assert demo.GOOD_QUESTION in ids
    assert demo.HALLUCINATED_QUESTION in ids


def test_the_hallucinated_question_really_is_a_retrieval_miss():
    """Part 2's whole narrative is that the needed row doesn't survive retrieval.

    If a future retriever change starts finding it, the demo would announce a
    hallucination and then show the pipeline working — so pin the recorded ranking.
    """
    recorded = demo.recorded("generation", demo.HALLUCINATED_QUESTION, "zero_shot")
    qa = {row["id"]: row for row in load_qa()}[demo.HALLUCINATED_QUESTION]
    assert set(qa["gold_evidence_ids"]).isdisjoint(recorded["retrieved_ids"][:5])
    assert not recorded["correct"]


def test_every_fallback_run_file_covers_both_demo_questions():
    """`--offline` replays these files, so a missing row means a blank verdict on screen."""
    for kind in demo._RUN_FILES:
        for question_id in (demo.GOOD_QUESTION, demo.HALLUCINATED_QUESTION):
            assert demo.recorded(kind, question_id, "zero_shot") is not None, (
                f"{kind} has no zero_shot row for {question_id}"
            )


def test_prf_counts_the_four_outcomes():
    # positive = hallucinated. Two caught, one missed, one false alarm, one clean.
    judged = [
        {"gold": True, "verdict": {"supported": False}},   # tp
        {"gold": True, "verdict": {"supported": False}},   # tp
        {"gold": True, "verdict": {"supported": True}},    # fn
        {"gold": False, "verdict": {"supported": False}},  # fp
        {"gold": False, "verdict": {"supported": True}},   # tn
    ]
    precision, recall, f1, accuracy = demo._prf(judged, lambda r: r["gold"])
    assert precision == 2 / 3
    assert recall == 2 / 3
    assert f1 == 2 / 3
    assert accuracy == 3 / 5


def test_prf_reproduces_the_reported_scores():
    """The demo prints these on screen; the report prints them in a table. Same numbers."""
    rows = demo._replay("judge_grounding")
    precision, recall, f1, accuracy = demo._prf(rows, lambda r: not r["gold_supported"])
    assert (round(precision, 2), round(recall, 2)) == (0.61, 0.49)
    assert (round(f1, 2), round(accuracy, 2)) == (0.54, 0.76)


def test_clip_never_exceeds_the_width_it_is_given():
    assert demo.clip("x" * 200, 40) == "x" * 39 + "…"
    assert demo.clip("short", 40) == "short"
    # Newlines in evidence text would otherwise break the ranked-chunk table.
    assert demo.clip("two\nlines", 40) == "two lines"
