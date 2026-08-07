"""Tests for the rule-based verifier.

It's pure arithmetic, so there's nothing to mock — every test is a small worked example.
"""

from src.detection.rule_based import derive, extract_numbers, verify

EVIDENCE = [("table_1", "the finished goods of 2003 is $ 384.3 ; the finished goods of 2002 is $ 206.7 .")]


# ---- pulling numbers out of the evidence ---- #


def test_extract_numbers_reads_the_figures_and_skips_years():
    # 2003 and 2002 are dropped — filing pages are full of years and they only give the
    # derivation search extra operands to hit the answer with by accident.
    assert extract_numbers(EVIDENCE) == [384.3, 206.7]


def test_extract_numbers_can_keep_years_if_asked():
    assert extract_numbers(EVIDENCE, drop_years=False) == [2003.0, 384.3, 2002.0, 206.7]


def test_extract_numbers_handles_thousands_separators():
    assert extract_numbers([("t", "revenue was 1,234,567 last year")]) == [1234567.0]


def test_extract_numbers_of_empty_evidence_is_empty():
    assert extract_numbers([]) == []


# ---- deriving the answer ---- #


def test_derive_finds_a_subtraction():
    # 384.3 - 206.7 = 177.6, the real gold program for this row
    assert "384.3 - 206.7" in derive(177.6, extract_numbers(EVIDENCE))


def test_derive_finds_a_number_quoted_straight_off_the_page():
    assert "quoted directly" in derive(384.3, extract_numbers(EVIDENCE))


def test_derive_finds_a_percent_change():
    # (60 - 50) / 50 = 0.2 — the most common shape in FinQA
    assert derive(0.2, [60.0, 50.0]) is not None


def test_derive_accepts_a_percentage_written_either_way():
    # FinQA stores '20%' as 0.2 in some rows and as 20.0 in others, so both must match.
    assert derive(0.2, [60.0, 50.0]) is not None
    assert derive(20.0, [60.0, 50.0]) is not None


def test_derive_gives_up_on_a_made_up_number():
    # -2017 is a year the model grabbed off the page and reported as a financial answer.
    assert derive(-2017.0, [384.3, 206.7]) is None


def test_derive_needs_two_different_numbers():
    assert derive(0.0, [5.0]) is None


# ---- the full verdict ---- #


def test_a_derivable_answer_is_supported():
    v = verify("what is the change?", EVIDENCE, "177.6")
    assert v["supported"] is True
    assert v["category"] == "supported"
    assert "384.3 - 206.7" in v["derivation"]


def test_an_underivable_answer_is_flagged():
    v = verify("what is the change?", EVIDENCE, "4823.91")
    assert v["supported"] is False
    assert v["category"] == "numeric_error"


def test_a_year_reported_as_an_answer_is_flagged():
    # Dropping years from the operand pool means 2002 is no longer "quoted directly",
    # so this is caught. Real models do this — one of ours answered -2017 to a
    # percentage question.
    assert verify("what is the change?", EVIDENCE, "2002")["supported"] is False


def test_known_limitation_a_figure_lifted_off_the_page_still_looks_supported():
    """We can't catch "right figure, wrong question" when the figure isn't a year.

    384.3 is the 2003 finished-goods value, not the *change* the question asked for. It's
    printed on the page, so the quoted-directly check accepts it. Spotting this needs to
    know what each number represents, which arithmetic alone can't. Pinned so the gap
    stays visible.
    """
    assert verify("what is the change?", EVIDENCE, "384.3")["supported"] is True


def test_refusing_to_answer_is_supported():
    v = verify("q", EVIDENCE, "I cannot determine the answer from the provided evidence.")
    assert v["supported"] is True
    assert v["category"] == "abstention"


def test_evidence_with_no_numbers_cannot_support_a_number():
    v = verify("q", [("text_1", "the company had a good year")], "177.6")
    assert v["supported"] is False
    assert v["category"] == "out_of_context"


def test_a_yes_no_answer_is_not_checked():
    # This verifier only does arithmetic, so it can't judge yes/no answers.
    v = verify("did revenue grow?", EVIDENCE, "yes")
    assert v["supported"] is True


def test_verdict_has_the_same_shape_as_the_llm_judge():
    v = verify("q", EVIDENCE, "177.6")
    for key in ("supported", "partial", "category", "confidence", "reasoning", "cited_evidence"):
        assert key in v


def test_a_looser_tolerance_accepts_more():
    # 178.5 is 0.5% off 177.6 — rejected at the default 1%... accepted at 5%.
    assert verify("q", EVIDENCE, "179.5", tol=0.001)["supported"] is False
    assert verify("q", EVIDENCE, "179.5", tol=0.05)["supported"] is True
