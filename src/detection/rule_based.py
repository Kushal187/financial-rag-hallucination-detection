"""Rule-based hallucination verifier (Stage 4, Member 1).

Checks an answer by trying to **re-derive** it from the numbers in the evidence. If we
can find a calculation that produces the answer, we call it supported. If nothing we try
gets there, the answer didn't come from the evidence.

Why re-derive instead of just looking the number up? Because FinQA answers are almost
never written on the page. We measured it: only **3.4%** of correct answers appear
literally in their evidence, but **99.2%** of the numbers *used in the calculation* do. So
"is this number in the evidence?" would reject 96% of correct answers. We have to redo the
maths instead.

Which calculations we try comes from the FinQA dev programs themselves — divide is 64% of
them, subtract 19%, add 6%, multiply 5.5% — plus percent change, which is the single most
common shape in the dataset.

Same `verify(question, context, answer)` signature as the LLM judge (see
src/detection/protocol.py), so both can be scored by the same script.
"""

import re

from src.detection.llm_judge import _is_abstention

__all__ = ["verify", "derive", "extract_numbers", "operations_for"]

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")

# Answers this far apart count as the same number. 1% matches the tolerance we tell the
# LLM judge to use, so the two verifiers are held to the same standard.
DEFAULT_TOL = 0.01

# Words in the question that tell us which calculation is being asked for. Restricting
# the search to those operations cuts out a lot of accidental matches.
_OPERATION_WORDS = {
    "subtract": ("change", "increase", "decrease", "difference", "growth", "grew", "decline"),
    "percent": ("percent", "percentage", "%", "portion", "share", "ratio", "proportion"),
    "divide": ("percent", "percentage", "%", "portion", "share", "ratio", "average", "per "),
    "add": ("total", "sum", "combined", "average"),
    "multiply": ("times", "product"),
}
_ALL_OPERATIONS = frozenset(_OPERATION_WORDS)


def extract_numbers(context, drop_years=True):
    """Pull the numbers out of the evidence text.

    `drop_years` removes whole numbers between 1900 and 2030. Filing pages are full of
    years, they're never what a question is asking for, and leaving them in gives the
    search below a pile of extra operands to hit the answer with by accident. Dropping
    them raised precision from 0.38 to 0.47 on our evaluation set.
    """
    numbers = []
    for _chunk_id, text in context:
        for match in _NUM.findall(str(text)):
            try:
                value = float(match.replace(",", ""))
            except ValueError:
                continue
            if drop_years and value == int(value) and 1900 <= value <= 2030:
                continue
            numbers.append(value)
    return numbers


def operations_for(question):
    """Which calculations the question is asking for, based on its wording.

    "what is the change in X" wants a subtraction; "what percentage of X" wants a
    division. Searching only the relevant operations is both faster and more accurate —
    it stops us "deriving" an answer by multiplying two unrelated figures together.

    Falls back to trying everything when the wording gives us nothing to go on.
    """
    text = (question or "").lower()
    picked = {op for op, words in _OPERATION_WORDS.items() if any(w in text for w in words)}
    return picked or set(_ALL_OPERATIONS)


def _same(a, b, tol):
    """Are these the same number? Also checks x100 and /100.

    FinQA writes percentages both ways — '93.5%' is stored as 0.935 but '24.69%' is
    stored as 24.69 — so a raw comparison would reject correct answers.
    """
    if a is None or b is None:
        return False
    for scale in (1.0, 100.0, 0.01):
        target = b * scale
        if abs(a - target) <= tol * max(abs(target), 1e-9):
            return True
    return False


def derive(target, numbers, tol=DEFAULT_TOL, operations=None):
    """Try to build `target` out of `numbers`. Returns how, or None if we can't.

    `operations` limits which calculations to try — pass the result of
    `operations_for(question)`. Defaults to all of them.

    Only one operation on at most two numbers, which covers about 59% of FinQA questions.
    The rest chain several steps together and this won't find them. Note that widening
    the search does *not* help: the verifier already accepts too much (see the tolerance
    sweep in docs/detection_dataset.md), so more reachable values means more wrong answers
    slipping through, not fewer.
    """
    if target is None:
        return None
    ops = _ALL_OPERATIONS if operations is None else operations

    for a in numbers:
        if _same(target, a, tol):
            return f"{a:g} (quoted directly)"

    for a in numbers:
        for b in numbers:
            if a is b:
                continue
            if "subtract" in ops and _same(target, a - b, tol):
                return f"{a:g} - {b:g}"
            if "add" in ops and _same(target, a + b, tol):
                return f"{a:g} + {b:g}"
            if "divide" in ops and b and _same(target, a / b, tol):
                return f"{a:g} / {b:g}"
            if "percent" in ops and b and _same(target, (a - b) / b, tol):
                return f"({a:g} - {b:g}) / {b:g}   [percent change]"
            if "multiply" in ops and _same(target, a * b, tol):
                return f"{a:g} * {b:g}"
    return None


def _parse(text):
    """First number in a string, or None."""
    match = _NUM.search(str(text))
    if not match:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def verify(question, context, answer, tol=DEFAULT_TOL):
    """Decide whether `answer` can be derived from `context`.

    Returns the same shape as the LLM judge's verdict so score_detection.py can grade
    either one without changes.
    """
    def verdict(supported, category, reasoning, derivation=None):
        return {
            "supported": supported,
            "partial": False,
            "category": category,
            "computed_value": None,
            "confidence": 1.0,
            "reasoning": reasoning,
            "cited_evidence": [],
            "derivation": derivation,
        }

    if _is_abstention(str(answer)):
        return verdict(True, "abstention", "The model declined to answer.")

    numbers = extract_numbers(context)
    if not numbers:
        return verdict(False, "out_of_context", "There are no numbers in the evidence at all.")

    target = _parse(answer)
    if target is None:
        # yes/no and other non-numeric answers can't be checked this way.
        return verdict(True, "supported", "Answer is not a number, so we cannot check it.")

    found = derive(target, numbers, tol, operations_for(question))
    if found:
        return verdict(True, "supported", f"The answer can be computed: {found}", found)

    # There is no `entity_error` branch. That category means "right kind of number, wrong
    # thing" — reporting the 2003 figure when asked for the change between 2003 and 2002.
    # Any figure printed on the page passes the "quoted directly" check above, so it looks
    # supported. Dropping years catches the subset where the model reports a year as an
    # answer, but the general case needs to know what each number represents, which this
    # approach doesn't.
    return verdict(
        False,
        "numeric_error",
        "The evidence has numbers, but none of the calculations we try produce this answer.",
    )
