"""Rule-based hallucination verifier.

Checks an answer by trying to re-derive it from the numbers in the evidence: if some
calculation over those numbers reaches the answer, it counts as supported. Looking the
number up instead would not work, because FinQA answers are almost never printed on the
page. The operations tried are the ones the FinQA dev programs actually use, plus percent
change. Same verify(question, context, answer) signature as the LLM judge.
"""

import re

from src.detection.llm_judge import is_abstention

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


DEFAULT_TOL = 0.01


_OPERATION_WORDS = {
    "subtract": ("change", "increase", "decrease", "difference", "growth", "grew", "decline"),
    "percent": ("percent", "percentage", "%", "portion", "share", "ratio", "proportion"),
    "divide": ("percent", "percentage", "%", "portion", "share", "ratio", "average", "per "),
    "add": ("total", "sum", "combined", "average"),
    "multiply": ("times", "product"),
}
_ALL_OPERATIONS = frozenset(_OPERATION_WORDS)


def extract_numbers(context, drop_years=True):
    """Pull the numbers out of the evidence. `drop_years` discards 1900-2030, which are
    almost always dates and just give `derive` extra operands to hit the answer with."""
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
    """Which calculations the question is asking for, based on its wording: "the change
    in X" wants a subtraction, "what percentage of X" a division. Falls back to all of
    them when the wording gives nothing to go on."""
    text = (question or "").lower()
    picked = {op for op, words in _OPERATION_WORDS.items() if any(w in text for w in words)}
    return picked or set(_ALL_OPERATIONS)


def _same(a, b, tol):
    """Are these the same number? Also checks x100 and /100, because FinQA stores '93.5%'
    as 0.935 but '24.69%' as 24.69, so a raw comparison would reject correct answers."""
    if a is None or b is None:
        return False
    for scale in (1.0, 100.0, 0.01):
        target = b * scale
        if abs(a - target) <= tol * max(abs(target), 1e-9):
            return True
    return False


def derive(target, numbers, tol=DEFAULT_TOL, operations=None):
    """Try to build `target` out of `numbers`. Returns the derivation, or None.

    One operation on at most two numbers, limited to `operations`. Multi-step programs are
    out of reach, but widening the search lets through more wrong answers, not fewer.
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
    """Decide whether `answer` can be derived from `context`, in the LLM judge's verdict
    shape."""
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

    if is_abstention(str(answer)):
        return verdict(True, "abstention", "The model declined to answer.")

    numbers = extract_numbers(context)
    if not numbers:
        return verdict(False, "out_of_context", "There are no numbers in the evidence at all.")

    target = _parse(answer)
    if target is None:
        return verdict(True, "supported", "Answer is not a number, so we cannot check it.")

    found = derive(target, numbers, tol, operations_for(question))
    if found:
        return verdict(True, "supported", f"The answer can be computed: {found}", found)

    return verdict(
        False,
        "numeric_error",
        "The evidence has numbers, but none of the calculations we try produce this answer.",
    )
