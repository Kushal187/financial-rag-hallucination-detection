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

__all__ = ["verify", "derive", "extract_numbers"]

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")

# Answers this far apart count as the same number. 1% matches the tolerance we tell the
# LLM judge to use, so the two verifiers are held to the same standard.
DEFAULT_TOL = 0.01


def extract_numbers(context):
    """Pull every number out of the evidence text."""
    numbers = []
    for _chunk_id, text in context:
        for match in _NUM.findall(str(text)):
            try:
                numbers.append(float(match.replace(",", "")))
            except ValueError:
                pass
    return numbers


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


def derive(target, numbers, tol=DEFAULT_TOL):
    """Try to build `target` out of `numbers`. Returns how, or None if we can't.

    Only tries one operation on at most two numbers. That covers about 59% of FinQA
    questions — the rest need several steps chained together, and this will not find
    those. That limitation is real and shows up in the results.
    """
    if target is None:
        return None

    for a in numbers:
        if _same(target, a, tol):
            return f"{a:g} (quoted directly)"

    for a in numbers:
        for b in numbers:
            if a is b:
                continue
            if _same(target, a - b, tol):
                return f"{a:g} - {b:g}"
            if _same(target, a + b, tol):
                return f"{a:g} + {b:g}"
            if b and _same(target, a / b, tol):
                return f"{a:g} / {b:g}"
            if b and _same(target, (a - b) / b, tol):
                return f"({a:g} - {b:g}) / {b:g}   [percent change]"
            if _same(target, a * b, tol):
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

    found = derive(target, numbers, tol)
    if found:
        return verdict(True, "supported", f"The answer can be computed: {found}", found)

    # Note there is no `entity_error` branch. That category means "right number, wrong
    # thing" — e.g. reporting 2002's figure when the question asked about 2003. We cannot
    # detect it: any number printed on the page passes the "quoted directly" check above,
    # so a wrong-but-present number always looks supported. Catching it would need to know
    # what each number *means*, which is exactly what this approach doesn't do.
    return verdict(
        False,
        "numeric_error",
        "The evidence has numbers, but none of the calculations we try produce this answer.",
    )
