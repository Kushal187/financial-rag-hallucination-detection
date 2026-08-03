"""Answer extraction + numeric-tolerant matching for FinQA generation.

FinQA answers are almost always numbers (sometimes yes/no). The matching has to
tolerate a known data hazard: `gold_answer` (display string) and `gold_answer_exe`
(executed program value) disagree on units for ~half the rows. The dominant case is
percentages stored two ways — `gold_answer="53%"` but `gold_answer_exe=0.53232`
(fraction). Matching against either field alone scores percentage questions near 0%.

`answers_match` therefore union-matches the predicted number against several
scale-variants of the gold, so a correct answer in *either* display convention is
accepted. All strategies get the same (small) upward bias, so it stays fair for a
strategy-comparison harness.

Pure stdlib only — no network, no model — so this is unit-testable in isolation.
"""

import json
import math
import re

__all__ = [
    "normalize_answer",
    "answers_match",
    "extract_final_answer",
    "parse_json_object",
]

# First numeric token, tolerating thousands separators and a leading sign/currency.
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d+")
_YESNO_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(r"final\s+answer\s*:?\s*(.+)", re.IGNORECASE | re.DOTALL)


def _parse_number(text: object) -> float | None:
    """Extract the first number from `text`, stripping commas. None if none found."""
    if text is None:
        return None
    m = _NUM_RE.search(str(text).replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def normalize_answer(raw: object) -> tuple[object, str]:
    """Reduce a model answer to ``(value, kind)`` where kind ∈ {number, yes_no, text, none}.

    - yes/no questions -> ``("yes"|"no", "yes_no")``
    - numeric answers -> ``(float, "number")``
    - unparseable text -> ``(text, "text")``
    - empty / abstention -> ``(None, "none")``
    """
    if raw is None:
        return (None, "none")
    text = str(raw).strip()
    if not text:
        return (None, "none")

    m = _YESNO_RE.match(text)
    if m:
        return ("yes" if m.group(1).lower() == "yes" else "no", "yes_no")

    num = _parse_number(text)
    if num is not None:
        return (num, "number")

    return (text, "text")


def _same_sign(a: float, b: float) -> bool:
    """True unless a and b have opposite signs (a wrong-sign number is never correct)."""
    if a == 0 or b == 0:
        return True
    return (a > 0) == (b > 0)


def answers_match(
    pred_raw: object,
    gold_answer: str,
    gold_answer_exe: object,
    rel_tol: float = 1e-2,
    abs_tol: float = 1e-2,
) -> bool:
    """Whether the predicted answer is correct, tolerant of FinQA's unit ambiguity.

    yes/no gold (``gold_answer_exe`` is a string) -> exact yes/no compare.
    numeric gold -> True if the predicted number is within tolerance of *any* of the
    gold scale-variants: ``{parse(gold_answer), gold_answer_exe, exe*100, exe/100}``
    (the *100 // /100 pair covers the fraction-vs-percent split). Sign flips never match.
    """
    # yes/no question
    if isinstance(gold_answer_exe, str):
        gold = gold_answer_exe.strip().lower()
        if not gold:  # fall back to the answer string
            gold = str(gold_answer).strip().lower()
        p_val, p_kind = normalize_answer(pred_raw)
        return p_kind == "yes_no" and p_val == gold

    p_val, p_kind = normalize_answer(pred_raw)
    if p_kind != "number" or p_val is None:
        return False  # abstention or unparseable -> not correct

    candidates: list[float] = []
    g_str = _parse_number(gold_answer)
    if g_str is not None:
        candidates.append(g_str)
    try:
        g_exe = float(gold_answer_exe)
    except (TypeError, ValueError):
        g_exe = None
    if g_exe is not None:
        candidates.extend([g_exe, g_exe * 100.0, g_exe / 100.0])

    return any(
        _same_sign(p_val, c) and math.isclose(p_val, c, rel_tol=rel_tol, abs_tol=abs_tol)
        for c in candidates
        if c is not None
    )


def parse_json_object(text: object) -> dict | None:
    """Best-effort JSON object parse: strips code fences, extracts the first {...}."""
    if text is None:
        return None
    s = str(text).strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def extract_final_answer(raw: object, strategy: str) -> str:
    """Pull the answer token out of a model response, strategy-aware.

    - ``structured`` -> the JSON ``answer`` field (falls back to ``FINAL ANSWER:``).
    - ``cot`` (and any fallback) -> text after ``FINAL ANSWER:``.
    - otherwise -> the raw response.
    """
    if raw is None:
        return ""
    text = str(raw).strip()

    if strategy == "structured":
        obj = parse_json_object(text)
        if obj is not None and obj.get("answer") is not None:
            return str(obj["answer"]).strip()

    m = _FINAL_ANSWER_RE.search(text)
    if m:
        # first line of the capture only
        return m.group(1).strip().splitlines()[0].strip()

    # No explicit marker -> free-form response. The final answer is almost always the
    # LAST number stated (a verbose model restates it in the closing sentence), not the
    # first (which is usually an intermediate value). Prefer an explicit yes/no when the
    # response opens with one.
    ym = _YESNO_RE.match(text)
    if ym:
        return "yes" if ym.group(1).lower() == "yes" else "no"
    nums = list(_NUM_RE.finditer(text.replace(",", "")))
    if nums:
        last = nums[-1]
        end = last.end()
        suffix = "%" if end < len(text) and text[end] == "%" else ""
        return last.group() + suffix

    return text
