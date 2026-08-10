"""Answer extraction and numeric-tolerant matching for FinQA.

`gold_answer` (display string) and `gold_answer_exe` (executed value) disagree on units
for about half the rows, mostly percentages stored as fractions. `answers_match` therefore
accepts the prediction against several scale-variants of the gold rather than one field.
"""

import json
import math
import re

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d+")
_YESNO_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(r"final\s+answer\s*:?\s*(.+)", re.IGNORECASE | re.DOTALL)
_ABSTENTION_RE = re.compile(
    r"cannot (?:be )?determin"                                 
    r"|unable to (?:determine|answer|provide)"
    r"|not (?:enough|sufficient) (?:info|information|evidence|data|context)"
    r"|insufficient (?:info|information|evidence|data|context)",
    re.IGNORECASE,
)


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
    """Reduce a model answer to `(value, kind)`, where kind is one of number, yes_no,
    abstention, text or none."""
    if raw is None:
        return (None, "none")
    text = str(raw).strip()
    if not text:
        return (None, "none")

    if _ABSTENTION_RE.search(text) and _parse_number(text) is None:
        return (None, "abstention")

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
    """Whether the prediction is correct, tolerant of FinQA's unit ambiguity.

    yes/no gold compares exactly; numeric gold matches the prediction against the gold
    value and its x100 and /100 variants. Sign flips never match.
    """
    if isinstance(gold_answer_exe, str):
        gold = gold_answer_exe.strip().lower()
        if not gold: 
            gold = str(gold_answer).strip().lower()
        p_val, p_kind = normalize_answer(pred_raw)
        return p_kind == "yes_no" and p_val == gold

    p_val, p_kind = normalize_answer(pred_raw)
    if p_kind != "number" or p_val is None:
        return False  

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
    """Pull the answer token out of a model response: the JSON `answer` field for
    `structured`, the text after `FINAL ANSWER:` for `cot`, otherwise the last number."""
    if raw is None:
        return ""
    text = str(raw).strip()

    if strategy == "structured":
        obj = parse_json_object(text)
        if obj is not None and obj.get("answer") is not None:
            return str(obj["answer"]).strip()

    m = _FINAL_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip().splitlines()[0].strip()

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
