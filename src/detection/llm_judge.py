"""LLM-as-a-judge hallucination verifier (Stage 4, Member 2).

Decides whether a generated answer is **supported by the retrieved evidence** — which
is the right definition of "not hallucinated" for RAG (an answer can be supported by
evidence yet still numerically wrong vs the gold label; that's a generation error, not
a hallucination, and the two metrics are reported separately).

Design choices:
- The judge NEVER sees the gold answer (that would make verification circular).
- Binary primary verdict ``supported`` for head-to-head parity with Member 1's
  rule-based verifier; ``category`` is the analytical layer.
- Structured (JSON) output for parseable, consistent verdicts.
- Obvious abstentions are short-circuited (supported=True, category="abstention")
  without an LLM call — an abstention is the safe failure mode, not a hallucination.
"""

import time

from src.detection.categories import (
    CATEGORIES,
    DEFAULT_CATEGORY,
    SUPPORTED_CATEGORY,
)
from src.generation.answer import parse_json_object
from src.generation.client import chat_json
from src.generation.context import format_context

__all__ = ["verify", "parse_verdict", "build_judge_messages"]

_ABSTENTION_PHRASES = ("cannot determine", "i cannot determine", "not enough", "insufficient")

_JUDGE_SYSTEM = (
    "You are a strict evaluator for financial question answering. Decide whether the "
    "candidate ANSWER is SUPPORTED by the EVIDENCE provided below.\n\n"
    "Rules:\n"
    "- Judge ONLY against the evidence. Never use outside knowledge, and never assume "
    "the answer is true just because it sounds plausible.\n"
    "- Financial answers are usually COMPUTED from the evidence, not quoted verbatim: "
    "a percentage = part / whole, a change = A - B, a growth rate, a ratio, an average, "
    "a share of a total. Do NOT mark an answer unsupported just because its final number "
    "does not appear literally in the evidence — it usually will not.\n"
    "- Work out the answer yourself first, using numbers from the evidence, and put that "
    'number in "computed_value" (null if the question is not numeric). Then compare it '
    "to the candidate answer.\n"
    "- An answer is SUPPORTED only if BOTH hold: the values needed appear in the "
    "evidence, AND your computed value matches the candidate answer within 1%.\n"
    "- Set supported=false when the answer uses a value not in the evidence, uses the "
    "wrong entity/year/line item, invents information, OR when your computed value "
    "differs from the candidate by more than 1%. A wrong result derived from the right "
    "numbers is still unsupported — that is exactly what numeric_error means. Never "
    "excuse a gap larger than 1% as rounding, a unit misunderstanding, or 'the evidence "
    "still supports the calculation'.\n"
    "- An answer that declines ('I cannot determine') is supported=true with category "
    '"abstention" — it is not a hallucination.\n\n'
    "Categories (use when supported=false):\n"
    + "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())
    + '\n\nSet "confidence" (0.0-1.0) to how clearly the evidence supports your verdict. '
    "Respond ONLY with valid json (no prose, no code fences) with exactly these keys: "
    '"supported" (boolean), "partial" (boolean), "category" (one of the category names '
    'above, or "supported" when supported=true), "computed_value" (the number your own '
    'calculation produced, or null), "confidence" (0.0-1.0), "reasoning" (one sentence), '
    '"cited_evidence" (list of evidence local_ids like "text_3"/"table_1" that back the '
    "answer, or [])."
)


def build_judge_messages(question: str, context: list[tuple[str, str]], answer: str) -> list[dict]:
    """Build the judge prompt. Deliberately excludes the gold answer."""
    user = (
        f"Evidence:\n{format_context(context)}\n\n"
        f"Question: {question}\n\n"
        f"Candidate answer: {answer}\n\n"
        "Is the candidate answer supported by the evidence? Respond with the json object."
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _coerce_cited(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return []


def parse_verdict(raw: object) -> dict:
    """Parse the judge's JSON into a normalized verdict, filling sane defaults.

    Robust to code fences, trailing prose, and missing keys — a run must never crash
    on a malformed judge response.
    """
    obj = parse_json_object(raw) or {}
    supported = bool(obj.get("supported", False))

    category = obj.get("category")
    if not category:
        category = SUPPORTED_CATEGORY if supported else DEFAULT_CATEGORY

    confidence = obj.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "supported": supported,
        "partial": bool(obj.get("partial", False)),
        "category": str(category),
        # The value the judge's own calculation produced. Kept so we can check whether it
        # actually compared its result to the candidate answer, rather than deciding the
        # evidence "supports the calculation" and waving the mismatch through.
        "computed_value": obj.get("computed_value"),
        "confidence": confidence,
        "reasoning": str(obj.get("reasoning", "")),
        "cited_evidence": _coerce_cited(obj.get("cited_evidence")),
    }


def _is_abstention(answer: str) -> bool:
    a = (answer or "").strip().lower()
    if not a:
        return True
    return any(phrase in a for phrase in _ABSTENTION_PHRASES)


def verify(
    question: str,
    context: list[tuple[str, str]],
    answer: str,
    temperature: float = 0.0,
) -> dict:
    """Return a verdict dict: ``{supported, partial, category, confidence, reasoning,
    cited_evidence, latency_ms}``. Abstentions are short-circuited (no LLM call)."""
    if _is_abstention(answer):
        return {
            "supported": True,
            "partial": False,
            "category": "abstention",
            "confidence": 1.0,
            "reasoning": "Answer is an abstention — no hallucination.",
            "cited_evidence": [],
            "latency_ms": 0.0,
        }

    messages = build_judge_messages(question, context, answer)
    start = time.perf_counter()
    raw = chat_json(messages, temperature=temperature)
    latency_ms = (time.perf_counter() - start) * 1000.0

    verdict = parse_verdict(raw)
    verdict["latency_ms"] = round(latency_ms, 1)
    return verdict
