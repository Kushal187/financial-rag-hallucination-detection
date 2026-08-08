"""LLM-as-a-judge hallucination verifier.

Decides whether an answer is supported by the retrieved evidence, not whether it matches
the gold label: an answer can follow from the evidence and still be numerically wrong,
which is a generation error rather than a hallucination. The judge never sees the gold
answer, since that would make the check circular.
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


_EVIDENCE_SYSTEM = (
    "You are checking whether a financial question CAN BE ANSWERED from the evidence "
    "provided. You are not checking anyone's answer, and you are not answering the "
    "question yourself.\n\n"
    "Rules:\n"
    "- Do NOT do any arithmetic. Do NOT work out the answer. Decide only whether the "
    "figures the question needs are present.\n"
    "- Financial answers are computed, not quoted: a percentage is part / whole, a change "
    "is one year minus another. So the evidence is SUFFICIENT when it contains the values "
    "such a calculation would need, even though the final number will not appear anywhere "
    "on the page. Do not require the answer itself to be present.\n"
    "- The evidence is INSUFFICIENT when a figure the question asks about is missing — a "
    "year that isn't shown, a line item that isn't listed, one side of a comparison that "
    "isn't there.\n"
    "- Judge only what is in front of you. Never use outside knowledge about the company.\n\n"
    "Respond ONLY with valid json (no prose, no code fences) with exactly these keys: "
    '"sufficient" (boolean), "missing" (one short phrase naming what is absent, or "" '
    'when sufficient), "confidence" (0.0-1.0).'
)


def build_evidence_messages(question: str, context: list[tuple[str, str]]) -> list[dict]:
    """Prompt for the evidence-sufficiency check. Deliberately excludes the answer, so
    the judge cannot anchor its own arithmetic on the number it is checking."""
    user = (
        f"Evidence:\n{format_context(context)}\n\n"
        f"Question: {question}\n\n"
        "Does this evidence contain what is needed to answer the question? "
        "Respond with the json object."
    )
    return [
        {"role": "system", "content": _EVIDENCE_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_evidence_verdict(raw: object) -> dict:
    """Parse the evidence-sufficiency reply into the shared verdict shape."""
    obj = parse_json_object(raw) or {}
    sufficient = bool(obj.get("sufficient", False))

    confidence = obj.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0

    missing = str(obj.get("missing", ""))
    return {
        "supported": sufficient,
        "partial": False,
        # Answering without the necessary figures is the out_of_context case.
        "category": SUPPORTED_CATEGORY if sufficient else "out_of_context",
        "computed_value": None,
        "confidence": confidence,
        "reasoning": "Evidence is sufficient." if sufficient else f"Evidence is missing: {missing}",
        "cited_evidence": [],
    }


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
        # Kept so we can check the judge actually compared its own result to the answer.
        "computed_value": obj.get("computed_value"),
        "confidence": confidence,
        "reasoning": str(obj.get("reasoning", "")),
        "cited_evidence": _coerce_cited(obj.get("cited_evidence")),
    }


def is_abstention(answer: str) -> bool:
    a = (answer or "").strip().lower()
    if not a:
        return True
    return any(phrase in a for phrase in _ABSTENTION_PHRASES)


def verify(
    question: str,
    context: list[tuple[str, str]],
    answer: str,
    temperature: float = 0.0,
    mode: str = "grounding",
) -> dict:
    
    if is_abstention(answer):
        return {
            "supported": True,
            "partial": False,
            "category": "abstention",
            "confidence": 1.0,
            "reasoning": "Answer is an abstention — no hallucination.",
            "cited_evidence": [],
            "latency_ms": 0.0,
        }

    if mode == "evidence":
        messages = build_evidence_messages(question, context)
        parse = parse_evidence_verdict
    else:
        messages = build_judge_messages(question, context, answer)
        parse = parse_verdict

    start = time.perf_counter()
    raw = chat_json(messages, temperature=temperature)
    latency_ms = (time.perf_counter() - start) * 1000.0

    verdict = parse(raw)
    verdict["latency_ms"] = round(latency_ms, 1)
    return verdict
