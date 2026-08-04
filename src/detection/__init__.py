"""Hallucination detection stage (Member 2): LLM-as-a-judge verifier.

Public API:
    verify(question, context, answer) -> dict   # verdict {supported, category, ...}
    parse_verdict(raw)                           # robust JSON -> normalized verdict
    build_judge_messages(question, context, answer)
    CATEGORIES                                   # hallucination category taxonomy
    Verifier                                     # shared verifier protocol (Member 1 parity)
"""

from src.detection.categories import CATEGORIES, DEFAULT_CATEGORY
from src.detection.llm_judge import build_judge_messages, parse_verdict, verify
from src.detection.protocol import Verifier

__all__ = [
    "verify",
    "parse_verdict",
    "build_judge_messages",
    "CATEGORIES",
    "DEFAULT_CATEGORY",
    "Verifier",
]
