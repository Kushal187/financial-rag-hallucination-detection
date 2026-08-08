"""Hallucination detection: an LLM-as-a-judge verifier and a rule-based one."""

from src.detection.categories import CATEGORIES, DEFAULT_CATEGORY
from src.detection.llm_judge import build_judge_messages, parse_verdict, verify

__all__ = [
    "CATEGORIES",
    "DEFAULT_CATEGORY",
    "build_judge_messages",
    "parse_verdict",
    "verify",
]
