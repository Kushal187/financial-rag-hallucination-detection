"""Shared verifier contract for the detection stage.

Makes Member 2's LLM-as-a-judge and Member 1's rule-based verifier interchangeable in
the detection run script: both are callable as

    verify(question: str, context: list[tuple[str, str]], answer: str) -> dict

returning at least ``{"supported": bool, "category": str}``. A plain function
satisfies this protocol, so the LLM judge's module-level ``verify`` is a drop-in.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Verifier(Protocol):
    """Callable contract: ``(question, context, answer) -> verdict dict``."""

    def __call__(
        self, question: str, context: list[tuple[str, str]], answer: str
    ) -> dict: ...
