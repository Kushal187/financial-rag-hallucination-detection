"""Generate a FinQA answer from retrieved evidence.

The orchestrator ties together prompt building, the Groq client, and answer
extraction. It is deliberately **decoupled from retrieval**: it takes a pre-assembled
``context`` (``list[(local_id, content)]``), so Member 1's baseline pipeline and
Member 3's reranked variant both call it identically — only the context they pass
differs.

Flow: build messages for the strategy -> call Groq (JSON mode for ``structured``) ->
extract the answer token -> normalize. Structured-output JSON parse failures degrade
gracefully (fall back to ``FINAL ANSWER:`` extraction) rather than crashing a run.
"""

import time

from src.generation import client
from src.generation.answer import extract_final_answer, normalize_answer
from src.generation.prompts import build_messages

__all__ = ["generate_answer"]


def generate_answer(
    question: str,
    context: list[tuple[str, str]],
    strategy: str = "zero_shot",
    temperature: float = 0.0,
    few_shot_examples: list[dict] | None = None,
    save_messages: bool = False,
) -> dict:
    """Generate an answer for ``question`` grounded in ``context``.

    Returns ``{raw, answer, answer_type, strategy, model, latency_ms}`` (plus
    ``messages`` when ``save_messages=True`` for debugging).
    """
    messages, needs_json = build_messages(strategy, question, context, few_shot_examples)

    start = time.perf_counter()
    if needs_json:
        raw = client.chat_json(messages, temperature=temperature)
    else:
        raw = client.chat(messages, temperature=temperature)
    latency_ms = (time.perf_counter() - start) * 1000.0

    answer = extract_final_answer(raw, strategy)
    _value, kind = normalize_answer(answer)

    result = {
        "raw": raw,
        "answer": answer,
        "answer_type": kind,
        "strategy": strategy,
        "model": client._MODEL,
        "latency_ms": round(latency_ms, 1),
    }
    if save_messages:
        result["messages"] = messages
    return result
