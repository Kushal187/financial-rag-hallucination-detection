"""Generate an answer from retrieved evidence, using one of the prompt strategies."""

import time

from src.generation import client
from src.generation.answer import extract_final_answer, normalize_answer
from src.generation.prompts import build_messages

def generate_answer(
    question: str,
    context: list[tuple[str, str]],
    strategy: str = "zero_shot",
    temperature: float = 0.0,
    few_shot_examples: list[dict] | None = None,
    save_messages: bool = False,
) -> dict:
    """Generate an answer for `question` grounded in `context`. Returns
    `{raw, answer, answer_type, strategy, model, latency_ms}`."""
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
