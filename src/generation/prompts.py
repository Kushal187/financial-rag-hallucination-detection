"""Prompt strategies for FinQA answer generation: zero_shot, few_shot, cot, structured.

All four share one signature, `(question, context, few_shot_examples) -> (messages,
needs_json)`. Few-shot examples come from the train split only, sampled stratified across
answer shape so the model sees percent and yes/no formats rather than three numeric rows.
"""

import random

from src.data.load_data import load_qa
from src.generation.context import format_context, get_chunk_contents


_SYSTEM_BASE = (
    "You are a financial question-answering assistant. Answer the user's question "
    "using ONLY the evidence provided below. Do not use outside knowledge. "
    "If the evidence does not contain the answer, respond exactly: "
    "'I cannot determine the answer from the provided evidence.' "
    "Otherwise give the answer as a single concise number or yes/no, "
    "matching the units shown in the evidence."
)

_FINAL_ANSWER_INSTRUCTION = (
    " Think step by step about which evidence is relevant and any calculation needed, "
    "then end your response with a final line in the exact form: "
    "FINAL ANSWER: <answer>"
)

_JSON_INSTRUCTION = (
    " Respond ONLY with a single valid json object (no prose, no code fences) with "
    'exactly these keys: "reasoning" (your brief step-by-step), "answer" (the final '
    'answer as a number or yes/no, matching the evidence units), and "answer_type" '
    '("number", "yes_no", or "text").'
)


def _evidence_block(context: list[tuple[str, str]]) -> str:
    return f"Evidence:\n{format_context(context)}\n"


def _question_block(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def build_zero_shot(question, context, few_shot_examples=None):
    messages = [
        {"role": "system", "content": _SYSTEM_BASE},
        {"role": "user", "content": _evidence_block(context) + _question_block(question)},
    ]
    return messages, False


def build_few_shot(question, context, few_shot_examples=None):
    examples = few_shot_examples or []
    messages = [{"role": "system", "content": _SYSTEM_BASE}]
    for ex in examples:
        messages.append(
            {"role": "user", "content": f"Evidence:\n{ex['evidence']}\nQuestion: {ex['question']}\nAnswer:"}
        )
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append(
        {"role": "user", "content": _evidence_block(context) + _question_block(question)}
    )
    return messages, False


def build_cot(question, context, few_shot_examples=None):
    messages = [
        {"role": "system", "content": _SYSTEM_BASE + _FINAL_ANSWER_INSTRUCTION},
        {"role": "user", "content": _evidence_block(context) + _question_block(question)},
    ]
    return messages, False


def build_structured(question, context, few_shot_examples=None):
    messages = [
        {"role": "system", "content": _SYSTEM_BASE + _JSON_INSTRUCTION},
        {"role": "user", "content": _evidence_block(context) + _question_block(question)},
    ]
    return messages, True


STRATEGIES = {
    "zero_shot": build_zero_shot,
    "few_shot": build_few_shot,
    "cot": build_cot,
    "structured": build_structured,
}


def build_messages(strategy, question, context, few_shot_examples=None):
    """Dispatch to a strategy builder. Raises ValueError on an unknown strategy."""
    try:
        builder = STRATEGIES[strategy]
    except KeyError:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {list(STRATEGIES)}")
    return builder(question, context, few_shot_examples=few_shot_examples)


def _classify(qa_row: dict) -> str:
    exe = qa_row["gold_answer_exe"]
    if isinstance(exe, str):
        return "yes_no"
    if "%" in str(qa_row.get("gold_answer", "")):
        return "percent"
    return "numeric"


def _example_from_qa(qa_row: dict) -> dict:
    chunks = get_chunk_contents(qa_row["doc_id"], qa_row["gold_evidence_ids"][:2])
    answer = str(qa_row.get("gold_answer", "")).strip()
    if not answer:
        answer = str(qa_row["gold_answer_exe"])
    return {
        "question": qa_row["question"],
        "evidence": format_context(chunks, with_citations=False),
        "answer": answer,
    }


def load_few_shot_examples(n: int = 3, seed: int = 42) -> list[dict]:
    """Return n stratified train demonstrations (yes-no, percent, then numeric), seeded
    so the sample is reproducible."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {"yes_no": [], "percent": [], "numeric": []}
    for q in load_qa():
        if q["split"] != "train":
            continue
        buckets[_classify(q)].append(q)

    chosen: list[dict] = []
    
    while len(chosen) < n:
        progressed = False
        for key in ("yes_no", "percent", "numeric"):
            if len(chosen) >= n:
                break
            if buckets[key]:
                idx = rng.randrange(len(buckets[key]))
                chosen.append(buckets[key].pop(idx))
                progressed = True
        if not progressed:
            break  

    return [_example_from_qa(q) for q in chosen[:n]]
