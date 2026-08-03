"""RAG generation stage: turn retrieved evidence into answers.

Public API:
    generate_answer(question, context, strategy="zero_shot", ...) -> dict
    STRATEGIES                          # {name: builder} (zero_shot, few_shot, cot, structured)
    load_few_shot_examples(n, seed)     # stratified train demonstrations
    normalize_answer / answers_match    # numeric-tolerant answer handling
    get_chunk_contents / format_context # local_id -> prompt-ready text
"""

from src.generation.answer import answers_match, extract_final_answer, normalize_answer
from src.generation.context import format_context, get_chunk_contents, get_chunks_by_doc
from src.generation.generate import generate_answer
from src.generation.prompts import STRATEGIES, build_messages, load_few_shot_examples

__all__ = [
    "generate_answer",
    "STRATEGIES",
    "build_messages",
    "load_few_shot_examples",
    "normalize_answer",
    "answers_match",
    "extract_final_answer",
    "get_chunk_contents",
    "get_chunks_by_doc",
    "format_context",
]
