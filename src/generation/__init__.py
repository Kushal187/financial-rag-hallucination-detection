"""RAG generation stage: turn retrieved evidence into answers."""

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
