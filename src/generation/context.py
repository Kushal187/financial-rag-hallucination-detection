"""Resolve retrieved chunk `local_id`s back to their text and format them for a prompt.

Retrieval returns only ``local_id`` strings (the shared contract in
``src.retrieval.*``); the generator needs the actual ``content`` to build a prompt,
and the judge needs the ``local_id``s to cite evidence. This module bridges the two
by building a doc-scoped ``{local_id: content}`` index **once** (the corpus is ~86k
rows — never re-read it per question; ``src.retrieval.bm25`` caches the same way via
``load_chunks_by_doc``).

The context seam used across generation + detection is ``list[tuple[local_id, content]]``
— bare strings lose the id the judge cites; the full chunk dict couples generation to
storage. Tuples are the minimal contract.
"""

from src.data.load_data import load_chunks

__all__ = ["get_chunk_contents", "get_chunks_by_doc", "format_context", "NO_EVIDENCE"]

NO_EVIDENCE = "[no evidence retrieved]"

# {doc_id: {local_id: content}} — built once, reused across all calls.
_INDEX: dict[str, dict[str, str]] | None = None


def _get_index() -> dict[str, dict[str, str]]:
    """Lazily build the doc-scoped index from the processed corpus."""
    global _INDEX
    if _INDEX is None:
        index: dict[str, dict[str, str]] = {}
        for chunk in load_chunks():
            index.setdefault(chunk["doc_id"], {})[chunk["local_id"]] = chunk["content"]
        _INDEX = index
    return _INDEX


def get_chunks_by_doc(doc_id: str) -> dict[str, str]:
    """Return ``{local_id: content}`` for every chunk in a document (empty if unknown)."""
    return dict(_get_index().get(doc_id, {}))


def get_chunk_contents(doc_id: str, local_ids: list[str]) -> list[tuple[str, str]]:
    """Map ``local_ids`` -> ``[(local_id, content), ...]`` in input order.

    Unknown ids are skipped (a retriever could in principle return an id that doesn't
    resolve; better to drop it silently than crash a run).
    """
    by_id = _get_index().get(doc_id, {})
    out: list[tuple[str, str]] = []
    for lid in local_ids:
        content = by_id.get(lid)
        if content is not None:
            out.append((lid, content))
    return out


def format_context(chunks: list[tuple[str, str]], with_citations: bool = True) -> str:
    """Render retrieved chunks into a prompt-ready evidence block.

    Numbered so the judge can cite ``[1]``/``[2]`` and so the generator sees ordering.
    An empty list yields ``NO_EVIDENCE`` so downstream code can detect abstention.
    """
    if not chunks:
        return NO_EVIDENCE
    lines: list[str] = []
    for i, (local_id, content) in enumerate(chunks, start=1):
        if with_citations:
            lines.append(f"Evidence [{i}] ({local_id}): {content}")
        else:
            lines.append(content)
    return "\n".join(lines)
