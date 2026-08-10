"""Resolve retrieved chunk local_ids back to their text and format them for a prompt.

Retrieval returns only local_ids, but the generator needs the content and the judge needs
the ids to cite. The index is built once and reused, since the corpus is ~86k rows. The
shared context type is `list[tuple[local_id, content]]`.
"""

from src.data.load_data import load_chunks

NO_EVIDENCE = "[no evidence retrieved]"

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
    """Map `local_ids` to `[(local_id, content), ...]` in input order, skipping any id
    that doesn't resolve."""
    by_id = _get_index().get(doc_id, {})
    out: list[tuple[str, str]] = []
    for lid in local_ids:
        content = by_id.get(lid)
        if content is not None:
            out.append((lid, content))
    return out


def format_context(chunks: list[tuple[str, str]], with_citations: bool = True) -> str:
    """Render retrieved chunks as a numbered evidence block so the judge can cite them.
    An empty list yields NO_EVIDENCE."""
    if not chunks:
        return NO_EVIDENCE
    lines: list[str] = []
    for i, (local_id, content) in enumerate(chunks, start=1):
        if with_citations:
            lines.append(f"Evidence [{i}] ({local_id}): {content}")
        else:
            lines.append(content)
    return "\n".join(lines)
