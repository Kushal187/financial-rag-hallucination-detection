"""Convert raw FinQA records into corpus chunks and answer-key rows.

The local_id scheme must stay identical to FinQA's gold_inds keys so retrieval can be
graded against them: text_N is the Nth sentence, table_N the Nth table row linearized
against the header. Every row is emitted, including table_0, since gold_inds can cite it.
"""


def linearize_table_row(header: list[str], row: list[str]) -> str:
    """Turn a table row into a sentence using FinQA's template."""
    row_label = row[0].strip()
    parts = []
    for j in range(1, len(row)):
        col = header[j].strip() if j < len(header) else ""
        parts.append(f"the {row_label} of {col} is {row[j].strip()}")
    body = " ; ".join(parts)
    prefix = f"{header[0].strip()} " if header and header[0].strip() else ""
    return f"{prefix}{body} ."


def record_to_chunks(record: dict) -> list[dict]:
    """Emit one corpus chunk per text sentence and per table data row."""
    doc_id = record["filename"]
    pre_text = record.get("pre_text", [])
    post_text = record.get("post_text", [])
    table = record.get("table", [])

    chunks: list[dict] = []
    position = 0

    for i, sentence in enumerate(pre_text + post_text):
        local_id = f"text_{i}"
        chunks.append(
            {
                "chunk_id": f"{doc_id}::{local_id}",
                "doc_id": doc_id,
                "local_id": local_id,
                "chunk_type": "text",
                "content": sentence.strip(),
                "position": position,
            }
        )
        position += 1

    header = table[0] if table else []
    for i in range(len(table)):
        local_id = f"table_{i}"
        chunks.append(
            {
                "chunk_id": f"{doc_id}::{local_id}",
                "doc_id": doc_id,
                "local_id": local_id,
                "chunk_type": "table_row",
                "content": linearize_table_row(header, table[i]),
                "position": position,
            }
        )
        position += 1

    return chunks


def record_to_qa(record: dict) -> dict:
    """Emit the answer-key row for a FinQA record."""
    qa = record["qa"]
    n_text = len(record.get("pre_text", [])) + len(record.get("post_text", []))
    evidence = [_normalize_evidence_id(k, n_text) for k in qa.get("gold_inds", {})]
    return {
        "id": record["id"],
        "question": qa["question"],
        "doc_id": record["filename"],
        "gold_answer": qa.get("answer", ""),
        "gold_answer_exe": qa.get("exe_ans"),
        "gold_evidence_ids": evidence,
        "gold_program": qa.get("program", ""),
    }


def _normalize_evidence_id(local_id: str, n_text: int) -> str:
    """Resolve FinQA's occasional negative text index (e.g. text_-1 = last sentence)."""
    if local_id.startswith("text_"):
        idx = int(local_id.split("_", 1)[1])
        if idx < 0:
            return f"text_{n_text + idx}"
    return local_id
