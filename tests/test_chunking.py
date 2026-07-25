"""Validate that chunking preserves the FinQA gold-evidence ID scheme.

The grading premise is that every `gold_inds` key resolves to a produced chunk's
`local_id`. If that breaks, retrieval can never be scored, so these tests guard it.
"""

import json
import os

import pytest

from src.data.chunking import linearize_table_row, record_to_chunks, record_to_qa

RAW = os.path.join(os.getenv("DATA_RAW_DIR", "data/raw"), "finqa", "train.json")


@pytest.fixture(scope="module")
def records() -> list[dict]:
    if not os.path.exists(RAW):
        pytest.skip(f"{RAW} not present")
    with open(RAW, encoding="utf-8") as f:
        return json.load(f)


def test_gold_evidence_ids_all_resolve(records):
    """Every gold_inds key must appear as a chunk local_id (sampled for speed)."""
    for record in records[:500]:
        local_ids = {c["local_id"] for c in record_to_chunks(record)}
        for ev in record_to_qa(record)["gold_evidence_ids"]:
            assert ev in local_ids, f"{record['id']}: {ev} missing from chunks"


def test_text_and_table_indexing(records):
    """text_0 == pre_text[0]; table_N == table[N] for every row including table_0."""
    record = next(r for r in records if r.get("pre_text") and len(r.get("table", [])) > 1)
    chunks = {c["local_id"]: c for c in record_to_chunks(record)}

    assert chunks["text_0"]["content"] == record["pre_text"][0].strip()
    table_ids = {lid for lid in chunks if lid.startswith("table_")}
    assert table_ids == {f"table_{i}" for i in range(len(record["table"]))}
    assert chunks["table_0"]["chunk_type"] == "table_row"


def test_linearize_table_row():
    header = ["", "2009", "2008"]
    row = ["revenue", "$ 100", "$ 90"]
    assert linearize_table_row(header, row) == "the revenue of 2009 is $ 100 ; the revenue of 2008 is $ 90 ."

    header2 = ["year", "gallons"]
    row2 = ["2018", "4447"]
    assert linearize_table_row(header2, row2) == "year the 2018 of gallons is 4447 ."
