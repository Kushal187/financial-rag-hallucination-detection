"""Evaluation metrics for the FinQA RAG + hallucination-detection pipeline.

Retrieval metrics live here. `evaluate_retriever` is retriever-agnostic: it takes a
`retrieve(question, doc_id, k) -> ranked local_ids` function, so dense, BM25, hybrid and
the cross-encoder reranker are all scored by the same harness (and their numbers stay
comparable).

Retrieval is the expensive step once a reranker is in the mix (a cross-encoder runs
~`pool` forward passes per question, versus one vector lookup). So the harness splits
into `collect_rankings`, which retrieves **once** per question, and pure aggregators
over those records — recall, per-type recall, latency — none of which re-retrieve.

Still to implement in later stages:
  - generation:  numeric-tolerant exact match (FinQA answers are mostly numbers)
                 -- lives in src/generation/answer.py::answers_match for now
  - detection:   precision / recall / f1
"""

import time
from collections.abc import Callable, Sequence


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Per-query recall@k: fraction of gold evidence ids found in the top-k retrieved ids.

    `|gold ∩ top_k| / |gold|`. Returns 1.0 when there is no gold evidence (vacuously
    complete), so such rows don't drag the average down.
    """
    gold = set(gold_ids)
    if not gold:
        return 1.0
    top_k = set(retrieved_ids[:k])
    return len(gold & top_k) / len(gold)


def collect_rankings(
    retrieve_fn: Callable[[str, str, int], list[str]],
    qa_rows: list[dict],
    k: int,
) -> list[dict]:
    """Run `retrieve_fn` once per question at depth `k`, keeping the ranking and latency.

    Every metric below is derived from these records instead of re-retrieving, and any
    k' <= k is a prefix slice of `ranked`. Keeping the raw rankings around also makes
    failure analysis possible (which questions did a reranker help or hurt?) without
    paying for retrieval a second time.

    Returns `[{"id", "doc_id", "gold_evidence_ids", "ranked", "latency_ms"}, ...]`.
    """
    records: list[dict] = []
    for row in qa_rows:
        start = time.perf_counter()
        ranked = retrieve_fn(row["question"], row["doc_id"], k)
        latency_ms = (time.perf_counter() - start) * 1000.0
        records.append(
            {
                # `id` is only needed to join records back to questions for failure
                # analysis; synthetic rows in tests don't carry one.
                "id": row.get("id"),
                "doc_id": row["doc_id"],
                "gold_evidence_ids": list(row["gold_evidence_ids"]),
                "ranked": list(ranked),
                "latency_ms": latency_ms,
            }
        )
    return records


def summarize_rankings(
    records: list[dict], ks: Sequence[int] = (1, 5, 10)
) -> dict[int, dict[str, float]]:
    """Aggregate `collect_rankings` output into recall@k / full@k for each k in `ks`.

    Returns `{k: {"recall": mean per-query recall@k,
                  "full":   fraction of questions with ALL gold evidence in the top-k}}`.
    "full" matters for FinQA: a question is only answerable if every gold fact is retrieved.
    """
    out: dict[int, dict[str, float]] = {}
    for k in sorted(ks):
        vals = [recall_at_k(r["ranked"], r["gold_evidence_ids"], k) for r in records]
        out[k] = {
            "recall": sum(vals) / len(vals) if vals else 0.0,
            "full": sum(v == 1.0 for v in vals) / len(vals) if vals else 0.0,
        }
    return out


def evaluate_retriever(
    retrieve_fn: Callable[[str, str, int], list[str]],
    qa_rows: list[dict],
    ks: Sequence[int] = (1, 5, 10),
) -> dict[int, dict[str, float]]:
    """Score a retriever over QA rows, reporting recall@k for each k in `ks`.

    `retrieve_fn(question, doc_id, k)` must return ranked chunk local_ids. Retrieval runs
    once per question at the largest k, then each k is a prefix slice — so cost is one
    retrieval per question, not one per k.
    """
    ks = tuple(sorted(ks))
    return summarize_rankings(collect_rankings(retrieve_fn, qa_rows, max(ks)), ks)


def _gold_type(local_id: str) -> str:
    """`"table_3" -> "table"`, `"text_1" -> "text"`.

    The chunk type is encoded in the `local_id` prefix (docs/data_schema.md), so this
    needs no corpus lookup.
    """
    return local_id.split("_", 1)[0]


def per_type_recall(
    records: list[dict], ks: Sequence[int] = (1, 5, 10)
) -> dict[str, dict[int, float]]:
    """Recall@k split by gold evidence type (`table` vs `text`).

    ~60% of FinQA gold evidence is table-only (docs/dataset_stats.json), and linearized
    table rows are near-duplicates of one another — so a retriever can look healthy
    overall while being much weaker on exactly the rows that decide the answer. This
    separates the two.

    Micro-averaged over gold *items*, not questions: a question with mixed evidence
    contributes to both buckets.

    Returns `{"table": {k: recall}, "text": {k: recall}}`.
    """
    ks = tuple(sorted(ks))
    # type -> k -> [found, total]
    tallies: dict[str, dict[int, list[int]]] = {}
    for rec in records:
        for k in ks:
            top_k = set(rec["ranked"][:k])
            for gold_id in rec["gold_evidence_ids"]:
                bucket = tallies.setdefault(_gold_type(gold_id), {kk: [0, 0] for kk in ks})[k]
                bucket[0] += gold_id in top_k
                bucket[1] += 1
    return {
        gold_type: {k: (found / total if total else 0.0) for k, (found, total) in by_k.items()}
        for gold_type, by_k in tallies.items()
    }


def mean_latency_ms(records: list[dict]) -> float:
    """Mean wall-clock retrieval latency per question, in milliseconds."""
    if not records:
        return 0.0
    return sum(r["latency_ms"] for r in records) / len(records)


def compare_retrievers(
    results_by_name: dict[str, dict[int, dict[str, float]]],
    ks: Sequence[int] = (1, 5, 10),
    latency_by_name: dict[str, float] | None = None,
) -> str:
    """Render a markdown comparison table, paste-ready for docs/.

    Recall@k for every k, then Full@k at the largest k (the "is this question even
    answerable" number), then optional mean latency so an accuracy gain can be read
    against what it costs.
    """
    ks = tuple(sorted(ks))
    max_k = max(ks) if ks else 0

    headers = ["retriever"] + [f"Recall@{k}" for k in ks] + [f"Full@{max_k}"]
    if latency_by_name:
        headers.append("ms/query")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for name, results in results_by_name.items():
        cells = [name] + [f"{results[k]['recall']:.1%}" for k in ks]
        cells.append(f"{results[max_k]['full']:.1%}")
        if latency_by_name:
            cells.append(f"{latency_by_name.get(name, 0.0):.0f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
