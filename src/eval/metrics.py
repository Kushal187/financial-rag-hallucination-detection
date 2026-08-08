"""Retrieval metrics for the FinQA RAG pipeline.

`evaluate_retriever` takes any `retrieve(question, doc_id, k) -> ranked local_ids`
function, so BM25, dense, hybrid and the reranker are scored by the same harness and
their numbers stay comparable. Retrieval happens once per question in `collect_rankings`;
everything else is a pure aggregation over those records, which matters once a
cross-encoder is in the mix and retrieval is the expensive step.
"""

import time
from collections.abc import Callable, Sequence


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Fraction of gold evidence ids found in the top-k. Returns 1.0 when there is no gold
    evidence, so those rows don't drag the average down."""
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

    Any k' <= k is then a prefix slice of `ranked`, and keeping the rankings makes failure
    analysis possible without retrieving again.

    Returns `[{"id", "doc_id", "gold_evidence_ids", "ranked", "latency_ms"}, ...]`.
    """
    records: list[dict] = []
    for row in qa_rows:
        start = time.perf_counter()
        ranked = retrieve_fn(row["question"], row["doc_id"], k)
        latency_ms = (time.perf_counter() - start) * 1000.0
        records.append(
            {
                # optional; only used to join records back to questions
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
    """Aggregate `collect_rankings` output into recall@k and full@k for each k in `ks`.

    `full` is the fraction of questions with *all* their gold evidence in the top-k, which
    is what decides whether a FinQA question is answerable at all.
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

    Retrieval runs once per question at the largest k, then each k is a prefix slice, so
    the cost is one retrieval per question rather than one per k.
    """
    ks = tuple(sorted(ks))
    return summarize_rankings(collect_rankings(retrieve_fn, qa_rows, max(ks)), ks)


def _gold_type(local_id: str) -> str:
    """`"table_3" -> "table"`. The chunk type is encoded in the local_id prefix, so this
    needs no corpus lookup."""
    return local_id.split("_", 1)[0]


def per_type_recall(
    records: list[dict], ks: Sequence[int] = (1, 5, 10)
) -> dict[str, dict[int, float]]:
    """Recall@k split by gold evidence type (`table` vs `text`).

    Most FinQA gold evidence is table-only and linearized table rows look alike, so a
    retriever can score well overall while being much weaker on the rows that actually
    decide the answer.

    Micro-averaged over gold items rather than questions, so a question with mixed
    evidence counts in both buckets. Returns `{"table": {k: recall}, "text": {k: recall}}`.
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
    """Render a markdown comparison table: recall@k for each k, full@k at the largest k,
    and mean latency when it is supplied, so an accuracy gain can be read against its cost."""
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
