"""Compare the cross-encoder reranker against the first-stage retrievers, with recall
split by evidence type and a win/loss analysis of the questions reranking changed."""

import argparse
import functools
import os
import sys
from collections.abc import Callable

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_qa  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    collect_rankings,
    compare_retrievers,
    mean_latency_ms,
    per_type_recall,
    recall_at_k,
    summarize_rankings,
)
from src.retrieval import bm25, dense, hybrid, rerank, weaviate_store  # noqa: E402

load_dotenv()


def _configs(args) -> list[tuple[str, Callable[[str, str, int], list[str]]]]:
    """Named `(label, retrieve_fn)` pairs: baselines first, then the rerank grid."""
    configs: list[tuple[str, Callable[[str, str, int], list[str]]]] = []
    for name in args.baselines:
        if name == "bm25":
            configs.append(("bm25", bm25.retrieve))
        elif name == "dense":
            configs.append(("dense", dense.retrieve))
        elif name == "hybrid":
            configs.append(
                (
                    f"hybrid (a={hybrid.DEFAULT_ALPHA})",
                    functools.partial(hybrid.retrieve, alpha=hybrid.DEFAULT_ALPHA),
                )
            )
    for base in args.bases:
        for pool in args.pools:
            configs.append(
                (
                    f"rerank ({base}, pool={pool})",
                    functools.partial(rerank.retrieve, pool=pool, base=base),
                )
            )
    return configs


def _flips(baseline_records: list[dict], variant_records: list[dict], k: int):
    baseline_by_id = {r["id"]: r for r in baseline_records}
    wins, losses = [], []
    for rec in variant_records:
        base = baseline_by_id.get(rec["id"])
        if base is None:
            continue
        gold = rec["gold_evidence_ids"]
        before = recall_at_k(base["ranked"], gold, k)
        after = recall_at_k(rec["ranked"], gold, k)
        if after > before:
            wins.append((rec, base, before, after))
        elif after < before:
            losses.append((rec, base, before, after))
    return wins, losses


def _print_flip_examples(title: str, flips: list, qa_by_id: dict, k: int, limit: int) -> None:
    print(f"\n{title} ({len(flips)} total, showing up to {limit})")
    for rec, base, before, after in flips[:limit]:
        question = qa_by_id.get(rec["id"], {}).get("question", "")
        print(f"  [{rec['id']}] recall@{k} {before:.0%} -> {after:.0%}")
        print(f"    q:        {question[:110]}")
        print(f"    gold:     {rec['gold_evidence_ids']}")
        print(f"    baseline: {base['ranked'][:k]}")
        print(f"    rerank:   {rec['ranked'][:k]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--pools", nargs="+", type=int, default=[10, 20, 30])
    parser.add_argument("--bases", nargs="+", choices=["hybrid", "bm25"], default=["hybrid"])
    parser.add_argument(
        "--baselines", nargs="+", choices=["bm25", "dense", "hybrid"], default=["bm25", "dense", "hybrid"]
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--examples", type=int, default=4, help="failure-analysis examples to print")
    parser.add_argument("--out", default=None, help="write a markdown report here (e.g. docs/rerank_results.md)")
    args = parser.parse_args()

    qa = [q for q in load_qa() if q["split"] == args.split]
    if args.limit:
        qa = qa[: args.limit]
    qa_by_id = {q["id"]: q for q in qa}

    ks = tuple(sorted(args.k))
    max_k = max(ks)
    configs = _configs(args)

    print(
        f"Evaluating {len(configs)} configs on {len(qa)} '{args.split}' questions "
        f"| ks={list(ks)} | rerank model={rerank.DEFAULT_MODEL}\n"
    )

    records_by_name: dict[str, list[dict]] = {}
    results_by_name: dict[str, dict[int, dict[str, float]]] = {}
    latency_by_name: dict[str, float] = {}

    try:
        for label, retrieve_fn in configs:
            print(f"  running {label} ...", flush=True)
            records = collect_rankings(retrieve_fn, qa, max_k)
            records_by_name[label] = records
            results_by_name[label] = summarize_rankings(records, ks)
            latency_by_name[label] = mean_latency_ms(records)
    finally:
        weaviate_store.close_client()

    table = compare_retrievers(results_by_name, ks, latency_by_name)
    print(f"\n## Retrieval comparison ({args.split}, n={len(qa)})\n")
    print(table)

    type_lines = [
        "| retriever | " + " | ".join(f"table@{k} | text@{k}" for k in ks) + " |",
        "|" + "|".join(["---"] * (1 + 2 * len(ks))) + "|",
    ]
    for label, records in records_by_name.items():
        by_type = per_type_recall(records, ks)
        cells = [label]
        for k in ks:
            cells.append(f"{by_type.get('table', {}).get(k, 0.0):.1%}")
            cells.append(f"{by_type.get('text', {}).get(k, 0.0):.1%}")
        type_lines.append("| " + " | ".join(cells) + " |")
    type_table = "\n".join(type_lines)
    print(f"\n## Recall@k by gold evidence type ({args.split}, n={len(qa)})\n")
    print(type_table)

    rerank_labels = [n for n in results_by_name if n.startswith("rerank")]
    baseline_label = next((n for n in results_by_name if n.startswith("hybrid")), None)
    flip_summary = ""
    if rerank_labels and baseline_label:
        best = max(rerank_labels, key=lambda n: results_by_name[n][ks[0]]["recall"])
        wins, losses = _flips(records_by_name[baseline_label], records_by_name[best], ks[0])
        flip_summary = (
            f"At k={ks[0]}, `{best}` vs `{baseline_label}`: "
            f"**{len(wins)} questions improved**, **{len(losses)} regressed** "
            f"(net {len(wins) - len(losses):+d} of {len(qa)})."
        )
        print(f"\n## Failure analysis\n\n{flip_summary}")
        _print_flip_examples("FIXED by reranking", wins, qa_by_id, ks[0], args.examples)
        _print_flip_examples("BROKEN by reranking", losses, qa_by_id, ks[0], args.examples)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(f"# Cross-encoder reranking results ({args.split} split)\n\n")
            f.write(
                f"Model: `{rerank.DEFAULT_MODEL}` · first-stage bases: {args.bases} · "
                f"pools: {args.pools} · n = {len(qa)} questions\n\n"
            )
            f.write("## Retrieval comparison\n\n" + table + "\n\n")
            f.write("## Recall@k by gold evidence type\n\n" + type_table + "\n\n")
            if flip_summary:
                f.write("## Failure analysis\n\n" + flip_summary + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
