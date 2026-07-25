# scripts/eval_retrieval.py
"""Compute Recall@k for a retrieval predictions file.

Generic across retrievers — works for BM25, dense, or hybrid as long as
they all produce predictions in the shared format.

Run from repo root:
    python scripts/eval_retrieval.py data/processed/bm25_predictions_dev.json
"""
import argparse
import json
from typing import Dict


def recall_at_k(predictions: Dict, k: int) -> float:
    """Average per-example recall: fraction of gold items appearing in top-k."""
    total = 0
    running = 0.0
    for pred in predictions.values():
        gold = set(pred["gold_inds"])
        if not gold:
            continue
        retrieved = {r["id"] for r in pred["retrieved"][:k]}
        running += len(gold & retrieved) / len(gold)
        total += 1
    return running / total if total else 0.0


def recall_by_evidence_type(predictions: Dict, k: int) -> Dict:
    """Break out recall@k by whether gold evidence is text-only, table-only, or mixed."""
    buckets = {"text_only": [], "table_only": [], "mixed": []}
    for pred in predictions.values():
        gold = set(pred["gold_inds"])
        if not gold:
            continue
        has_text = any(g.startswith("text_") for g in gold)
        has_table = any(g.startswith("table_") for g in gold)
        if has_text and has_table:
            bucket = "mixed"
        elif has_text:
            bucket = "text_only"
        else:
            bucket = "table_only"
        retrieved = {r["id"] for r in pred["retrieved"][:k]}
        buckets[bucket].append(len(gold & retrieved) / len(gold))
    return {
        b: {"recall": (sum(vs) / len(vs) if vs else 0.0), "n": len(vs)}
        for b, vs in buckets.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", help="Path to predictions JSON")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    args = ap.parse_args()

    with open(args.predictions) as f:
        preds = json.load(f)

    n_with_gold = sum(1 for p in preds.values() if p["gold_inds"])
    print(f"\nPredictions: {len(preds)}  |  With gold evidence: {n_with_gold}\n")

    print("Overall Recall@k")
    print("-" * 32)
    for k in args.ks:
        print(f"  Recall@{k:<2}: {recall_at_k(preds, k):.4f}")

    print("\nRecall@5 by evidence type")
    print("-" * 32)
    breakdown = recall_by_evidence_type(preds, k=5)
    for bucket, info in breakdown.items():
        print(f"  {bucket:<11} (n={info['n']:>5}): {info['recall']:.4f}")


if __name__ == "__main__":
    main()