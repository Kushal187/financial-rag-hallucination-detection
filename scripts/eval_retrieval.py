"""Evaluate a retriever on FinQA: Recall@k over a split.

Reports, per k:
  Recall@k  mean per-query fraction of gold evidence retrieved
  Full@k    fraction of questions with ALL gold evidence in the top-k (answerable)

Usage:
    python scripts/eval_retrieval.py --split test
    python scripts/eval_retrieval.py --split dev --k 1 3 5 10
    python scripts/eval_retrieval.py --split dev --retriever hybrid --alpha 0.7
"""

import argparse
import functools
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_qa  # noqa: E402
from src.eval.metrics import evaluate_retriever  # noqa: E402
from src.retrieval import dense, hybrid, weaviate_store  # noqa: E402

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", help="split to evaluate (default: test)")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    parser.add_argument("--retriever", choices=["dense", "hybrid"], default="dense")
    parser.add_argument("--alpha", type=float, default=hybrid.DEFAULT_ALPHA, help="hybrid only: 0=BM25, 1=vector")
    args = parser.parse_args()

    qa = [q for q in load_qa() if q["split"] == args.split]
    if args.limit:
        qa = qa[: args.limit]

    if args.retriever == "hybrid":
        retrieve_fn = functools.partial(hybrid.retrieve, alpha=args.alpha)
        print(f"Evaluating hybrid retrieval (alpha={args.alpha}) on {len(qa)} '{args.split}' questions...")
    else:
        retrieve_fn = dense.retrieve
        print(f"Evaluating dense retrieval on {len(qa)} '{args.split}' questions...")

    try:
        results = evaluate_retriever(retrieve_fn, qa, ks=tuple(args.k))
    finally:
        weaviate_store.close_client()

    print(f"\n{'k':>4}  {'Recall@k':>9}  {'Full@k':>7}")
    for k in sorted(results):
        r = results[k]
        print(f"{k:>4}  {r['recall']:>8.1%}  {r['full']:>6.1%}")


if __name__ == "__main__":
    main()

