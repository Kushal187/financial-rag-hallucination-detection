"""Sweep the hybrid retriever's alpha (BM25<->vector blend) on the dev split to pick the
best setting before running the final held-out evaluation on test.

Usage:
    python scripts/sweep_hybrid_alpha.py --split dev --alphas 0.0 0.25 0.5 0.75 1.0
"""

import argparse
import functools
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_qa  # noqa: E402
from src.eval.metrics import evaluate_retriever  # noqa: E402
from src.retrieval import hybrid, weaviate_store  # noqa: E402

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    qa = [q for q in load_qa() if q["split"] == args.split]
    if args.limit:
        qa = qa[: args.limit]
    print(f"Sweeping hybrid alpha on {len(qa)} '{args.split}' questions...\n")

    ks = tuple(sorted(args.k))
    header = "alpha".rjust(6) + "".join(f"Recall@{k}".rjust(11) for k in ks)
    print(header)

    try:
        for alpha in args.alphas:
            retrieve_fn = functools.partial(hybrid.retrieve, alpha=alpha)
            results = evaluate_retriever(retrieve_fn, qa, ks=ks)
            row = f"{alpha:>6.2f}" + "".join(f"{results[k]['recall']:>10.1%} " for k in ks)
            print(row)
    finally:
        weaviate_store.close_client()


if __name__ == "__main__":
    main()
