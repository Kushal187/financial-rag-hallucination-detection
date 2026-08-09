
import argparse
import functools
import json
import os
import sys
import time

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_qa  # noqa: E402
from src.generation import (  # noqa: E402
    answers_match,
    generate_answer,
    get_chunk_contents,
    load_few_shot_examples,
)
from src.generation.prompts import STRATEGIES  # noqa: E402

load_dotenv()

_RUNS_DIR = "data/runs"


def _build_retriever(name: str, alpha: float | None, pool: int | None = None):
    if name == "bm25":
        from src.retrieval import bm25

        return bm25.retrieve
    if name == "hybrid":
        from src.retrieval import hybrid

        if alpha is None:
            alpha = hybrid.DEFAULT_ALPHA
        return functools.partial(hybrid.retrieve, alpha=alpha)
    if name == "dense":
        from src.retrieval import dense

        return dense.retrieve
    if name == "rerank":
        from src.retrieval import rerank

        if pool is None:
            pool = rerank.DEFAULT_POOL
        return functools.partial(rerank.retrieve, pool=pool)
    raise ValueError(f"unknown retriever {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES)
    )
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid", "rerank"], default="hybrid")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=None, help="hybrid only (default: hybrid.DEFAULT_ALPHA)")
    parser.add_argument("--pool", type=int, default=None, help="rerank only (default: rerank.DEFAULT_POOL)")
    parser.add_argument("--seed", type=int, default=42, help="few-shot example sampling seed")
    parser.add_argument("--save-messages", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    unknown = [s for s in args.strategies if s not in STRATEGIES]
    if unknown:
        parser.error(f"unknown strategies {unknown}; choose from {list(STRATEGIES)}")

    qa = [q for q in load_qa() if q["split"] == args.split]
    if args.limit:
        qa = qa[: args.limit]

    few_shot = (
        load_few_shot_examples(n=3, seed=args.seed)
        if "few_shot" in args.strategies
        else None
    )

    retrieve_fn = _build_retriever(args.retriever, args.alpha, args.pool)
    os.makedirs(_RUNS_DIR, exist_ok=True)
    out_path = args.out or os.path.join(
        _RUNS_DIR, f"gen_{args.split}_{args.retriever}_k{args.k}_{int(time.time())}.jsonl"
    )

    print(
        f"Generating on {len(qa)} '{args.split}' questions | "
        f"retriever={args.retriever} k={args.k} alpha={args.alpha if args.alpha is not None else 'n/a'} | "
        f"strategies={args.strategies}"
    )

    stats = {s: {"correct": 0, "answered": 0, "latency_ms": 0.0, "n": 0} for s in args.strategies}

    try:
        with open(out_path, "w", encoding="utf-8") as out_f:
            for i, row in enumerate(qa, 1):
                retrieved = retrieve_fn(row["question"], row["doc_id"], args.k)
                context = get_chunk_contents(row["doc_id"], retrieved)
                for strategy in args.strategies:
                    gen = generate_answer(
                        row["question"],
                        context,
                        strategy=strategy,
                        few_shot_examples=few_shot,
                        save_messages=args.save_messages,
                    )
                    abstained = gen["answer_type"] in ("abstention", "none")
                    correct = answers_match(
                        gen["answer"], row["gold_answer"], row["gold_answer_exe"]
                    )
                    st = stats[strategy]
                    st["n"] += 1
                    st["correct"] += int(correct)
                    st["answered"] += int(not abstained)
                    st["latency_ms"] += gen["latency_ms"]

                    out_f.write(
                        json.dumps(
                            {
                                "id": row["id"],
                                "doc_id": row["doc_id"],
                                "question": row["question"],
                                "split": row["split"],
                                "strategy": strategy,
                                "retriever": args.retriever,
                                "k": args.k,
                                "retrieved_ids": retrieved,
                                "answer": gen["answer"],
                                "answer_type": gen["answer_type"],
                                "correct": correct,
                                "abstained": abstained,
                                "gold_answer": row["gold_answer"],
                                "gold_answer_exe": row["gold_answer_exe"],
                                "latency_ms": gen["latency_ms"],
                                "raw": gen["raw"],
                                **({"messages": gen["messages"]} if args.save_messages else {}),
                            }
                        )
                        + "\n"
                    )
                if i % 10 == 0:
                    print(f"  ...{i}/{len(qa)}")
    finally:
        if args.retriever != "bm25":
            from src.retrieval import weaviate_store

            weaviate_store.close_client()

    print(f"\nResults (n per strategy shown; wrote {out_path})")
    print(f"{'strategy':<12} {'accuracy':>9} {'attempted':>10} {'%answered':>10} {'avg_lat_ms':>11} {'n':>5}")
    for s in args.strategies:
        st = stats[s]
        n = st["n"] or 1
        attempted = st["answered"] or 1
        print(
            f"{s:<12} {st['correct'] / n:>8.1%} {st['correct'] / attempted:>9.1%} "
            f"{st['answered'] / n:>9.1%} {st['latency_ms'] / n:>10.0f} {st['n']:>5}"
        )


if __name__ == "__main__":
    main()
