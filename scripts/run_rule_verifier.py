
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection import rule_based  # noqa: E402
from src.generation.context import get_chunk_contents  # noqa: E402


def context_for(row):
    ctx = row.get("context")
    if isinstance(ctx, list) and ctx and isinstance(ctx[0], (list, tuple)):
        return [(str(a), str(b)) for a, b in ctx]
    return get_chunk_contents(row["doc_id"], row.get("retrieved_ids", []))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="labeled set from build_detection_set.py")
    parser.add_argument("--tol", type=float, default=rule_based.DEFAULT_TOL,
                        help="how close a calculation has to be to count (default 0.01 = 1%%)")
    parser.add_argument("--out", default="data/runs/verdicts_rule_based.jsonl")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8") if line.strip()]

    categories = Counter()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            verdict = rule_based.verify(
                row["question"], context_for(row), str(row.get("answer", "")), tol=args.tol
            )
            categories[verdict["category"]] += 1
            f.write(json.dumps({**row, "verdict": verdict}) + "\n")

    print(f"Checked {len(rows)} answers at {args.tol:.0%} tolerance -> {args.out}\n")
    for category, n in categories.most_common():
        print(f"  {category:<20} {n:>4}  ({n / len(rows):.0%})")
    print("\nScore it with:")
    print(f"  python scripts/score_detection.py --input {args.out}")


if __name__ == "__main__":
    main()
