

import argparse
import json
import os
import sys
import time
from collections import Counter

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection import verify  # noqa: E402
from src.generation.context import get_chunk_contents  # noqa: E402

load_dotenv()

_RUNS_DIR = "data/runs"


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _context_for(row: dict) -> list[tuple[str, str]]:
    ctx = row.get("context")
    if isinstance(ctx, list) and ctx and isinstance(ctx[0], (list, tuple)):
        return [(str(a), str(b)) for a, b in ctx]
    return get_chunk_contents(row["doc_id"], row.get("retrieved_ids", []))


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL of generated answers to judge")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["grounding", "evidence"],
        default="grounding",
        help="grounding: does the answer follow from the evidence (default). "
        "evidence: was there enough evidence to answer at all — no arithmetic, and the "
        "candidate answer is never shown to the judge",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = _read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]

    os.makedirs(_RUNS_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_path = args.out or os.path.join(_RUNS_DIR, f"judge_{base}_{int(time.time())}.jsonl")

    print(f"Judging {len(rows)} answers from {args.input}")
    verdict_counts = Counter()
    category_counts = Counter()
    tp = fp = fn = 0  
    has_labels = False

    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows, 1):
            context = _context_for(row)
            verdict = verify(
                row["question"], context, str(row.get("answer", "")), mode=args.mode
            )
            verdict_counts["supported" if verdict["supported"] else "hallucinated"] += 1
            category_counts[verdict["category"]] += 1

            if "gold_supported" in row:
                has_labels = True
                gold_halluc = not bool(row["gold_supported"])
                pred_halluc = not verdict["supported"]
                if pred_halluc and gold_halluc:
                    tp += 1
                elif pred_halluc:
                    fp += 1
                elif gold_halluc:
                    fn += 1

            out_f.write(json.dumps({**row, "verdict": verdict}) + "\n")
            if i % 10 == 0:
                print(f"  ...{i}/{len(rows)}")

    print(f"\nWrote {out_path}\n")
    print("Verdict distribution:")
    for label, n in verdict_counts.most_common():
        print(f"  {label:<14} {n:>5}  ({n / len(rows):.1%})")
    print("\nCategory distribution:")
    for label, n in category_counts.most_common():
        print(f"  {label:<20} {n:>5}  ({n / len(rows):.1%})")

    if has_labels:
        precision, recall, f1 = _prf(tp, fp, fn)
        print(
            f"\nvs gold labels (positive = hallucinated): "
            f"P={precision:.2f} R={recall:.2f} F1={f1:.2f}  (tp={tp} fp={fp} fn={fn})"
        )


if __name__ == "__main__":
    main()
