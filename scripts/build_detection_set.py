"""Label generated answers as supported or hallucinated to produce the detection
evaluation set."""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_qa  # noqa: E402
from src.eval.metrics import recall_at_k  # noqa: E402
from src.generation.answer import normalize_answer  # noqa: E402
from src.generation.context import get_chunk_contents  # noqa: E402


def declined(row):
    answer = str(row.get("answer", ""))
    if row.get("abstained") or not answer.strip():
        return True
    # Older runs didn't set `abstained` properly, so also check the text itself.
    return normalize_answer(answer)[1] == "abstention"


def label_row(row, gold_ids, k):
    if not gold_ids:
        return None, "no_gold_evidence"

    has_evidence = recall_at_k(row.get("retrieved_ids", []), gold_ids, k) == 1.0

    if declined(row):
        return True, "abstention" if has_evidence else "abstention_no_evidence"

    if has_evidence:
        return (True, "correct") if row.get("correct") else (False, "wrong_despite_evidence")

    if row.get("correct"):
        return None, "correct_without_evidence"
    return False, "fabricated_without_evidence"


def build(run_paths, k):
    qa_by_id = {q["id"]: q for q in load_qa()}
    rows = []
    counts = Counter()

    for path in run_paths:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            run = json.loads(line)
            qa = qa_by_id[run["id"]]

            supported, reason = label_row(run, qa["gold_evidence_ids"], k)
            counts[reason] += 1
            if supported is None:
                continue

            context = get_chunk_contents(run["doc_id"], run["retrieved_ids"][:k])
            rows.append(
                {
                    "id": run["id"],
                    "doc_id": run["doc_id"],
                    "question": run["question"],
                    "context": [[chunk_id, text] for chunk_id, text in context],
                    "answer": str(run.get("answer", "")),
                    "gold_supported": supported,
                    "label_reason": reason,
                    "gold_answer": qa["gold_answer"],
                    "gold_answer_exe": qa["gold_answer_exe"],
                    "gold_evidence_ids": qa["gold_evidence_ids"],
                    "retriever": run.get("retriever"),
                    "strategy": run.get("strategy"),
                    "abstained": declined(run),
                }
            )
    return rows, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="output files from compare_prompts.py")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="data/processed/detection_eval.jsonl")
    args = parser.parse_args()

    rows, counts = build(args.runs, args.k)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    supported = sum(r["gold_supported"] for r in rows)
    print(f"Read {sum(counts.values())} answers, kept {len(rows)}\n")
    for reason, n in counts.most_common():
        print(f"  {reason:<24} {n:>4}")
    print(f"\nsupported {supported}   hallucinated {len(rows) - supported}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
