"""Build the labeled dataset for hallucination detection (Stage 4, Member 3).

Takes answers our RAG pipeline already generated and labels each one as supported or
hallucinated, so we can measure how well the verifiers work.

The rule:
    gold evidence retrieved + answer correct  ->  supported
    gold evidence retrieved + answer wrong    ->  hallucinated  (it had what it needed)
    gold evidence MISSING   + refused         ->  supported     (refusing is the right call)
    gold evidence MISSING   + answer wrong    ->  hallucinated  (it made something up)
    gold evidence MISSING   + answer correct  ->  skip the row

Only the last case gets skipped. If the evidence wasn't there and the model still got it
right, we can't tell whether it guessed luckily or found the answer some other way, so
the label would be a coin flip.

The "evidence missing" rows matter a lot: that's where the model invents things. One of
ours answered '-2017' to a percentage question — it grabbed a year off the page and
reported it as a financial figure. Dropping those rows would leave the dataset with only
arithmetic slips and none of the outright fabrication a detector most needs to catch.

Usage:
    python scripts/build_detection_set.py \
        --runs data/runs/paired_hybrid.jsonl data/runs/paired_rerank.jsonl \
        --out data/processed/detection_eval.jsonl
"""

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
    """Did the model refuse to answer? Refusing is the safe option, not a hallucination."""
    answer = str(row.get("answer", ""))
    if row.get("abstained") or not answer.strip():
        return True
    # Older runs didn't set `abstained` properly, so also check the text itself.
    return normalize_answer(answer)[1] == "abstention"


def label_row(row, gold_ids, k):
    """Return (supported, reason). `supported` is None when we should skip the row."""
    if not gold_ids:
        return None, "no_gold_evidence"

    has_evidence = recall_at_k(row.get("retrieved_ids", []), gold_ids, k) == 1.0

    # Refusing is the safe failure mode either way — it's never a hallucination.
    if declined(row):
        return True, "abstention" if has_evidence else "abstention_no_evidence"

    if has_evidence:
        return (True, "correct") if row.get("correct") else (False, "wrong_despite_evidence")

    # No gold evidence in the context. A wrong answer here had nothing to stand on, so
    # it's a fabrication. A *right* answer is ambiguous — lucky guess, or the figure was
    # reachable some other way — so we can't label it and skip it.
    if row.get("correct"):
        return None, "correct_without_evidence"
    return False, "fabricated_without_evidence"


def build(run_paths, k):
    """Label every answer in the given run files. Returns (rows, counts)."""
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

            # Save the evidence text with the row so the file works on its own.
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
                    # Kept for our own analysis. The judge never sees these.
                    "gold_answer": qa["gold_answer"],
                    "gold_answer_exe": qa["gold_answer_exe"],
                    "gold_evidence_ids": qa["gold_evidence_ids"],
                    "retriever": run.get("retriever"),
                    # Kept so results can be split by prompt strategy — one question can
                    # appear several times here, once per strategy that answered it.
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
