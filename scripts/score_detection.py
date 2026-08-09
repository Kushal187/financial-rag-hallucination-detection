
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.answer import _parse_number  # noqa: E402


def near_miss(row, tol=0.10):
    if row.get("gold_supported"):
        return False
    predicted = _parse_number(row.get("answer"))
    gold = row.get("gold_answer_exe")
    if predicted is None or not isinstance(gold, (int, float)) or not gold:
        return False
    error = min(
        abs(predicted - gold) / abs(gold),
        abs(predicted - gold * 100) / abs(gold * 100),
        abs(predicted - gold / 100) / abs(gold / 100),
    )
    return error < tol


def score(rows):
    tp = fp = fn = tn = 0
    for row in rows:
        gold_hallucinated = not row["gold_supported"]
        judge_hallucinated = not row["verdict"]["supported"]
        if judge_hallucinated and gold_hallucinated:
            tp += 1
        elif judge_hallucinated:
            fp += 1
        elif gold_hallucinated:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="judged file from run_llm_judge.py")
    parser.add_argument("--skip-near-miss", action="store_true",
                        help="ignore answers within 10%% of the right answer")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8") if line.strip()]
    if args.skip_near_miss:
        kept = [r for r in rows if not near_miss(r)]
        print(f"Skipped {len(rows) - len(kept)} near-miss rows (within 10% of the right answer)")
        rows = kept

    tp, fp, fn, tn = score(rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print(f"\n{os.path.basename(args.input)} — {len(rows)} answers\n")
    print(f"  precision  {precision:.2f}")
    print(f"  recall     {recall:.2f}")
    print(f"  F1         {f1:.2f}")
    print(f"  accuracy   {(tp + tn) / len(rows):.2f}")
    print(f"\n  judge caught {tp} of {tp + fn} hallucinations, and wrongly flagged {fp} good answers")

    wrong = [r for r in rows if bool(r["gold_supported"]) != bool(r["verdict"]["supported"])]
    if wrong:
        print(f"\nDisagreements ({len(wrong)}):\n")
        for row in wrong:
            kind = "judge flagged a good answer" if row["gold_supported"] else "judge missed one"
            print(f"  [{kind}] {row['id']}")
            print(f"    answer {row['answer']!r}, right answer {row['gold_answer']!r}")


if __name__ == "__main__":
    main()
