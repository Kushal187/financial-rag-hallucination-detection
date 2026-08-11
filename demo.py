"""Demo: retrieval -> generation -> hallucination detection on FinQA.

    python demo.py                            replay the recorded runs
    python demo.py --live                     call the retriever and the LLM
    python demo.py AAPL/2004/page_36.pdf-2    one question, by id
    python demo.py "gross margin"             or by a piece of the question
    python demo.py --random                   a random evaluated question
    python demo.py AAPL/2004/page_36.pdf-2 --live   one question, live
"""

import argparse
import json
import os
import random
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.load_data import load_qa  # noqa: E402
from src.detection import llm_judge, rule_based  # noqa: E402
from src.generation import answers_match, generate_answer, get_chunk_contents  # noqa: E402

K = 5
STRATEGY = "zero_shot"
JUDGE_MODEL = os.getenv("JUDGE_MODEL")

GOOD_QUESTION = "V/2008/page_17.pdf-1"
BAD_QUESTION = "ADBE/2018/page_86.pdf-1"

RUNS = {
    "gen": "data/runs/prompts_dev_rerank.jsonl",
    "grounding": "data/runs/judge_v4_claude_haiku.jsonl",
    "evidence": "data/runs/judge_v5_evidence_mode.jsonl",
    "selfgraded": "data/runs/judge_v3_with_fabrications.jsonl",
}
LABELED = "data/processed/detection_eval.jsonl"

_runs = {}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def recorded(kind, question_id):
    """The row this question produced in an earlier run, or None."""
    if kind not in _runs:
        rows = read_jsonl(RUNS[kind])
        _runs[kind] = {r["id"]: r for r in rows if r.get("strategy") == STRATEGY}
    return _runs[kind].get(question_id)


def replayable():
    """Questions with a recorded run at every stage, so they work without --live."""
    for kind in RUNS:
        recorded(kind, "")
    return sorted(set(_runs["gen"]) & set(_runs["grounding"]) & set(_runs["evidence"]))


def judge(question, context, answer, question_id, mode, live):
    """One LLM judge verdict, live or replayed. None if there is no recorded verdict."""
    if not live:
        row = recorded(mode, question_id)
        return row["verdict"] if row else None

    if not JUDGE_MODEL:
        return llm_judge.verify(question, context, answer, mode=mode)

    # Grade with a different model than the one that wrote the answer.
    from src.generation import bedrock, client
    previous = (client._MODEL, bedrock.MODEL)
    client._MODEL = bedrock.MODEL = JUDGE_MODEL
    try:
        return llm_judge.verify(question, context, answer, mode=mode)
    finally:
        client._MODEL, bedrock.MODEL = previous


def run(qa, retrieve_fn, live):
    """Retrieve, answer, then verify: one question through the whole pipeline."""
    print("=" * 78)
    print(qa["question"])
    print(f"{qa['doc_id']}  ({qa['id']})")
    print("=" * 78)

    #retrieval
    if live:
        ids = retrieve_fn(qa["question"], qa["doc_id"], K)
    else:
        ids = recorded("gen", qa["id"])["retrieved_ids"][:K]
    context = get_chunk_contents(qa["doc_id"], ids)
    gold = set(qa["gold_evidence_ids"])
    missing = gold - set(ids)

    print(f"\nRETRIEVAL   top {K} chunks of this filing page")
    for i, (chunk_id, text) in enumerate(context, start=1):
        mark = "GOLD" if chunk_id in gold else "    "
        print(f"  {i}. {chunk_id:<10} {mark}  {' '.join(text.split())[:52]}")
    if missing:
        print(f"  gold evidence: MISSING {', '.join(sorted(missing))}")
    else:
        print(f"  gold evidence: all {len(gold)} retrieved")

    #generation
    if live:
        answer = generate_answer(qa["question"], context, strategy=STRATEGY)["answer"]
    else:
        answer = recorded("gen", qa["id"])["answer"]
    correct = answers_match(answer, qa["gold_answer"], qa["gold_answer_exe"])

    print("\nGENERATION   the model sees only those chunks")
    print(f"  answer: {answer}")
    print(f"  gold:   {qa['gold_answer']}   -> {'CORRECT' if correct else 'WRONG'}")

    #detection
    print("\nDETECTION   none of these is shown the gold answer")
    verdicts = {"rule-based": rule_based.verify(qa["question"], context, answer)}
    for mode in ("grounding", "evidence"):
        verdicts["judge/" + mode] = judge(
            qa["question"], context, answer, qa["id"], mode, live)

    for name, verdict in verdicts.items():
        if verdict is None:
            print(f"  {name:<16} no recorded verdict")
            continue
        label = "supported" if verdict["supported"] else "HALLUCINATED"
        print(f"  {name:<16} {label:<14} {verdict['category']}")
        why = verdict.get("derivation") or verdict.get("reasoning", "")
        print(f"                   {' '.join(why.split())[:120]}")

    flagged = [n for n, v in verdicts.items() if v and not v["supported"]]
    print(f"\n  => the answer is {'correct' if correct else 'wrong'}, and "
          + (f"flagged by {', '.join(flagged)}" if flagged else "nothing flagged it"))


def prf(rows, hallucinated):
    """Precision, recall and F1 with hallucinated as the positive class."""
    tp = sum(1 for r in rows if hallucinated(r) and not r["verdict"]["supported"])
    fp = sum(1 for r in rows if not hallucinated(r) and not r["verdict"]["supported"])
    fn = sum(1 for r in rows if hallucinated(r) and r["verdict"]["supported"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def table(title, entries, hallucinated, rows):
    print(f"\n{title}")
    print(f"  {'verifier':<30}{'P':>6}{'R':>6}{'F1':>6}")
    for name, judged in entries:
        p, r, f1 = prf(judged, hallucinated)
        print(f"  {name:<30}{p:>6.2f}{r:>6.2f}{f1:>6.2f}")
    share = sum(1 for r in rows if hallucinated(r)) / len(rows)
    print(f"  {'flag everything (baseline)':<30}{share:>6.2f}{1.0:>6.2f}"
          f"{2 * share / (share + 1):>6.2f}")


def scores():
    """Both verifiers over the whole labeled set, so two examples are not the evidence."""
    rows = read_jsonl(LABELED)
    rule = [{**r, "verdict": rule_based.verify(r["question"],
                                               [(a, b) for a, b in r["context"]],
                                               str(r["answer"]))} for r in rows]
    print("\n" + "=" * 78)
    print(f"ALL {len(rows)} LABELED ANSWERS   (positive class = hallucinated)")
    print("=" * 78)

    table("A. the answer does not follow from the evidence", [
        ("rule-based", rule),
        ("judge, llama (self-graded)", read_jsonl(RUNS["selfgraded"])),
        ("judge, claude-haiku", read_jsonl(RUNS["grounding"])),
    ], lambda r: not r["gold_supported"], rows)

    table("B. the model answered without enough evidence", [
        ("judge, grounding mode", read_jsonl(RUNS["grounding"])),
        ("judge, evidence mode", read_jsonl(RUNS["evidence"])),
    ], lambda r: r["label_reason"] == "fabricated_without_evidence", rows)

    print("\nChanging only the judging model moved F1 from 0.41 to 0.54.")


def pick(qa_rows, needle):
    """The question matching `needle`, by exact id or a substring of the id or text."""
    for row in qa_rows:
        if row["id"] == needle:
            return needle
    hits = [r for r in qa_rows if needle.lower() in r["id"].lower()
            or needle.lower() in r["question"].lower()]
    if not hits:
        sys.exit(f"nothing matches {needle!r}")
    if len(hits) > 1:
        print(f"{len(hits)} questions match {needle!r}:\n")
        for row in hits[:10]:
            print(f"  {row['id']:<28} {row['question'][:60]}")
        sys.exit("\nuse a full id")
    return hits[0]["id"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", help="question id, or part of one")
    parser.add_argument("--live", action="store_true", help="call the retriever and the LLM")
    parser.add_argument("--random", action="store_true", help="pick a random question")
    args = parser.parse_args()

    qa_rows = load_qa()
    qa = {row["id"]: row for row in qa_rows}

    if args.random:
        chosen = [random.choice(replayable())]
    elif args.question:
        chosen = [pick(qa_rows, args.question)]
    else:
        chosen = [GOOD_QUESTION, BAD_QUESTION]

    retrieve_fn = None
    if args.live:
        from src.retrieval import rerank
        retrieve_fn = rerank.retrieve
        print("live: cross-encoder rerank + LLM")
        print(f"judge: {JUDGE_MODEL or 'the same model that writes the answers'}")
    else:
        print("replaying recorded runs (--live to call the APIs)")
        print("judge: claude-haiku-4.5")
        for question_id in chosen:
            if question_id not in replayable():
                sys.exit(f"{question_id} has no recorded verdicts; use --live")

    for question_id in chosen:
        print()
        run(qa[question_id], retrieve_fn, args.live)
    scores()

    if args.live:
        from src.retrieval import weaviate_store
        weaviate_store.close_client()


if __name__ == "__main__":
    main()
