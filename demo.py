#!/usr/bin/env python
"""One run of the whole system end to end: retrieval, generation, detection.

    python demo.py           # replays the recorded runs, no setup, no network
    python demo.py --live    # actually calls the retriever and the LLM

Three parts, on real FinQA questions: the pipeline working, the same pipeline
hallucinating and being caught, then precision/recall/F1 over 560 labeled answers.
"""

import argparse
import contextlib
import json
import os
import sys
import textwrap
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.load_data import load_qa  # noqa: E402
from src.detection import llm_judge, rule_based  # noqa: E402
from src.generation import answers_match, generate_answer, get_chunk_contents  # noqa: E402

GOOD_QUESTION = "V/2008/page_17.pdf-1"
HALLUCINATED_QUESTION = "ADBE/2018/page_86.pdf-1"

K = 5
STRATEGY = "zero_shot"
WIDTH = 84

RUN_FILES = {
    "generation": "data/runs/prompts_dev_rerank.jsonl",
    "judge_grounding": "data/runs/judge_v4_claude_haiku.jsonl",
    "judge_evidence": "data/runs/judge_v5_evidence_mode.jsonl",
    "judge_selfgraded": "data/runs/judge_v3_with_fabrications.jsonl",
}
LABELED_SET = "data/processed/detection_eval.jsonl"


_ANSI = {"bold": "\033[1m", "dim": "\033[2m", "red": "\033[31m",
         "green": "\033[32m", "yellow": "\033[33m", "cyan": "\033[36m"}
_USE_COLOR = True


def c(text, *styles):
    styles = [s for s in styles if s]
    if not _USE_COLOR or not styles:
        return str(text)
    return "".join(_ANSI[s] for s in styles) + str(text) + "\033[0m"


def title(text):
    print()
    print(c("=" * WIDTH, "cyan"))
    print(c(f"  {text}", "bold", "cyan"))
    print(c("=" * WIDTH, "cyan"))


def step(text):
    print()
    print(c(f"  {text}", "bold"))


def field(label, value, style=None):
    pad = " " * 19
    body = textwrap.wrap(str(value), width=WIDTH - len(pad)) or [""]
    print(f"     {c(label.ljust(14), 'dim')}{c(body[0], style)}")
    for line in body[1:]:
        print(pad + c(line, style))


def note(text, style="dim"):
    print()
    for line in textwrap.wrap(str(text), width=WIDTH - 6) or [""]:
        print(c("     " + line, style))


def clip(text, width):
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


_cache = {}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def recorded(kind, question_id):
    """The row this question produced in a previous run, or None."""
    if kind not in _cache:
        path = RUN_FILES[kind]
        rows = read_jsonl(path) if os.path.exists(path) else []
        _cache[kind] = {(r["id"], r.get("strategy")): r for r in rows}
    return _cache[kind].get((question_id, STRATEGY))


def retrieve_stage(qa, retrieve_fn):
    """Retrieve the evidence, print the ranking, and say whether the gold row is in it."""
    step("STEP 1 · RETRIEVAL   which chunks of this page are relevant?")

    if retrieve_fn:
        start = time.perf_counter()
        ids = retrieve_fn(qa["question"], qa["doc_id"], K)
        source = f"live, {(time.perf_counter() - start) * 1000:.0f} ms"
    else:
        ids = recorded("generation", qa["id"])["retrieved_ids"][:K]
        source = "recorded run"

    context = get_chunk_contents(qa["doc_id"], ids)
    gold = set(qa["gold_evidence_ids"])
    found = gold & set(ids)

    print()
    for rank, (local_id, text) in enumerate(context, start=1):
        marker = c(" GOLD", "green", "bold") if local_id in gold else "     "
        print(f"     {rank}. {c(local_id.ljust(9), 'cyan')}{marker}  {clip(text, WIDTH - 26)}")
    print()

    if found == gold:
        field("gold evidence", f"{len(found)} of {len(gold)} in the top {K}, retrieval succeeded", "green")
    else:
        field("gold evidence", f"{len(found)} of {len(gold)} in the top {K}, MISSING "
                               f"{', '.join(sorted(gold - found))}", "red")
    field("source", source)
    return context, found == gold


def generate_stage(qa, context, live):
    """Answer from the retrieved context only, then score it against the gold answer."""
    step("STEP 2 · GENERATION   what is the answer, given only those chunks?")

    if live:
        gen = generate_answer(qa["question"], context, strategy=STRATEGY)
        answer, model = gen["answer"], gen["model"]
    else:
        gen = recorded("generation", qa["id"])
        answer, model = gen["answer"], "llama-3.3-70b (recorded)"

    correct = answers_match(answer, qa["gold_answer"], qa["gold_answer_exe"])
    print()
    field("model", model)
    field("answer", answer, "bold")
    field("gold answer", f"{qa['gold_answer']}   ->   " + ("CORRECT" if correct else "WRONG"),
          "green" if correct else "red")
    return answer, correct


VERIFIERS = [
    ("rule-based", "no LLM, re-derives the answer from the page's numbers"),
    ("LLM judge · grounding", "computes the answer itself, then compares"),
    ("LLM judge · evidence", "no arithmetic, never shown the answer"),
]


@contextlib.contextmanager
def judging_as(model_id):
    """Point the LLM client at a different model for the duration of the block, so the
    judge is not the model that wrote the answer. Safe because both client backends read
    their module-level model at call time."""
    if not model_id:
        yield
        return
    from src.generation import bedrock, client
    previous = (client._MODEL, bedrock.MODEL)
    client._MODEL = bedrock.MODEL = model_id
    try:
        yield
    finally:
        client._MODEL, bedrock.MODEL = previous


def detect_stage(qa, context, answer, live, judge_model=None):
    """Run all three verifiers and print what each concluded."""
    step("STEP 3 · DETECTION   does that answer follow from those chunks?")
    note("None of these is given the gold answer. They see the question, the evidence and "
         "the candidate answer, which is all you have at deployment time.")
    print()

    verdicts = {"rule-based": rule_based.verify(qa["question"], context, answer)}
    for mode, kind in (("grounding", "judge_grounding"), ("evidence", "judge_evidence")):
        if live:
            with judging_as(judge_model):
                verdicts[f"LLM judge · {mode}"] = llm_judge.verify(
                    qa["question"], context, answer, mode=mode)
        else:
            row = recorded(kind, qa["id"])
            verdicts[f"LLM judge · {mode}"] = row["verdict"] if row else {
                "supported": None, "category": "unavailable", "reasoning": ""}

    for name, how in VERIFIERS:
        print_verdict(name, how, verdicts[name])
    return verdicts


def print_verdict(name, how, verdict):
    supported = verdict.get("supported")
    label, style = (("SUPPORTED", "green") if supported
                    else ("HALLUCINATED", "red") if supported is False
                    else ("UNAVAILABLE", "yellow"))
    print(f"     {c(name.ljust(24), 'bold')}{c(label.ljust(15), style, 'bold')}"
          f"{c(verdict.get('category', ''), 'dim')}")
    print(f"       {c(how, 'dim')}")

    if verdict.get("derivation"):
        print(f"       {c('re-derived as: ' + verdict['derivation'], 'green')}")
        print()
        return

    # Worth showing next to the verdict: a judge that computed the candidate's own
    # number and still flagged it has contradicted itself, and you can see it here.
    if verdict.get("computed_value") is not None:
        print(f"       {c('the judge worked it out as: ' + str(verdict['computed_value']), 'cyan')}")
    for line in textwrap.wrap(clip(verdict.get("reasoning", ""), 400), WIDTH - 8):
        print(f"       {c(line, 'dim')}")
    print()


def part_one(qa, retrieve_fn, live, judge_model=None):
    title("PART 1 · THE RAG PIPELINE, WORKING")
    note("A question about Visa's 2008 filing. The answer is not written anywhere on the "
         "page, so it has to be computed from the table.", "cyan")
    print()
    field("question", qa["question"], "bold")

    context, _hit = retrieve_stage(qa, retrieve_fn)
    answer, correct = generate_stage(qa, context, live)
    verdicts = detect_stage(qa, context, answer, live, judge_model)

    flagged = [n for n, v in verdicts.items() if v.get("supported") is False]
    if correct and not flagged:
        note("Nothing was flagged. That matters as much as catching hallucinations: a "
             "detector that fires on good answers is useless.", "green")
    elif correct:
        note(f"Flagged by {', '.join(flagged)}, on an answer that is correct. This is a "
             f"false alarm, and precision is the harder half of the problem."
             + (" It is also what self-grading looks like: the judge is the model that "
                "wrote this answer. The recorded run, graded by a different model, does "
                "not flag it." if live and not judge_model else ""), "yellow")
    else:
        note("The model got this one wrong on this run, so part 1 is showing a "
             "hallucination too. The verdicts below are still the point.", "yellow")


def part_two(qa, retrieve_fn, live, judge_model=None):
    title("PART 2 · THE SAME PIPELINE, HALLUCINATING")
    note(f"Same code path, different question. The row it needs "
         f"({', '.join(qa['gold_evidence_ids'])}) does not survive retrieval.", "cyan")
    print()
    field("question", qa["question"], "bold")

    context, gold_hit = retrieve_stage(qa, retrieve_fn)
    if not gold_hit:
        note("Retrieval missed, and nothing in the pipeline knows it. The generator is "
             "never told its evidence is incomplete.", "yellow")

    answer, correct = generate_stage(qa, context, live)
    if correct and live:
        answer = recorded("generation", qa["id"])["answer"]
        note(f"The model got this right on this run. Continuing with the recorded answer "
             f"({answer!r}) so there is still a hallucination to catch.", "yellow")
    else:
        note("The model answered anyway, with no hedging and no signal that anything was "
             "missing. A wrong answer looks exactly like a right one.", "red")

    verdicts = detect_stage(qa, context, answer, live, judge_model)

    flagged = [n for n, v in verdicts.items() if v.get("supported") is False]
    if len(flagged) == len(verdicts):
        note("All three caught it, each for a different reason, and none of them saw the "
             "gold answer.", "green")
    elif flagged:
        note(f"Caught by {', '.join(flagged)}; the rest missed it. Part 3 has the rates.",
             "yellow")
    else:
        note("Nothing flagged it, a false negative. Both verifiers miss roughly half of "
             "all hallucinations.", "red")


def part_three():
    title("PART 3 · HOW OFTEN DOES THAT WORK?")
    note("Two questions prove nothing. Below is every verifier over the same 560 labeled "
         "answers, positive class = hallucinated.", "cyan")

    rows = read_jsonl(LABELED_SET)
    start = time.perf_counter()
    rule_rows = [{**r, "verdict": rule_based.verify(r["question"],
                                                    [(a, b) for a, b in r["context"]],
                                                    str(r["answer"]))} for r in rows]
    elapsed = time.perf_counter() - start

    print()
    print(c("     A · the answer does not follow from the evidence", "bold"))
    score_table([
        ("rule-based", rule_rows, f"re-run now, {elapsed:.2f}s, $0"),
        ("LLM judge · llama-3.3-70b", replay("judge_selfgraded"), "graded its own answers"),
        ("LLM judge · claude-haiku", replay("judge_grounding"), "a different model grades"),
    ], lambda r: not r["gold_supported"], rows)

    print()
    print(c("     B · the model answered without sufficient evidence", "bold"))
    score_table([
        ("LLM judge · grounding", replay("judge_grounding"), "sees the answer, computes"),
        ("LLM judge · evidence", replay("judge_evidence"), "no answer shown, no arithmetic"),
    ], lambda r: r["label_reason"] == "fabricated_without_evidence", rows)

    note("Changing only the judging model moved F1 from 0.41 to 0.54, a bigger effect than "
         "any prompt change we tried.")


def replay(kind):
    path = RUN_FILES[kind]
    return read_jsonl(path) if os.path.exists(path) else []


def score_table(entries, positive, all_rows):
    """Print P/R/F1/accuracy per verifier, plus the flag-everything baseline."""
    print()
    print(c(f"     {'verifier':<28}{'P':>6}{'R':>6}{'F1':>6}{'acc':>6}", "dim"))
    for name, judged, comment in entries:
        if not judged:
            print(f"     {name:<28}{c('(run file not found)', 'yellow')}")
            continue
        p, r, f1, acc = prf(judged, positive)
        print(f"     {name:<28}{p:>6.2f}{r:>6.2f}{f1:>6.2f}{acc:>6.2f}   {c(comment, 'dim')}")

    p = sum(1 for r in all_rows if positive(r)) / len(all_rows)
    print(c(f"     {'baseline: flag everything':<28}{p:>6.2f}{1.0:>6.2f}"
            f"{2 * p / (p + 1):>6.2f}{p:>6.2f}   the bar to clear", "dim"))


def prf(judged, positive):
    tp = fp = fn = tn = 0
    for row in judged:
        gold, pred = positive(row), not row["verdict"]["supported"]
        if pred and gold:
            tp += 1
        elif pred:
            fp += 1
        elif gold:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, (tp + tn) / len(judged)


def build_retriever(name):
    """Return `(retrieve_fn, label)` for a live run, or None if it cannot be set up."""
    try:
        if name == "bm25":
            from src.retrieval import bm25
            return bm25.retrieve, "bm25"

        # Load the models and open the Weaviate connection now. Left lazy, they happen
        # inside the first timed retrieval and get reported as query latency.
        from src.retrieval import embed, weaviate_store
        embed.get_model()
        weaviate_store.get_client()
        if name == "rerank":
            from src.retrieval import rerank
            rerank.get_model()
            return rerank.retrieve, f"rerank (pool={rerank.DEFAULT_POOL})"
        if name == "hybrid":
            from src.retrieval import hybrid
            return hybrid.retrieve, f"hybrid (alpha={hybrid.DEFAULT_ALPHA})"
        from src.retrieval import dense
        return dense.retrieve, "dense"
    except Exception as err:
        note(f"{name} is unavailable ({type(err).__name__}: {clip(err, 70)}). "
             f"Replaying the recorded runs instead.", "yellow")
        return None


def llm_ready():
    """Whether the configured LLM backend has credentials, without spending a call."""
    try:
        from src.generation import client
        if client._PROVIDER == "bedrock":
            import boto3
            # boto3 builds a client even with no credentials, so resolve them directly.
            if boto3.Session().get_credentials() is None:
                raise RuntimeError("no AWS credentials found")
        else:
            client.get_client()
        return True
    except Exception as err:
        note(f"No LLM credentials ({type(err).__name__}: {clip(err, 70)}). "
             f"Replaying the recorded runs instead.", "yellow")
        return False


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true",
                        help="call the retriever and the LLM instead of replaying")
    parser.add_argument("--retriever", default="rerank",
                        choices=["rerank", "hybrid", "dense", "bm25"], help="--live only")
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL"),
                        help="grade with a different model than the one that wrote the "
                             "answers (env: JUDGE_MODEL)")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    global _USE_COLOR
    _USE_COLOR = not args.no_color and sys.stdout.isatty() and not os.getenv("NO_COLOR")

    qa = {row["id"]: row for row in load_qa()}
    for question_id in (GOOD_QUESTION, HALLUCINATED_QUESTION):
        if question_id not in qa:
            sys.exit(f"{question_id} is not in the corpus. Build it first.")

    title("FINANCIAL RAG WITH HALLUCINATION DETECTION")
    print()
    field("pipeline", "retrieve -> generate -> verify")
    field("dataset", "FinQA, questions over single pages of company 10-K filings")

    retriever = build_retriever(args.retriever) if args.live and llm_ready() else None
    live = retriever is not None
    retrieve_fn, label = retriever if retriever else (None, "recorded run")

    if live:
        from src.generation import client
        field("mode", f"live, retriever = {label}")
        field("generator", client._MODEL)
        field("judge", args.judge_model or f"{client._MODEL}  (the same model)")
        if not args.judge_model:
            note("This run has one model both writing and grading, so the judge is marking "
                 "its own homework. That is the weakest configuration we measured: F1 0.41, "
                 "against 0.54 when a different model grades. Set JUDGE_MODEL to split "
                 "them.", "yellow")
    else:
        field("mode", "replaying recorded runs (--live to call the APIs)", "yellow")
        field("judge", "claude-haiku-4.5, the configuration the report recommends")

    try:
        part_one(qa[GOOD_QUESTION], retrieve_fn, live, args.judge_model)
        part_two(qa[HALLUCINATED_QUESTION], retrieve_fn, live, args.judge_model)
        part_three()
        print()
    finally:
        if live and args.retriever != "bm25":
            try:
                from src.retrieval import weaviate_store
                weaviate_store.close_client()
            except Exception:
                pass


if __name__ == "__main__":
    main()
