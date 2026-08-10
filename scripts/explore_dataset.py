"""Summarize the raw FinQA splits: field coverage, answer types, program operations and
the gold evidence mix. Saves the per-split statistics as JSON."""

import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median

DATA_DIR = Path("data/raw/finqa")
OUT_DIR = Path("docs")
SPLITS = ["train", "dev", "test"]


def load_split(name):
    with open(DATA_DIR / f"{name}.json") as f:
        return json.load(f)


def classify_answer(ans):
    if ans is None or ans == "":
        return "empty"
    s = str(ans).strip()
    if "%" in s:
        return "percentage"
    if "$" in s:
        return "currency"
    cleaned = s.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        f = float(cleaned)
        return "integer" if f.is_integer() else "decimal"
    except ValueError:
        return "string"


def summarize(vals):
    return {"mean": round(mean(vals), 1), "median": median(vals), "max": max(vals)} if vals else {}


def analyze_split(name, data):
    stats = {"split": name, "n_examples": len(data)}

    field_counts = Counter()
    for ex in data:
        field_counts.update(ex.keys())
    stats["top_level_fields"] = dict(field_counts)

    qa_field_counts = Counter()
    for ex in data:
        qa_field_counts.update((ex.get("qa") or {}).keys())
    stats["qa_fields"] = dict(qa_field_counts)

    pre_lens = [len(ex.get("pre_text", [])) for ex in data]
    post_lens = [len(ex.get("post_text", [])) for ex in data]
    table_rows = [len(ex.get("table", [])) for ex in data]
    table_cols = [len(ex["table"][0]) if ex.get("table") else 0 for ex in data]
    stats["pre_text_sentences"] = summarize(pre_lens)
    stats["post_text_sentences"] = summarize(post_lens)
    stats["table_rows"] = summarize(table_rows)
    stats["table_cols"] = summarize(table_cols)

    q_lens = [len(ex["qa"]["question"].split())
              for ex in data if ex.get("qa", {}).get("question")]
    stats["question_words"] = summarize(q_lens)

    ans_types = Counter()
    for ex in data:
        ans_types[classify_answer(ex.get("qa", {}).get("answer"))] += 1
    stats["answer_types"] = dict(ans_types)

    op_counts = Counter()
    n_ops_per_prog = []
    for ex in data:
        prog = ex.get("qa", {}).get("program", "")
        if prog:
            ops = re.findall(r"([a-zA-Z_]+)\(", prog)
            op_counts.update(ops)
            n_ops_per_prog.append(len(ops))
    stats["operations"] = dict(op_counts.most_common())
    stats["ops_per_program"] = summarize(n_ops_per_prog)

    gold_text_only = gold_table_only = gold_both = gold_missing = 0
    n_gold_items = []
    for ex in data:
        gi = ex.get("qa", {}).get("gold_inds") or {}
        has_text = any(k.startswith("text_") for k in gi)
        has_table = any(k.startswith("table_") for k in gi)
        if not gi:
            gold_missing += 1
        elif has_text and has_table:
            gold_both += 1
        elif has_text:
            gold_text_only += 1
        elif has_table:
            gold_table_only += 1
        n_gold_items.append(len(gi))
    stats["gold_evidence"] = {
        "text_only": gold_text_only,
        "table_only": gold_table_only,
        "both_text_and_table": gold_both,
        "missing_or_empty": gold_missing,
        "items_per_example": summarize(n_gold_items),
    }

    companies = Counter()
    years = Counter()
    for ex in data:
        fn = ex.get("filename", "")
        parts = fn.split("/")
        if len(parts) >= 2:
            companies[parts[0]] += 1
            years[parts[1]] += 1
    stats["n_unique_companies"] = len(companies)
    stats["top_10_companies"] = dict(companies.most_common(10))
    stats["year_distribution"] = dict(sorted(years.items()))

    mismatches = 0
    checked = 0
    for ex in data:
        qa = ex.get("qa", {})
        ans, exe = qa.get("answer"), qa.get("exe_ans")
        if ans in (None, "") or exe is None:
            continue
        try:
            a = float(str(ans).replace(",", "").replace("$", "").replace("%", ""))
            e = float(exe)
            checked += 1
            if (abs(a - e) > 0.01
                    and abs(a - e * 100) > 0.01
                    and abs(a - e / 100) > 0.01):
                mismatches += 1
        except (ValueError, TypeError):
            continue
    stats["answer_vs_exe_ans"] = {
        "checked": checked,
        "mismatches_after_scale_normalization": mismatches,
    }

    return stats


def print_split_summary(stats):
    print(f"\n{'=' * 60}")
    print(f"SPLIT: {stats['split'].upper()}  |  {stats['n_examples']} examples")
    print("=" * 60)
    print(f"  Questions (words): mean={stats['question_words'].get('mean')}, "
          f"median={stats['question_words'].get('median')}, "
          f"max={stats['question_words'].get('max')}")
    print(f"  pre_text sentences: mean={stats['pre_text_sentences'].get('mean')}")
    print(f"  post_text sentences: mean={stats['post_text_sentences'].get('mean')}")
    print(f"  Table rows x cols: {stats['table_rows'].get('mean')} x {stats['table_cols'].get('mean')}")
    print(f"  Answer types: {stats['answer_types']}")
    print(f"  Gold evidence: {stats['gold_evidence']}")
    print(f"  Top operations: {dict(list(stats['operations'].items())[:6])}")
    print(f"  Unique companies: {stats['n_unique_companies']}")
    print(f"  answer/exe_ans mismatches: {stats['answer_vs_exe_ans']}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    all_stats = {}
    for split in SPLITS:
        path = DATA_DIR / f"{split}.json"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        data = load_split(split)
        stats = analyze_split(split, data)
        all_stats[split] = stats
        print_split_summary(stats)

    out = OUT_DIR / "dataset_stats.json"
    with open(out, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()