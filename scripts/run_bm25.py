# scripts/run_bm25.py
"""Run BM25 retrieval over a FinQA split and save top-k predictions.

Run from repo root:
    python scripts/run_bm25.py --split dev
    python scripts/run_bm25.py --split test
"""
import argparse
import json
import sys
from pathlib import Path

# Make `src` importable when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm
from src.retrieval.bm25 import BM25Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--top-k", type=int, default=10,
                    help="How many candidates to retrieve per query")
    ap.add_argument("--data-dir", default="data/raw/finqa")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    data_path = Path(args.data_dir) / f"{args.split}.json"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bm25_predictions_{args.split}.json"

    with open(data_path) as f:
        data = json.load(f)

    predictions = {}
    for ex in tqdm(data, desc=f"BM25 on {args.split}"):
        retriever = BM25Retriever(ex)
        question = ex["qa"]["question"]
        top_k = retriever.retrieve(question, k=args.top_k)
        predictions[ex["id"]] = {
            "question": question,
            "gold_inds": list((ex["qa"].get("gold_inds") or {}).keys()),
            "retrieved": [{"id": cid, "score": float(s)} for cid, s in top_k],
        }

    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"[saved] {len(predictions)} predictions to {out_path}")


if __name__ == "__main__":
    main()