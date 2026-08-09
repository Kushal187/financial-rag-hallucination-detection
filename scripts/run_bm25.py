
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm
from src.retrieval.bm25 import BM25Retriever, load_chunks_by_doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--raw-data-dir", default="data/raw/finqa")
    ap.add_argument("--chunks-path", default="data/processed/finqa_chunks.jsonl")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    print(f"Loading chunks from {args.chunks_path} (split={args.split})...")
    chunks_by_doc = load_chunks_by_doc(args.chunks_path, split=args.split)
    print(f"  Loaded chunks for {len(chunks_by_doc)} documents")

    raw_path = Path(args.raw_data_dir) / f"{args.split}.json"
    with open(raw_path) as f:
        data = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bm25_predictions_{args.split}.json"

    predictions = {}
    missing = 0
    for ex in tqdm(data, desc=f"BM25 on {args.split}"):
        doc_id = ex["filename"]
        chunks = chunks_by_doc.get(doc_id)
        if not chunks:
            missing += 1
            continue
        retriever = BM25Retriever(chunks)
        question = ex["qa"]["question"]
        top_k = retriever.rank(question, k=args.top_k)
        predictions[ex["id"]] = {
            "question": question,
            "doc_id": doc_id,
            "gold_inds": list((ex["qa"].get("gold_inds") or {}).keys()),
            "retrieved": [{"id": lid, "score": float(s)} for lid, s in top_k],
        }

    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"[saved] {len(predictions)} predictions to {out_path}")
    if missing:
        print(f"[warning] {missing} questions had no matching chunks in the corpus")


if __name__ == "__main__":
    main()

