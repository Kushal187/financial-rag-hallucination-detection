"""Embed the processed FinQA chunks and load them into Weaviate Cloud.

Usage:
    python scripts/ingest_weaviate.py --recreate            # full ingest
    python scripts/ingest_weaviate.py --recreate --limit 500  # smoke test

Requires WEAVIATE_URL, WEAVIATE_API_KEY (and optionally EMBED_MODEL) in .env.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_chunks  # noqa: E402
from src.retrieval.embed import embed_texts  # noqa: E402
from src.retrieval import weaviate_store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default="FinqaChunk")
    parser.add_argument("--recreate", action="store_true", help="drop and recreate the collection")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None, help="only ingest the first N chunks")
    parser.add_argument(
        "--split",
        nargs="+",
        default=["dev", "test"],
        help="which splits to ingest (default: dev test -- the queried/eval splits). "
        "Use 'all' for every split. Note the sandbox cluster caps at 100k objects, "
        "so train (~191k chunks) does not fit.",
    )
    args = parser.parse_args()

    chunks = load_chunks()
    if args.split != ["all"]:
        wanted = set(args.split)
        chunks = [c for c in chunks if c.get("split") in wanted]
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"Loaded {len(chunks)} chunks (splits={args.split}).")

    print("Embedding...")
    vectors = embed_texts([c["content"] for c in chunks])

    try:
        weaviate_store.create_collection(args.collection, recreate=args.recreate)
        print(f"Ingesting into '{args.collection}'...")
        weaviate_store.ingest_chunks(chunks, vectors, name=args.collection, batch_size=args.batch_size)
        total = weaviate_store.count(args.collection)
        print(f"Done. Collection now holds {total} objects.")
    finally:
        weaviate_store.close_client()


if __name__ == "__main__":
    main()
