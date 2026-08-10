"""Embed the chunk corpus and ingest it into the Weaviate collection."""

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
        default=["train", "dev", "test"],
        help="which splits to ingest (default: train dev test). Deduped these total "
        "~86k chunks, under the 100k Weaviate sandbox cap. Use 'all' for every split.",
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
