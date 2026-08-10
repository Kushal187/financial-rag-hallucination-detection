"""Weaviate Cloud vector store for FinQA chunks (bring-your-own vectors).

One global collection, filtered by `doc_id` at query time, since FinQA gold evidence
always lives on a single filing page.
"""

import os

import numpy as np
import weaviate
from dotenv import load_dotenv
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery

load_dotenv()

_COLLECTION = "FinqaChunk"
_client: weaviate.WeaviateClient | None = None


def get_client() -> weaviate.WeaviateClient:
    global _client
    if _client is None:
        url = os.getenv("WEAVIATE_URL")
        api_key = os.getenv("WEAVIATE_API_KEY")
        if not url or not api_key:
            raise RuntimeError(
                "WEAVIATE_URL and WEAVIATE_API_KEY must be set (check your .env file)"
            )
        _client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key),
        )
    return _client


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


def create_collection(name: str = _COLLECTION, recreate: bool = False):
    """Create the chunk collection (vectorizer=none, cosine). Returns the collection."""
    client = get_client()
    if recreate and client.collections.exists(name):
        client.collections.delete(name)
    if not client.collections.exists(name):
        client.collections.create(
            name=name,
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hfresh(
                    distance_metric=VectorDistances.COSINE
                )
            ),
            properties=[
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="local_id", data_type=DataType.TEXT),
                Property(name="chunk_type", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="position", data_type=DataType.INT),
            ],
        )
    return client.collections.get(name)


def ingest_chunks(
    chunks: list[dict], vectors: np.ndarray, name: str = _COLLECTION, batch_size: int = 200
) -> int:
    """Batch-insert chunks with their precomputed vectors. Returns count attempted."""
    collection = get_client().collections.get(name)
    with collection.batch.fixed_size(batch_size=batch_size) as batch:
        for chunk, vector in zip(chunks, vectors):
            batch.add_object(
                properties={
                    "doc_id": chunk["doc_id"],
                    "local_id": chunk["local_id"],
                    "chunk_type": chunk["chunk_type"],
                    "content": chunk["content"],
                    "position": chunk["position"],
                },
                vector=vector.tolist(),
            )
    failed = collection.batch.failed_objects
    if failed:
        raise RuntimeError(f"{len(failed)} objects failed to ingest; first: {failed[0]}")
    return len(chunks)


def search(query_vector: np.ndarray, doc_id: str, k: int = 5, name: str = _COLLECTION) -> list[dict]:
    """Return the top-k chunks within a single document, ranked by cosine similarity."""
    collection = get_client().collections.get(name)
    result = collection.query.near_vector(
        near_vector=query_vector.tolist(),
        limit=k,
        filters=Filter.by_property("doc_id").equal(doc_id),
        return_metadata=MetadataQuery(distance=True),
    )
    return [
        {
            "local_id": o.properties["local_id"],
            "content": o.properties["content"],
            "distance": o.metadata.distance,
        }
        for o in result.objects
    ]


def hybrid_search(
    query: str,
    query_vector: np.ndarray,
    doc_id: str,
    k: int = 5,
    alpha: float = 0.5,
    name: str = _COLLECTION,
) -> list[dict]:
    """Return the top-k chunks in one document by hybrid BM25 + vector fusion."""
    collection = get_client().collections.get(name)
    result = collection.query.hybrid(
        query=query,
        vector=query_vector.tolist(),
        alpha=alpha,
        query_properties=["content"],
        limit=k,
        filters=Filter.by_property("doc_id").equal(doc_id),
        return_metadata=MetadataQuery(score=True),
    )
    return [
        {
            "local_id": o.properties["local_id"],
            "content": o.properties["content"],
            "score": o.metadata.score,
        }
        for o in result.objects
    ]


def count(name: str = _COLLECTION) -> int:
    """Total objects currently in the collection."""
    return get_client().collections.get(name).aggregate.over_all(total_count=True).total_count
