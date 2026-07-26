"""Dense embeddings via sentence-transformers (bring-your-own vectors for Weaviate)."""

import os

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

_EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_EMBED_MODEL)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed texts into L2-normalized vectors (float32) suitable for cosine search."""
    return get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 1000,
    ).astype("float32")
