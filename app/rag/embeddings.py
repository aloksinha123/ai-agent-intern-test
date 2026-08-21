"""Local Sentence Transformers embedding client and utilities."""

import os
from typing import Any, List, Optional, Union

from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer

# Load environment variables from .env if present
load_dotenv()

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# In-memory singleton cache to avoid reloading weights across calls
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def get_embedding_model_name(model_name: Optional[str] = None) -> str:
    """Resolve the embedding model name from parameter, environment, or default."""
    if model_name:
        return model_name
    return os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_embedding_model(model: Optional[Union[SentenceTransformer, Any, str]] = None) -> Any:
    """Retrieve or load a cached SentenceTransformer model instance.

    Args:
        model: An existing encoder instance, mock object, or a model identifier string.

    Returns:
        SentenceTransformer model or custom encoder ready for inference.
    """
    if model is not None and not isinstance(model, str):
        return model

    resolved_name = get_embedding_model_name(model if isinstance(model, str) else None)
    if resolved_name not in _MODEL_CACHE:
        _MODEL_CACHE[resolved_name] = SentenceTransformer(resolved_name)
    return _MODEL_CACHE[resolved_name]


def get_embedding_dimension(model: Optional[Union[SentenceTransformer, str]] = None) -> int:
    """Get the output vector dimension for the resolved embedding model."""
    inst = get_embedding_model(model)
    dim = inst.get_sentence_embedding_dimension()
    return int(dim) if dim is not None else 384


def embed_texts(
    texts: List[str],
    model: Optional[Union[SentenceTransformer, str]] = None,
    model_name: Optional[str] = None,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """Generate dense embeddings for a list of text strings using SentenceTransformer.

    Args:
        texts: List of strings to embed.
        model: Optional SentenceTransformer model instance or model identifier.
        model_name: Optional model identifier string (alias for model).
        batch_size: Batch size for encoding.
        normalize_embeddings: Whether to L2-normalize vectors for cosine similarity.

    Returns:
        A 2D numpy array of shape (len(texts), embedding_dim) with dtype float32.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    encoder = get_embedding_model(model or model_name)
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
    )

    return np.asarray(embeddings, dtype=np.float32)


def embed_query(
    query: str,
    model: Optional[Union[SentenceTransformer, str]] = None,
    model_name: Optional[str] = None,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """Generate a dense embedding vector for a single query string.

    Args:
        query: Query text string.
        model: Optional SentenceTransformer model instance.
        model_name: Optional model identifier string.
        normalize_embeddings: Whether to L2-normalize vector.

    Returns:
        A 1D numpy array of shape (embedding_dim,) with dtype float32.
    """
    embeddings = embed_texts(
        [query],
        model=model,
        model_name=model_name,
        normalize_embeddings=normalize_embeddings,
    )
    return embeddings[0]
