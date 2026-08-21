"""FAISS vector store and deterministic chunk-mapped similarity retrieval."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np

from app.rag.embeddings import embed_query
from app.rag.models import Chunk, SearchResult


def _chunk_to_dict(chunk: Chunk) -> Dict[str, Any]:
    """Convert a Chunk object to a JSON-serializable dictionary."""
    return {
        "chunk_id": chunk.chunk_id,
        "filename": chunk.filename,
        "document_title": chunk.document_title,
        "heading": chunk.heading,
        "content": chunk.content,
        "metadata": chunk.metadata,
    }


def _dict_to_chunk(data: Dict[str, Any]) -> Chunk:
    """Reconstruct a Chunk object from a dictionary."""
    return Chunk(
        chunk_id=data["chunk_id"],
        filename=data["filename"],
        document_title=data["document_title"],
        heading=data["heading"],
        content=data["content"],
        metadata=data.get("metadata", {}),
    )


class VectorStore:
    """FAISS-backed vector store maintaining 1-to-1 chunk provenance mapping."""

    def __init__(self, index: faiss.Index, chunks: List[Chunk]):
        self.index = index
        self.chunks = chunks
        self.dimension = index.d

    @classmethod
    def build(cls, chunks: List[Chunk], embeddings: np.ndarray) -> "VectorStore":
        """Build a FAISS IndexFlatIP store from chunks and dense embeddings.

        Embeddings are L2-normalized prior to insertion so inner product equals cosine similarity.

        Args:
            chunks: List of Chunk objects.
            embeddings: 2D numpy array of shape (N, D) and dtype float32.

        Returns:
            A populated VectorStore instance.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)})."
            )

        if len(chunks) == 0:
            raise ValueError("Cannot build an empty VectorStore.")

        emb_matrix = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(emb_matrix)

        dim = emb_matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb_matrix)

        return cls(index=index, chunks=list(chunks))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """Search the FAISS index using a query embedding vector.

        Args:
            query_embedding: 1D (D,) or 2D (1, D) numpy array of dtype float32.
            top_k: Number of nearest chunks to retrieve.

        Returns:
            List of SearchResult objects sorted by descending similarity score.
        """
        q_vec = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)

        faiss.normalize_L2(q_vec)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(q_vec, k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append(
                SearchResult(
                    score=float(score),
                    chunk_id=chunk.chunk_id,
                    filename=chunk.filename,
                    document_title=chunk.document_title,
                    heading=chunk.heading,
                    content=chunk.content,
                    metadata=dict(chunk.metadata),
                )
            )

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "customer",
        model: Optional[Any] = None,
        model_name: Optional[str] = None,
    ) -> List[SearchResult]:
        """Perform two-stage metadata-aware retrieval.

        Stage 1: Retrieve dense semantic candidates from FAISS index.
        Stage 2: Apply RetrievalPolicy to filter for authority/eligibility and rank by precedence.

        Args:
            query: User search query string.
            top_k: Number of final results to return.
            mode: Retrieval mode ('customer', 'internal', or 'all').
            model: Optional SentenceTransformer model instance.
            model_name: Optional model identifier string.

        Returns:
            List of scored and policy-filtered SearchResult objects.
        """
        from app.rag.policy import RetrievalPolicy

        query_vec = embed_query(query, model=model, model_name=model_name)

        # Stage 1: Over-fetch candidates to ensure sufficient valid pool for metadata filtering
        candidate_k = max(top_k * 4, 20)
        candidates = self.search(query_vec, top_k=candidate_k)

        # Stage 2: Apply metadata-aware eligibility and precedence ranking
        return RetrievalPolicy.apply(candidates, query=query, mode=mode, top_k=top_k)

    def save(self, storage_dir: Union[str, Path]) -> None:
        """Persist FAISS index and chunk metadata mapping to disk.

        Args:
            storage_dir: Directory where kb_index.faiss and kb_chunks.json will be saved.
        """
        out_dir = Path(storage_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        index_path = out_dir / "kb_index.faiss"
        chunks_path = out_dir / "kb_chunks.json"

        faiss.write_index(self.index, str(index_path))

        serialized_chunks = [_chunk_to_dict(c) for c in self.chunks]
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(serialized_chunks, f, indent=2)

    @classmethod
    def load(cls, storage_dir: Optional[Union[str, Path]] = None) -> "VectorStore":
        """Load persisted FAISS index and chunk metadata from disk.

        Args:
            storage_dir: Directory containing kb_index.faiss and kb_chunks.json.
                         Defaults to repository root / storage.

        Returns:
            A restored VectorStore instance.
        """
        if storage_dir is None:
            storage_dir = Path(__file__).resolve().parent.parent.parent / "storage"

        in_dir = Path(storage_dir)
        index_path = in_dir / "kb_index.faiss"
        chunks_path = in_dir / "kb_chunks.json"

        if not index_path.is_file():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        if not chunks_path.is_file():
            raise FileNotFoundError(f"Chunk metadata file not found: {chunks_path}")

        index = faiss.read_index(str(index_path))

        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = [_dict_to_chunk(item) for item in data]
        return cls(index=index, chunks=chunks)
