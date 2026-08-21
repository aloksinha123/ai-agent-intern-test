"""Unit tests for FAISS vector store, SentenceTransformer embedding normalization, and retrieval.

All tests use deterministic mock embeddings to guarantee zero network calls during tests.
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from app.rag.loader import load_knowledge_base
from app.rag.models import Chunk, SearchResult
from app.rag.splitter import split_documents
from app.rag.vector_store import VectorStore

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = REPO_ROOT / "knowledge-base"


def _generate_dummy_embeddings(count: int, dim: int = 64, seed: int = 42) -> np.ndarray:
    """Generate reproducible synthetic embedding vectors for testing."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(size=(count, dim)).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / np.maximum(norms, 1e-12)


class TestVectorStore(unittest.TestCase):
    """Tests for FAISS index creation, persistence, and retrieval with mocked embeddings."""

    @classmethod
    def setUpClass(cls):
        cls.docs = load_knowledge_base(KB_DIR)
        cls.chunks = split_documents(cls.docs)
        cls.dim = 64
        cls.embeddings = _generate_dummy_embeddings(len(cls.chunks), dim=cls.dim)

    def test_build_vector_store_from_all_53_chunks(self):
        """Verify that a VectorStore can be built from all 53 KB chunks."""
        self.assertEqual(len(self.chunks), 53)
        store = VectorStore.build(chunks=self.chunks, embeddings=self.embeddings)

        self.assertEqual(store.dimension, self.dim)
        self.assertEqual(len(store.chunks), 53)
        self.assertEqual(store.index.ntotal, 53)

    def test_vector_store_save_and_load_roundtrip(self):
        """Verify that VectorStore correctly saves and restores index and full metadata."""
        store = VectorStore.build(chunks=self.chunks, embeddings=self.embeddings)

        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)

            index_path = Path(tmpdir) / "kb_index.faiss"
            chunks_path = Path(tmpdir) / "kb_chunks.json"

            self.assertTrue(index_path.is_file())
            self.assertTrue(chunks_path.is_file())

            restored_store = VectorStore.load(tmpdir)
            self.assertEqual(restored_store.dimension, self.dim)
            self.assertEqual(len(restored_store.chunks), 53)
            self.assertEqual(restored_store.index.ntotal, 53)

            # Check metadata preservation on restored chunks
            first_chunk = restored_store.chunks[0]
            self.assertEqual(first_chunk.chunk_id, self.chunks[0].chunk_id)
            self.assertEqual(first_chunk.filename, self.chunks[0].filename)
            self.assertEqual(first_chunk.heading, self.chunks[0].heading)
            self.assertEqual(first_chunk.metadata.get("document_id"), "RET-2026-01")
            self.assertEqual(first_chunk.metadata.get("status"), "active")

    def test_search_returns_structured_results_and_valid_scores(self):
        """Verify that search returns properly typed SearchResult objects with numeric scores."""
        store = VectorStore.build(chunks=self.chunks, embeddings=self.embeddings)
        query_vec = _generate_dummy_embeddings(1, dim=self.dim, seed=99)[0]

        top_k = 5
        results = store.search(query_vec, top_k=top_k)

        self.assertEqual(len(results), top_k)
        for i, res in enumerate(results):
            self.assertIsInstance(res, SearchResult)
            self.assertIsInstance(res.score, float)
            self.assertTrue(res.chunk_id)
            self.assertTrue(res.filename.endswith(".md"))
            self.assertTrue(res.document_title)
            self.assertTrue(res.heading)
            self.assertTrue(res.content)
            self.assertIsInstance(res.metadata, dict)
            self.assertIn("document_id", res.metadata)

            # Verify descending score ordering
            if i > 0:
                self.assertGreaterEqual(results[i - 1].score, res.score)

    def test_top_k_bounds(self):
        """Verify top_k parameter respects chunk count limits."""
        store = VectorStore.build(chunks=self.chunks, embeddings=self.embeddings)
        query_vec = self.embeddings[0]

        res_1 = store.search(query_vec, top_k=1)
        self.assertEqual(len(res_1), 1)
        # Highest match for its own vector should be the first chunk with score approx 1.0
        self.assertEqual(res_1[0].chunk_id, self.chunks[0].chunk_id)
        self.assertAlmostEqual(res_1[0].score, 1.0, places=4)

        res_10 = store.search(query_vec, top_k=10)
        self.assertEqual(len(res_10), 10)

    @patch("app.rag.vector_store.embed_query")
    def test_retrieve_with_mocked_embedding(self, mock_embed_query):
        """Verify the retrieve(query, top_k) interface using a mocked embedding generator."""
        query_vec = _generate_dummy_embeddings(1, dim=self.dim, seed=7)[0]
        mock_embed_query.return_value = query_vec

        store = VectorStore.build(chunks=self.chunks, embeddings=self.embeddings)
        results = store.retrieve("return policy for normal items", top_k=3)

        self.assertEqual(len(results), 3)
        mock_embed_query.assert_called_once()
        self.assertIsInstance(results[0], SearchResult)

    def test_mismatched_chunks_and_embeddings_raises_error(self):
        """Verify that building a store with mismatched chunk/embedding lengths raises ValueError."""
        wrong_embeddings = _generate_dummy_embeddings(10, dim=self.dim)
        with self.assertRaises(ValueError):
            VectorStore.build(chunks=self.chunks, embeddings=wrong_embeddings)


class TestSentenceTransformerEmbeddings(unittest.TestCase):
    """Tests for local SentenceTransformer embedding wrapper with mocks."""

    def test_get_embedding_model_name_resolution(self):
        """Verify model name resolution with argument and fallback."""
        from app.rag.embeddings import get_embedding_model_name
        self.assertEqual(get_embedding_model_name("custom-model"), "custom-model")
        with patch.dict("os.environ", {"EMBEDDING_MODEL": "env-model"}):
            self.assertEqual(get_embedding_model_name(), "env-model")

    def test_get_embedding_dimension(self):
        """Verify dimension extraction from model."""
        from app.rag.embeddings import get_embedding_dimension

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        self.assertEqual(get_embedding_dimension(mock_model), 384)

    def test_embed_texts_with_mock_model(self):
        """Verify embed_texts returns float32 numpy array with normalized outputs."""
        from app.rag.embeddings import embed_texts

        mock_model = MagicMock()
        raw_output = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        mock_model.encode.return_value = raw_output

        texts = ["text A", "text B"]
        result = embed_texts(texts, model=mock_model)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (2, 2))
        mock_model.encode.assert_called_once()

    def test_embed_query_with_mock_model(self):
        """Verify embed_query returns 1D vector."""
        from app.rag.embeddings import embed_query

        mock_model = MagicMock()
        raw_output = np.array([[0.6, 0.8]], dtype=np.float32)
        mock_model.encode.return_value = raw_output

        result = embed_query("hello", model=mock_model)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (2,))

    def test_empty_texts_returns_empty_array(self):
        """Verify empty input returns empty float32 array without calling model."""
        from app.rag.embeddings import embed_texts
        result = embed_texts([])
        self.assertEqual(len(result), 0)
        self.assertEqual(result.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
