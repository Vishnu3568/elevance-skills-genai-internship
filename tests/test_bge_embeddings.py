"""Focused unit tests for the centralized BGE-small embedding model provider."""

import os
import sys
import unittest
import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_helper import get_instructor_embeddings


class TestBGEEmbeddings(unittest.TestCase):
    """Test suite verifying BGE-small embedding contract, dimensions, and normalization."""

    def test_bge_small_embedding_contract(self):
        """Verify BGE-small embedding provider yields 384-dimensional unit vectors."""
        embedding = get_instructor_embeddings()
        self.assertIsNotNone(embedding)

        vector = embedding.embed_query("test query")
        self.assertEqual(len(vector), 384)

        norm = float(np.linalg.norm(vector))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_bge_small_document_embedding_contract(self):
        """Verify document embeddings also produce 384-dimensional unit vectors."""
        embedding = get_instructor_embeddings()
        docs = ["Sample customer support document", "Clinical trial evidence"]
        vectors = embedding.embed_documents(docs)

        self.assertEqual(len(vectors), 2)
        for vec in vectors:
            self.assertEqual(len(vec), 384)
            norm = float(np.linalg.norm(vec))
            self.assertAlmostEqual(norm, 1.0, places=4)

    def test_singleton_behavior(self):
        """Verify get_instructor_embeddings maintains single instance."""
        emb1 = get_instructor_embeddings()
        emb2 = get_instructor_embeddings()
        self.assertIs(emb1, emb2)


if __name__ == "__main__":
    unittest.main()
