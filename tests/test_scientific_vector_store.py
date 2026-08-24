import unittest
import sys
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb import (  # type: ignore
        ScientificPaper,
        create_paper_documents,
        build_scientific_vector_store,
        load_scientific_vector_store,
        add_papers_to_vector_store,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb import (  # type: ignore
        ScientificPaper,
        create_paper_documents,
        build_scientific_vector_store,
        load_scientific_vector_store,
        add_papers_to_vector_store,
    )

try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    from langchain.embeddings.base import Embeddings


import re

class DeterministicFakeEmbeddings(Embeddings):
    """Fast, deterministic local embeddings for unit testing without external model overhead."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def _embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        words = re.findall(r"\w+", text.lower())
        for word in words:
            idx = abs(hash(word)) % self.dim
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


class TestScientificVectorStore(unittest.TestCase):
    """Unit and isolation tests for the scientific vector store module."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base = Path(self.temp_dir)
        self.embeddings = DeterministicFakeEmbeddings()

        self.paper1 = ScientificPaper(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
            categories=["cs.CL", "cs.LG"],
            primary_category="cs.CL",
            published_date="2017-06-12",
            doi="10.48550/arXiv.1706.03762",
            journal_ref="NeurIPS 2017",
            concepts=["Transformer", "Self-Attention", "Multi-Head Attention"],
        )

        self.paper2 = ScientificPaper(
            arxiv_id="2005.11401",
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            authors=["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus"],
            abstract="Large pre-trained language models store factual knowledge in their parameters...",
            categories=["cs.CL", "cs.AI"],
            primary_category="cs.CL",
            published_date="2020-05-22",
            doi="10.48550/arXiv.2005.11401",
            journal_ref="NeurIPS 2020",
            concepts=["RAG", "Dense Retrieval", "Parametric Memory"],
        )

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_paper_documents(self):
        docs = create_paper_documents([self.paper1, self.paper2])

        self.assertEqual(len(docs), 2)

        # Verify page_content structure
        content = docs[0].page_content
        self.assertIn("Title: Attention Is All You Need", content)
        self.assertIn("Categories: cs.CL (cs.CL, cs.LG)", content)
        self.assertIn("Concepts: Transformer, Self-Attention, Multi-Head Attention", content)
        self.assertIn("Abstract: The dominant sequence transduction models", content)

        # Verify metadata
        meta = docs[0].metadata
        self.assertEqual(meta["arxiv_id"], "1706.03762")
        self.assertEqual(meta["title"], "Attention Is All You Need")
        self.assertEqual(meta["authors"], "Ashish Vaswani, Noam Shazeer, Niki Parmar")
        self.assertEqual(meta["primary_category"], "cs.CL")
        self.assertEqual(meta["categories"], ["cs.CL", "cs.LG"])
        self.assertEqual(meta["published_date"], "2017-06-12")
        self.assertEqual(meta["url"], "https://arxiv.org/abs/1706.03762")
        self.assertEqual(meta["doi"], "10.48550/arXiv.1706.03762")
        self.assertEqual(meta["journal_ref"], "NeurIPS 2017")
        self.assertEqual(meta["concepts"], ["Transformer", "Self-Attention", "Multi-Head Attention"])

    def test_document_metadata_separated_from_embedding_content(self):
        docs = create_paper_documents([self.paper1])
        content = docs[0].page_content
        meta = docs[0].metadata

        # Metadata fields should exist in metadata dict
        self.assertEqual(meta["doi"], "10.48550/arXiv.1706.03762")
        self.assertEqual(meta["url"], "https://arxiv.org/abs/1706.03762")

        # But should NOT be embedded into page_content
        self.assertNotIn("https://arxiv.org/abs/", content)
        self.assertNotIn("10.48550/", content)

    def test_create_documents_does_not_mutate_papers(self):
        orig_concepts = list(self.paper1.concepts)
        orig_authors = list(self.paper1.authors)

        create_paper_documents([self.paper1])

        self.assertEqual(self.paper1.concepts, orig_concepts)
        self.assertEqual(self.paper1.authors, orig_authors)

    def test_build_scientific_vector_store(self):
        store_path = self.base / "faiss_test"
        vector_store = build_scientific_vector_store(
            papers=[self.paper1, self.paper2],
            store_path=str(store_path),
            embeddings=self.embeddings,
        )

        self.assertTrue(store_path.exists())
        self.assertEqual(vector_store.index.ntotal, 2)
        self.assertEqual(len(vector_store.docstore._dict), 2)

    def test_load_scientific_vector_store(self):
        store_path = self.base / "faiss_test"
        build_scientific_vector_store(
            papers=[self.paper1, self.paper2],
            store_path=str(store_path),
            embeddings=self.embeddings,
        )

        loaded_store = load_scientific_vector_store(
            store_path=str(store_path),
            embeddings=self.embeddings,
        )

        self.assertEqual(loaded_store.index.ntotal, 2)
        self.assertEqual(len(loaded_store.docstore._dict), 2)

        # Check docstore metadata persistence
        first_doc = list(loaded_store.docstore._dict.values())[0]
        self.assertIn("arxiv_id", first_doc.metadata)

    def test_empty_build_is_rejected(self):
        store_path = self.base / "faiss_empty"
        with self.assertRaises(ValueError):
            build_scientific_vector_store(
                papers=[],
                store_path=str(store_path),
                embeddings=self.embeddings,
            )

    def test_missing_store_raises_file_not_found(self):
        missing_path = self.base / "non_existent_faiss"
        with self.assertRaises(FileNotFoundError):
            load_scientific_vector_store(
                store_path=str(missing_path),
                embeddings=self.embeddings,
            )

    def test_add_papers_to_vector_store(self):
        store_path = self.base / "faiss_incremental"
        vector_store = build_scientific_vector_store(
            papers=[self.paper1],
            store_path=str(store_path),
            embeddings=self.embeddings,
        )
        self.assertEqual(vector_store.index.ntotal, 1)

        updated_store = add_papers_to_vector_store(vector_store, [self.paper2])
        self.assertEqual(updated_store.index.ntotal, 2)
        self.assertEqual(len(updated_store.docstore._dict), 2)

    def test_empty_add_is_safe(self):
        store_path = self.base / "faiss_safe"
        vector_store = build_scientific_vector_store(
            papers=[self.paper1],
            store_path=str(store_path),
            embeddings=self.embeddings,
        )

        same_store = add_papers_to_vector_store(vector_store, [])
        self.assertEqual(same_store.index.ntotal, 1)

    def test_semantic_retrieval_returns_relevant_paper(self):
        store_path = self.base / "faiss_retrieval"
        vector_store = build_scientific_vector_store(
            papers=[self.paper1, self.paper2],
            store_path=str(store_path),
            embeddings=self.embeddings,
        )

        # Query relevant to RAG paper
        results = vector_store.similarity_search("Retrieval-Augmented Generation factual knowledge", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["arxiv_id"], "2005.11401")

        # Query relevant to Attention paper
        results_att = vector_store.similarity_search("Attention Transformer dominant sequence models", k=1)
        self.assertEqual(len(results_att), 1)
        self.assertEqual(results_att[0].metadata["arxiv_id"], "1706.03762")

    def test_scientific_store_isolation(self):
        """Verify that scientific vector store construction does NOT touch customer or medical stores."""
        proj_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        customer_faiss = proj_root / "faiss_index"
        medical_faiss = proj_root / "faiss_index_medical"

        # Record modification times if directories exist
        cust_mtime = customer_faiss.stat().st_mtime if customer_faiss.exists() else None
        med_mtime = medical_faiss.stat().st_mtime if medical_faiss.exists() else None

        # Execute build and load operations in temp directory
        temp_store = self.base / "temp_faiss_isolated"
        build_scientific_vector_store(
            papers=[self.paper1, self.paper2],
            store_path=str(temp_store),
            embeddings=self.embeddings,
        )
        load_scientific_vector_store(
            store_path=str(temp_store),
            embeddings=self.embeddings,
        )

        # Assert production directories are unmodified
        if customer_faiss.exists():
            self.assertEqual(customer_faiss.stat().st_mtime, cust_mtime)
        if medical_faiss.exists():
            self.assertEqual(medical_faiss.stat().st_mtime, med_mtime)


if __name__ == "__main__":
    unittest.main()
