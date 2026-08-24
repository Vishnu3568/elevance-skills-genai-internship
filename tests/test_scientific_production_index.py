import unittest
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb import (  # type: ignore
        DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
        ScientificRetriever,
        ingest_jsonl,
        load_scientific_vector_store,
    )
    # pyrefly: ignore [missing-import]
    from src.langchain_helper import get_instructor_embeddings  # type: ignore
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb import (  # type: ignore
        DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
        ScientificRetriever,
        ingest_jsonl,
        load_scientific_vector_store,
    )
    # pyrefly: ignore [missing-import]
    from langchain_helper import get_instructor_embeddings  # type: ignore


class TestScientificProductionIndex(unittest.TestCase):
    """Integration test suite for the production scientific dataset and FAISS store."""

    @classmethod
    def setUpClass(cls):
        cls.proj_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.dataset_path = cls.proj_root / "dataset" / "arxiv_ai_ml_subset.jsonl"
        cls.store_path = cls.proj_root / DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH

    def test_production_dataset_file_and_schema_validation(self):
        """Verify that dataset/arxiv_ai_ml_subset.jsonl contains 100 valid scientific records."""
        self.assertTrue(self.dataset_path.exists(), f"Dataset not found at {self.dataset_path}")
        result = ingest_jsonl(str(self.dataset_path))

        self.assertEqual(result.accepted, 100)
        self.assertEqual(result.invalid, 0)
        self.assertEqual(result.out_of_domain, 0)
        self.assertEqual(result.duplicates, 0)
        self.assertEqual(len(result.errors), 0)

        # Inspect first paper
        paper = result.papers[0]
        self.assertTrue(paper.arxiv_id)
        self.assertTrue(paper.title)
        self.assertTrue(len(paper.authors) > 0)
        self.assertTrue(paper.abstract)
        self.assertTrue(paper.primary_category)
        self.assertTrue(paper.published_date)

    def test_production_store_exists_and_vector_count_matches(self):
        """Verify that faiss_index_scientific contains 100 vectors and valid docstore documents."""
        if not self.store_path.exists():
            self.skipTest("faiss_index_scientific/ not yet built")

        embeddings = get_instructor_embeddings()
        vector_store = load_scientific_vector_store(str(self.store_path), embeddings=embeddings)

        docstore = vector_store.docstore._dict
        self.assertEqual(len(docstore), 100)
        self.assertEqual(vector_store.index.ntotal, 100)

        # Check metadata consistency across docstore
        for doc in docstore.values():
            meta = doc.metadata
            self.assertIn("arxiv_id", meta)
            self.assertIn("title", meta)
            self.assertIn("authors", meta)
            self.assertIn("primary_category", meta)
            self.assertIn("published_date", meta)
            self.assertTrue(meta["arxiv_id"])
            self.assertTrue(meta["title"])

    def test_production_semantic_retrieval(self):
        """Verify semantic retrieval on the production index across target domain topics."""
        if not self.store_path.exists():
            self.skipTest("faiss_index_scientific/ not yet built")

        embeddings = get_instructor_embeddings()
        vector_store = load_scientific_vector_store(str(self.store_path), embeddings=embeddings)
        retriever = ScientificRetriever(vector_store, default_k=3)

        results = retriever.retrieve("speech recognition and speaker diarization", k=3)
        self.assertGreaterEqual(len(results), 1)
        top = results[0]
        self.assertTrue(top.arxiv_id)
        self.assertTrue(top.title)
        self.assertGreater(top.score, 0.0)

    def test_production_isolation(self):
        """Verify that customer and medical stores are completely unaffected."""
        customer_csv = self.proj_root / "dataset" / "knowledge_base.csv"
        customer_faiss = self.proj_root / "faiss_index"
        medical_faiss = self.proj_root / "faiss_index_medical"

        self.assertTrue(customer_csv.exists())
        self.assertTrue(customer_faiss.exists())
        self.assertTrue(medical_faiss.exists())

        # Customer CSV must remain 79 rows
        import pandas as pd
        df = pd.read_csv(customer_csv)
        self.assertEqual(len(df), 79)

        # Customer FAISS must remain 79 vectors
        import faiss
        c_idx = faiss.read_index(str(customer_faiss / "index.faiss"))
        self.assertEqual(c_idx.ntotal, 79)


if __name__ == "__main__":
    unittest.main()
