"""Integration benchmark and verification suite for production Scientific FAISS retrieval.

Validates unfiltered queries, multi-criteria metadata filtering (category, year, author,
combined), data contract integrity, store isolation, and proves the Step 2 deep-recall
improvement on the real 100-paper Kaggle-derived corpus.
"""

import json
import os
import re
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Set

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb import (  # type: ignore
        DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
        ScientificRetrievalResult,
        ScientificRetriever,
        load_scientific_vector_store,
    )
    # pyrefly: ignore [missing-import]
    from src.langchain_helper import get_instructor_embeddings  # type: ignore
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb import (  # type: ignore
        DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
        ScientificRetrievalResult,
        ScientificRetriever,
        load_scientific_vector_store,
    )
    # pyrefly: ignore [missing-import]
    from langchain_helper import get_instructor_embeddings  # type: ignore


class TestScientificRetrievalBenchmark(unittest.TestCase):
    """Benchmark test suite validating ScientificRetriever against production Kaggle FAISS store."""

    @classmethod
    def setUpClass(cls):
        cls.proj_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.kaggle_jsonl = cls.proj_root / "dataset" / "arxiv_ai_ml_subset_kaggle.jsonl"
        cls.store_path = cls.proj_root / DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH

        if not cls.store_path.exists():
            raise unittest.SkipTest("faiss_index_scientific/ does not exist. Build it first.")

        # Load real 100 Kaggle paper records for reference
        cls.kaggle_records: Dict[str, dict] = {}
        with open(cls.kaggle_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cls.kaggle_records[rec["id"]] = rec

        cls.embeddings = get_instructor_embeddings()
        cls.vector_store = load_scientific_vector_store(str(cls.store_path), embeddings=cls.embeddings)
        cls.retriever = ScientificRetriever(cls.vector_store, default_k=3)
        cls.iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}")

    def _assert_valid_retrieval_result(self, res: ScientificRetrievalResult):
        """Helper to enforce strict data contracts on every retrieved document."""
        self.assertIsInstance(res, ScientificRetrievalResult)
        self.assertIn(res.arxiv_id, self.kaggle_records, f"arXiv ID {res.arxiv_id} not in Kaggle corpus!")
        self.assertTrue(res.title and len(res.title.strip()) > 0, "Empty title returned")
        self.assertTrue(self.iso_re.match(res.published_date), f"Invalid ISO published_date: {res.published_date}")
        self.assertTrue(res.url.startswith("https://arxiv.org/abs/"), f"Invalid URL: {res.url}")
        self.assertIsInstance(res.score, float)
        self.assertGreaterEqual(res.score, 0.0, "Score should be non-negative")

    def test_unfiltered_retrieval_production_corpus(self):
        """A. Unfiltered retrieval on 3 representative queries respecting target_k, no duplicates, valid IDs."""
        test_queries = [
            ("natural language processing and language models", 3),
            ("deep convolutional neural networks for computer vision", 2),
            ("gradient optimization and machine learning representations", 4),
        ]

        for query, target_k in test_queries:
            results = self.retriever.retrieve(query, k=target_k)
            self.assertEqual(len(results), target_k, f"Query '{query}' did not return target_k={target_k}")

            seen_ids: Set[str] = set()
            for r in results:
                self._assert_valid_retrieval_result(r)
                self.assertNotIn(r.arxiv_id, seen_ids, f"Duplicate paper {r.arxiv_id} in query '{query}'")
                seen_ids.add(r.arxiv_id)

    def test_category_filtering_production_corpus(self):
        """B. Category filtering across cs.LG, cs.CV, cs.CL, cs.AI."""
        categories_to_test = ["cs.LG", "cs.CV", "cs.CL", "cs.AI"]

        for cat in categories_to_test:
            results = self.retriever.retrieve(
                "machine learning representation models",
                k=2,
                category_filter=cat,
            )
            self.assertGreaterEqual(len(results), 1, f"No results returned for category {cat}")
            for r in results:
                self._assert_valid_retrieval_result(r)
                self.assertTrue(
                    r.primary_category.lower() == cat.lower() or any(c.lower() == cat.lower() for c in r.categories),
                    f"Result {r.arxiv_id} does not match requested category {cat}",
                )

    def test_year_filtering_production_corpus(self):
        """C. Publication year filtering for min_year=2019 and min_year=2020."""
        for min_yr in [2019, 2020]:
            results = self.retriever.retrieve(
                "neural networks deep learning",
                k=3,
                min_year=min_yr,
            )
            self.assertGreaterEqual(len(results), 1)
            for r in results:
                self._assert_valid_retrieval_result(r)
                yr = int(r.published_date[:4])
                self.assertGreaterEqual(yr, min_yr, f"Paper {r.arxiv_id} published in {yr} < min_year {min_yr}")

    def test_author_filtering_production_corpus(self):
        """D. Author filtering for real authors present in the 100-paper Kaggle corpus."""
        # 'Urvashi Khandelwal' is an author of 1911.00172 in the corpus
        results = self.retriever.retrieve(
            "language modeling and nearest neighbors",
            k=1,
            author_filter="Khandelwal",
        )
        self.assertEqual(len(results), 1)
        self._assert_valid_retrieval_result(results[0])
        self.assertEqual(results[0].arxiv_id, "1911.00172")
        self.assertIn("Khandelwal", results[0].authors)

    def test_combined_filters_production_corpus(self):
        """F. Multi-criteria combined filtering (category + year, category + author)."""
        # Combination 1: category=cs.CL + min_year=2020
        comb1_results = self.retriever.retrieve(
            "natural language representations",
            k=2,
            category_filter="cs.CL",
            min_year=2020,
        )
        self.assertGreaterEqual(len(comb1_results), 1)
        for r in comb1_results:
            self._assert_valid_retrieval_result(r)
            self.assertTrue(r.primary_category == "cs.CL" or "cs.CL" in r.categories)
            self.assertGreaterEqual(int(r.published_date[:4]), 2020)

        # Combination 2: category=cs.CV + author=Kim
        comb2_results = self.retriever.retrieve(
            "object detectors neural networks",
            k=1,
            category_filter="cs.CV",
            author_filter="Kim",
        )
        self.assertEqual(len(comb2_results), 1)
        self._assert_valid_retrieval_result(comb2_results[0])
        self.assertEqual(comb2_results[0].arxiv_id, "1912.00289")
        self.assertIn("Kim", comb2_results[0].authors)
        self.assertTrue(comb2_results[0].primary_category == "cs.CV" or "cs.CV" in comb2_results[0].categories)

    def test_step2_recall_improvement_proof_on_real_corpus(self):
        """5. Prove Step 2 deep-recall improvement on real production corpus.

        Case Study:
        Query: 'natural language processing'
        Paper 1910.00163 ('Specializing Word Embeddings...') is authored by 'Jason Eisner'.
        In raw semantic similarity for 'natural language processing', it ranks 7th.

        With old logic: fetch_k = target_k * 4 = 1 * 4 = 4 candidates (Ranks 1..4).
        Because 1910.00163 is at Rank 7, old logic returned 0 results.

        With Step 2 dynamic corpus search (fetch_k = 100), the retriever searches
        all 100 papers and successfully returns 1910.00163 at rank 1!
        """
        # 1. Verify rank of target paper in raw similarity search
        raw_candidates = self.vector_store.similarity_search_with_score("natural language processing", k=15)
        raw_ids = [doc.metadata["arxiv_id"] for doc, _ in raw_candidates]
        target_id = "1910.00163"

        self.assertIn(target_id, raw_ids, f"Expected {target_id} in top 15 raw candidates")
        raw_rank = raw_ids.index(target_id) + 1
        # Confirm it is beyond rank 4
        self.assertGreater(raw_rank, 4, f"Target paper rank ({raw_rank}) is not > 4")

        # 2. Query with target_k=1 and author_filter="Jason Eisner"
        results = self.retriever.retrieve(
            "natural language processing",
            k=1,
            author_filter="Jason Eisner",
        )

        # 3. Assert that paper is successfully retrieved
        self.assertEqual(len(results), 1, "Failed to retrieve paper beyond the old target_k * 4 ceiling!")
        self.assertEqual(results[0].arxiv_id, target_id)
        self.assertIn("Jason Eisner", results[0].authors)
        self.assertEqual(results[0].title, "Specializing Word Embeddings (for Parsing) by Information Bottleneck")

    def test_production_stores_isolation(self):
        """Verify customer and medical vector stores remain strictly untouched."""
        customer_csv = self.proj_root / "dataset" / "knowledge_base.csv"
        customer_faiss = self.proj_root / "faiss_index"
        medical_faiss = self.proj_root / "faiss_index_medical"

        self.assertTrue(customer_csv.exists())
        self.assertTrue(customer_faiss.exists())
        self.assertTrue(medical_faiss.exists())

        import pandas as pd
        df = pd.read_csv(customer_csv)
        self.assertEqual(len(df), 79, "Customer CSV rows modified!")

        import faiss
        c_idx = faiss.read_index(str(customer_faiss / "index.faiss"))
        self.assertEqual(c_idx.ntotal, 79, "Customer FAISS vectors modified!")

        m_idx = faiss.read_index(str(medical_faiss / "index.faiss"))
        self.assertEqual(m_idx.ntotal, 5, "Medical FAISS vectors modified!")


if __name__ == "__main__":
    unittest.main()
