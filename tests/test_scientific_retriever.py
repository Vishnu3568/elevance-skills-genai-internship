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
        build_scientific_vector_store,
        ScientificRetriever,
        ScientificRetrievalResult,
        format_retrieval_context,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb import (  # type: ignore
        ScientificPaper,
        build_scientific_vector_store,
        ScientificRetriever,
        ScientificRetrievalResult,
        format_retrieval_context,
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


class TestScientificRetriever(unittest.TestCase):
    """Unit tests for the scientific semantic retrieval and metadata filtering layer."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base = Path(self.temp_dir)
        self.embeddings = DeterministicFakeEmbeddings()

        # Seed 4 diverse papers for comprehensive filtering tests
        self.paper1 = ScientificPaper(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            abstract="The dominant sequence transduction models are based on complex neural networks with attention...",
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
            abstract="Large pre-trained language models store factual knowledge in parametric memory with retrieval...",
            categories=["cs.CL", "cs.AI"],
            primary_category="cs.CL",
            published_date="2020-05-22",
            doi="10.48550/arXiv.2005.11401",
            journal_ref="NeurIPS 2020",
            concepts=["RAG", "Dense Retrieval", "Parametric Memory"],
        )

        self.paper3 = ScientificPaper(
            arxiv_id="2106.09685",
            title="LoRA: Low-Rank Adaptation of Large Language Models",
            authors=["Edward J. Hu", "Yelong Shen", "Phillip Wallis"],
            abstract="An important paradigm consists of fine-tuning large models using low rank adaptation matrices...",
            categories=["cs.LG", "cs.AI"],
            primary_category="cs.LG",
            published_date="2021-06-17",
            doi="10.48550/arXiv.2106.09685",
            journal_ref="ICLR 2022",
            concepts=["LoRA", "PEFT", "Fine-Tuning"],
        )

        self.paper4 = ScientificPaper(
            arxiv_id="2010.11929",
            title="An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
            authors=["Alexey Dosovitskiy", "Lucas Beyer", "Alexander Kolesnikov"],
            abstract="While the Transformer architecture has become the de-facto standard for NLP, we show Vision Transformers applied directly to images...",
            categories=["cs.CV", "cs.LG"],
            primary_category="cs.CV",
            published_date="2020-10-22",
            doi="10.48550/arXiv.2010.11929",
            journal_ref="ICLR 2021",
            concepts=["Vision Transformer", "ViT", "Computer Vision"],
        )

        store_path = self.base / "faiss_test"
        self.vector_store = build_scientific_vector_store(
            papers=[self.paper1, self.paper2, self.paper3, self.paper4],
            store_path=str(store_path),
            embeddings=self.embeddings,
        )
        self.retriever = ScientificRetriever(self.vector_store, default_k=3)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_semantic_ranking_and_top_k(self):
        results = self.retriever.retrieve("Retrieval-Augmented Generation factual knowledge", k=2)

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], ScientificRetrievalResult)
        # Top result should be the RAG paper
        self.assertEqual(results[0].arxiv_id, "2005.11401")
        self.assertIn("Retrieval-Augmented Generation", results[0].title)
        self.assertIsInstance(results[0].score, float)

    def test_score_threshold_filtering(self):
        # Setting a very strict L2 threshold filters out less relevant papers
        all_results = self.retriever.retrieve("Attention Transformer dominant sequence models", k=4)
        self.assertGreater(len(all_results), 0)

        # Filter with threshold slightly above top score
        top_score = all_results[0].score
        filtered_results = self.retriever.retrieve(
            "Attention Transformer dominant sequence models",
            k=4,
            score_threshold=top_score + 0.001,
        )
        self.assertGreaterEqual(len(filtered_results), 1)
        for res in filtered_results:
            self.assertLessEqual(res.score, top_score + 0.001)

    def test_category_filtering(self):
        # Search for vision transformer with CV category filter
        results = self.retriever.retrieve(
            "Transformers for image recognition at scale",
            k=4,
            category_filter="cs.CV",
        )
        self.assertGreater(len(results), 0)
        for res in results:
            self.assertTrue("cs.CV" in res.categories or res.primary_category == "cs.CV")

    def test_min_year_filtering(self):
        # Search with minimum publication year 2021 (only LoRA paper published in 2021)
        results = self.retriever.retrieve(
            "language models adaptation neural networks",
            k=4,
            min_year=2021,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].arxiv_id, "2106.09685")
        self.assertEqual(results[0].published_date, "2021-06-17")

    def test_author_filtering(self):
        # Filter by author name (case-insensitive substring)
        results = self.retriever.retrieve(
            "neural networks transformer",
            k=4,
            author_filter="Vaswani",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].arxiv_id, "1706.03762")
        self.assertIn("Vaswani", results[0].authors)

    def test_concept_filtering(self):
        results = self.retriever.retrieve(
            "adaptation fine-tuning models",
            k=4,
            concept_filter="PEFT",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].arxiv_id, "2106.09685")
        self.assertIn("PEFT", results[0].concepts)

    def test_format_retrieval_context(self):
        results = self.retriever.retrieve("Attention Is All You Need", k=1)
        context = format_retrieval_context(results)

        self.assertIn("[Paper 1]", context)
        self.assertIn("Title: Attention Is All You Need", context)
        self.assertIn("arXiv ID: 1706.03762", context)
        self.assertIn("Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar", context)
        self.assertIn("Categories: cs.CL (cs.CL, cs.LG)", context)
        self.assertIn("URL: https://arxiv.org/abs/1706.03762", context)
        self.assertIn("Content:", context)

    def test_format_retrieval_context_empty(self):
        context = format_retrieval_context([])
        self.assertEqual(context, "No relevant scientific papers found.")

    def test_query_validation(self):
        with self.assertRaises(TypeError):
            self.retriever.retrieve(12345)  # type: ignore

        with self.assertRaises(ValueError):
            self.retriever.retrieve("   ")

    def test_zero_k_returns_empty_list(self):
        results = self.retriever.retrieve("Attention", k=0)
        self.assertEqual(results, [])

    def test_retriever_initialization_validation(self):
        with self.assertRaises(ValueError):
            ScientificRetriever(None)  # type: ignore

    def test_category_filter_beyond_target_k_times_four(self):
        """Verify category filtering finds a matching document ranked beyond target_k * 4."""
        deep_papers = []
        # Create 12 high-similarity papers in cs.LG
        for i in range(1, 13):
            deep_papers.append(
                ScientificPaper(
                    arxiv_id=f"1901.{i:05d}",
                    title=f"Reinforcement Learning Optimization Paradigm {i}",
                    authors=["Common Author"],
                    abstract="Reinforcement learning optimization and policy gradients...",
                    categories=["cs.LG"],
                    primary_category="cs.LG",
                    published_date="2019-01-15",
                    concepts=["Reinforcement Learning"],
                )
            )
        # Create 13th paper in cs.CL (lower lexical similarity to RL query)
        deep_papers.append(
            ScientificPaper(
                arxiv_id="1901.99999",
                title="Syntactic Language Parsing and Dependency Trees",
                authors=["Linguistics Team"],
                abstract="Linguistic syntax trees and natural language dependency grammar...",
                categories=["cs.CL"],
                primary_category="cs.CL",
                published_date="2019-02-20",
                concepts=["NLP", "Parsing"],
            )
        )

        deep_store = build_scientific_vector_store(
            papers=deep_papers,
            store_path=str(self.base / "deep_store_cat"),
            embeddings=self.embeddings,
        )
        deep_retriever = ScientificRetriever(deep_store, default_k=1)

        # Query matches RL papers strongly, but we filter specifically for cs.CL
        # With old fetch_k = target_k * 4 = 4, the 13th paper would never be fetched.
        # With dynamic full-corpus search, it must be discovered.
        results = deep_retriever.retrieve(
            "Reinforcement Learning Optimization",
            k=1,
            category_filter="cs.CL",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].arxiv_id, "1901.99999")
        self.assertEqual(results[0].primary_category, "cs.CL")

    def test_author_filter_beyond_target_k_times_four(self):
        """Verify author filtering finds a matching paper ranked beyond target_k * 4."""
        deep_papers = []
        # Create 12 high-similarity papers with General Author
        for i in range(1, 13):
            deep_papers.append(
                ScientificPaper(
                    arxiv_id=f"2001.{i:05d}",
                    title=f"Computer Vision Feature Extraction {i}",
                    authors=["General Author"],
                    abstract="Convolutional visual feature representation extraction...",
                    categories=["cs.CV"],
                    primary_category="cs.CV",
                    published_date="2020-01-10",
                    concepts=["Computer Vision"],
                )
            )
        # Create 13th paper with unique target author
        deep_papers.append(
            ScientificPaper(
                arxiv_id="2001.88888",
                title="Deep Vision Classification Framework",
                authors=["UniqueTargetAuthor"],
                abstract="Deep convolutional networks for image classification...",
                categories=["cs.CV"],
                primary_category="cs.CV",
                published_date="2020-03-15",
                concepts=["Computer Vision"],
            )
        )

        deep_store = build_scientific_vector_store(
            papers=deep_papers,
            store_path=str(self.base / "deep_store_author"),
            embeddings=self.embeddings,
        )
        deep_retriever = ScientificRetriever(deep_store, default_k=1)

        # target_k = 1 with author filter for UniqueTargetAuthor
        results = deep_retriever.retrieve(
            "Computer Vision Feature Extraction",
            k=1,
            author_filter="UniqueTargetAuthor",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].arxiv_id, "2001.88888")
        self.assertIn("UniqueTargetAuthor", results[0].authors)

    def test_combined_metadata_filters_deep_recall(self):
        """Verify combined category, min_year, and concept filters work accurately."""
        results = self.retriever.retrieve(
            "transformers and neural networks",
            k=2,
            category_filter="cs.CL",
            min_year=2020,
            concept_filter="RAG",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].arxiv_id, "2005.11401")
        self.assertEqual(results[0].primary_category, "cs.CL")
        self.assertIn("RAG", results[0].concepts)

    def test_unfiltered_retrieval_respects_target_k_and_no_duplicates(self):
        """Verify unfiltered retrieval respects target_k and contains zero duplicate results."""
        results = self.retriever.retrieve("Transformer neural networks", k=3)
        self.assertEqual(len(results), 3)

        retrieved_ids = [r.arxiv_id for r in results]
        self.assertEqual(len(retrieved_ids), len(set(retrieved_ids)))

    def test_target_k_respected_when_many_matches_pass_filters(self):
        """Verify that when many papers pass filters, only target_k results are returned."""
        # Both paper1 and paper2 are in cs.CL
        results = self.retriever.retrieve(
            "neural networks",
            k=1,
            category_filter="cs.CL",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].primary_category, "cs.CL")

    def test_retriever_isolation_from_production_stores(self):
        """Verify retriever operations never modify production customer or medical stores."""
        proj_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        customer_faiss = proj_root / "faiss_index"
        medical_faiss = proj_root / "faiss_index_medical"

        cust_mtime = customer_faiss.stat().st_mtime if customer_faiss.exists() else None
        med_mtime = medical_faiss.stat().st_mtime if medical_faiss.exists() else None

        # Execute multiple queries and filters
        self.retriever.retrieve("Transformer", k=2)
        self.retriever.retrieve("RAG", category_filter="cs.AI", min_year=2020)

        if customer_faiss.exists():
            self.assertEqual(customer_faiss.stat().st_mtime, cust_mtime)
        if medical_faiss.exists():
            self.assertEqual(medical_faiss.stat().st_mtime, med_mtime)


if __name__ == "__main__":
    unittest.main()

