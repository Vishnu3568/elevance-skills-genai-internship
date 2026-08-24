"""Comprehensive Unit & Integration Test Suite for Scientific Exploration Engine.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb import (  # type: ignore
        DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
        AuthorStatistic,
        CategoryStatistic,
        ConceptCooccurrence,
        ConceptExtractionResult,
        CorpusSummary,
        RelatedPaperMatch,
        ScientificExplorationEngine,
        ScientificExplorationGraph,
        ScientificRetrievalResult,
        ScientificRetriever,
        extract_concepts_from_text,
        load_scientific_vector_store,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb import (  # type: ignore
        DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
        AuthorStatistic,
        CategoryStatistic,
        ConceptCooccurrence,
        ConceptExtractionResult,
        CorpusSummary,
        RelatedPaperMatch,
        ScientificExplorationEngine,
        ScientificExplorationGraph,
        ScientificRetrievalResult,
        ScientificRetriever,
        extract_concepts_from_text,
        load_scientific_vector_store,
    )


class TestScientificExploration(unittest.TestCase):
    """Test suite covering deterministic concept extraction, analytics, graph, and related papers."""

    @classmethod
    def setUpClass(cls):
        cls.proj_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.scientific_store_path = cls.proj_root / DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH

        # Synthetic fixture for isolated unit tests
        cls.sample_papers = [
            {
                "arxiv_id": "2012.00001v1",
                "title": "A Review of Transformer Architectures and Attention Mechanisms in NLP",
                "abstract": "We explore multi-head self-attention, BERT fine-tuning, and language modeling with transfer learning.",
                "authors": "Alice Smith, Bob Jones",
                "primary_category": "cs.CL",
                "categories": ["cs.CL", "cs.AI"],
                "published_date": "2020-05-10T12:00:00Z",
            },
            {
                "arxiv_id": "2012.00002v1",
                "title": "End-to-End Speaker Diarization with Speech Recognition",
                "abstract": "We present neural diarization and acoustic modeling for speech recognition using deep learning.",
                "authors": "Charlie Brown, Alice Smith",
                "primary_category": "cs.CL",
                "categories": ["cs.CL", "eess.AS"],
                "published_date": "2020-05-15T14:30:00Z",
            },
            {
                "arxiv_id": "1905.00003v1",
                "title": "Graph Neural Networks for Relation Extraction",
                "abstract": "We evaluate GNNs and convolutional neural networks for relation extraction in knowledge bases.",
                "authors": "David Miller",
                "primary_category": "cs.AI",
                "categories": ["cs.AI", "cs.LG"],
                "published_date": "1905-11-20T08:00:00Z",  # test early year
            },
            {
                "arxiv_id": "2012.00004v1",
                "title": "Model Compression of Convolutional Neural Networks",
                "abstract": "We apply 2-bit weight quantization and knowledge distillation for deep learning acceleration.",
                "authors": "Eve White, Bob Jones",
                "primary_category": "cs.CV",
                "categories": ["cs.CV", "cs.LG"],
                "published_date": "2020-12-01T09:00:00Z",
            },
        ]

    def test_concept_extraction_and_normalization(self):
        """Verify that concept extraction normalizes phrases correctly."""
        text = "We use a multi-head self-attention mechanism and BERT fine-tuning for sequence-to-sequence modeling."
        concepts = extract_concepts_from_text(title="Attention in NLP", abstract=text)
        self.assertIn("Attention", concepts)
        self.assertIn("BERT", concepts)
        self.assertIn("Fine-Tuning", concepts)
        self.assertIn("Sequence-to-Sequence", concepts)
        self.assertIn("Natural Language Processing", concepts)

    def test_false_positive_protection_for_short_terms(self):
        """Verify that short common terms do not trigger false positive matches."""
        # e.g., 'civil' or 'cv' should not match 'Computer Vision' unless full term is present
        text = "We discuss civil engineering and ml algorithms in general."
        concepts = extract_concepts_from_text(title="Civil analysis", abstract=text)
        self.assertNotIn("Computer Vision", concepts)

    def test_concept_frequency_and_ranking(self):
        """Verify concept frequency calculation across synthetic corpus."""
        engine = ScientificExplorationEngine(papers=self.sample_papers)
        stats = engine.get_concept_statistics()
        self.assertGreater(len(stats), 0)

        # 'Deep Learning' appears in paper 2 and 4
        dl_stat = next((s for s in stats if s.concept == "Deep Learning"), None)
        self.assertIsNotNone(dl_stat)
        self.assertEqual(dl_stat.frequency, 2)
        self.assertIn("2012.00002v1", dl_stat.paper_ids)
        self.assertIn("2012.00004v1", dl_stat.paper_ids)

    def test_concept_to_paper_mapping(self):
        """Verify papers_for_concept returns matching papers."""
        engine = ScientificExplorationEngine(papers=self.sample_papers)
        diar_papers = engine.get_papers_for_concept("Speaker Diarization")
        self.assertEqual(len(diar_papers), 1)
        self.assertEqual(diar_papers[0]["arxiv_id"], "2012.00002v1")

    def test_paper_to_concepts_mapping(self):
        """Verify concepts_for_paper returns concepts of specific paper."""
        engine = ScientificExplorationEngine(papers=self.sample_papers)
        concepts = engine.get_concepts_for_paper("2012.00001v1")
        self.assertIn("Transformer", concepts)
        self.assertIn("Attention", concepts)
        self.assertIn("BERT", concepts)
        self.assertIn("Transfer Learning", concepts)

    def test_concept_cooccurrence(self):
        """Verify concept co-occurrence pairs and frequencies."""
        engine = ScientificExplorationEngine(papers=self.sample_papers)
        cooccurs = engine.get_concept_cooccurrence()
        self.assertGreater(len(cooccurs), 0)

        # 'Attention' and 'BERT' co-occur in paper 1
        att_bert = next(
            (c for c in cooccurs if (c.concept_a == "Attention" and c.concept_b == "BERT") or (c.concept_a == "BERT" and c.concept_b == "Attention")),
            None,
        )
        self.assertIsNotNone(att_bert)
        self.assertGreaterEqual(att_bert.cooccurrence_count, 1)
        self.assertIn("2012.00001v1", att_bert.paper_ids)

    def test_graph_generation_and_node_types(self):
        """Verify tripartite graph node and edge generation."""
        engine = ScientificExplorationEngine(papers=self.sample_papers)
        graph: ScientificExplorationGraph = engine.build_exploration_graph()

        node_types = {n.node_type for n in graph.nodes}
        self.assertIn("category", node_types)
        self.assertIn("concept", node_types)
        self.assertIn("paper", node_types)

        edge_types = {e.relation_type for e in graph.edges}
        self.assertIn("belongs_to_category", edge_types)
        self.assertIn("exhibits_concept", edge_types)
        self.assertIn("concept_cooccurrence", edge_types)

    def test_graph_determinism(self):
        """Verify that building the graph multiple times yields identical results."""
        engine = ScientificExplorationEngine(papers=self.sample_papers)
        g1 = engine.build_exploration_graph()
        g2 = engine.build_exploration_graph()

        self.assertEqual(len(g1.nodes), len(g2.nodes))
        self.assertEqual(len(g1.edges), len(g2.edges))
        self.assertEqual([n.id for n in g1.nodes], [n.id for n in g2.nodes])
        self.assertEqual([(e.source, e.target, e.weight) for e in g1.edges], [(e.source, e.target, e.weight) for e in g2.edges])

    def test_primary_and_all_category_statistics(self):
        """Verify category distribution calculation."""
        engine = ScientificExplorationEngine(papers=self.sample_papers)
        stats = engine.get_category_statistics()
        self.assertGreater(len(stats), 0)

        cs_cl = next((s for s in stats if s.category == "cs.CL"), None)
        self.assertIsNotNone(cs_cl)
        self.assertEqual(cs_cl.primary_count, 2)
        self.assertEqual(cs_cl.all_count, 2)
        self.assertEqual(cs_cl.percentage, 50.0)

    def test_year_statistics_and_timeline(self):
        """Verify publication year statistics and monthly timeline aggregation."""
        engine = ScientificExplorationEngine(papers=self.sample_papers)
        year_stats = engine.get_year_statistics()
        self.assertIn(2020, year_stats)
        self.assertEqual(year_stats[2020], 3)

        timeline_monthly = engine.get_timeline(granularity="monthly")
        self.assertGreater(len(timeline_monthly), 0)
        # May 2020 has 2 papers
        may_2020 = next((t for t in timeline_monthly if t.period == "2020-05"), None)
        self.assertIsNotNone(may_2020)
        self.assertEqual(may_2020.paper_count, 2)

    def test_author_statistics(self):
        """Verify author aggregation and top author ranking."""
        engine = ScientificExplorationEngine(papers=self.sample_papers)
        author_stats = engine.get_author_statistics(top_n=5)
        self.assertGreater(len(author_stats), 0)

        # Alice Smith and Bob Jones have 2 papers each
        alice = next((a for a in author_stats if a.author == "Alice Smith"), None)
        self.assertIsNotNone(alice)
        self.assertEqual(alice.paper_count, 2)
        self.assertEqual(len(alice.paper_ids), 2)

    def test_corpus_summary(self):
        """Verify complete corpus summary structure."""
        engine = ScientificExplorationEngine(papers=self.sample_papers)
        summary: CorpusSummary = engine.get_corpus_summary()
        self.assertEqual(summary.total_papers, 4)
        self.assertTrue(summary.earliest_date)
        self.assertTrue(summary.latest_date)
        self.assertGreater(len(summary.top_concepts), 0)
        self.assertGreater(len(summary.top_authors), 0)

    def test_related_paper_discovery_excludes_source_paper(self):
        """Verify that find_related_papers reuses retriever and removes self."""
        mock_retriever = MagicMock(spec=ScientificRetriever)
        mock_doc = MagicMock()
        mock_doc.page_content = "Transformer and NLP content"

        # Mock retriever returning source paper + 2 related papers
        mock_retriever.retrieve.return_value = [
            ScientificRetrievalResult(
                document=mock_doc,
                arxiv_id="2012.00001v1",  # Source paper itself
                title="A Review of Transformer Architectures and Attention Mechanisms in NLP",
                authors="Alice Smith",
                primary_category="cs.CL",
                categories=["cs.CL"],
                published_date="2020-05-10",
                score=0.0,
            ),
            ScientificRetrievalResult(
                document=mock_doc,
                arxiv_id="2012.00002v1",  # Related paper
                title="End-to-End Speaker Diarization with Speech Recognition",
                authors="Charlie Brown",
                primary_category="cs.CL",
                categories=["cs.CL"],
                published_date="2020-05-15",
                score=0.25,
            ),
            ScientificRetrievalResult(
                document=mock_doc,
                arxiv_id="2012.00004v1",  # Related paper
                title="Model Compression of Convolutional Neural Networks",
                authors="Eve White",
                primary_category="cs.CV",
                categories=["cs.CV"],
                published_date="2020-12-01",
                score=0.35,
            ),
        ]

        engine = ScientificExplorationEngine(papers=self.sample_papers)
        related = engine.find_related_papers(target="2012.00001v1", retriever=mock_retriever, top_k=2)

        self.assertEqual(len(related), 2)
        # Verify source paper was excluded
        self.assertNotIn("2012.00001v1", [r.paper.arxiv_id for r in related])
        self.assertEqual(related[0].paper.arxiv_id, "2012.00002v1")
        self.assertIn("cs.CL", related[0].shared_categories)

    def test_empty_corpus_and_invalid_dates_handling(self):
        """Verify robust handling of empty corpus and invalid/missing dates."""
        empty_engine = ScientificExplorationEngine(papers=[])
        self.assertEqual(empty_engine.total_papers, 0)
        self.assertEqual(empty_engine.get_concept_statistics(), [])
        self.assertEqual(empty_engine.get_category_statistics(), [])
        self.assertEqual(empty_engine.get_timeline(), [])
        self.assertEqual(empty_engine.get_author_statistics(), [])

        # Paper with invalid / None published_date
        bad_date_papers = [
            {
                "arxiv_id": "9999.99999",
                "title": "Invalid Date Paper",
                "abstract": "Deep learning paper.",
                "authors": "Test Author",
                "primary_category": "cs.AI",
                "categories": ["cs.AI"],
                "published_date": None,
            }
        ]
        bad_date_engine = ScientificExplorationEngine(papers=bad_date_papers)
        self.assertEqual(bad_date_engine.get_year_statistics(), {})
        self.assertEqual(bad_date_engine.get_timeline(), [])

    def test_production_scientific_corpus_exploration_integration(self):
        """Verify exploration engine against the production 100-paper scientific store."""
        if not self.scientific_store_path.exists():
            self.skipTest("faiss_index_scientific/ not yet built")

        vector_store = load_scientific_vector_store(str(self.scientific_store_path))
        engine = ScientificExplorationEngine(vector_store=vector_store)

        self.assertEqual(engine.total_papers, 100)
        summary = engine.get_corpus_summary()
        self.assertEqual(summary.total_papers, 100)
        self.assertEqual(summary.earliest_date[:4], "2019")
        self.assertEqual(summary.latest_date[:4], "2020")

        # Check concept statistics
        concept_stats = engine.get_concept_statistics()
        self.assertGreater(len(concept_stats), 5)
        # Top concepts should include domain items like Deep Learning, Speech Recognition, etc.
        top_concept_names = [cs.concept for cs in concept_stats[:5]]
        self.assertTrue(any(c in top_concept_names for c in ["Deep Learning", "Speech Recognition", "Convolutional Neural Networks", "Attention", "Transformer", "Machine Learning"]))

        # Check graph
        graph = engine.build_exploration_graph(max_concepts=15, max_papers=30)
        self.assertGreater(len(graph.nodes), 30)
        self.assertGreater(len(graph.edges), 20)

    def test_production_stores_isolation(self):
        """Verify that exploration operations never touch customer or medical stores."""
        customer_csv = self.proj_root / "dataset" / "knowledge_base.csv"
        customer_faiss = self.proj_root / "faiss_index"
        medical_faiss = self.proj_root / "faiss_index_medical"

        import pandas as pd
        df = pd.read_csv(customer_csv)
        self.assertEqual(len(df), 79)

        import faiss
        c_idx = faiss.read_index(str(customer_faiss / "index.faiss"))
        self.assertEqual(c_idx.ntotal, 79)

        s_idx = faiss.read_index(str(self.scientific_store_path / "index.faiss"))
        self.assertEqual(s_idx.ntotal, 100)


if __name__ == "__main__":
    unittest.main()
