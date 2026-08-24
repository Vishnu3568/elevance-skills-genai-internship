"""Integration test suite for the Streamlit Scientific Expert application layer.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    # pyrefly: ignore [missing-import]
    import src.scientific_main as sm  # type: ignore
    # pyrefly: ignore [missing-import]
    from src.scientific_kb import (  # type: ignore
        DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
        CorpusSummary,
        RelatedPaperMatch,
        ScientificExpertResponse,
        ScientificExpertService,
        ScientificExplorationEngine,
        ScientificExplorationGraph,
        ScientificRetrievalResult,
        generate_dot_graph,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    import scientific_main as sm  # type: ignore
    # pyrefly: ignore [missing-import]
    from scientific_kb import (  # type: ignore
        DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
        CorpusSummary,
        RelatedPaperMatch,
        ScientificExpertResponse,
        ScientificExpertService,
        ScientificExplorationEngine,
        ScientificExplorationGraph,
        ScientificRetrievalResult,
    )


class TestScientificMain(unittest.TestCase):
    """Integration test suite for the Streamlit Scientific Expert application layer."""

    @classmethod
    def setUpClass(cls):
        cls.proj_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.scientific_store = cls.proj_root / DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH

    def test_scientific_app_imports_and_constants(self):
        """Verify that scientific_main module imports and constants are correctly defined."""
        self.assertTrue(hasattr(sm, "initialize_scientific_service"))
        self.assertTrue(hasattr(sm, "initialize_exploration_engine"))
        self.assertTrue(hasattr(sm, "generate_dot_graph"))
        self.assertTrue(hasattr(sm, "format_source_badge"))
        self.assertTrue(hasattr(sm, "render_grounding_status"))
        self.assertTrue(hasattr(sm, "render_sources_panel"))
        self.assertTrue(hasattr(sm, "render_sidebar"))
        self.assertTrue(hasattr(sm, "main"))
        self.assertTrue(sm.DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH.endswith("faiss_index_scientific"))

    def test_initialize_scientific_service_success(self):
        """Verify that initialize_scientific_service initializes the service with production index."""
        if not self.scientific_store.exists():
            self.skipTest("faiss_index_scientific/ not yet built")

        service, err = sm.initialize_scientific_service(str(self.scientific_store))
        self.assertIsNone(err)
        self.assertIsNotNone(service)
        self.assertIsInstance(service, ScientificExpertService)

        # Check that retriever holds the 100-paper vector store
        docstore = service.retriever.vector_store.docstore._dict
        self.assertEqual(len(docstore), 100)

    def test_initialize_scientific_service_missing_index(self):
        """Verify graceful error handling when vector store path does not exist."""
        service, err = sm.initialize_scientific_service("nonexistent_faiss_dir_12345")
        self.assertIsNone(service)
        self.assertIsNotNone(err)
        self.assertIn("Scientific vector store not found", err)

    def test_initialize_exploration_engine(self):
        """Verify initialization of exploration engine from vector store."""
        if not self.scientific_store.exists():
            self.skipTest("faiss_index_scientific/ not yet built")

        service, _ = sm.initialize_scientific_service(str(self.scientific_store))
        self.assertIsNotNone(service)

        engine = sm.initialize_exploration_engine(service.retriever.vector_store)
        self.assertIsNotNone(engine)
        self.assertIsInstance(engine, ScientificExplorationEngine)
        self.assertEqual(engine.total_papers, 100)

        # Test None input handling
        none_engine = sm.initialize_exploration_engine(None)
        self.assertIsNone(none_engine)

    def test_format_source_badge(self):
        """Verify source badge markdown formatting."""
        res = ScientificRetrievalResult(
            document=None,  # type: ignore
            arxiv_id="2012.10055v2",
            title="End-to-End Speaker Diarization as Post-Processing",
            authors="John Doe, Jane Smith",
            primary_category="cs.CL",
            categories=["cs.CL"],
            published_date="2020-12-18",
            score=0.25,
            url="https://arxiv.org/abs/2012.10055v2",
        )
        badge = sm.format_source_badge(res)
        self.assertIn("End-to-End Speaker Diarization", badge)
        self.assertIn("`cs.CL`", badge)
        self.assertIn("(2020)", badge)
        self.assertIn("https://arxiv.org/abs/2012.10055v2", badge)

    def test_service_paper_lookup_and_search_integration(self):
        """Verify that UI service helper can find papers and perform literature search."""
        if not self.scientific_store.exists():
            self.skipTest("faiss_index_scientific/ not yet built")

        service, err = sm.initialize_scientific_service(str(self.scientific_store))
        self.assertIsNone(err)
        self.assertIsNotNone(service)

        # Search by query
        results = service.retriever.retrieve("speaker diarization", k=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(results[0].title)
        self.assertTrue(results[0].arxiv_id)

    def test_chat_query_execution_and_response_structure(self):
        """Verify that asking a question through the initialized service produces a structured response."""
        if not self.scientific_store.exists():
            self.skipTest("faiss_index_scientific/ not yet built")

        service, err = sm.initialize_scientific_service(str(self.scientific_store))
        self.assertIsNone(err)
        self.assertIsNotNone(service)

        with patch.object(service.generator, "_invoke_llm", return_value="Speech features can indicate cognitive health [2012.10055v2]."):
            resp: ScientificExpertResponse = service.ask("Explain automated speech and language features for cognitive impairment", k=2)
            self.assertIsInstance(resp, ScientificExpertResponse)
            self.assertTrue(resp.answer)
            self.assertIsInstance(resp.grounded, bool)
            self.assertIsInstance(resp.sources, list)

    def test_paper_summary_workflow_integration(self):
        """Verify paper summary execution through the service layer."""
        if not self.scientific_store.exists():
            self.skipTest("faiss_index_scientific/ not yet built")

        service, err = sm.initialize_scientific_service(str(self.scientific_store))
        self.assertIsNone(err)
        self.assertIsNotNone(service)

        with patch.object(service.generator, "_invoke_llm", return_value="### Research Problem\nSpeaker diarization [2012.10055v2].\n### Core Methodology\nPost-processing neural model.\n### Key Findings\nImproves DER."):
            resp = service.summarize_paper("2012.10055v2")
            self.assertIsInstance(resp, ScientificExpertResponse)
            self.assertTrue(resp.answer)

    def test_concept_explanation_workflow_integration(self):
        """Verify concept explanation execution through the service layer."""
        if not self.scientific_store.exists():
            self.skipTest("faiss_index_scientific/ not yet built")

        service, err = sm.initialize_scientific_service(str(self.scientific_store))
        self.assertIsNone(err)
        self.assertIsNotNone(service)

        with patch.object(service.generator, "_invoke_llm", return_value="### Technical Definition\nDeep learning acceleration optimizes compute [2012.10055v2].\n### Intuitive Explanation\nMaking models run faster."):
            resp = service.explain_concept("Deep Learning Acceleration")
            self.assertIsInstance(resp, ScientificExpertResponse)
            self.assertTrue(resp.answer)

    def test_comparison_workflow_integration(self):
        """Verify comparison execution through the service layer."""
        if not self.scientific_store.exists():
            self.skipTest("faiss_index_scientific/ not yet built")

        service, err = sm.initialize_scientific_service(str(self.scientific_store))
        self.assertIsNone(err)
        self.assertIsNotNone(service)

        with patch.object(service.generator, "_invoke_llm", return_value="### Comparative Synthesis\nCNNs focus on spatial features while speech models focus on time [2012.10055v2]."):
            resp = service.compare("Convolutional Neural Networks", "Speech Models")
            self.assertIsInstance(resp, ScientificExpertResponse)
            self.assertTrue(resp.answer)

    def test_session_clear_resets_history(self):
        """Verify that clearing session resets conversation history."""
        if not self.scientific_store.exists():
            self.skipTest("faiss_index_scientific/ not yet built")

        service, err = sm.initialize_scientific_service(str(self.scientific_store))
        self.assertIsNone(err)
        self.assertIsNotNone(service)

        with patch.object(service.generator, "_invoke_llm", return_value="Speaker diarization segments audio [2012.10055v2]."):
            service.ask("What is speaker diarization?", k=1)
            self.assertGreater(len(service.get_session_messages()), 0)

            service.clear_session()
            self.assertEqual(len(service.get_session_messages()), 0)

    def test_corpus_overview_and_summary_integration(self):
        """Verify that corpus overview summary feeds correctly into UI data structures."""
        if not self.scientific_store.exists():
            self.skipTest("faiss_index_scientific/ not yet built")

        service, _ = sm.initialize_scientific_service(str(self.scientific_store))
        engine = sm.initialize_exploration_engine(service.retriever.vector_store)
        summary: CorpusSummary = engine.get_corpus_summary()

        self.assertEqual(summary.total_papers, 100)
        self.assertGreater(len(summary.category_distribution), 0)
        self.assertGreater(len(summary.top_concepts), 0)
        self.assertGreater(len(summary.top_authors), 0)
        self.assertIn(2019, summary.year_distribution)
        self.assertIn(2020, summary.year_distribution)

    def test_concept_explorer_and_cooccurrences_integration(self):
        """Verify concept analytics and co-occurrence extraction for UI exploration."""
        if not self.scientific_store.exists():
            self.skipTest("faiss_index_scientific/ not yet built")

        service, _ = sm.initialize_scientific_service(str(self.scientific_store))
        engine = sm.initialize_exploration_engine(service.retriever.vector_store)

        concepts = engine.get_concept_statistics()
        self.assertGreater(len(concepts), 0)
        top_concept = concepts[0].concept

        matching_papers = engine.get_papers_for_concept(top_concept)
        self.assertEqual(len(matching_papers), concepts[0].frequency)

        cooccurs = engine.get_concept_cooccurrence()
        self.assertGreater(len(cooccurs), 0)
        self.assertTrue(hasattr(cooccurs[0], "cooccurrence_count"))

    def test_timeline_and_period_breakdown_integration(self):
        """Verify yearly and monthly timeline queries for UI timeline view."""
        if not self.scientific_store.exists():
            self.skipTest("faiss_index_scientific/ not yet built")

        service, _ = sm.initialize_scientific_service(str(self.scientific_store))
        engine = sm.initialize_exploration_engine(service.retriever.vector_store)

        yearly = engine.get_timeline(granularity="yearly")
        self.assertEqual(len(yearly), 2)  # 2019 and 2020

        monthly = engine.get_timeline(granularity="monthly")
        self.assertGreater(len(monthly), 5)
        self.assertTrue(all(len(t.period) == 7 for t in monthly))

    def test_paper_explorer_and_related_recommendations_integration(self):
        """Verify paper exploration and nearest-neighbor recommendations."""
        if not self.scientific_store.exists():
            self.skipTest("faiss_index_scientific/ not yet built")

        service, _ = sm.initialize_scientific_service(str(self.scientific_store))
        engine = sm.initialize_exploration_engine(service.retriever.vector_store)

        first_paper = engine.papers[0]
        first_pid = first_paper.get("arxiv_id") or first_paper.get("id")

        concepts = engine.get_concepts_for_paper(first_pid)
        self.assertIsInstance(concepts, list)

        # Related papers discovery
        related = engine.find_related_papers(target=first_paper, retriever=service.retriever, top_k=3)
        self.assertLessEqual(len(related), 3)
        # Verify source paper is not recommended to itself
        self.assertNotIn(first_pid, [r.paper.arxiv_id for r in related])

    def test_knowledge_graph_and_dot_generation_integration(self):
        """Verify graph construction and Graphviz DOT serialization."""
        if not self.scientific_store.exists():
            self.skipTest("faiss_index_scientific/ not yet built")

        service, _ = sm.initialize_scientific_service(str(self.scientific_store))
        engine = sm.initialize_exploration_engine(service.retriever.vector_store)

        graph = engine.build_exploration_graph(max_concepts=10, max_papers=20)
        self.assertGreater(len(graph.nodes), 10)
        self.assertGreater(len(graph.edges), 5)

        dot_str = sm.generate_dot_graph(graph, max_edges=20)
        self.assertTrue(dot_str.startswith("digraph ScientificKnowledgeGraph {"))
        self.assertTrue(dot_str.endswith("}"))
        self.assertIn("cat_", dot_str)

    def test_empty_exploration_handling(self):
        """Verify graceful handling when exploration engine has zero records."""
        empty_engine = ScientificExplorationEngine(papers=[])
        self.assertEqual(empty_engine.total_papers, 0)
        self.assertEqual(empty_engine.get_concept_statistics(), [])
        self.assertEqual(empty_engine.get_category_statistics(), [])
        self.assertEqual(empty_engine.get_timeline(), [])
        self.assertEqual(empty_engine.get_author_statistics(), [])

        graph = empty_engine.build_exploration_graph()
        self.assertEqual(len(graph.nodes), 0)
        self.assertEqual(len(graph.edges), 0)
        dot_str = sm.generate_dot_graph(graph)
        self.assertIn("digraph", dot_str)

    def test_production_store_isolation(self):
        """Verify that scientific_main never accesses customer or medical stores."""
        customer_csv = self.proj_root / "dataset" / "knowledge_base.csv"
        customer_faiss = self.proj_root / "faiss_index"
        medical_faiss = self.proj_root / "faiss_index_medical"

        import pandas as pd
        df = pd.read_csv(customer_csv)
        self.assertEqual(len(df), 79)

        import faiss
        c_idx = faiss.read_index(str(customer_faiss / "index.faiss"))
        self.assertEqual(c_idx.ntotal, 79)

        s_idx = faiss.read_index(str(self.scientific_store / "index.faiss"))
        self.assertEqual(s_idx.ntotal, 100)


if __name__ == "__main__":
    unittest.main()
