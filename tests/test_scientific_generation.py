import unittest
import sys
import os
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb import (  # type: ignore
        ScientificAnswer,
        ScientificGenerator,
        ScientificRetrievalResult,
        build_scientific_prompt,
        INSUFFICIENT_EVIDENCE_PHRASE,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb import (  # type: ignore
        ScientificAnswer,
        ScientificGenerator,
        ScientificRetrievalResult,
        build_scientific_prompt,
        INSUFFICIENT_EVIDENCE_PHRASE,
    )

try:
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage
except ImportError:
    from langchain.docstore.document import Document
    from langchain.schema import AIMessage


class FakeScientificLLM:
    """Deterministic fake LLM for unit testing without downloading models or making network calls."""

    def __init__(self, response_text: str = "This is a grounded scientific explanation."):
        self.response_text = response_text
        self.last_prompt = ""

    def invoke(self, prompt: str):
        self.last_prompt = str(prompt)
        return AIMessage(content=self.response_text)


class TestScientificGeneration(unittest.TestCase):
    """Unit tests for the scientific LLM generation and answer synthesis layer."""

    def setUp(self):
        doc1 = Document(
            page_content="Title: Attention Is All You Need\nAbstract: Transformer architecture...",
            metadata={"arxiv_id": "1706.03762", "title": "Attention Is All You Need"},
        )
        self.sample_result1 = ScientificRetrievalResult(
            document=doc1,
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            authors="Ashish Vaswani, Noam Shazeer",
            primary_category="cs.CL",
            categories=["cs.CL", "cs.LG"],
            published_date="2017-06-12",
            score=0.15,
            concepts=["Transformer", "Self-Attention"],
            url="https://arxiv.org/abs/1706.03762",
        )

        doc2 = Document(
            page_content="Title: Retrieval-Augmented Generation\nAbstract: RAG models combine parametric memory...",
            metadata={"arxiv_id": "2005.11401", "title": "Retrieval-Augmented Generation"},
        )
        self.sample_result2 = ScientificRetrievalResult(
            document=doc2,
            arxiv_id="2005.11401",
            title="Retrieval-Augmented Generation",
            authors="Patrick Lewis, Ethan Perez",
            primary_category="cs.CL",
            categories=["cs.CL", "cs.AI"],
            published_date="2020-05-22",
            score=0.25,
            concepts=["RAG", "Dense Retrieval"],
            url="https://arxiv.org/abs/2005.11401",
        )

    def test_prompt_construction_with_evidence(self):
        query = "How does the Transformer model work?"
        context = "[Paper 1]\nTitle: Attention Is All You Need\narXiv ID: 1706.03762"

        prompt = build_scientific_prompt(query, context)

        self.assertIn("You are an expert scientific AI research assistant", prompt)
        self.assertIn("SCIENTIFIC EVIDENCE:", prompt)
        self.assertIn("Title: Attention Is All You Need", prompt)
        self.assertIn("QUESTION:", prompt)
        self.assertIn("How does the Transformer model work?", prompt)
        self.assertIn("GROUNDED SCIENTIFIC EXPLANATION:", prompt)

    def test_successful_grounded_answer_generation(self):
        llm_text = (
            "According to 'Attention Is All You Need' (arXiv: 1706.03762), the Transformer "
            "replaces recurrence with multi-head self-attention mechanisms."
        )
        fake_llm = FakeScientificLLM(response_text=llm_text)
        generator = ScientificGenerator(llm=fake_llm)

        res = generator.generate_answer(
            query="Explain self-attention in Transformers.",
            retrieval_results=[self.sample_result1],
        )

        self.assertIsInstance(res, ScientificAnswer)
        self.assertEqual(res.query, "Explain self-attention in Transformers.")
        self.assertEqual(res.answer, llm_text)
        self.assertEqual(len(res.sources), 1)
        self.assertEqual(res.sources[0].arxiv_id, "1706.03762")
        self.assertTrue(res.grounded)
        self.assertIn("Attention Is All You Need", fake_llm.last_prompt)

    def test_source_provenance_preservation(self):
        generator = ScientificGenerator(llm=FakeScientificLLM())
        sources = [self.sample_result1, self.sample_result2]

        res = generator.generate_answer(
            query="Compare Transformer and RAG architectures.",
            retrieval_results=sources,
        )

        self.assertEqual(len(res.sources), 2)
        self.assertEqual(res.sources[0].arxiv_id, "1706.03762")
        self.assertEqual(res.sources[1].arxiv_id, "2005.11401")

    def test_empty_retrieval_results_fallback(self):
        mock_llm = MagicMock()
        generator = ScientificGenerator(llm=mock_llm)

        res = generator.generate_answer(
            query="Tell me about dark matter particles.",
            retrieval_results=[],
        )

        # Should NOT call LLM when evidence is empty
        mock_llm.invoke.assert_not_called()
        self.assertIsInstance(res, ScientificAnswer)
        self.assertEqual(res.answer, INSUFFICIENT_EVIDENCE_PHRASE)
        self.assertEqual(res.sources, [])
        self.assertFalse(res.grounded)

    def test_ungrounded_detection_when_llm_states_insufficient_evidence(self):
        fake_llm = FakeScientificLLM(response_text=INSUFFICIENT_EVIDENCE_PHRASE)
        generator = ScientificGenerator(llm=fake_llm)

        res = generator.generate_answer(
            query="What is quantum gravity?",
            retrieval_results=[self.sample_result1],
        )

        self.assertEqual(res.answer, INSUFFICIENT_EVIDENCE_PHRASE)
        self.assertFalse(res.grounded)

    def test_query_input_validation(self):
        generator = ScientificGenerator(llm=FakeScientificLLM())

        with self.assertRaises(TypeError):
            generator.generate_answer(12345, [self.sample_result1])  # type: ignore

        with self.assertRaises(ValueError):
            generator.generate_answer("   ", [self.sample_result1])

    def test_no_llm_configured_raises_runtime_error(self):
        generator = ScientificGenerator(llm=None)

        with self.assertRaises(RuntimeError):
            generator.generate_answer(
                query="What is self-attention?",
                retrieval_results=[self.sample_result1],
            )

    def test_callable_llm_adapter(self):
        def plain_function_llm(prompt: str) -> str:
            return f"Answer generated from prompt of length {len(prompt)}"

        generator = ScientificGenerator(llm=plain_function_llm)
        res = generator.generate_answer(
            query="Test query",
            retrieval_results=[self.sample_result1],
        )

        self.assertIn("Answer generated from prompt", res.answer)
        self.assertTrue(res.grounded)

    def test_generation_isolation_from_production_stores(self):
        """Verify generation operations never modify production customer or medical stores."""
        proj_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        customer_faiss = proj_root / "faiss_index"
        medical_faiss = proj_root / "faiss_index_medical"

        cust_mtime = customer_faiss.stat().st_mtime if customer_faiss.exists() else None
        med_mtime = medical_faiss.stat().st_mtime if medical_faiss.exists() else None

        generator = ScientificGenerator(llm=FakeScientificLLM())
        generator.generate_answer("Test query", [self.sample_result1])
        generator.generate_answer("Empty query test", [])

        if customer_faiss.exists():
            self.assertEqual(customer_faiss.stat().st_mtime, cust_mtime)
        if medical_faiss.exists():
            self.assertEqual(medical_faiss.stat().st_mtime, med_mtime)


if __name__ == "__main__":
    unittest.main()
