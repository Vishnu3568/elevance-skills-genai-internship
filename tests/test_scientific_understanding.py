"""Unit tests for the PaperUnderstandingService and structured paper analysis extraction layer.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb import (  # type: ignore
        PaperUnderstandingService,
        ScientificPaper,
        ScientificRetrievalResult,
        StructuredPaperAnalysis,
        STRUCTURED_EXTRACTION_PROMPT_TEMPLATE,
        build_paper_understanding_prompt,
        validate_structured_analysis,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb import (  # type: ignore
        PaperUnderstandingService,
        ScientificPaper,
        ScientificRetrievalResult,
        StructuredPaperAnalysis,
        STRUCTURED_EXTRACTION_PROMPT_TEMPLATE,
        build_paper_understanding_prompt,
        validate_structured_analysis,
    )

try:
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage
except ImportError:
    from langchain.docstore.document import Document
    from langchain.schema import AIMessage


class FakeExtractionLLM:
    """Deterministic fake LLM returning predefined JSON extraction responses."""

    def __init__(self, response_json: Any = None, raw_text: str = ""):
        if raw_text:
            self.response_text = raw_text
        elif response_json is not None:
            self.response_text = json.dumps(response_json) if not isinstance(response_json, str) else response_json
        else:
            self.response_text = json.dumps({
                "research_problem": "Factual inaccuracy in generative LLMs.",
                "motivation": "Parametric models hallucinate.",
                "methodology": "Neural retrieval augmentation.",
                "model_architecture": "DPR + Seq2Seq Generator.",
                "key_contributions": ["End-to-end differentiable RAG framework."],
                "datasets_benchmarks": ["Natural Questions", "TriviaQA"],
                "key_findings": ["RAG generates more factual text than standard BART."],
                "quantitative_results": ["44.5 EM on Natural Questions."],
                "limitations": ["Inference overhead of multiple retrieved passages."],
                "open_questions": ["Extension to multi-modal document retrieval."],
                "technical_concepts": ["RAG", "DPR", "Marginalization"],
            })
        self.last_prompt = ""

    def invoke(self, prompt: str):
        self.last_prompt = str(prompt)
        return AIMessage(content=self.response_text)


class PredictAdapterLLM:
    """Fake LLM supporting .predict() interface."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_prompt = ""

    def predict(self, prompt: str) -> str:
        self.last_prompt = str(prompt)
        return self.response_text


class TestScientificUnderstanding(unittest.TestCase):
    """Unit tests for PaperUnderstandingService and structured research understanding extraction."""

    def setUp(self):
        self.sample_paper = ScientificPaper(
            arxiv_id="2005.11401",
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            authors=["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus"],
            abstract="Large pre-trained language models store factual knowledge in their parameters. However, their ability to access and manipulate knowledge is limited...",
            categories=["cs.CL", "cs.AI"],
            primary_category="cs.CL",
            published_date="2020-05-22",
            doi="10.48550/arXiv.2005.11401",
            concepts=["RAG", "Dense Passage Retrieval"],
        )

        doc = Document(
            page_content="Title: Attention Is All You Need\nAbstract: The dominant sequence transduction models are based on complex recurrent networks...",
            metadata={"arxiv_id": "1706.03762", "title": "Attention Is All You Need"},
        )
        self.sample_retrieval_result = ScientificRetrievalResult(
            document=doc,
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            authors="Ashish Vaswani, Noam Shazeer",
            primary_category="cs.CL",
            categories=["cs.CL", "cs.LG"],
            published_date="2017-06-12",
            score=0.12,
            concepts=["Transformer", "Self-Attention"],
        )

    def test_successful_structured_extraction(self):
        """Test 1: Given a realistic paper and LLM JSON, returns valid StructuredPaperAnalysis."""
        expected_json = {
            "research_problem": "Pre-trained LMs lack verifiable factual grounding.",
            "motivation": "Parametric knowledge cannot be easily updated or audited.",
            "methodology": "Combine parametric seq2seq with non-parametric dense neural retriever.",
            "model_architecture": "DPR retriever encoder + BART sequence-to-sequence generator.",
            "key_contributions": [
                "Proposed RAG-Sequence and RAG-Token probabilistic formulations.",
                "Demonstrated state-of-the-art results across 5 open-domain benchmarks.",
            ],
            "datasets_benchmarks": ["Natural Questions", "TriviaQA", "CuratedTREC"],
            "key_findings": [
                "Non-parametric memory produces more specific and factual generations.",
            ],
            "quantitative_results": [
                "44.5 EM on Natural Questions benchmark.",
            ],
            "limitations": [
                "Marginalization over top-K documents increases latency.",
            ],
            "open_questions": [
                "Applying RAG to structured table and multi-modal knowledge bases.",
            ],
            "technical_concepts": [
                "Retrieval-Augmented Generation",
                "Dense Passage Retrieval",
                "Non-Parametric Memory",
            ],
        }
        fake_llm = FakeExtractionLLM(response_json=expected_json)
        service = PaperUnderstandingService(llm=fake_llm)

        analysis = service.analyze_paper_structure(self.sample_paper)

        self.assertIsInstance(analysis, StructuredPaperAnalysis)
        self.assertEqual(analysis.arxiv_id, "2005.11401")
        self.assertEqual(analysis.title, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks")
        self.assertEqual(analysis.research_problem, expected_json["research_problem"])
        self.assertEqual(analysis.model_architecture, expected_json["model_architecture"])
        self.assertEqual(len(analysis.key_contributions), 2)
        self.assertEqual(len(analysis.datasets_benchmarks), 3)
        self.assertEqual(len(analysis.quantitative_results), 1)
        self.assertEqual(len(analysis.limitations), 1)
        self.assertEqual(len(analysis.technical_concepts), 3)

    def test_missing_optional_information_in_abstract(self):
        """Test 2: Abstract lacking benchmarks or limitations preserves empty defaults without fabricating."""
        minimal_json = {
            "research_problem": "Language representation learning.",
            "motivation": "",
            "methodology": "Masked language modeling.",
            "model_architecture": "Bidirectional Transformer.",
            "key_contributions": ["Bidirectional pre-training."],
            "datasets_benchmarks": [],
            "key_findings": [],
            "quantitative_results": [],
            "limitations": [],
            "open_questions": [],
            "technical_concepts": ["BERT", "Transformer"],
        }
        service = PaperUnderstandingService(llm=FakeExtractionLLM(response_json=minimal_json))

        analysis = service.analyze_paper_structure(self.sample_paper)

        self.assertEqual(analysis.motivation, "")
        self.assertEqual(analysis.datasets_benchmarks, [])
        self.assertEqual(analysis.quantitative_results, [])
        self.assertEqual(analysis.limitations, [])
        self.assertEqual(analysis.open_questions, [])
        self.assertEqual(len(analysis.key_contributions), 1)

    def test_markdown_wrapped_json_parsing(self):
        """Test markdown code fences (```json ... ```) are safely stripped and parsed."""
        raw_markdown_json = (
            "```json\n"
            "{\n"
            '  "research_problem": "Problem X",\n'
            '  "motivation": "Motivation Y",\n'
            '  "methodology": "Method Z",\n'
            '  "model_architecture": "Arch W",\n'
            '  "key_contributions": ["Contribution 1"],\n'
            '  "datasets_benchmarks": [],\n'
            '  "key_findings": [],\n'
            '  "quantitative_results": [],\n'
            '  "limitations": [],\n'
            '  "open_questions": [],\n'
            '  "technical_concepts": ["Concept 1"]\n'
            "}\n"
            "```"
        )
        service = PaperUnderstandingService(llm=FakeExtractionLLM(raw_text=raw_markdown_json))
        analysis = service.analyze_paper_structure(self.sample_paper)

        self.assertEqual(analysis.research_problem, "Problem X")
        self.assertEqual(analysis.key_contributions, ["Contribution 1"])

    def test_malformed_json_raises_value_error(self):
        """Test 3: Malformed JSON output from LLM raises clear ValueError."""
        service = PaperUnderstandingService(llm=FakeExtractionLLM(raw_text="This is not JSON at all! {unclosed"))

        with self.assertRaises(ValueError) as ctx:
            service.analyze_paper_structure(self.sample_paper)
        self.assertIn("Failed to parse LLM extraction response as valid JSON", str(ctx.exception))

    def test_non_object_json_raises_value_error(self):
        """Test 4: Non-object JSON payload (e.g. list or string) raises ValueError."""
        service_list = PaperUnderstandingService(llm=FakeExtractionLLM(raw_text='["item1", "item2"]'))
        with self.assertRaises(ValueError) as ctx:
            service_list.analyze_paper_structure(self.sample_paper)
        self.assertIn("Expected JSON object (dict)", str(ctx.exception))

        service_str = PaperUnderstandingService(llm=FakeExtractionLLM(raw_text='"just a string"'))
        with self.assertRaises(ValueError):
            service_str.analyze_paper_structure(self.sample_paper)

    def test_invalid_structured_field_fails_validation(self):
        """Test 5: Invalid field types in JSON trigger validate_structured_analysis error."""
        invalid_json = {
            "research_problem": "Valid problem",
            "motivation": "Valid motivation",
            "methodology": "Valid method",
            "model_architecture": "Valid arch",
            "key_contributions": "Should have been a list, not string",
            "datasets_benchmarks": [],
            "key_findings": [],
            "quantitative_results": [],
            "limitations": [],
            "open_questions": [],
            "technical_concepts": [],
        }
        service = PaperUnderstandingService(llm=FakeExtractionLLM(response_json=invalid_json))

        with self.assertRaises(TypeError) as ctx:
            service.analyze_paper_structure(self.sample_paper)
        self.assertIn("key_contributions must be a list of strings", str(ctx.exception))

    def test_empty_abstract_raises_error_without_calling_llm(self):
        """Test 6: Empty abstract raises ValueError and does not call LLM."""
        mock_llm = MagicMock()
        service = PaperUnderstandingService(llm=mock_llm)

        with self.assertRaises(ValueError) as ctx:
            service.analyze_paper_structure(
                paper={"arxiv_id": "2005.11401", "title": "Some Title", "abstract": ""}
            )
        self.assertIn("Abstract cannot be empty or whitespace-only", str(ctx.exception))
        mock_llm.invoke.assert_not_called()

    def test_whitespace_only_abstract_raises_error(self):
        """Test 7: Whitespace-only abstract raises ValueError and does not call LLM."""
        mock_llm = MagicMock()
        service = PaperUnderstandingService(llm=mock_llm)

        with self.assertRaises(ValueError) as ctx:
            service.analyze_paper_structure(
                paper={"arxiv_id": "2005.11401", "title": "Some Title", "abstract": "   \n\t  "}
            )
        self.assertIn("Abstract cannot be empty or whitespace-only", str(ctx.exception))
        mock_llm.invoke.assert_not_called()

    def test_source_identity_preservation_over_llm_hallucinations(self):
        """Test 8: Authoritative source arxiv_id and title strictly override LLM hallucinations."""
        hallucinated_json = {
            "arxiv_id": "9999.99999",
            "title": "Completely Hallucinated Title",
            "research_problem": "Real problem",
            "motivation": "Real motivation",
            "methodology": "Real method",
            "model_architecture": "Real arch",
            "key_contributions": ["Real contribution"],
            "datasets_benchmarks": [],
            "key_findings": [],
            "quantitative_results": [],
            "limitations": [],
            "open_questions": [],
            "technical_concepts": [],
        }
        service = PaperUnderstandingService(llm=FakeExtractionLLM(response_json=hallucinated_json))

        analysis = service.analyze_paper_structure(self.sample_paper)

        self.assertEqual(analysis.arxiv_id, "2005.11401")
        self.assertEqual(analysis.title, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks")

    def test_llm_invocation_compatibility(self):
        """Test 9: Supports .invoke(), callable, and .predict() LLM adapters."""
        resp_json = {
            "research_problem": "P",
            "motivation": "M",
            "methodology": "Meth",
            "model_architecture": "Arch",
            "key_contributions": ["C1"],
            "datasets_benchmarks": [],
            "key_findings": [],
            "quantitative_results": [],
            "limitations": [],
            "open_questions": [],
            "technical_concepts": ["T1"],
        }
        resp_text = json.dumps(resp_json)

        # 1. Callable LLM
        def callable_llm(prompt: str) -> str:
            return resp_text

        service_callable = PaperUnderstandingService(llm=callable_llm)
        res1 = service_callable.analyze_paper_structure(self.sample_paper)
        self.assertEqual(res1.research_problem, "P")

        # 2. Predict LLM
        service_predict = PaperUnderstandingService(llm=PredictAdapterLLM(resp_text))
        res2 = service_predict.analyze_paper_structure(self.sample_paper)
        self.assertEqual(res2.model_architecture, "Arch")

        # 3. None LLM raises RuntimeError
        service_none = PaperUnderstandingService(llm=None)
        with self.assertRaises(RuntimeError):
            service_none.analyze_paper_structure(self.sample_paper)

    def test_prompt_grounding_and_instructions(self):
        """Test 10: Verify extraction prompt contains metadata, abstract, schema, and grounding rules."""
        prompt = build_paper_understanding_prompt(self.sample_paper)

        self.assertIn("Title: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", prompt)
        self.assertIn("arXiv ID: 2005.11401", prompt)
        self.assertIn("Patrick Lewis", prompt)
        self.assertIn("cs.CL", prompt)
        self.assertIn("PAPER ABSTRACT / SCIENTIFIC EVIDENCE:", prompt)
        self.assertIn("Large pre-trained language models store factual knowledge", prompt)
        self.assertIn("Do NOT fabricate or invent datasets", prompt)
        self.assertIn('"research_problem":', prompt)
        self.assertIn('"model_architecture":', prompt)
        self.assertIn('"key_contributions":', prompt)
        self.assertIn('"datasets_benchmarks":', prompt)
        self.assertIn('"quantitative_results":', prompt)
        self.assertIn("Return ONLY a valid, parseable JSON object", prompt)

    def test_retrieval_result_input_compatibility(self):
        """Test compatibility when passing a ScientificRetrievalResult object."""
        service = PaperUnderstandingService(llm=FakeExtractionLLM())
        analysis = service.analyze_paper_structure(self.sample_retrieval_result)

        self.assertEqual(analysis.arxiv_id, "1706.03762")
        self.assertEqual(analysis.title, "Attention Is All You Need")
        self.assertIsInstance(analysis, StructuredPaperAnalysis)

    def test_understanding_isolation_from_production_stores(self):
        """Test 11: Understanding operations never modify production customer or medical stores."""
        proj_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        customer_faiss = proj_root / "faiss_index"
        medical_faiss = proj_root / "faiss_index_medical"

        cust_mtime = customer_faiss.stat().st_mtime if customer_faiss.exists() else None
        med_mtime = medical_faiss.stat().st_mtime if medical_faiss.exists() else None

        service = PaperUnderstandingService(llm=FakeExtractionLLM())
        service.analyze_paper_structure(self.sample_paper)

        if customer_faiss.exists():
            self.assertEqual(customer_faiss.stat().st_mtime, cust_mtime)
        if medical_faiss.exists():
            self.assertEqual(medical_faiss.stat().st_mtime, med_mtime)


if __name__ == "__main__":
    unittest.main()
