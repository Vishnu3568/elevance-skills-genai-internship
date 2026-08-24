import unittest
import sys
import os
from pathlib import Path
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb import (  # type: ignore
        Citation,
        GroundingValidationResult,
        ScientificAnswer,
        ScientificRetrievalResult,
        extract_cited_arxiv_ids,
        format_grounded_answer,
        validate_answer_grounding,
        INSUFFICIENT_EVIDENCE_PHRASE,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb import (  # type: ignore
        Citation,
        GroundingValidationResult,
        ScientificAnswer,
        ScientificRetrievalResult,
        extract_cited_arxiv_ids,
        format_grounded_answer,
        validate_answer_grounding,
        INSUFFICIENT_EVIDENCE_PHRASE,
    )

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document


class TestScientificGrounding(unittest.TestCase):
    """Unit tests for the evidence grounding and citation validation layer."""

    def setUp(self):
        doc1 = Document(
            page_content="Title: Attention Is All You Need\nAbstract: Transformer architecture...",
            metadata={"arxiv_id": "1706.03762", "title": "Attention Is All You Need"},
        )
        self.source1 = ScientificRetrievalResult(
            document=doc1,
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            authors="Ashish Vaswani, Noam Shazeer",
            primary_category="cs.CL",
            categories=["cs.CL", "cs.LG"],
            published_date="2017-06-12",
            score=0.12,
            concepts=["Transformer", "Self-Attention"],
            doi="10.48550/arXiv.1706.03762",
            url="https://arxiv.org/abs/1706.03762",
        )

        doc2 = Document(
            page_content="Title: LoRA: Low-Rank Adaptation\nAbstract: Fine-tuning large models...",
            metadata={"arxiv_id": "2106.09685", "title": "LoRA: Low-Rank Adaptation of Large Language Models"},
        )
        self.source2 = ScientificRetrievalResult(
            document=doc2,
            arxiv_id="2106.09685",
            title="LoRA: Low-Rank Adaptation of Large Language Models",
            authors="Edward J. Hu, Yelong Shen",
            primary_category="cs.LG",
            categories=["cs.LG", "cs.AI"],
            published_date="2021-06-17",
            score=0.20,
            concepts=["LoRA", "PEFT"],
            doi="10.48550/arXiv.2106.09685",
            url="https://arxiv.org/abs/2106.09685",
        )

    def test_standard_arxiv_id_extraction(self):
        text = "As shown in 1706.03762 and discussed by Vaswani et al."
        ids = extract_cited_arxiv_ids(text)
        self.assertEqual(ids, ["1706.03762"])

    def test_versioned_arxiv_id_extraction(self):
        text = "Refer to the paper arXiv:2005.11401v2 for details on RAG architecture."
        ids = extract_cited_arxiv_ids(text)
        self.assertEqual(ids, ["2005.11401v2"])

    def test_multiple_citations_and_duplicate_removal(self):
        text = (
            "We build upon 1706.03762 (Transformers) and LoRA (2106.09685). "
            "Re-visiting 1706.03762 confirms the self-attention mechanism."
        )
        ids = extract_cited_arxiv_ids(text)
        self.assertEqual(ids, ["1706.03762", "2106.09685"])

    def test_valid_retrieved_citations(self):
        answer = "The Transformer architecture was introduced in 1706.03762."
        val = validate_answer_grounding(answer, [self.source1, self.source2])

        self.assertTrue(val.is_grounded)
        self.assertEqual(len(val.valid_citations), 1)
        self.assertEqual(val.valid_citations[0].arxiv_id, "1706.03762")
        self.assertEqual(val.valid_citations[0].title, "Attention Is All You Need")
        self.assertEqual(val.unsupported_citations, [])
        self.assertIsNone(val.warning_message)

    def test_unsupported_hallucinated_citations(self):
        answer = "This finding was originally published in 9901.12345 and later verified in 1810.04805."
        val = validate_answer_grounding(answer, [self.source1, self.source2])

        self.assertFalse(val.is_grounded)
        self.assertEqual(val.valid_citations, [])
        self.assertEqual(val.unsupported_citations, ["9901.12345", "1810.04805"])
        self.assertIn("unsupported citation", str(val.warning_message))

    def test_mixed_valid_and_unsupported_citations(self):
        answer = "While 1706.03762 introduced self-attention, paper 9901.12345 claims otherwise."
        val = validate_answer_grounding(answer, [self.source1])

        self.assertFalse(val.is_grounded)
        self.assertEqual(len(val.valid_citations), 1)
        self.assertEqual(val.valid_citations[0].arxiv_id, "1706.03762")
        self.assertEqual(val.unsupported_citations, ["9901.12345"])

    def test_answers_containing_no_citations(self):
        answer = "Transformers rely entirely on self-attention mechanisms to compute representations."
        val = validate_answer_grounding(answer, [self.source1])

        # No hallucinated citations -> grounded
        self.assertTrue(val.is_grounded)
        self.assertEqual(val.unsupported_citations, [])

    def test_insufficient_evidence_responses(self):
        val = validate_answer_grounding(INSUFFICIENT_EVIDENCE_PHRASE, [self.source1])
        self.assertFalse(val.is_grounded)
        self.assertEqual(val.valid_citations, [])
        self.assertEqual(val.unsupported_citations, [])
        self.assertEqual(val.warning_message, "Insufficient evidence to answer the question.")

    def test_markdown_formatting_and_provenance_links(self):
        sci_ans = ScientificAnswer(
            query="What is LoRA?",
            answer="LoRA (2106.09685) freezes the pre-trained model weights and injects trainable rank decomposition matrices.",
            sources=[self.source2],
            grounded=True,
        )
        formatted = format_grounded_answer(sci_ans)

        self.assertIn("### 🧠 Grounded Scientific Explanation", formatted)
        self.assertIn("LoRA (2106.09685)", formatted)
        self.assertIn("### 📚 Verified Sources & References", formatted)
        self.assertIn("**LoRA: Low-Rank Adaptation of Large Language Models**", formatted)
        self.assertIn("[`arXiv:2106.09685`](https://arxiv.org/abs/2106.09685)", formatted)
        self.assertIn("[DOI: 10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)", formatted)

    def test_markdown_formatting_with_unsupported_citations_warning(self):
        answer_text = "Refer to 9901.00001 for non-existent evidence."
        val = validate_answer_grounding(answer_text, [self.source1])
        formatted = format_grounded_answer(answer_text, val)

        self.assertIn("⚠️ **Citation Warning**", formatted)
        self.assertIn("9901.00001", formatted)

    def test_input_validation(self):
        with self.assertRaises(TypeError):
            validate_answer_grounding(12345, [self.source1])  # type: ignore

        with self.assertRaises(TypeError):
            validate_answer_grounding("Answer", "not a list")  # type: ignore

    def test_grounding_isolation_from_production_stores(self):
        """Verify grounding operations never modify customer or medical vector stores."""
        proj_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        customer_faiss = proj_root / "faiss_index"
        medical_faiss = proj_root / "faiss_index_medical"

        cust_mtime = customer_faiss.stat().st_mtime if customer_faiss.exists() else None
        med_mtime = medical_faiss.stat().st_mtime if medical_faiss.exists() else None

        validate_answer_grounding("Test 1706.03762", [self.source1])
        format_grounded_answer("Test answer", fallback_sources=[self.source1])

        if customer_faiss.exists():
            self.assertEqual(customer_faiss.stat().st_mtime, cust_mtime)
        if medical_faiss.exists():
            self.assertEqual(medical_faiss.stat().st_mtime, med_mtime)


if __name__ == "__main__":
    unittest.main()
