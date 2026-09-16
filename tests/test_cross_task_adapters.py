"""Unit Tests for Cross-Task Response Adapters (Day 36).

Validates information preservation, confidence normalization, citation extraction,
and evidence mapping across all 5 specialist response formats.
"""

import unittest
import os
import sys
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src.chatbot_service import ChatbotResponse
from src.cross_task.adapters import (
    adapt_chatbot_response,
    adapt_medical_response,
    adapt_multilingual_response,
    adapt_multimodal_response,
    adapt_scientific_response,
)
from src.cross_task.models import ConfidenceTier, DomainType
from src.medical_qa_service import MedicalQAResponse
from src.multilingual.models import MultilingualResponse
from src.multimodal.models import (
    AmbiguityDetails,
    ModalityType,
    MultimodalResponse,
    VisualEvidenceItem,
)
from src.scientific_kb.grounding import Citation
from src.scientific_kb.retriever import ScientificRetrievalResult
from src.scientific_kb.service import ScientificExpertResponse


class TestCrossTaskAdapters(unittest.TestCase):
    """Test suite for adapter functions converting specialist responses to UnifiedResponse."""

    def test_adapt_chatbot_response(self):
        """Verify Task 1/2 ChatbotResponse adaptation."""
        mock_doc = MagicMock()
        mock_doc.page_content = "The data science course duration is 6 months."
        mock_doc.metadata = {"source": "faq_42"}

        cb_resp = ChatbotResponse(
            query="How long is the course?",
            final_answer="The data science course duration is 6 months.",
            raw_answer="The data science course duration is 6 months.",
            sentiment_label="POSITIVE",
            confidence_score=0.92,
            is_ood=False,
            source_documents=[mock_doc],
        )

        unified = adapt_chatbot_response(cb_resp, query="How long is the course?")
        self.assertEqual(unified.domain, DomainType.CUSTOMER_SUPPORT.value)
        self.assertEqual(unified.final_text_response, "The data science course duration is 6 months.")
        self.assertEqual(unified.sentiment, "POSITIVE")
        self.assertEqual(unified.confidence_score, 0.92)
        self.assertEqual(unified.confidence_tier, ConfidenceTier.HIGH.value)
        self.assertEqual(len(unified.evidence), 1)
        self.assertEqual(len(unified.citations), 1)
        self.assertEqual(unified.citations[0].source_id, "faq_42")

    def test_adapt_medical_response(self):
        """Verify Task 3 MedicalQAResponse adaptation."""
        mock_doc = MagicMock()
        mock_doc.page_content = "Mammography is the gold standard for screening."
        mock_doc.metadata = {"source_id": "MED_101"}

        med_resp = MedicalQAResponse(
            query="What is breast cancer screening?",
            final_answer="Mammography is commonly used for screening.",
            is_grounded=True,
            confidence_score=0.88,
            analysis=MagicMock(),
            status="GROUNDED",
            evidence_documents=[mock_doc],
            citations=[{"title": "NIH Clinical Guide", "source_id": "MED_101", "url": "https://nih.gov"}],
        )

        unified = adapt_medical_response(med_resp, query="What is breast cancer screening?")
        self.assertEqual(unified.domain, DomainType.MEDICAL.value)
        self.assertEqual(unified.confidence_score, 0.88)
        self.assertEqual(unified.confidence_tier, ConfidenceTier.HIGH.value)
        self.assertTrue(unified.is_grounded)
        self.assertEqual(len(unified.citations), 1)
        self.assertEqual(unified.citations[0].url, "https://nih.gov")

    def test_adapt_scientific_response(self):
        """Verify Task 4 ScientificExpertResponse adaptation."""
        mock_doc = MagicMock()
        mock_doc.page_content = "We propose the Transformer."

        src = ScientificRetrievalResult(
            document=mock_doc,
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            authors="Vaswani et al.",
            primary_category="cs.CL",
            categories=["cs.CL", "cs.AI"],
            published_date="2017-06-12",
            score=0.5,
        )
        cit = Citation(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            url="https://arxiv.org/abs/1706.03762",
            primary_category="cs.CL",
        )

        sci_resp = ScientificExpertResponse(
            query="Explain Transformer attention",
            condensed_query="Transformer attention",
            answer="Transformers use multi-head self-attention.",
            intent="concept_explanation",
            sources=[src],
            citations=[cit],
            grounded=True,
        )

        unified = adapt_scientific_response(sci_resp, query="Explain Transformer attention")
        self.assertEqual(unified.domain, DomainType.SCIENTIFIC.value)
        self.assertEqual(unified.confidence_score, 0.95)
        self.assertTrue(unified.is_grounded)
        self.assertEqual(len(unified.citations), 1)
        self.assertEqual(unified.citations[0].source_id, "1706.03762")

    def test_adapt_multimodal_response(self):
        """Verify Task 5 MultimodalResponse adaptation."""
        ev = VisualEvidenceItem(
            description="Detected boundary box for thoracic region",
            region_label="thoracic_region",
            confidence=0.96,
        )
        mm_resp = MultimodalResponse(
            query="Describe this image",
            answer="The image shows a thoracic visual rendering.",
            modality=ModalityType.TEXT_AND_IMAGE,
            ambiguity=AmbiguityDetails(is_ambiguous=False),
            visual_evidence=[ev],
            grounded=True,
        )

        unified = adapt_multimodal_response(mm_resp, query="Describe this image")
        self.assertEqual(unified.domain, DomainType.MULTIMODAL.value)
        self.assertEqual(len(unified.evidence), 1)
        self.assertEqual(unified.evidence[0].region_label, "thoracic_region")

    def test_adapt_multilingual_response(self):
        """Verify Task 6 MultilingualResponse adaptation."""
        ml_resp = MultilingualResponse(
            query="¿Cuál es el precio?",
            language="es",
            intent="fees",
            aligned_query="What is the price of the course?",
            final_answer="El precio del curso es $500.",
            raw_answer="The price is $500.",
            confidence_score=0.91,
            is_grounded=True,
            source_documents=[{"prompt": "What is the fee?", "response": "The fee is $500."}],
        )

        unified = adapt_multilingual_response(ml_resp, query="¿Cuál es el precio?")
        self.assertEqual(unified.detected_language, "es")
        self.assertEqual(unified.final_text_response, "El precio del curso es $500.")
        self.assertEqual(len(unified.evidence), 1)


if __name__ == "__main__":
    unittest.main()
