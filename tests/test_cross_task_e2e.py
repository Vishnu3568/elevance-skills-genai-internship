"""End-to-End Integration Tests for Cross-Task Service (Day 36).

Validates end-to-end processing across Customer Support, Medical, Scientific,
Multimodal, and Multilingual requests using CrossTaskService.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.cross_task.models import DomainType, UnifiedRequest
from src.cross_task.service import CrossTaskService


class TestCrossTaskE2E(unittest.TestCase):
    """End-to-end integration test suite for CrossTaskService."""

    def setUp(self):
        self.service = CrossTaskService()

    def test_customer_support_e2e(self):
        """Verify Customer Support query execution through CrossTaskService."""
        with patch.object(self.service.orchestrator, "get_chatbot_service") as mock_get_cb:
            mock_cb = MagicMock()
            mock_get_cb.return_value = mock_cb
            mock_doc = MagicMock()
            mock_doc.page_content = "Course fee is $500."
            mock_doc.metadata = {"source": "faq_1"}

            from src.chatbot_service import ChatbotResponse
            mock_cb.process_query.return_value = ChatbotResponse(
                query="What is the course fee?",
                final_answer="The course fee is $500.",
                raw_answer="The course fee is $500.",
                sentiment_label="NEUTRAL",
                confidence_score=0.95,
                is_ood=False,
                source_documents=[mock_doc],
            )

            resp = self.service.process(
                query="What is the course fee?",
                session_id="e2e_session_1",
            )
            self.assertEqual(resp.domain, DomainType.CUSTOMER_SUPPORT.value)
            self.assertEqual(resp.final_text_response, "The course fee is $500.")
            self.assertEqual(resp.session_id, "e2e_session_1")
            self.assertTrue(resp.is_grounded)
            self.assertEqual(len(resp.evidence), 1)

    def test_medical_qa_e2e(self):
        """Verify Medical query execution through CrossTaskService."""
        with patch.object(self.service.orchestrator, "get_medical_qa_service") as mock_get_med:
            mock_med = MagicMock()
            mock_get_med.return_value = mock_med

            from src.medical_qa_service import MedicalQAResponse
            mock_med.process_query.return_value = MedicalQAResponse(
                query="What are symptoms of diabetes?",
                final_answer="Frequent urination and increased thirst are common symptoms.",
                is_grounded=True,
                confidence_score=0.89,
                analysis=MagicMock(),
                status="GROUNDED",
            )

            resp = self.service.process(
                query="What are symptoms of diabetes?",
                session_id="e2e_session_2",
            )
            self.assertEqual(resp.domain, DomainType.MEDICAL.value)
            self.assertIn("Frequent urination", resp.final_text_response)
            self.assertTrue(resp.is_grounded)

    def test_scientific_e2e(self):
        """Verify Scientific research query execution through CrossTaskService."""
        with patch.object(self.service.orchestrator, "get_scientific_expert_service") as mock_get_sci:
            mock_sci = MagicMock()
            mock_get_sci.return_value = mock_sci

            from src.scientific_kb.service import ScientificExpertResponse
            mock_sci.ask.return_value = ScientificExpertResponse(
                query="Explain Tree-LSTM with attention",
                condensed_query="Tree-LSTM attention",
                answer="Tree-LSTM utilizes hierarchical tree attention over syntactic parses.",
                intent="concept_explanation",
                grounded=True,
            )

            resp = self.service.process(
                query="Explain Tree-LSTM with attention in neural networks",
                session_id="e2e_session_3",
            )
            self.assertEqual(resp.domain, DomainType.SCIENTIFIC.value)
            self.assertTrue(resp.is_grounded)


if __name__ == "__main__":
    unittest.main()
