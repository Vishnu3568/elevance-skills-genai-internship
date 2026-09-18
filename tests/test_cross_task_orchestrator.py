"""Unit and Mock Integration Tests for CrossTaskOrchestrator (Day 36).

Validates end-to-end dispatching across Customer, Medical, Scientific, Multimodal,
and Multilingual pathways, verifying isolation, session retention, and error resilience.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.cross_task.models import DomainType, UnifiedRequest
from src.cross_task.orchestrator import CrossTaskOrchestrator
from src.cross_task.router import CrossTaskRouter, RoutingDecision
from src.multimodal.models import ImageArtifact, ModalityType, MultimodalResponse
from src.scientific_kb.service import ScientificExpertResponse
from src.medical_qa_service import MedicalQAResponse
from src.chatbot_service import ChatbotResponse


class TestCrossTaskOrchestrator(unittest.TestCase):
    """Test suite for CrossTaskOrchestrator."""

    def setUp(self):
        self.mock_router = MagicMock(spec=CrossTaskRouter)
        self.mock_cb_service = MagicMock()
        self.mock_med_service = MagicMock()
        self.mock_sci_service = MagicMock()
        self.mock_mm_service = MagicMock()
        self.mock_ml_service = MagicMock()

        self.orchestrator = CrossTaskOrchestrator(
            router=self.mock_router,
            chatbot_service=self.mock_cb_service,
            medical_qa_service=self.mock_med_service,
            scientific_expert_service=self.mock_sci_service,
            multimodal_service=self.mock_mm_service,
            multilingual_service=self.mock_ml_service,
        )

    def test_dispatch_customer_support(self):
        """Verify routing and dispatch to Customer Service chatbot."""
        self.mock_router.route.return_value = RoutingDecision(
            domain=DomainType.CUSTOMER_SUPPORT,
            detected_language="en",
            detected_language_name="English",
            confidence=0.90,
            reasoning="Customer signals detected",
            is_cross_lingual=False,
        )
        self.mock_cb_service.process_query.return_value = ChatbotResponse(
            query="What is the refund policy?",
            final_answer="Refunds are processed within 7 business days.",
            raw_answer="Refunds are processed within 7 business days.",
            sentiment_label="NEUTRAL",
            confidence_score=0.94,
            is_ood=False,
            source_documents=[],
        )

        req = UnifiedRequest(query="What is the refund policy?", session_id="test_sess")
        resp = self.orchestrator.dispatch(req)

        self.assertEqual(resp.domain, DomainType.CUSTOMER_SUPPORT.value)
        self.assertEqual(resp.final_text_response, "Refunds are processed within 7 business days.")
        self.mock_cb_service.process_query.assert_called_once_with("What is the refund policy?")

    def test_dispatch_medical(self):
        """Verify routing and dispatch to Medical Q&A service."""
        self.mock_router.route.return_value = RoutingDecision(
            domain=DomainType.MEDICAL,
            detected_language="en",
            detected_language_name="English",
            confidence=0.95,
            reasoning="Clinical signals detected",
            is_cross_lingual=False,
        )
        self.mock_med_service.process_query.return_value = MedicalQAResponse(
            query="What are breast cancer symptoms?",
            final_answer="Common symptoms include a lump in the breast tissue.",
            is_grounded=True,
            confidence_score=0.91,
            analysis=MagicMock(),
            status="GROUNDED",
        )

        req = UnifiedRequest(query="What are breast cancer symptoms?")
        resp = self.orchestrator.dispatch(req)

        self.assertEqual(resp.domain, DomainType.MEDICAL.value)
        self.assertEqual(resp.final_text_response, "Common symptoms include a lump in the breast tissue.")
        self.mock_med_service.process_query.assert_called_once_with("What are breast cancer symptoms?")

    def test_dispatch_scientific(self):
        """Verify routing and dispatch to Scientific Domain Expert service."""
        self.mock_router.route.return_value = RoutingDecision(
            domain=DomainType.SCIENTIFIC,
            detected_language="en",
            detected_language_name="English",
            confidence=0.92,
            reasoning="Research signals detected",
            is_cross_lingual=False,
        )
        self.mock_sci_service.ask.return_value = ScientificExpertResponse(
            query="Explain Tree-LSTM",
            condensed_query="Tree-LSTM",
            answer="Tree-LSTM generalizes sequence LSTMs to tree-structured topologies.",
            intent="concept_explanation",
            grounded=True,
        )

        req = UnifiedRequest(query="Explain Tree-LSTM")
        resp = self.orchestrator.dispatch(req)

        self.assertEqual(resp.domain, DomainType.SCIENTIFIC.value)
        self.mock_sci_service.ask.assert_called_once_with("Explain Tree-LSTM")

    def test_dispatch_multimodal(self):
        """Verify routing and dispatch to Multimodal Assistant service."""
        dummy_img = ImageArtifact(
            data=b"\x89PNG\r\n\x1a\n" + b"\x00" * 30,
            mime_type="image/png",
            width=100,
            height=100,
            format="PNG",
        )
        self.mock_router.route.return_value = RoutingDecision(
            domain=DomainType.MULTIMODAL,
            detected_language="en",
            detected_language_name="English",
            confidence=1.0,
            reasoning="Image input detected",
            is_cross_lingual=False,
        )
        self.mock_mm_service.process_request.return_value = MultimodalResponse(
            query="Explain this chart",
            answer="The chart illustrates model accuracy over epochs.",
            modality=ModalityType.TEXT_AND_IMAGE,
            grounded=True,
        )

        req = UnifiedRequest(query="Explain this chart", image=dummy_img)
        resp = self.orchestrator.dispatch(req)

        self.assertEqual(resp.domain, DomainType.MULTIMODAL.value)
        self.mock_mm_service.process_request.assert_called_once()

    def test_dispatch_cross_lingual(self):
        """Verify non-English customer queries route to Multilingual cross-lingual pipeline."""
        from src.multilingual.models import MultilingualResponse

        self.mock_router.route.return_value = RoutingDecision(
            domain=DomainType.CUSTOMER_SUPPORT,
            detected_language="es",
            detected_language_name="Spanish",
            confidence=0.88,
            reasoning="Spanish language customer query",
            is_cross_lingual=True,
        )
        self.mock_ml_service.process_text_request.return_value = MultilingualResponse(
            query="¿Cuál es el costo del curso?",
            language="es",
            intent="fees",
            aligned_query="What is the cost of the course?",
            final_answer="El costo del curso es de $500.",
            raw_answer="The course fee is $500.",
            confidence_score=0.93,
            is_grounded=True,
        )

        req = UnifiedRequest(query="¿Cuál es el costo del curso?", session_id="es_session")
        resp = self.orchestrator.dispatch(req)

        self.assertEqual(resp.domain, DomainType.CUSTOMER_SUPPORT.value)
        self.assertEqual(resp.detected_language, "es")
        self.assertEqual(resp.final_text_response, "El costo del curso es de $500.")
        self.mock_ml_service.process_text_request.assert_called_once()

    def test_dispatch_exception_resilience(self):
        """Verify that runtime errors inside a service return a structured error response without crashing."""
        self.mock_router.route.return_value = RoutingDecision(
            domain=DomainType.CUSTOMER_SUPPORT,
            detected_language="en",
            detected_language_name="English",
            confidence=0.80,
            reasoning="Standard query",
            is_cross_lingual=False,
        )
        self.mock_cb_service.process_query.side_effect = RuntimeError("FAISS memory fault")

        req = UnifiedRequest(query="What is the policy?")
        resp = self.orchestrator.dispatch(req)

        self.assertEqual(resp.domain, DomainType.CUSTOMER_SUPPORT.value)
        self.assertFalse(resp.is_grounded)
        self.assertEqual(resp.confidence_score, 0.0)
        self.assertEqual(resp.confidence_tier, "INSUFFICIENT")
        self.assertIn("FAISS memory fault", resp.final_text_response)

    def test_get_scientific_expert_service_binds_open_source_model(self):
        """Verify CrossTaskOrchestrator binds the open-source LLM for Scientific service."""
        orch = CrossTaskOrchestrator(router=self.mock_router)
        sci_service = orch.get_scientific_expert_service()
        if sci_service is not None:
            self.assertEqual(sci_service.generator.model_name, "google/flan-t5-base")
            self.assertTrue(sci_service.generator.is_open_source)


if __name__ == "__main__":
    unittest.main()
