"""Unit Tests for Cross-Task Domain Router (Day 36).

Validates modality routing, domain keyword matching, arXiv ID detection, explicit overrides,
and cross-cutting multilingual language detection and query alignment.
"""

import unittest
from unittest.mock import MagicMock

from src.cross_task.models import DomainType, UnifiedRequest
from src.cross_task.router import CrossTaskRouter, RoutingDecision
from src.multimodal.models import ImageArtifact


class TestCrossTaskRouter(unittest.TestCase):
    """Test suite for CrossTaskRouter."""

    def setUp(self):
        self.router = CrossTaskRouter()

    def test_multimodal_image_routing(self):
        """Verify that any request containing an ImageArtifact routes to MULTIMODAL."""
        dummy_image = ImageArtifact(
            data=b"\x89PNG\r\n\x1a\n" + b"\x00" * 50,
            mime_type="image/png",
            width=200,
            height=200,
            format="PNG",
        )
        req = UnifiedRequest(
            query="Analyze this cardiac scan",
            image=dummy_image,
        )
        decision = self.router.route(req)
        self.assertEqual(decision.domain, DomainType.MULTIMODAL)
        self.assertEqual(decision.confidence, 1.0)
        self.assertIn("Multimodal Assistant Service", decision.reasoning)

    def test_explicit_domain_override(self):
        """Verify that explicit valid domain_override takes precedence."""
        req = UnifiedRequest(
            query="Tell me about pricing",
            domain_override="scientific",
        )
        decision = self.router.route(req)
        self.assertEqual(decision.domain, DomainType.SCIENTIFIC)
        self.assertEqual(decision.confidence, 1.0)

    def test_medical_domain_routing(self):
        """Verify clinical and healthcare questions route to MEDICAL."""
        req = UnifiedRequest(query="What are the symptoms and treatments for breast cancer?")
        decision = self.router.route(req)
        self.assertEqual(decision.domain, DomainType.MEDICAL)
        self.assertGreaterEqual(decision.confidence, 0.80)

        req2 = UnifiedRequest(query="What is the dosage for diabetes medication?")
        decision2 = self.router.route(req2)
        self.assertEqual(decision2.domain, DomainType.MEDICAL)

    def test_scientific_domain_routing(self):
        """Verify AI/ML research queries and arXiv IDs route to SCIENTIFIC."""
        req = UnifiedRequest(query="Explain how the attention mechanism works in Transformer models")
        decision = self.router.route(req)
        self.assertEqual(decision.domain, DomainType.SCIENTIFIC)
        self.assertGreaterEqual(decision.confidence, 0.80)

        # Standalone arXiv ID
        req2 = UnifiedRequest(query="1901.00066")
        decision2 = self.router.route(req2)
        self.assertEqual(decision2.domain, DomainType.SCIENTIFIC)
        self.assertEqual(decision2.confidence, 1.0)

    def test_customer_support_domain_routing(self):
        """Verify ed-tech customer FAQ queries route to CUSTOMER_SUPPORT."""
        req = UnifiedRequest(query="What is your course fee, refund policy, and certificate validity?")
        decision = self.router.route(req)
        self.assertEqual(decision.domain, DomainType.CUSTOMER_SUPPORT)
        self.assertGreaterEqual(decision.confidence, 0.80)

    def test_multilingual_cross_lingual_routing(self):
        """Verify non-English queries detect language and route to appropriate domain."""
        # Spanish customer query
        req_es = UnifiedRequest(query="¿Cuál es el precio del curso y la política de reembolso?")
        decision_es = self.router.route(req_es)
        self.assertEqual(decision_es.detected_language, "es")
        self.assertTrue(decision_es.is_cross_lingual)
        self.assertEqual(decision_es.domain, DomainType.CUSTOMER_SUPPORT)

        # French medical query
        req_fr = UnifiedRequest(query="Quels sont les symptômes du cancer?")
        decision_fr = self.router.route(req_fr)
        self.assertEqual(decision_fr.detected_language, "fr")
        self.assertTrue(decision_fr.is_cross_lingual)
        self.assertEqual(decision_fr.domain, DomainType.MEDICAL)

    def test_empty_query_fallback(self):
        """Verify empty query string routes to fallback safely."""
        req = UnifiedRequest(query="")
        decision = self.router.route(req)
        self.assertEqual(decision.domain, DomainType.CUSTOMER_SUPPORT)
        self.assertEqual(decision.confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
