"""Unit Tests for Cross-Task Data Models (Day 36).

Validates schema correctness, type-checking invariants, serialization, and post-init
checks for UnifiedRequest, UnifiedResponse, EvidenceItem, and CitationItem.
"""

import unittest
from src.cross_task.models import (
    CitationItem,
    ConfidenceTier,
    DomainType,
    EvidenceItem,
    UnifiedRequest,
    UnifiedResponse,
)


class TestCrossTaskModels(unittest.TestCase):
    """Test suite for cross-task models and contracts."""

    def test_evidence_item_valid(self):
        """Test valid EvidenceItem creation and serialization."""
        item = EvidenceItem(
            description="Visual canvas shows 3-channel RGB profile",
            source="visual_analysis",
            confidence=0.98,
            region_label="canvas_bounds",
            metadata={"width": 800, "height": 600},
        )
        self.assertEqual(item.description, "Visual canvas shows 3-channel RGB profile")
        self.assertEqual(item.source, "visual_analysis")
        self.assertEqual(item.confidence, 0.98)
        self.assertEqual(item.region_label, "canvas_bounds")
        d = item.to_dict()
        self.assertEqual(d["confidence"], 0.98)
        self.assertEqual(d["metadata"]["width"], 800)

    def test_citation_item_valid(self):
        """Test valid CitationItem creation and serialization."""
        cit = CitationItem(
            title="Attention Is All You Need",
            source_id="1706.03762",
            url="https://arxiv.org/abs/1706.03762",
            authors="Vaswani et al.",
        )
        self.assertEqual(cit.title, "Attention Is All You Need")
        self.assertEqual(cit.source_id, "1706.03762")
        d = cit.to_dict()
        self.assertEqual(d["authors"], "Vaswani et al.")

    def test_unified_request_valid(self):
        """Test valid UnifiedRequest creation and serialization."""
        req = UnifiedRequest(
            query="What are the symptoms of breast cancer?",
            session_id="session_123",
            language_hint="en",
            domain_override="medical",
        )
        self.assertEqual(req.query, "What are the symptoms of breast cancer?")
        self.assertEqual(req.session_id, "session_123")
        self.assertEqual(req.domain_override, "medical")
        d = req.to_dict()
        self.assertFalse(d["has_image"])
        self.assertEqual(d["domain_override"], "medical")

    def test_unified_request_invalid_type(self):
        """Test UnifiedRequest type validation on query and session_id."""
        with self.assertRaises(TypeError):
            # pyrefly: ignore [arg-type]
            UnifiedRequest(query=12345)  # type: ignore

        with self.assertRaises(ValueError):
            UnifiedRequest(query="valid query", session_id="")

        with self.assertRaises(ValueError):
            UnifiedRequest(query="valid query", domain_override="invalid_domain_name")

    def test_unified_response_valid(self):
        """Test valid UnifiedResponse creation and serialization."""
        resp = UnifiedResponse(
            query="Explain Tree-LSTM with attention",
            final_text_response="Tree-LSTM incorporates hierarchical tree attention.",
            domain=DomainType.SCIENTIFIC.value,
            detected_language="en",
            detected_language_name="English",
            confidence_score=0.95,
            confidence_tier=ConfidenceTier.HIGH.value,
            is_grounded=True,
            sentiment="NEUTRAL",
            execution_time_ms=45.2,
        )
        self.assertEqual(resp.domain, "scientific")
        self.assertEqual(resp.confidence_score, 0.95)
        self.assertTrue(resp.is_grounded)
        d = resp.to_dict()
        self.assertEqual(d["confidence_score"], 0.95)
        self.assertEqual(d["confidence_tier"], "HIGH")
        self.assertEqual(d["execution_time_ms"], 45.2)

    def test_unified_response_invalid_confidence(self):
        """Test UnifiedResponse confidence bounds enforcement."""
        with self.assertRaises(ValueError):
            UnifiedResponse(
                query="test",
                final_text_response="test response",
                domain="customer_support",
                confidence_score=1.5,
            )


if __name__ == "__main__":
    unittest.main()
