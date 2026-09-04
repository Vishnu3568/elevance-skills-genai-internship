"""Unit tests for Multimodal Evidence Confidence Assessment (Phase 5 — Day 28 Step 2).
"""

import json
import os
import sys
import unittest

# Ensure src is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from multimodal.confidence import (
    ConfidenceAssessmentResult,
    ConfidenceStatus,
    ItemConfidence,
    MultimodalConfidenceAssessor,
    assess_multimodal_confidence,
)
from multimodal.context import (
    EvidenceProvenance,
    MultimodalContext,
    UnifiedEvidenceItem,
    build_multimodal_context,
)
from multimodal.evidence_validator import (
    EvidenceStatus,
    EvidenceValidationResult,
    MultimodalEvidenceValidator,
)
from multimodal.models import (
    ImageArtifact,
    ModalityType,
    MultimodalResponse,
    VisualEvidenceItem,
)
from multimodal.vision import StructuredVisualOutput


class TestMultimodalConfidenceAssessment(unittest.TestCase):
    """Tests for Day 28 Step 2 confidence assessment component."""

    def setUp(self):
        self.assessor = MultimodalConfidenceAssessor(threshold=0.7)

        # High-confidence item
        self.high_conf_item = VisualEvidenceItem(
            description="Clear architectural block diagram",
            region_label="upper_center",
            confidence=0.95,
        )

        # Low-confidence item
        self.low_conf_item = VisualEvidenceItem(
            description="Blurred text in peripheral margin",
            region_label="bottom_left",
            confidence=0.35,
        )

        # Dummy byte buffer for artifact testing
        self.dummy_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 45
        self.artifact = ImageArtifact(
            data=self.dummy_bytes,
            mime_type="image/png",
            width=640,
            height=480,
            format="PNG",
            file_name="schematic.png",
        )

    def test_single_high_confidence_evidence_item(self):
        """1. Verify assessment of a single high-confidence evidence item."""
        result = self.assessor.assess([self.high_conf_item])

        self.assertEqual(result.status, ConfidenceStatus.AVAILABLE)
        self.assertTrue(result.is_confident)
        self.assertEqual(result.total_items, 1)
        self.assertEqual(result.valid_confidence_count, 1)
        self.assertEqual(result.aggregate_confidence, 0.95)
        self.assertEqual(result.min_confidence, 0.95)
        self.assertEqual(result.max_confidence, 0.95)
        self.assertEqual(result.confidence_distribution["high"], 1)
        self.assertEqual(result.confidence_distribution["low"], 0)

    def test_single_low_confidence_evidence_item(self):
        """2. Verify assessment of a single low-confidence evidence item below threshold."""
        result = self.assessor.assess([self.low_conf_item])

        self.assertEqual(result.status, ConfidenceStatus.AVAILABLE)
        self.assertFalse(result.is_confident)
        self.assertEqual(result.total_items, 1)
        self.assertEqual(result.aggregate_confidence, 0.35)
        self.assertEqual(result.confidence_distribution["low"], 1)
        self.assertEqual(result.confidence_distribution["high"], 0)

    def test_multiple_evidence_items_preservation(self):
        """3. Verify preservation of individual scores across multiple evidence items."""
        items = [
            VisualEvidenceItem(description="Item 1", confidence=0.9),
            VisualEvidenceItem(description="Item 2", confidence=0.8),
            VisualEvidenceItem(description="Item 3", confidence=0.7),
        ]

        result = self.assessor.assess(items)

        self.assertEqual(result.status, ConfidenceStatus.AVAILABLE)
        self.assertEqual(result.total_items, 3)
        self.assertEqual(result.valid_confidence_count, 3)
        self.assertEqual(len(result.item_confidences), 3)

        expected_scores = [0.9, 0.8, 0.7]
        for idx, ic in enumerate(result.item_confidences):
            self.assertEqual(ic.item_index, idx)
            self.assertEqual(ic.confidence_score, expected_scores[idx])
            self.assertEqual(ic.status, ConfidenceStatus.AVAILABLE)

    def test_aggregate_confidence_calculation(self):
        """4. Verify aggregate confidence calculation with mean, min, and harmonic methods."""
        items = [
            {"description": "A", "confidence": 0.9},
            {"description": "B", "confidence": 0.6},
        ]

        # Default mean: (0.9 + 0.6) / 2 = 0.75
        mean_assessor = MultimodalConfidenceAssessor(threshold=0.7, aggregation_method="mean")
        res_mean = mean_assessor.assess(items)
        self.assertEqual(res_mean.aggregate_confidence, 0.75)
        self.assertTrue(res_mean.is_confident)

        # Min method: min(0.9, 0.6) = 0.60
        min_assessor = MultimodalConfidenceAssessor(threshold=0.7, aggregation_method="min")
        res_min = min_assessor.assess(items)
        self.assertEqual(res_min.aggregate_confidence, 0.60)
        self.assertFalse(res_min.is_confident)

        # Harmonic mean
        harm_assessor = MultimodalConfidenceAssessor(threshold=0.7, aggregation_method="harmonic")
        res_harm = harm_assessor.assess(items)
        self.assertIsNotNone(res_harm.aggregate_confidence)
        self.assertAlmostEqual(res_harm.aggregate_confidence, 0.72, delta=0.01)

    def test_missing_confidence_handling(self):
        """5. Verify missing confidence is flagged as unavailable and never defaulted to high."""
        items = [
            {"description": "No conf specified"},              # missing key
            {"description": "Explicit None", "confidence": None}, # None
        ]

        result = self.assessor.assess(items)

        self.assertEqual(result.status, ConfidenceStatus.UNAVAILABLE)
        self.assertFalse(result.is_confident)
        self.assertIsNone(result.aggregate_confidence)
        self.assertEqual(result.total_items, 2)
        self.assertEqual(result.unavailable_confidence_count, 2)
        self.assertEqual(result.valid_confidence_count, 0)

        for ic in result.item_confidences:
            self.assertEqual(ic.status, ConfidenceStatus.UNAVAILABLE)
            self.assertIsNone(ic.confidence_score)

    def test_invalid_confidence_type(self):
        """6. Verify detection of invalid confidence types (boolean, string, NaN, Inf, non-numeric)."""
        invalid_items = [
            {"description": "Bool True", "confidence": True},       # boolean True
            {"description": "Bool False", "confidence": False},     # boolean False
            {"description": "String conf", "confidence": "0.9"},    # string
            {"description": "NaN conf", "confidence": float("nan")}, # NaN
            {"description": "Inf conf", "confidence": float("inf")}, # Inf
        ]

        result = self.assessor.assess(invalid_items)

        self.assertEqual(result.status, ConfidenceStatus.INVALID)
        self.assertFalse(result.is_confident)
        self.assertIsNone(result.aggregate_confidence)
        self.assertEqual(result.invalid_confidence_count, 5)
        self.assertEqual(result.valid_confidence_count, 0)

        for ic in result.item_confidences:
            self.assertEqual(ic.status, ConfidenceStatus.INVALID)
            self.assertFalse(ic.is_valid)
            self.assertIsNotNone(ic.error_message)

    def test_out_of_range_confidence(self):
        """7. Verify confidence values outside [0.0, 1.0] are classified as invalid."""
        out_of_bounds_items = [
            {"description": "Negative", "confidence": -0.1},
            {"description": "Too high", "confidence": 1.001},
            {"description": "Extremely high", "confidence": 100.0},
        ]

        result = self.assessor.assess(out_of_bounds_items)

        self.assertEqual(result.status, ConfidenceStatus.INVALID)
        self.assertFalse(result.is_confident)
        self.assertEqual(result.invalid_confidence_count, 3)
        self.assertEqual(result.valid_confidence_count, 0)

        for ic in result.item_confidences:
            self.assertEqual(ic.status, ConfidenceStatus.INVALID)
            self.assertIn("out of bounds", ic.error_message.lower())

    def test_empty_or_no_evidence(self):
        """8. Verify handling when evidence is None, an empty list, or has no evidence items."""
        # Case A: None
        res_none = self.assessor.assess(None)
        self.assertEqual(res_none.status, ConfidenceStatus.UNAVAILABLE)
        self.assertFalse(res_none.is_confident)
        self.assertIsNone(res_none.aggregate_confidence)
        self.assertEqual(res_none.total_items, 0)

        # Case B: Empty list
        res_empty = self.assessor.assess([])
        self.assertEqual(res_empty.status, ConfidenceStatus.UNAVAILABLE)
        self.assertFalse(res_empty.is_confident)
        self.assertEqual(res_empty.total_items, 0)

        # Case C: MultimodalResponse without evidence
        resp_no_ev = MultimodalResponse(
            query="test",
            answer="test ans",
            modality=ModalityType.TEXT_ONLY,
            visual_evidence=[],
        )
        res_resp = self.assessor.assess(resp_no_ev)
        self.assertEqual(res_resp.status, ConfidenceStatus.UNAVAILABLE)
        self.assertFalse(res_resp.is_confident)

    def test_deterministic_results(self):
        """9. Verify identical repeated assessment calls produce identical deterministic outputs."""
        items = [self.high_conf_item, self.low_conf_item]

        res1 = self.assessor.assess(items)
        res2 = self.assessor.assess(items)

        self.assertEqual(res1.status, res2.status)
        self.assertEqual(res1.aggregate_confidence, res2.aggregate_confidence)
        self.assertEqual(res1.to_dict(), res2.to_dict())

    def test_json_serialization(self):
        """10. Verify full JSON serializability of ConfidenceAssessmentResult without error."""
        items = [
            self.high_conf_item,
            {"description": "Missing conf", "confidence": None},
            {"description": "Bad conf", "confidence": "invalid"},
        ]

        result = self.assessor.assess(items)
        self.assertEqual(result.status, ConfidenceStatus.MIXED)

        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["status"], "mixed")
        self.assertEqual(d["total_items"], 3)
        self.assertEqual(d["valid_confidence_count"], 1)
        self.assertEqual(d["unavailable_confidence_count"], 1)
        self.assertEqual(d["invalid_confidence_count"], 1)

        # Must serialize cleanly to JSON string
        json_str = json.dumps(d)
        self.assertIn('"status": "mixed"', json_str)
        self.assertIn('"valid_confidence_count": 1', json_str)

    def test_raw_image_bytes_never_exposed(self):
        """11. Verify raw image byte payloads are never exposed in confidence assessment outputs."""
        item_with_bytes = {
            "description": "Visual diagram with raw payload",
            "confidence": 0.88,
            "data": self.dummy_bytes,
            "base64_str": "iVBORw0KGgoAAAANSUhEUgAA...",
            "metadata": {
                "raw_bytes": self.dummy_bytes,
                "image_data": self.dummy_bytes,
            },
        }

        result = self.assessor.assess([item_with_bytes])
        self.assertEqual(result.status, ConfidenceStatus.AVAILABLE)

        d = result.to_dict()
        d_str = str(d)

        self.assertNotIn("b'\\x89PNG", d_str)
        self.assertNotIn("b'\\x00", d_str)
        self.assertNotIn("data", d["metadata"])

    def test_functional_interface_and_containers(self):
        """12. Verify assess_multimodal_confidence functional interface with MultimodalContext."""
        vis_out = StructuredVisualOutput(
            scene_description="Technical flowchart",
            detected_objects=["processor", "memory_unit"],
        )
        ctx = build_multimodal_context(
            query="Analyze hardware layout",
            artifact=self.artifact,
            visual_output=vis_out,
        )

        res = assess_multimodal_confidence(ctx, threshold=0.6)
        self.assertEqual(res.status, ConfidenceStatus.AVAILABLE)
        self.assertTrue(res.is_confident)
        self.assertGreater(res.total_items, 0)
        self.assertIsNotNone(res.aggregate_confidence)


if __name__ == "__main__":
    unittest.main()
