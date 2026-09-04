"""Unit tests for Multimodal Evidence Validation (Phase 5 — Day 28 Step 1).
"""

import json
import os
import sys
import unittest

# Ensure src is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from multimodal.context import (
    EvidenceProvenance,
    MultimodalContext,
    UnifiedEvidenceItem,
    build_multimodal_context,
)
from multimodal.evidence_validator import (
    EvidenceItemValidation,
    EvidenceStatus,
    EvidenceValidationResult,
    MultimodalEvidenceValidator,
    validate_multimodal_evidence,
)
from multimodal.models import (
    ImageArtifact,
    ModalityType,
    MultimodalResponse,
    VisualEvidenceItem,
)
from multimodal.vision import StructuredVisualOutput


class TestMultimodalEvidenceValidation(unittest.TestCase):
    """Tests for Day 28 Step 1 evidence validation component."""

    def setUp(self):
        self.validator = MultimodalEvidenceValidator()

        # Canonical valid visual evidence item
        self.valid_vis_item = VisualEvidenceItem(
            description="High-frequency activation in the upper convolution block",
            region_label="upper_left",
            confidence=0.92,
        )

        # Canonical valid unified evidence item with provenance
        self.valid_unified_item = UnifiedEvidenceItem(
            description="Documented transformer self-attention layer",
            modality=ModalityType.TEXT_ONLY,
            region_label=None,
            confidence=0.88,
            source_type="academic_text",
            provenance=EvidenceProvenance(
                source_type="academic_text",
                modality=ModalityType.TEXT_ONLY,
                source_id="arxiv_1706_03762",
                provider="arxiv",
                confidence=0.88,
            ),
        )

        # Dummy byte payload
        self.dummy_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 40
        self.artifact = ImageArtifact(
            data=self.dummy_bytes,
            mime_type="image/png",
            width=640,
            height=480,
            format="PNG",
            file_name="diagram.png",
        )

    def test_valid_evidence_items(self):
        """1. Verify validation of valid VisualEvidenceItem, UnifiedEvidenceItem, and dicts."""
        items = [
            self.valid_vis_item,
            self.valid_unified_item,
            {
                "description": "Bounding box around input tensor",
                "region_label": "bottom_right",
                "confidence": 0.95,
                "modality": "image_only",
                "source_type": "object_detection",
            },
        ]

        result = self.validator.validate_evidence(items)

        self.assertEqual(result.status, EvidenceStatus.VALID)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.total_items, 3)
        self.assertEqual(result.valid_items_count, 3)
        self.assertEqual(result.invalid_items_count, 0)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.valid_evidence), 3)

    def test_empty_evidence_handling(self):
        """2. Verify handling of None, empty lists, and responses with no evidence."""
        # Case A: None
        res_none = self.validator.validate_evidence(None)
        self.assertEqual(res_none.status, EvidenceStatus.NO_EVIDENCE)
        self.assertFalse(res_none.is_valid)
        self.assertEqual(res_none.total_items, 0)
        self.assertEqual(res_none.valid_items_count, 0)
        self.assertEqual(res_none.invalid_items_count, 0)

        # Case B: Empty list
        res_empty = self.validator.validate_evidence([])
        self.assertEqual(res_empty.status, EvidenceStatus.NO_EVIDENCE)
        self.assertFalse(res_empty.is_valid)
        self.assertEqual(res_empty.total_items, 0)

        # Case C: MultimodalResponse with no visual_evidence
        resp_no_ev = MultimodalResponse(
            query="Hello",
            answer="Hi there",
            modality=ModalityType.TEXT_ONLY,
            visual_evidence=[],
        )
        res_resp = self.validator.validate_evidence(resp_no_ev)
        self.assertEqual(res_resp.status, EvidenceStatus.NO_EVIDENCE)
        self.assertFalse(res_resp.is_valid)

    def test_missing_required_evidence_fields(self):
        """3. Verify rejection of items with missing, empty, or whitespace-only descriptions."""
        items = [
            {"region_label": "center", "confidence": 0.9},  # missing description
            {"description": "", "confidence": 0.9},         # empty description
            {"description": "   ", "confidence": 0.9},       # whitespace-only
        ]

        result = self.validator.validate_evidence(items)

        self.assertEqual(result.status, EvidenceStatus.INVALID)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.total_items, 3)
        self.assertEqual(result.valid_items_count, 0)
        self.assertEqual(result.invalid_items_count, 3)
        self.assertEqual(len(result.errors), 3)
        for err in result.errors:
            self.assertIn("description", err.lower())

    def test_malformed_evidence_items(self):
        """4. Verify detection of malformed evidence (invalid types, out-of-bounds confidence)."""
        malformed_items = [
            "raw_string_not_an_item",                     # unsupported type
            12345,                                        # integer
            {"description": "Valid desc", "confidence": -0.5},   # negative confidence
            {"description": "Valid desc", "confidence": 1.5},    # confidence > 1.0
            {"description": "Valid desc", "confidence": True},   # boolean confidence
            {"description": "Valid desc", "region_label": 999},  # non-string region label
            {"description": "Valid desc", "modality": 42},       # non-string modality
        ]

        result = self.validator.validate_evidence(malformed_items)

        self.assertEqual(result.status, EvidenceStatus.INVALID)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.total_items, 7)
        self.assertEqual(result.valid_items_count, 0)
        self.assertEqual(result.invalid_items_count, 7)

    def test_multiple_evidence_items_ordering(self):
        """5. Verify preservation of item indices and ordering across multiple evidence items."""
        items = [
            VisualEvidenceItem(description=f"Evidence {i}", confidence=0.8)
            for i in range(5)
        ]

        result = self.validator.validate_evidence(items)

        self.assertEqual(result.status, EvidenceStatus.VALID)
        self.assertEqual(result.total_items, 5)
        self.assertEqual(result.valid_items_count, 5)
        for idx, item_val in enumerate(result.item_validations):
            self.assertEqual(item_val.item_index, idx)
            self.assertEqual(item_val.description, f"Evidence {idx}")

    def test_mixed_valid_and_invalid_evidence(self):
        """6. Verify appropriate partition and reporting for mixed valid and invalid evidence."""
        items = [
            self.valid_vis_item,                       # valid
            {"description": "", "confidence": 0.8},    # invalid (empty desc)
            self.valid_unified_item,                   # valid
            {"description": "Bad conf", "confidence": 2.0},  # invalid (conf out of bounds)
        ]

        result = self.validator.validate_evidence(items)

        self.assertEqual(result.status, EvidenceStatus.PARTIAL)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.total_items, 4)
        self.assertEqual(result.valid_items_count, 2)
        self.assertEqual(result.invalid_items_count, 2)
        self.assertEqual(len(result.valid_evidence), 2)
        self.assertEqual(len(result.invalid_evidence), 2)

    def test_no_raw_image_bytes_leakage(self):
        """7. Verify raw image byte payloads are strictly excluded from validation results."""
        item_with_bytes = {
            "description": "Visual schematic diagram",
            "region_label": "full_canvas",
            "confidence": 0.95,
            "data": self.dummy_bytes,
            "base64_str": "iVBORw0KGgoAAAANSUhEUgAA...",
            "metadata": {
                "raw_buffer": self.dummy_bytes,
                "image_data": self.dummy_bytes,
                "label": "encoder",
            },
        }

        result = self.validator.validate_evidence([item_with_bytes])
        self.assertEqual(result.status, EvidenceStatus.VALID)

        d = result.to_dict()
        d_str = str(d)

        # Assert no raw bytes or byte literals leak into the serialized output
        self.assertNotIn("b'\\x89PNG", d_str)
        self.assertNotIn("b'\\x00", d_str)
        self.assertNotIn("data", d["valid_evidence"][0])
        self.assertNotIn("base64_str", d["valid_evidence"][0])

    def test_validation_result_serialization(self):
        """8. Verify deterministic JSON serialization of EvidenceValidationResult."""
        items = [self.valid_vis_item, {"description": "Invalid item", "confidence": 99}]
        result = self.validator.validate_evidence(items)

        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["status"], "partial")
        self.assertEqual(d["total_items"], 2)
        self.assertEqual(d["valid_items_count"], 1)
        self.assertEqual(d["invalid_items_count"], 1)

        # Must be serializable by standard json without error
        json_str = json.dumps(d)
        self.assertIn('"status": "partial"', json_str)
        self.assertIn("Invalid item", json_str)

    def test_validation_from_multimodal_context_and_response(self):
        """9. Verify seamless extraction and validation from MultimodalContext and MultimodalResponse."""
        # MultimodalContext with structured output
        vis_out = StructuredVisualOutput(
            scene_description="Technical flowchart",
            detected_objects=["arrow", "block"],
        )
        ctx = build_multimodal_context(
            query="Inspect flow",
            artifact=self.artifact,
            visual_output=vis_out,
        )

        res_ctx = validate_multimodal_evidence(ctx)
        self.assertEqual(res_ctx.status, EvidenceStatus.VALID)
        self.assertTrue(res_ctx.is_valid)
        self.assertGreater(res_ctx.total_items, 0)
        self.assertEqual(res_ctx.source_summary.get("container"), "MultimodalContext")

        # MultimodalResponse with visual evidence
        resp = MultimodalResponse(
            query="Inspect flow",
            answer="Flowchart detected.",
            modality=ModalityType.TEXT_AND_IMAGE,
            visual_evidence=[self.valid_vis_item],
        )

        res_resp = validate_multimodal_evidence(resp)
        self.assertEqual(res_resp.status, EvidenceStatus.VALID)
        self.assertTrue(res_resp.is_valid)
        self.assertEqual(res_resp.total_items, 1)
        self.assertEqual(res_resp.source_summary.get("container"), "MultimodalResponse")


if __name__ == "__main__":
    unittest.main()
