"""Unit tests for Unified Multimodal Context Representation (Phase 5 — Day 26 Step 1).
"""

import io
import os
import sys
from typing import Any
import unittest
from PIL import Image

# Ensure src is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from multimodal.context import (
    EvidenceProvenance,
    MultimodalContext,
    TextualContext,
    UnifiedEvidenceItem,
    VisualContext,
    build_multimodal_context,
    validate_multimodal_context,
)
from multimodal.models import (
    ImageArtifact,
    ModalityType,
    MultimodalRequest,
    VisualEvidenceItem,
)
from multimodal.preprocessing import preprocess_image
from multimodal.vision import StructuredVisualOutput, VisionService


class TestMultimodalContext(unittest.TestCase):
    """Tests for the unified multimodal context container and builder."""

    @classmethod
    def setUpClass(cls):
        # Create valid test image bytes and artifact
        img = Image.new("RGB", (120, 80), color="purple")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        cls.png_bytes = buf.getvalue()

        cls.artifact = ImageArtifact(
            data=cls.png_bytes,
            mime_type="image/png",
            width=120,
            height=80,
            format="PNG",
            file_name="chart_q3.png",
        )
        cls.preprocessed = preprocess_image(cls.artifact)
        cls.vision_service = VisionService()
        cls.visual_output = cls.vision_service.analyze(cls.preprocessed)

    def test_text_only_context(self):
        text_ctx = TextualContext(
            raw_query="What was the net revenue in 2023?",
            normalized_query="what was the net revenue in 2023?",
            extracted_entities=["net revenue", "2023"],
            provenance=EvidenceProvenance(
                source_type="user_query",
                modality=ModalityType.TEXT_ONLY,
                source_id="sess_123",
                provider="user",
            ),
        )

        ctx = MultimodalContext(
            session_id="sess_123",
            modality=ModalityType.TEXT_ONLY,
            text_context=text_ctx,
        )

        self.assertEqual(ctx.session_id, "sess_123")
        self.assertEqual(ctx.modality, ModalityType.TEXT_ONLY)
        self.assertTrue(ctx.text_context.has_text)
        self.assertFalse(ctx.visual_context.has_visuals)
        self.assertIn("sess_123", ctx.get_summary_description())

    def test_image_only_context(self):
        vis_ctx = VisualContext(
            artifact=self.artifact,
            preprocessed=self.preprocessed,
            visual_output=self.visual_output,
            scene_description=self.visual_output.scene_description,
            detected_objects=self.visual_output.detected_objects,
            visible_text=self.visual_output.visible_text,
            spatial_observations=self.visual_output.spatial_observations,
            visual_attributes=self.visual_output.visual_attributes,
            provenance=EvidenceProvenance(
                source_type="visual_inspection",
                modality=ModalityType.IMAGE_ONLY,
                source_id="chart_q3.png",
                provider="deterministic_mock",
            ),
        )

        ctx = MultimodalContext(
            session_id="sess_img_only",
            modality=ModalityType.IMAGE_ONLY,
            visual_context=vis_ctx,
        )

        self.assertEqual(ctx.modality, ModalityType.IMAGE_ONLY)
        self.assertFalse(ctx.text_context.has_text)
        self.assertTrue(ctx.visual_context.has_visuals)
        self.assertEqual(ctx.visual_context.artifact.format, "PNG")

    def test_text_and_image_context(self):
        ctx = build_multimodal_context(
            query="Analyze the trends shown in this bar chart.",
            artifact=self.artifact,
            preprocessed=self.preprocessed,
            visual_output=self.visual_output,
            session_id="sess_joint",
        )

        self.assertEqual(ctx.modality, ModalityType.TEXT_AND_IMAGE)
        self.assertTrue(ctx.text_context.has_text)
        self.assertTrue(ctx.visual_context.has_visuals)
        self.assertEqual(ctx.text_context.raw_query, "Analyze the trends shown in this bar chart.")
        self.assertTrue(len(ctx.evidence_items) > 0)
        self.assertIsNotNone(ctx.visual_context.provenance)
        self.assertEqual(ctx.visual_context.provenance.source_id, "chart_q3.png")

    def test_visual_evidence_attachment_and_conversion(self):
        ctx = build_multimodal_context(
            artifact=self.artifact,
            visual_output=self.visual_output,
            session_id="sess_ev",
        )

        evidence = ctx.get_visual_evidence()
        self.assertTrue(len(evidence) > 0)
        self.assertTrue(all(isinstance(item, VisualEvidenceItem) for item in evidence))

        # Add custom evidence item
        custom_item = UnifiedEvidenceItem(
            description="High sales growth in Q3",
            modality=ModalityType.IMAGE_ONLY,
            region_label="Quadrant Top-Right",
            confidence=0.92,
            source_type="ocr_text",
            provenance=EvidenceProvenance(
                source_type="ocr",
                modality=ModalityType.IMAGE_ONLY,
                source_id="chart_q3.png",
                provider="tesseract",
            ),
        )
        ctx.add_evidence(custom_item)
        self.assertIn(custom_item, ctx.evidence_items)

        converted = custom_item.to_visual_evidence_item()
        self.assertEqual(converted.description, "High sales growth in Q3")
        self.assertEqual(converted.confidence, 0.92)

    def test_modality_consistency_enforcement(self):
        # 1. TEXT_ONLY with visual data must fail
        with self.assertRaises(ValueError) as ctx1:
            MultimodalContext(
                session_id="s1",
                modality=ModalityType.TEXT_ONLY,
                text_context=TextualContext(raw_query="Hello"),
                visual_context=VisualContext(artifact=self.artifact),
            )
        self.assertIn("visual_context contains visual data", str(ctx1.exception))

        # 2. IMAGE_ONLY with query text must fail
        with self.assertRaises(ValueError) as ctx2:
            MultimodalContext(
                session_id="s2",
                modality=ModalityType.IMAGE_ONLY,
                text_context=TextualContext(raw_query="Hello"),
                visual_context=VisualContext(artifact=self.artifact),
            )
        self.assertIn("text_context contains query text", str(ctx2.exception))

        # 3. TEXT_AND_IMAGE missing either component must fail
        with self.assertRaises(ValueError) as ctx3:
            MultimodalContext(
                session_id="s3",
                modality=ModalityType.TEXT_AND_IMAGE,
                text_context=TextualContext(raw_query="Hello"),
                visual_context=VisualContext(),  # empty visual
            )
        self.assertIn("visual_context has no visual data", str(ctx3.exception))

    def test_session_propagation(self):
        req = MultimodalRequest(
            query="Summarize findings",
            session_id="custom_session_999",
            metadata_filters={"domain": "finance"},
        )
        ctx = build_multimodal_context(request=req)

        self.assertEqual(ctx.session_id, "custom_session_999")
        self.assertEqual(ctx.modality, ModalityType.TEXT_ONLY)
        self.assertEqual(ctx.metadata.get("filters"), {"domain": "finance"})

    def test_invalid_and_empty_context_rejection(self):
        # Empty session id
        with self.assertRaises(ValueError):
            MultimodalContext(
                session_id="   ",
                modality=ModalityType.TEXT_ONLY,
                text_context=TextualContext(raw_query="query"),
            )

        # Entirely empty context (no text, no visuals, no evidence)
        with self.assertRaises(ValueError):
            MultimodalContext(
                session_id="valid_id",
                modality=ModalityType.TEXT_ONLY,
                text_context=TextualContext(),
                visual_context=VisualContext(),
            )

        # Non-ModalityType modality
        with self.assertRaises(TypeError):
            MultimodalContext(
                session_id="valid_id",
                modality="text_only",  # type: ignore
                text_context=TextualContext(raw_query="query"),
            )

    def test_mutable_default_isolation(self):
        # Verify two instances created with default factories do not share state
        ctx1 = MultimodalContext(
            session_id="sess_1",
            modality=ModalityType.TEXT_ONLY,
            text_context=TextualContext(raw_query="query 1"),
        )
        ctx2 = MultimodalContext(
            session_id="sess_2",
            modality=ModalityType.TEXT_ONLY,
            text_context=TextualContext(raw_query="query 2"),
        )

        item = UnifiedEvidenceItem(
            description="Unique Evidence",
            modality=ModalityType.TEXT_ONLY,
            confidence=0.9,
        )
        ctx1.add_evidence(item)
        ctx1.metadata["key"] = "val"

        self.assertEqual(len(ctx1.evidence_items), 1)
        self.assertEqual(len(ctx2.evidence_items), 0)
        self.assertNotIn("key", ctx2.metadata)

    def test_deterministic_serialization(self):
        ctx = build_multimodal_context(
            query="Compare regional sales",
            artifact=self.artifact,
            visual_output=self.visual_output,
            session_id="serial_sess",
            metadata={"source": "unit_test"},
        )

        d = ctx.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["session_id"], "serial_sess")
        self.assertEqual(d["modality"], "text_and_image")
        self.assertEqual(d["metadata"]["source"], "unit_test")
        self.assertIn("text_context", d)
        self.assertIn("visual_context", d)
        self.assertIn("evidence_items", d)

        # Ensure raw binary data bytes are NOT serialized in to_dict()
        vis_dict = d["visual_context"]
        if vis_dict.get("artifact_summary"):
            self.assertNotIn("data", vis_dict["artifact_summary"])
            self.assertIn("data_size_bytes", vis_dict["artifact_summary"])


if __name__ == "__main__":
    unittest.main()
