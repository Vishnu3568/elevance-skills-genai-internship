"""Unit tests for Evidence-Aware Multimodal Response Generation (Phase 5 — Day 26 Step 3).
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
    UnifiedEvidenceItem,
    build_multimodal_context,
)
from multimodal.models import (
    ImageArtifact,
    ModalityType,
    MultimodalResponse,
    VisualEvidenceItem,
)
from multimodal.preprocessing import preprocess_image
from multimodal.reasoning import (
    MultimodalReasoningEngine,
    ReasoningResult,
)
from multimodal.response_generator import (
    MultimodalResponseGenerator,
    generate_multimodal_response,
)
from multimodal.vision import VisionService


class TestMultimodalResponseGeneration(unittest.TestCase):
    """Tests for evidence-aware multimodal response transformation and guarantees."""

    @classmethod
    def setUpClass(cls):
        # Create test image and visual artifacts
        img = Image.new("RGB", (160, 120), color="purple")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        cls.png_bytes = buf.getvalue()

        cls.artifact = ImageArtifact(
            data=cls.png_bytes,
            mime_type="image/png",
            width=160,
            height=120,
            format="PNG",
            file_name="flowchart.png",
        )
        cls.preprocessed = preprocess_image(cls.artifact)
        cls.vision_service = VisionService()
        cls.visual_output = cls.vision_service.analyze(cls.preprocessed)
        cls.engine = MultimodalReasoningEngine()
        cls.generator = MultimodalResponseGenerator(default_confidence_threshold=0.5)

    def test_text_only_response_generation(self):
        ctx = build_multimodal_context(
            query="What is the function of attention heads?",
            session_id="sess_text_resp",
        )
        res = self.engine.reason(ctx)
        resp = self.generator.generate_response(res, context=ctx)

        self.assertIsInstance(resp, MultimodalResponse)
        self.assertEqual(resp.modality, ModalityType.TEXT_ONLY)
        self.assertEqual(resp.session_id, "sess_text_resp")
        self.assertEqual(resp.query, "What is the function of attention heads?")
        self.assertIn("function of attention heads", resp.answer)
        self.assertEqual(len(resp.visual_evidence), 0)
        self.assertTrue(resp.grounded)
        self.assertIn("### Textual Analysis", resp.formatted_markdown)
        self.assertIn("Textual Evidence", resp.formatted_markdown)

    def test_image_only_response_generation(self):
        ctx = build_multimodal_context(
            artifact=self.artifact,
            visual_output=self.visual_output,
            session_id="sess_img_resp",
        )
        res = self.engine.reason(ctx)
        resp = self.generator.generate_response(res, context=ctx)

        self.assertIsInstance(resp, MultimodalResponse)
        self.assertEqual(resp.modality, ModalityType.IMAGE_ONLY)
        self.assertEqual(resp.session_id, "sess_img_resp")
        self.assertEqual(resp.query, "")
        self.assertTrue(len(resp.visual_evidence) > 0)
        self.assertTrue(resp.grounded)
        self.assertIn("### Visual Inspection", resp.formatted_markdown)
        self.assertIn("Visual Evidence", resp.formatted_markdown)

    def test_text_and_image_response_generation(self):
        ctx = build_multimodal_context(
            query="Trace the flow direction indicated in the image",
            artifact=self.artifact,
            visual_output=self.visual_output,
            session_id="sess_joint_resp",
        )
        res = self.engine.reason(ctx)
        resp = self.generator.generate_response(res, context=ctx)

        self.assertIsInstance(resp, MultimodalResponse)
        self.assertEqual(resp.modality, ModalityType.TEXT_AND_IMAGE)
        self.assertEqual(resp.session_id, "sess_joint_resp")
        self.assertEqual(resp.query, "Trace the flow direction indicated in the image")
        self.assertTrue(len(resp.visual_evidence) > 0)
        self.assertTrue(resp.grounded)
        self.assertIn("### Multimodal Analysis", resp.formatted_markdown)
        self.assertIn("Textual Evidence", resp.formatted_markdown)
        self.assertIn("Visual Evidence", resp.formatted_markdown)

    def test_evidence_and_provenance_preservation(self):
        prov = EvidenceProvenance(
            source_type="diagram_extractor",
            modality=ModalityType.TEXT_AND_IMAGE,
            source_id="flowchart.png",
            provider="ocr_engine_v1",
            confidence=0.97,
            metadata={"node_count": 4},
        )
        ev_item = UnifiedEvidenceItem(
            description="Decision diamond connected to terminal state",
            modality=ModalityType.TEXT_AND_IMAGE,
            confidence=0.97,
            provenance=prov,
        )

        ctx = build_multimodal_context(
            query="Explain logic flow",
            artifact=self.artifact,
            visual_output=self.visual_output,
            session_id="sess_prov_check",
        )
        ctx.add_evidence(ev_item)

        res = self.engine.reason(ctx)
        resp = self.generator.generate_response(res, context=ctx)

        self.assertIsNotNone(resp.image_metadata)
        prov_records = resp.image_metadata.get("provenance_records", [])
        self.assertTrue(len(prov_records) > 0)
        matching = [r for r in prov_records if r.get("provider") == "ocr_engine_v1"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["confidence"], 0.97)

    def test_confidence_propagation_and_validation(self):
        res = ReasoningResult(
            interpreted_query="Confidence test",
            reasoning_summary="High confidence test deduction",
            confidence=0.92,
            modality=ModalityType.TEXT_ONLY,
            session_id="sess_conf",
            textual_evidence=["Evidence line"],
        )
        resp = self.generator.generate_response(res)

        self.assertEqual(resp.image_metadata.get("confidence"), 0.92)
        self.assertIn("**Confidence Assessment:** 0.92", resp.formatted_markdown)

    def test_insufficient_evidence_low_confidence(self):
        # Result with confidence lower than generator threshold (0.5)
        res = ReasoningResult(
            interpreted_query="Uncertain inquiry",
            reasoning_summary="Tentative interpretation",
            confidence=0.35,
            modality=ModalityType.TEXT_ONLY,
            session_id="sess_low_conf",
            textual_evidence=["minimal evidence"],
        )
        resp = self.generator.generate_response(res)

        self.assertFalse(resp.grounded)
        self.assertIsNotNone(resp.warning_message)
        self.assertIn("Low confidence", resp.warning_message)
        self.assertIn("[Caution: Evidence Limited]", resp.answer)
        self.assertIn("Evidence Warning", resp.formatted_markdown)

    def test_insufficient_evidence_missing_modalities_in_joint(self):
        # Joint request missing visual evidence
        res = ReasoningResult(
            interpreted_query="Query with missing image data",
            reasoning_summary="Incomplete joint analysis",
            confidence=0.95,
            modality=ModalityType.TEXT_AND_IMAGE,
            session_id="sess_missing_vis",
            textual_evidence=["Some text query"],
            visual_evidence=[],  # Missing visual evidence!
        )
        resp = self.generator.generate_response(res)

        self.assertFalse(resp.grounded)
        self.assertIsNotNone(resp.warning_message)
        self.assertIn("missing either visual or textual evidence", resp.warning_message)
        self.assertIn("[Caution: Evidence Limited]", resp.answer)

    def test_serialization_excludes_raw_bytes(self):
        ctx = build_multimodal_context(
            query="Serialization audit",
            artifact=self.artifact,
            visual_output=self.visual_output,
            session_id="sess_serial",
        )
        res = self.engine.reason(ctx)
        resp = self.generator.generate_response(res, context=ctx)
        data = resp.to_dict()

        self.assertIsInstance(data, dict)
        self.assertEqual(data["session_id"], "sess_serial")
        self.assertEqual(data["modality"], "text_and_image")
        self.assertIn("visual_evidence", data)
        self.assertIn("image_metadata", data)
        self.assertNotIn("data", data)
        self.assertNotIn("png_bytes", str(data))

    def test_deterministic_output(self):
        ctx = build_multimodal_context(
            query="Determinism audit",
            artifact=self.artifact,
            visual_output=self.visual_output,
            session_id="sess_det",
        )
        res = self.engine.reason(ctx)

        resp1 = self.generator.generate_response(res)
        resp2 = self.generator.generate_response(res)

        self.assertEqual(resp1.answer, resp2.answer)
        self.assertEqual(resp1.formatted_markdown, resp2.formatted_markdown)
        self.assertEqual(resp1.grounded, resp2.grounded)
        self.assertEqual(resp1.warning_message, resp2.warning_message)
        self.assertEqual(resp1.to_dict(), resp2.to_dict())

    def test_mutable_default_isolation(self):
        resp1 = MultimodalResponse(
            query="Q1",
            answer="A1",
            modality=ModalityType.TEXT_ONLY,
            session_id="s1",
        )
        resp2 = MultimodalResponse(
            query="Q2",
            answer="A2",
            modality=ModalityType.TEXT_ONLY,
            session_id="s2",
        )

        dummy_item = VisualEvidenceItem(description="vis1")
        resp1.visual_evidence.append(dummy_item)

        self.assertEqual(len(resp1.visual_evidence), 1)
        self.assertEqual(len(resp2.visual_evidence), 0)

    def test_convenience_function_integration(self):
        res = ReasoningResult(
            interpreted_query="Direct call query",
            reasoning_summary="Direct call summary",
            confidence=0.88,
            modality=ModalityType.TEXT_ONLY,
            session_id="sess_conv",
            textual_evidence=["Direct text evidence"],
        )
        resp = generate_multimodal_response(res)

        self.assertIsInstance(resp, MultimodalResponse)
        self.assertEqual(resp.session_id, "sess_conv")
        self.assertEqual(resp.modality, ModalityType.TEXT_ONLY)
        self.assertTrue(resp.grounded)

    def test_invalid_reasoning_result_rejection(self):
        with self.assertRaises(TypeError):
            self.generator.generate_response("not_a_reasoning_result")  # type: ignore

        with self.assertRaises(TypeError):
            self.generator.generate_response(None)  # type: ignore


if __name__ == "__main__":
    unittest.main()
