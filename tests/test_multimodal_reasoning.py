"""Unit tests for Cross-Modal Reasoning Foundation (Phase 5 — Day 26 Step 2).
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
)
from multimodal.models import (
    ImageArtifact,
    ModalityType,
    MultimodalRequest,
    MultimodalResponse,
    VisualEvidenceItem,
)
from multimodal.orchestrator import MultimodalOrchestrator
from multimodal.preprocessing import preprocess_image
from multimodal.reasoning import (
    DeterministicReasoningEngine,
    MultimodalReasoningEngine,
    MultimodalReasoningResult,
    MultimodalReasoningService,
    ReasoningEngine,
    ReasoningResult,
)
from multimodal.vision import VisionService


class TestMultimodalReasoning(unittest.TestCase):
    """Tests for cross-modal reasoning foundation, modality dispatch, and evidence provenance."""

    @classmethod
    def setUpClass(cls):
        # Create valid test image bytes and artifact
        img = Image.new("RGB", (140, 90), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        cls.png_bytes = buf.getvalue()

        cls.artifact = ImageArtifact(
            data=cls.png_bytes,
            mime_type="image/png",
            width=140,
            height=90,
            format="PNG",
            file_name="growth_curve.png",
        )
        cls.preprocessed = preprocess_image(cls.artifact)
        cls.vision_service = VisionService()
        cls.visual_output = cls.vision_service.analyze(cls.preprocessed)

    def test_text_only_reasoning(self):
        ctx = build_multimodal_context(
            query="Explain transformer self-attention mechanisms.",
            session_id="sess_text_1",
        )
        engine = MultimodalReasoningEngine()
        result = engine.reason(ctx)

        self.assertIsInstance(result, ReasoningResult)
        self.assertEqual(result.modality, ModalityType.TEXT_ONLY)
        self.assertIn("[Text Reasoning]", result.reasoning_summary)
        self.assertIn("transformer self-attention", result.reasoning_summary)
        self.assertEqual(result.session_id, "sess_text_1")
        self.assertTrue(len(result.reasoning_steps) >= 3)
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(result.interpreted_query, "Explain transformer self-attention mechanisms.")
        self.assertEqual(len(result.visual_evidence), 0)

    def test_image_only_reasoning(self):
        ctx = build_multimodal_context(
            artifact=self.artifact,
            preprocessed=self.preprocessed,
            visual_output=self.visual_output,
            session_id="sess_vis_1",
        )
        engine = MultimodalReasoningEngine()
        result = engine.reason(ctx)

        self.assertIsInstance(result, ReasoningResult)
        self.assertEqual(result.modality, ModalityType.IMAGE_ONLY)
        self.assertIn("[Visual Reasoning]", result.reasoning_summary)
        self.assertTrue(len(result.visual_evidence) > 0)
        self.assertTrue(len(result.combined_evidence) > 0)
        self.assertIn("visual_attributes", result.metadata)
        self.assertEqual(result.session_id, "sess_vis_1")

    def test_text_and_image_reasoning(self):
        query_str = "What trend does this growth curve demonstrate?"
        ctx = build_multimodal_context(
            query=query_str,
            artifact=self.artifact,
            preprocessed=self.preprocessed,
            visual_output=self.visual_output,
            session_id="sess_joint_1",
        )
        engine = MultimodalReasoningEngine()
        result = engine.reason(ctx)

        self.assertIsInstance(result, ReasoningResult)
        self.assertEqual(result.modality, ModalityType.TEXT_AND_IMAGE)
        self.assertIn("[Joint Multimodal Reasoning]", result.reasoning_summary)

        # Confirm BOTH modalities are explicitly reflected in summary, evidence, and metadata
        self.assertIn(query_str, result.reasoning_summary)
        self.assertTrue(len(result.textual_evidence) > 0)
        self.assertTrue(len(result.visual_evidence) > 0)
        self.assertEqual(result.metadata.get("fused_modalities"), ["text", "image"])
        self.assertTrue(len(result.reasoning_steps) >= 4)

    def test_evidence_combination_and_provenance(self):
        custom_prov = EvidenceProvenance(
            source_type="chart_parser",
            modality=ModalityType.TEXT_AND_IMAGE,
            source_id="growth_curve.png",
            provider="mock_detector",
            confidence=0.96,
        )
        custom_ev = UnifiedEvidenceItem(
            description="Upward sloping trajectory in quadrant 1",
            modality=ModalityType.TEXT_AND_IMAGE,
            confidence=0.96,
            provenance=custom_prov,
        )

        ctx = build_multimodal_context(
            query="Verify slope",
            artifact=self.artifact,
            visual_output=self.visual_output,
            session_id="sess_prov",
        )
        ctx.add_evidence(custom_ev)

        engine = MultimodalReasoningEngine()
        result = engine.reason(ctx)

        self.assertIn(custom_ev, result.combined_evidence)
        matching = [ev for ev in result.combined_evidence if ev.description == "Upward sloping trajectory in quadrant 1"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].provenance.provider, "mock_detector")
        self.assertEqual(matching[0].provenance.confidence, 0.96)

    def test_confidence_validation(self):
        # ReasoningResult confidence must be in range [0.0, 1.0]
        with self.assertRaises(ValueError):
            ReasoningResult(
                reasoning_summary="Test",
                confidence=1.5,
                session_id="s1",
            )
        with self.assertRaises(ValueError):
            ReasoningResult(
                reasoning_summary="Test",
                confidence=-0.1,
                session_id="s1",
            )

    def test_conversion_to_multimodal_response(self):
        ctx = build_multimodal_context(
            query="Analyze chart",
            artifact=self.artifact,
            visual_output=self.visual_output,
            session_id="resp_sess",
        )
        engine = MultimodalReasoningEngine()
        result = engine.reason(ctx)

        resp = result.to_multimodal_response()
        self.assertIsInstance(resp, MultimodalResponse)
        self.assertEqual(resp.session_id, "resp_sess")
        self.assertEqual(resp.modality, ModalityType.TEXT_AND_IMAGE)
        self.assertTrue(resp.grounded)
        self.assertTrue(len(resp.visual_evidence) > 0)
        self.assertIn("### Multimodal", resp.formatted_markdown)

    def test_deterministic_output(self):
        ctx = build_multimodal_context(
            query="Repeatability test",
            artifact=self.artifact,
            visual_output=self.visual_output,
            session_id="repeat_sess",
        )
        engine = MultimodalReasoningEngine()

        res1 = engine.reason(ctx)
        res2 = engine.reason(ctx)

        self.assertEqual(res1.reasoning_summary, res2.reasoning_summary)
        self.assertEqual(res1.reasoning_steps, res2.reasoning_steps)
        self.assertEqual(res1.confidence, res2.confidence)
        self.assertEqual(res1.to_dict(), res2.to_dict())

    def test_serialization_excludes_raw_bytes(self):
        ctx = build_multimodal_context(
            query="Serialization check",
            artifact=self.artifact,
            visual_output=self.visual_output,
            session_id="serial_sess",
        )
        engine = MultimodalReasoningEngine()
        res = engine.reason(ctx)
        d = res.to_dict()

        self.assertIsInstance(d, dict)
        self.assertEqual(d["session_id"], "serial_sess")
        self.assertIn("reasoning_summary", d)
        self.assertIn("combined_evidence", d)
        self.assertIn("visual_evidence", d)
        self.assertNotIn("data", d)

    def test_mutable_default_isolation(self):
        res1 = ReasoningResult(reasoning_summary="R1", session_id="s1")
        res2 = ReasoningResult(reasoning_summary="R2", session_id="s2")

        res1.reasoning_steps.append("Step 1")
        res1.metadata["tag"] = "test"

        self.assertEqual(len(res1.reasoning_steps), 1)
        self.assertEqual(len(res2.reasoning_steps), 0)
        self.assertNotIn("tag", res2.metadata)

    def test_empty_and_invalid_context_rejection(self):
        engine = MultimodalReasoningEngine()

        with self.assertRaises(TypeError):
            engine.reason("invalid_string_context")  # type: ignore

        # Context with invalid session ID
        bad_ctx = object.__new__(MultimodalContext)
        bad_ctx.session_id = "   "
        bad_ctx.modality = ModalityType.TEXT_ONLY
        bad_ctx.text_context = TextualContext(raw_query="hello")
        bad_ctx.visual_context = VisualContext()
        bad_ctx.evidence_items = []
        bad_ctx.metadata = {}

        with self.assertRaises(ValueError):
            engine.reason(bad_ctx)

    def test_custom_reasoning_engine_extensibility(self):
        class CustomAnalyticalEngine(ReasoningEngine):
            def reason(self, context: MultimodalContext) -> ReasoningResult:
                return ReasoningResult(
                    reasoning_summary="Custom analytical deduction executed.",
                    modality=context.modality,
                    reasoning_steps=["Custom step 1", "Custom step 2"],
                    confidence=0.99,
                    session_id=context.session_id,
                    engine_name="custom_analytical_engine",
                )

        custom_engine = CustomAnalyticalEngine()
        ctx = build_multimodal_context(query="Sample query", session_id="cust_sess")
        result = custom_engine.reason(ctx)

        self.assertEqual(result.engine_name, "custom_analytical_engine")
        self.assertEqual(result.reasoning_summary, "Custom analytical deduction executed.")
        self.assertEqual(result.confidence, 0.99)

    def test_orchestrator_hook_integration(self):
        engine = MultimodalReasoningEngine()
        hook = engine.as_orchestrator_hook(vision_service=self.vision_service)

        orchestrator = MultimodalOrchestrator(reasoning_engine_hook=hook)

        # Process a joint text and image request through the full orchestrator
        req = MultimodalRequest(
            query="Analyze quarterly performance metrics",
            images=[self.artifact],
            session_id="hook_sess_1",
        )
        resp = orchestrator.process(req)

        self.assertIsInstance(resp, MultimodalResponse)
        self.assertEqual(resp.modality, ModalityType.TEXT_AND_IMAGE)
        self.assertIn("[Joint Multimodal Reasoning]", resp.answer)
        self.assertIn("quarterly performance metrics", resp.answer)
        self.assertTrue(len(resp.visual_evidence) > 0)
        self.assertEqual(resp.session_id, "hook_sess_1")


if __name__ == "__main__":
    unittest.main()
