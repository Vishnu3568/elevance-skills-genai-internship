"""Unit tests for Phase 5 Multimodal AI Assistant vision model integration (Day 25 Step 3).
"""

import io
import os
import sys
from typing import Any
import unittest
from PIL import Image

# Ensure src is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from multimodal.models import (
    ImageArtifact,
    ModalityType,
    MultimodalRequest,
    MultimodalResponse,
    VisualEvidenceItem,
)
from multimodal.orchestrator import MultimodalOrchestrator
from multimodal.preprocessing import PreprocessedImage, preprocess_image
from multimodal.vision import (
    DeterministicMockVisionProvider,
    GeminiVisionProvider,
    StructuredVisualOutput,
    VisionModelProvider,
    VisionService,
)


class TestVisionModelIntegration(unittest.TestCase):
    """Tests for vision model abstraction, deterministic mock, and orchestrator integration."""

    @classmethod
    def setUpClass(cls):
        # Create valid test image bytes
        img = Image.new("RGB", (240, 160), color=(80, 120, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        cls.png_bytes = buf.getvalue()

        cls.artifact = ImageArtifact(
            data=cls.png_bytes,
            mime_type="image/png",
            width=240,
            height=160,
            format="PNG",
            file_name="test_chart.png",
        )
        cls.preprocessed = preprocess_image(cls.artifact)

    def test_preprocessed_image_to_visual_result(self):
        service = VisionService()
        output = service.analyze(self.preprocessed)

        self.assertIsInstance(output, StructuredVisualOutput)
        self.assertEqual(output.provider_name, "deterministic_mock")
        self.assertEqual(output.confidence, 1.0)
        self.assertIn("PNG", output.scene_description)
        self.assertIn("240x160", output.scene_description)
        self.assertTrue(len(output.detected_objects) > 0)
        self.assertTrue(len(output.spatial_observations) > 0)
        self.assertEqual(output.visual_attributes["orientation"], "landscape")

    def test_structured_visual_output_evidence_conversion(self):
        output = StructuredVisualOutput(
            scene_description="Bar chart showing quarterly performance",
            detected_objects=["bar_chart", "legend", "x_axis"],
            visible_text=["Q1", "Q2", "Revenue"],
            visual_attributes={"palette": "blues"},
            spatial_observations=["Top quadrant holds legend"],
            confidence=0.95,
            provider_name="test_mock",
        )

        evidence_items = output.to_visual_evidence_items()
        self.assertTrue(len(evidence_items) >= 4)
        self.assertTrue(all(isinstance(item, VisualEvidenceItem) for item in evidence_items))

        descriptions = [item.description for item in evidence_items]
        self.assertIn("Bar chart showing quarterly performance", descriptions)
        self.assertTrue(any("Visible text: 'Q1'" in d for d in descriptions))
        self.assertTrue(any("Detected entity/object: bar_chart" in d for d in descriptions))

        # Test dictionary serialization
        d = output.to_dict()
        self.assertEqual(d["confidence"], 0.95)
        self.assertEqual(d["provider_name"], "test_mock")

    def test_custom_provider_invocation(self):
        class CustomTestProvider(VisionModelProvider):
            def analyze(self, image: PreprocessedImage) -> StructuredVisualOutput:
                return StructuredVisualOutput(
                    scene_description="Custom domain scene description",
                    detected_objects=["custom_entity"],
                    confidence=0.99,
                    provider_name="custom_test_provider",
                )

        custom_service = VisionService(provider=CustomTestProvider())
        output = custom_service.analyze(self.preprocessed)

        self.assertEqual(output.provider_name, "custom_test_provider")
        self.assertEqual(output.scene_description, "Custom domain scene description")
        self.assertEqual(output.detected_objects, ["custom_entity"])

    def test_direct_image_artifact_analysis_auto_preprocesses(self):
        # Passing an ImageArtifact directly should automatically trigger preprocessing
        service = VisionService()
        output = service.analyze(self.artifact)

        self.assertIsInstance(output, StructuredVisualOutput)
        self.assertEqual(output.visual_attributes["width"], 240)
        self.assertEqual(output.visual_attributes["height"], 160)

    def test_invalid_input_type_rejection(self):
        service = VisionService()
        with self.assertRaises(TypeError):
            service.analyze(b"raw bytes instead of PreprocessedImage or ImageArtifact")  # type: ignore

    def test_backend_failure_handling(self):
        class FailingProvider(VisionModelProvider):
            def analyze(self, image: PreprocessedImage) -> StructuredVisualOutput:
                raise RuntimeError("Hardware accelerator timeout")

        failing_service = VisionService(provider=FailingProvider())
        with self.assertRaises(RuntimeError) as ctx:
            failing_service.analyze(self.preprocessed)
        self.assertIn("Hardware accelerator timeout", str(ctx.exception))

    def test_empty_result_handling(self):
        empty_output = StructuredVisualOutput(
            scene_description="",
            detected_objects=[],
            visible_text=[],
            visual_attributes={},
            spatial_observations=[],
        )
        evidence = empty_output.to_visual_evidence_items()
        self.assertEqual(evidence, [])

    def test_metadata_propagation(self):
        service = VisionService()
        output = service.analyze(self.preprocessed)

        self.assertEqual(output.metadata.get("file_name"), "test_chart.png")
        self.assertEqual(output.metadata.get("original_width"), 240)
        self.assertEqual(output.metadata.get("format"), "PNG")

    def test_orchestrator_preprocessing_vision_integration(self):
        vision_service = VisionService()
        orchestrator = MultimodalOrchestrator(vision_service=vision_service)

        req = MultimodalRequest(images=[self.artifact], session_id="test_vision_sess")
        resp = orchestrator.process(req)

        self.assertIsInstance(resp, MultimodalResponse)
        self.assertEqual(resp.modality, ModalityType.IMAGE_ONLY)
        self.assertIn("[Visual Understanding]", resp.answer)
        self.assertTrue(len(resp.visual_evidence) > 0)
        self.assertIn("visual_canvas", resp.visual_evidence[1].description)

    def test_text_and_image_does_not_perform_day26_reasoning(self):
        # Verify text+image requests ONLY extract visual information and do not perform LLM fusion reasoning
        vision_service = VisionService()
        orchestrator = MultimodalOrchestrator(vision_service=vision_service)

        req = MultimodalRequest(
            query="Which product had the highest sales growth?",
            images=[self.artifact],
            session_id="joint_sess",
        )
        resp = orchestrator.process(req)

        self.assertEqual(resp.modality, ModalityType.TEXT_AND_IMAGE)
        # Verify it reports extracted features without synthesizing ungrounded business answers
        self.assertIn("[Visual Understanding] Extracted visual features", resp.answer)
        self.assertIn("Which product had the highest sales growth?", resp.answer)
        self.assertTrue(len(resp.visual_evidence) > 0)

    def test_gemini_vision_provider_missing_key_rejection(self):
        # GeminiVisionProvider should raise ValueError if API key is empty
        provider = GeminiVisionProvider(api_key="")
        with self.assertRaises(ValueError) as ctx:
            provider.analyze(self.preprocessed)
        self.assertIn("GOOGLE_API_KEY is not configured", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
