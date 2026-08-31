"""Unit tests for Phase 5 Multimodal AI Assistant orchestrator and pipeline routing.
"""

import sys
import os
import unittest

# Ensure src is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from multimodal.models import (
    AmbiguityDetails,
    ImageArtifact,
    ModalityType,
    MultimodalRequest,
    MultimodalResponse,
    VisualEvidenceItem,
)
from multimodal.orchestrator import MultimodalOrchestrator


class TestMultimodalOrchestrator(unittest.TestCase):
    """Tests for MultimodalOrchestrator pipeline and modality dispatch."""

    def setUp(self):
        self.valid_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10"
        self.artifact = ImageArtifact(
            data=self.valid_bytes,
            mime_type="image/png",
            width=640,
            height=480,
            format="PNG",
            file_name="chart.png",
        )
        self.orchestrator = MultimodalOrchestrator()

    def test_text_only_routing(self):
        req = MultimodalRequest(query="Explain attention mechanisms", session_id="sess_text")
        resp = self.orchestrator.process(req)

        self.assertIsInstance(resp, MultimodalResponse)
        self.assertEqual(resp.query, "Explain attention mechanisms")
        self.assertEqual(resp.modality, ModalityType.TEXT_ONLY)
        self.assertTrue(resp.grounded)
        self.assertEqual(resp.session_id, "sess_text")
        self.assertIn("Explain attention mechanisms", resp.answer)

    def test_image_only_routing(self):
        req = MultimodalRequest(images=[self.artifact], session_id="sess_img")
        resp = self.orchestrator.process(req)

        self.assertIsInstance(resp, MultimodalResponse)
        self.assertEqual(resp.modality, ModalityType.IMAGE_ONLY)
        self.assertTrue(resp.grounded)
        self.assertEqual(resp.session_id, "sess_img")
        self.assertIsNotNone(resp.image_metadata)
        self.assertEqual(resp.image_metadata.get("format"), "PNG")
        self.assertEqual(resp.image_metadata.get("width"), 640)
        self.assertEqual(len(resp.visual_evidence), 1)

    def test_text_and_image_routing(self):
        req = MultimodalRequest(
            query="What are the axis labels on this graph?",
            images=[self.artifact],
            session_id="sess_joint",
        )
        resp = self.orchestrator.process(req)

        self.assertIsInstance(resp, MultimodalResponse)
        self.assertEqual(resp.query, "What are the axis labels on this graph?")
        self.assertEqual(resp.modality, ModalityType.TEXT_AND_IMAGE)
        self.assertTrue(resp.grounded)
        self.assertEqual(resp.session_id, "sess_joint")
        self.assertIsNotNone(resp.image_metadata)
        self.assertEqual(len(resp.visual_evidence), 1)

    def test_dict_payload_coercion(self):
        dict_payload = {
            "query": "Summarize findings",
            "session_id": "dict_session",
            "temperature": 0.2,
        }
        resp = self.orchestrator.process(dict_payload)

        self.assertIsInstance(resp, MultimodalResponse)
        self.assertEqual(resp.query, "Summarize findings")
        self.assertEqual(resp.modality, ModalityType.TEXT_ONLY)
        self.assertEqual(resp.session_id, "dict_session")

    def test_invalid_payload_type_rejection(self):
        with self.assertRaises(TypeError):
            self.orchestrator.process(12345)  # type: ignore

    def test_empty_request_rejection(self):
        with self.assertRaises(ValueError):
            self.orchestrator.process(MultimodalRequest(query=None, images=[]))

    def test_process_query_convenience_method(self):
        resp = self.orchestrator.process_query(
            query="What is in this diagram?",
            images=[self.artifact],
            session_id="convenience_sess",
        )
        self.assertEqual(resp.modality, ModalityType.TEXT_AND_IMAGE)
        self.assertEqual(resp.session_id, "convenience_sess")

    def test_ambiguity_hook_integration(self):
        def mock_ambiguity_detector(req: MultimodalRequest) -> AmbiguityDetails:
            if req.query == "What is that?":
                return AmbiguityDetails(
                    is_ambiguous=True,
                    clarification_question="Please specify which part of the image you want explained.",
                    missing_aspects=["region_of_interest"],
                )
            return AmbiguityDetails(is_ambiguous=False)

        orch_with_ambiguity = MultimodalOrchestrator(ambiguity_detector_hook=mock_ambiguity_detector)

        amb_req = MultimodalRequest(query="What is that?", images=[self.artifact])
        resp = orch_with_ambiguity.process(amb_req)

        self.assertTrue(resp.ambiguity.is_ambiguous)
        self.assertIn("Please specify which part", resp.answer)
        self.assertEqual(resp.ambiguity.missing_aspects, ["region_of_interest"])

    def test_reasoning_hook_integration(self):
        def mock_reasoning_engine(req: MultimodalRequest) -> MultimodalResponse:
            return MultimodalResponse(
                query=req.query or "",
                answer="Custom reasoning response from Day 26 engine",
                modality=req.modality,
                grounded=True,
                session_id=req.session_id,
            )

        orch_with_reasoning = MultimodalOrchestrator(reasoning_engine_hook=mock_reasoning_engine)
        req = MultimodalRequest(query="Explain chart", images=[self.artifact])
        resp = orch_with_reasoning.process(req)

        self.assertEqual(resp.answer, "Custom reasoning response from Day 26 engine")

    def test_grounding_validator_hook_integration(self):
        def mock_grounding_validator(resp: MultimodalResponse) -> MultimodalResponse:
            resp.grounded = True
            resp.visual_evidence.append(
                VisualEvidenceItem(description="Verified high-contrast title banner", confidence=0.99)
            )
            return resp

        orch_with_grounding = MultimodalOrchestrator(grounding_validator_hook=mock_grounding_validator)
        req = MultimodalRequest(query="Analyze", images=[self.artifact])
        resp = orch_with_grounding.process(req)

        self.assertEqual(len(resp.visual_evidence), 2)
        self.assertEqual(resp.visual_evidence[-1].description, "Verified high-contrast title banner")


if __name__ == "__main__":
    unittest.main()
