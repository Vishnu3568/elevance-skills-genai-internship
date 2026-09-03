"""Unit tests for Multimodal Follow-Up Questions and Context Resolution (Phase 5 — Day 27 Step 3).
"""

import os
import sys
import unittest

# Ensure src is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from multimodal.context import build_multimodal_context
from multimodal.context_retention import (
    MultimodalContextRetriever,
    RetainedConversationContext,
)
from multimodal.conversation import (
    MultimodalConversationSession,
    MultimodalSessionManager,
)
from multimodal.followup import (
    FollowUpResolution,
    MultimodalFollowUpResolver,
    resolve_multimodal_followup,
)
from multimodal.models import (
    ImageArtifact,
    ModalityType,
    MultimodalResponse,
    VisualEvidenceItem,
)
from multimodal.vision import StructuredVisualOutput


class TestMultimodalFollowUp(unittest.TestCase):
    """Tests for Day 27 Step 3 follow-up question and multimodal context resolution."""

    def setUp(self):
        self.resolver = MultimodalFollowUpResolver()

        # Dummy byte payload for artifact
        self.dummy_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 50
        self.artifact = ImageArtifact(
            data=self.dummy_bytes,
            mime_type="image/png",
            width=800,
            height=600,
            format="PNG",
            file_name="network_arch.png",
        )

        self.visual_output = StructuredVisualOutput(
            scene_description="Diagram of a neural-network architecture",
            detected_objects=["blue component", "encoder_layer", "decoder_layer"],
            visible_text=["INPUT", "ATTENTION", "OUTPUT"],
            visual_attributes={"color_scheme": "blue_accent", "complexity": "high"},
        )

        self.context = build_multimodal_context(
            query="What is shown in this image?",
            artifact=self.artifact,
            visual_output=self.visual_output,
            session_id="sess_followup",
        )

    def test_followup_after_text_only_conversation(self):
        """1. Verify follow-up pronoun resolution after a text-only turn."""
        session = MultimodalConversationSession(session_id="sess_text_followup")
        resp = MultimodalResponse(
            query="Explain transformer self-attention",
            answer="Transformer self-attention computes query-key dot products to dynamically weigh sequence representations.",
            modality=ModalityType.TEXT_ONLY,
            session_id="sess_text_followup",
        )
        session.add_turn(query="Explain transformer self-attention", response=resp)

        # User asks: "What is its computational complexity?"
        res = self.resolver.resolve_followup("What is its computational complexity?", session=session)

        self.assertTrue(res.is_followup)
        self.assertIn("its", res.detected_references)
        self.assertIn("computational complexity", res.resolved_query)
        self.assertIn("transformer self-attention", res.resolved_query.lower())
        self.assertEqual(res.referenced_modality, ModalityType.TEXT_ONLY)

    def test_followup_after_image_only_conversation(self):
        """2. Verify follow-up resolution after an image-only upload turn."""
        session = MultimodalConversationSession(session_id="sess_img_followup")
        resp = MultimodalResponse(
            query="",
            answer="The image shows a neural-network architecture with multi-layer perceptrons.",
            modality=ModalityType.IMAGE_ONLY,
            session_id="sess_img_followup",
        )
        session.add_turn(query="", response=resp, context=self.context)

        # Follow-up: "How does it work?"
        res = self.resolver.resolve_followup("How does it work?", session=session)

        self.assertTrue(res.is_followup)
        self.assertIn("it", res.detected_references)
        self.assertIn("neural-network architecture", res.resolved_query.lower())
        self.assertIn("How does", res.resolved_query)
        self.assertEqual(res.referenced_modality, ModalityType.IMAGE_ONLY)

    def test_followup_after_text_and_image_conversation(self):
        """3. Verify follow-up resolution after a joint text + image turn."""
        session = MultimodalConversationSession(session_id="sess_joint")
        resp = MultimodalResponse(
            query="What is shown in this image?",
            answer="The image shows a neural-network architecture with deep transformer layers.",
            modality=ModalityType.TEXT_AND_IMAGE,
            session_id="sess_joint",
        )
        session.add_turn(query="What is shown in this image?", response=resp, context=self.context)

        # Follow-up: "Can you explain it in detail?"
        res = self.resolver.resolve_followup("Can you explain it in detail?", session=session)

        self.assertTrue(res.is_followup)
        self.assertIn("neural-network architecture", res.resolved_query.lower())
        self.assertIn("explain", res.resolved_query.lower())

    def test_pronoun_reference_resolution(self):
        """4. Verify pronoun resolution for 'it', 'this', 'that', and 'its'."""
        session = MultimodalConversationSession(session_id="sess_pronouns")
        resp = MultimodalResponse(
            query="Explain convolutional networks",
            answer="Convolutional networks apply localized kernels for spatial pattern extraction.",
            modality=ModalityType.TEXT_ONLY,
            session_id="sess_pronouns",
        )
        session.add_turn(query="Explain convolutional networks", response=resp)

        # "How does it work?"
        res_it = self.resolver.resolve_followup("How does it work?", session=session)
        self.assertTrue(res_it.is_followup)
        self.assertIn("convolutional networks", res_it.resolved_query.lower())

        # "Why is that?"
        res_that = self.resolver.resolve_followup("Why is that?", session=session)
        self.assertTrue(res_that.is_followup)
        self.assertIn("convolutional networks", res_that.resolved_query.lower())

        # "What does this do?"
        res_this = self.resolver.resolve_followup("What does this do?", session=session)
        self.assertTrue(res_this.is_followup)
        self.assertIn("convolutional networks", res_this.resolved_query.lower())

    def test_previous_image_and_diagram_references(self):
        """5. Verify references to 'this image', 'the previous image', and 'the diagram'."""
        session = MultimodalConversationSession(session_id="sess_vis_ref")
        resp = MultimodalResponse(
            query="Explain this diagram",
            answer="The diagram illustrates a neural-network architecture.",
            modality=ModalityType.TEXT_AND_IMAGE,
            session_id="sess_vis_ref",
        )
        session.add_turn(query="Explain this diagram", response=resp, context=self.context)

        # Reference: "What is shown in the previous image?"
        res_img = self.resolver.resolve_followup("What is shown in the previous image?", session=session)
        self.assertTrue(res_img.is_followup)
        self.assertIn("the previous image", res_img.detected_references)
        self.assertIn("neural-network architecture", res_img.resolved_query.lower())

        # Reference: "Explain the diagram"
        res_diag = self.resolver.resolve_followup("Explain the diagram", session=session)
        self.assertTrue(res_diag.is_followup)
        self.assertIn("the diagram", res_diag.detected_references)
        self.assertIn("neural-network architecture", res_diag.resolved_query.lower())

    def test_previous_answer_and_component_references(self):
        """6. Verify references to 'the previous answer' and 'the blue component'."""
        session = MultimodalConversationSession(session_id="sess_ans_comp")
        resp = MultimodalResponse(
            query="Inspect the diagram",
            answer="The diagram illustrates a neural-network architecture.",
            modality=ModalityType.TEXT_AND_IMAGE,
            session_id="sess_ans_comp",
        )
        session.add_turn(query="Inspect the diagram", response=resp, context=self.context)

        # "What does the blue component do?"
        res_comp = self.resolver.resolve_followup("What does the blue component do?", session=session)
        self.assertTrue(res_comp.is_followup)
        self.assertIn("the blue component", res_comp.detected_references)
        self.assertIn("blue component", res_comp.resolved_query)
        self.assertIn("neural-network architecture", res_comp.resolved_query.lower())

        # "Can you elaborate on the previous answer?"
        res_ans = self.resolver.resolve_followup("Can you elaborate on the previous answer?", session=session)
        self.assertTrue(res_ans.is_followup)
        self.assertIn("the previous answer", res_ans.detected_references)
        self.assertIn("neural-network architecture", res_ans.resolved_query.lower())

    def test_insufficient_context_fallback(self):
        """7. Verify graceful fallback to the raw query when there is insufficient context."""
        # Case A: Empty session
        empty_session = MultimodalConversationSession(session_id="sess_empty")
        res_empty = self.resolver.resolve_followup("How does it work?", session=empty_session)
        self.assertFalse(res_empty.is_followup)
        self.assertEqual(res_empty.resolved_query, "How does it work?")
        self.assertEqual(res_empty.resolution_strategy, "no_history_fallback")

        # Case B: None session and None context
        res_none = self.resolver.resolve_followup("Explain this diagram", context=None, session=None)
        self.assertFalse(res_none.is_followup)
        self.assertEqual(res_none.resolved_query, "Explain this diagram")

        # Case C: Standalone, independent question with existing history
        session = MultimodalConversationSession(session_id="sess_with_hist")
        resp = MultimodalResponse(
            query="What is AI?",
            answer="Artificial Intelligence is the simulation of human intelligence.",
            modality=ModalityType.TEXT_ONLY,
            session_id="sess_with_hist",
        )
        session.add_turn(query="What is AI?", response=resp)

        res_indep = self.resolver.resolve_followup("What is the speed of light?", session=session)
        self.assertFalse(res_indep.is_followup)
        self.assertEqual(res_indep.resolved_query, "What is the speed of light?")
        self.assertEqual(res_indep.resolution_strategy, "independent_query")

    def test_session_isolation(self):
        """8. Verify session A context cannot resolve or leak into session B."""
        manager = MultimodalSessionManager()
        sess_a = manager.get_or_create_session("session_alpha")
        sess_b = manager.get_or_create_session("session_beta")

        resp_a = MultimodalResponse(
            query="What is in the image?",
            answer="The image shows a neural-network architecture.",
            modality=ModalityType.TEXT_AND_IMAGE,
            session_id="session_alpha",
        )
        sess_a.add_turn(query="What is in the image?", response=resp_a, context=self.context)

        # Resolve against Session A -> should resolve to neural-network architecture
        res_a = self.resolver.resolve_followup("How does it work?", session=sess_a)
        self.assertTrue(res_a.is_followup)
        self.assertIn("neural-network architecture", res_a.resolved_query.lower())

        # Resolve against Session B (empty) -> must NOT use Session A's context
        res_b = self.resolver.resolve_followup("How does it work?", session=sess_b)
        self.assertFalse(res_b.is_followup)
        self.assertEqual(res_b.resolved_query, "How does it work?")
        self.assertNotIn("neural-network architecture", res_b.resolved_query.lower())

    def test_retained_visual_context(self):
        """9. Verify retained visual metadata is attached to FollowUpResolution."""
        session = MultimodalConversationSession(session_id="sess_vis_meta")
        resp = MultimodalResponse(
            query="Analyze the diagram",
            answer="The diagram illustrates a neural-network architecture.",
            modality=ModalityType.TEXT_AND_IMAGE,
            session_id="sess_vis_meta",
        )
        session.add_turn(query="Analyze the diagram", response=resp, context=self.context)

        res = self.resolver.resolve_followup("What does the blue component do?", session=session)
        self.assertIsNotNone(res.retained_visual_summary)
        self.assertEqual(
            res.retained_visual_summary.get("scene_description"),
            "Diagram of a neural-network architecture",
        )
        self.assertIn("blue component", res.retained_visual_summary.get("detected_objects", []))
        self.assertEqual(res.retained_visual_summary.get("image_metadata", {}).get("format"), "PNG")

    def test_no_raw_image_bytes_in_resolution(self):
        """10. Verify FollowUpResolution dictionary contains zero raw image bytes."""
        session = MultimodalConversationSession(session_id="sess_bytes_chk")
        resp = MultimodalResponse(
            query="Analyze the diagram",
            answer="The diagram illustrates a neural-network architecture.",
            modality=ModalityType.TEXT_AND_IMAGE,
            session_id="sess_bytes_chk",
        )
        session.add_turn(query="Analyze the diagram", response=resp, context=self.context)

        res = resolve_multimodal_followup("How does it work?", session=session)
        d = res.to_dict()
        d_str = str(d)

        self.assertNotIn("data", d["retained_visual_summary"]["image_metadata"])
        self.assertNotIn("b'\\x89PNG", d_str)
        self.assertNotIn("b'\\x00", d_str)
        self.assertEqual(d["target_topic"], "neural-network architecture")


if __name__ == "__main__":
    unittest.main()
