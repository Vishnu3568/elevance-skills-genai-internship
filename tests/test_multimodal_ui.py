"""Unit and Component Tests for Multimodal Streamlit UI (Phase 5 — Day 29 Step 1).

Tests the Streamlit UI components, MultimodalAssistantService adapter, request construction,
response rendering, evidence visualization, ambiguity/missing info presentation,
session isolation, and raw image byte protection.
"""

import io
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import multimodal_main
from src.multimodal import (
    FallbackAction,
    FallbackDecision,
    ModalityType,
    MultimodalAssistantService,
    MultimodalResponse,
    MultimodalUIResult,
    VisualEvidenceItem,
)


class TestMultimodalUI(unittest.TestCase):
    """Test suite for Multimodal Streamlit UI and MultimodalAssistantService."""

    @classmethod
    def setUpClass(cls):
        """Create sample in-memory image bytes."""
        img = Image.new("RGB", (128, 64), color="darkblue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        cls.sample_png_bytes = buf.getvalue()

    def setUp(self):
        """Clear cache and initialize fresh service for each test."""
        clear_cache = getattr(multimodal_main.initialize_multimodal_service, "clear", None)
        if clear_cache:
            clear_cache()
        self.service = MultimodalAssistantService()

    # -------------------------------------------------------------------------
    # 1. Module & Symbol Inspections
    # -------------------------------------------------------------------------

    def test_01_module_exports(self):
        """Verify that multimodal_main exports expected UI and service functions."""
        self.assertTrue(hasattr(multimodal_main, "initialize_multimodal_service"))
        self.assertTrue(hasattr(multimodal_main, "render_multimodal_response"))
        self.assertTrue(hasattr(multimodal_main, "main"))

    def test_02_service_initialization(self):
        """Verify initialize_multimodal_service returns a valid cached service."""
        service, error = multimodal_main.initialize_multimodal_service()
        self.assertIsNotNone(service)
        self.assertIsNone(error)
        self.assertIsInstance(service, MultimodalAssistantService)

    # -------------------------------------------------------------------------
    # 2. Text-Only Interaction Flow
    # -------------------------------------------------------------------------

    def test_03_text_only_interaction(self):
        """Verify text-only user input creates valid request, response, and turn."""
        res = self.service.process_interaction(
            session_id="test_sess_text",
            query="Explain attention mechanisms",
        )
        self.assertIsInstance(res, MultimodalUIResult)
        self.assertEqual(res.modality, ModalityType.TEXT_ONLY)
        self.assertEqual(res.turn_index, 1)
        self.assertTrue(res.is_proceed)
        self.assertIn("attention", res.response.query.lower())
        self.assertIsNotNone(res.response.answer)

    # -------------------------------------------------------------------------
    # 3. Image-Only Interaction Flow
    # -------------------------------------------------------------------------

    def test_04_image_only_interaction(self):
        """Verify image-only upload without text query creates valid image-only response."""
        res = self.service.process_interaction(
            session_id="test_sess_image",
            image_bytes=self.sample_png_bytes,
            file_name="sample.png",
        )
        self.assertIsInstance(res, MultimodalUIResult)
        self.assertEqual(res.modality, ModalityType.IMAGE_ONLY)
        self.assertEqual(res.turn_index, 1)
        self.assertIsNotNone(res.response.answer)

    # -------------------------------------------------------------------------
    # 4. Text + Image Interaction Flow
    # -------------------------------------------------------------------------

    def test_05_text_and_image_interaction(self):
        """Verify joint text and image input creates joint request with visual evidence."""
        res = self.service.process_interaction(
            session_id="test_sess_joint",
            query="What is shown in this chart?",
            image_bytes=self.sample_png_bytes,
            file_name="chart.png",
        )
        self.assertIsInstance(res, MultimodalUIResult)
        self.assertEqual(res.modality, ModalityType.TEXT_AND_IMAGE)
        self.assertEqual(res.turn_index, 1)
        self.assertTrue(res.is_proceed)
        self.assertGreater(len(res.response.visual_evidence), 0)

    # -------------------------------------------------------------------------
    # 5. Missing Information Fallback Detection
    # -------------------------------------------------------------------------

    def test_06_missing_information_fallback(self):
        """Verify visual query without image attachment triggers missing information fallback."""
        res = self.service.process_interaction(
            session_id="test_sess_missing",
            query="What is the color of the second chart in this image?",
            # No image bytes provided
        )
        self.assertIsInstance(res, MultimodalUIResult)
        self.assertTrue(res.is_missing_information)
        self.assertEqual(res.fallback_decision.action, FallbackAction.REQUEST_MISSING_INFORMATION)
        self.assertIn("image", res.response.answer.lower())

    # -------------------------------------------------------------------------
    # 6. Ambiguity Fallback Detection
    # -------------------------------------------------------------------------

    def test_07_ambiguity_fallback(self):
        """Verify ambiguous request triggers clarification fallback decision."""
        # Query matching vague query pattern
        res = self.service.process_interaction(
            session_id="test_sess_ambig",
            query="What is this?",
        )
        self.assertIsInstance(res, MultimodalUIResult)
        self.assertTrue(res.is_clarification)
        self.assertEqual(res.fallback_decision.action, FallbackAction.ASK_CLARIFICATION)

    # -------------------------------------------------------------------------
    # 7. Session Isolation & Reset Functionality
    # -------------------------------------------------------------------------

    def test_08_session_isolation_and_clear(self):
        """Verify session turns are isolated and clear_session resets only target session."""
        self.service.process_interaction(session_id="sess_A", query="Question 1")
        self.service.process_interaction(session_id="sess_A", query="Question 2")
        self.service.process_interaction(session_id="sess_B", query="Question 1")

        sess_a = self.service.get_or_create_session("sess_A")
        sess_b = self.service.get_or_create_session("sess_B")

        self.assertEqual(sess_a.get_turn_count(), 2)
        self.assertEqual(sess_b.get_turn_count(), 1)

        # Clear session A
        cleared = self.service.clear_session("sess_A")
        self.assertTrue(cleared)

        sess_a_new = self.service.get_or_create_session("sess_A")
        self.assertEqual(sess_a_new.get_turn_count(), 0)
        self.assertEqual(sess_b.get_turn_count(), 1)  # Session B unaffected

    # -------------------------------------------------------------------------
    # 8. Multi-Turn Follow-Up Query Resolution
    # -------------------------------------------------------------------------

    def test_09_multiturn_followup_resolution(self):
        """Verify multi-turn interaction tracks conversation turns and resolves follow-ups."""
        res1 = self.service.process_interaction(
            session_id="sess_followup",
            query="Analyze the neural network architecture",
            image_bytes=self.sample_png_bytes,
            file_name="arch.png",
        )
        self.assertEqual(res1.turn_index, 1)

        res2 = self.service.process_interaction(
            session_id="sess_followup",
            query="Why does it use that?",
        )
        self.assertEqual(res2.turn_index, 2)
        self.assertIsNotNone(res2.response.answer)

    # -------------------------------------------------------------------------
    # 9. Raw Image Byte Protection Invariant
    # -------------------------------------------------------------------------

    def test_10_raw_byte_protection(self):
        """Verify session turns and summaries never retain raw image byte payloads."""
        self.service.process_interaction(
            session_id="sess_bytes",
            query="Inspect diagram",
            image_bytes=self.sample_png_bytes,
            file_name="diagram.png",
        )
        session = self.service.get_or_create_session("sess_bytes")
        session_dict = session.to_dict()

        for turn in session_dict.get("turns", []):
            context_summary = turn.get("context_summary")
            if context_summary:
                self.assertNotIn("data", context_summary)
                self.assertNotIn(self.sample_png_bytes, str(context_summary).encode())

    # -------------------------------------------------------------------------
    # 10. UI Response Rendering Verification (Mocked Streamlit)
    # -------------------------------------------------------------------------

    @patch("streamlit.expander")
    @patch("streamlit.caption")
    @patch("streamlit.write")
    @patch("streamlit.markdown")
    @patch("streamlit.success")
    def test_11_render_proceed_response(self, mock_success, mock_md, mock_write, mock_caption, mock_exp):
        """Verify render_multimodal_response handles PROCEED status with evidence."""
        mock_exp.return_value.__enter__.return_value = MagicMock()
        ev_item = VisualEvidenceItem(description="Bar chart showing growth", region_label="Chart A", confidence=0.95)
        response = MultimodalResponse(
            query="What is this?",
            answer="This is a bar chart showing growth.",
            modality=ModalityType.TEXT_AND_IMAGE,
            visual_evidence=[ev_item],
            grounded=True,
        )
        decision = FallbackDecision(
            action=FallbackAction.PROCEED,
            should_proceed=True,
            reason="Reliable evidence available",
            safe_message="This is a bar chart showing growth.",
        )
        result = MultimodalUIResult(
            response=response,
            fallback_decision=decision,
            turn_index=1,
            modality=ModalityType.TEXT_AND_IMAGE,
        )

        multimodal_main.render_multimodal_response(result)
        mock_success.assert_called_once()
        self.assertIn("Grounded Analysis", mock_success.call_args[0][0])
        mock_write.assert_any_call(response.answer)

    @patch("streamlit.expander")
    @patch("streamlit.caption")
    @patch("streamlit.write")
    @patch("streamlit.markdown")
    @patch("streamlit.warning")
    @patch("streamlit.info")
    def test_12_render_clarification_response(self, mock_info, mock_warning, mock_md, mock_write, mock_caption, mock_exp):
        """Verify render_multimodal_response handles ASK_CLARIFICATION status."""
        mock_exp.return_value.__enter__.return_value = MagicMock()
        decision = FallbackDecision(
            action=FallbackAction.ASK_CLARIFICATION,
            should_proceed=False,
            reason="Ambiguous input",
            safe_message="Which component would you like to inspect?",
            clarification_needed="Which component would you like to inspect?",
        )
        response = decision.to_multimodal_response(query="Explain that component")
        result = MultimodalUIResult(
            response=response,
            fallback_decision=decision,
            turn_index=1,
            modality=ModalityType.TEXT_ONLY,
        )

        multimodal_main.render_multimodal_response(result)
        mock_warning.assert_called()
        self.assertIn("Clarification Required", mock_warning.call_args[0][0])

    @patch("streamlit.expander")
    @patch("streamlit.caption")
    @patch("streamlit.write")
    @patch("streamlit.markdown")
    @patch("streamlit.error")
    def test_13_render_unsupported_claim_response(self, mock_error, mock_md, mock_write, mock_caption, mock_exp):
        """Verify render_multimodal_response handles DECLINE_UNSUPPORTED_CLAIM status."""
        mock_exp.return_value.__enter__.return_value = MagicMock()
        decision = FallbackDecision(
            action=FallbackAction.DECLINE_UNSUPPORTED_CLAIM,
            should_proceed=False,
            reason="Claim unsupported by evidence",
            safe_message="I cannot verify the claim based on the provided evidence.",
        )
        response = decision.to_multimodal_response(query="Verify false claim")
        result = MultimodalUIResult(
            response=response,
            fallback_decision=decision,
            turn_index=1,
            modality=ModalityType.TEXT_ONLY,
        )

        multimodal_main.render_multimodal_response(result)
        mock_error.assert_called()
        self.assertIn("Unsupported Claim Declined", mock_error.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
