"""End-to-End Integration Test Suite for Multimodal AI Assistant (Phase 5 — Day 29 Step 2).

Verifies the complete multimodal interaction pipeline from UI adaptation -> service ->
orchestrator -> reasoning -> evidence validation -> confidence assessment -> ambiguity
detection -> missing information detection -> safe fallback -> response generation -> UI rendering.

Tests real component integration with deterministic in-memory images across:
- Text-only flow
- Image-only flow
- Text + image joint flow
- Multi-turn multimodal flow & follow-up resolution
- Ambiguous input & clarification fallback
- Missing information detection & fallback
- Unsupported claim detection & safe refusal
- Session isolation & independent clearing
- Raw byte protection invariants
- Supported image formats (PNG, JPEG, WebP) and failure handling
- Streamlit UI adaptation layer
"""

import io
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

# Ensure project root and src/ are in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import multimodal_main
from src.multimodal import (
    ConfidenceStatus,
    EvidenceStatus,
    FallbackAction,
    ModalityType,
    MultimodalAssistantService,
    MultimodalFallbackHandler,
    MultimodalResponse,
    MultimodalUIResult,
    VisualEvidenceItem,
)


class TestMultimodalE2E(unittest.TestCase):
    """End-to-End Integration test suite for the complete Multimodal AI Assistant pipeline."""

    @classmethod
    def setUpClass(cls):
        """Generate deterministic in-memory images across all supported formats."""
        # 1. Valid PNG (Chart-like diagram)
        png_img = Image.new("RGB", (200, 150), color=(10, 50, 120))
        png_buf = io.BytesIO()
        png_img.save(png_buf, format="PNG")
        cls.valid_png_bytes = png_buf.getvalue()

        # 2. Valid JPEG (Architecture diagram)
        jpg_img = Image.new("RGB", (300, 200), color=(200, 100, 50))
        jpg_buf = io.BytesIO()
        jpg_img.save(jpg_buf, format="JPEG")
        cls.valid_jpg_bytes = jpg_buf.getvalue()

        # 3. Valid WebP (Flowchart)
        webp_img = Image.new("RGB", (180, 120), color=(50, 180, 80))
        webp_buf = io.BytesIO()
        webp_img.save(webp_buf, format="WEBP")
        cls.valid_webp_bytes = webp_buf.getvalue()

        # 4. Corrupted / Malformed Bytes
        cls.malformed_bytes = b"NOT_A_VALID_IMAGE_PAYLOAD_DATA_12345"

    def setUp(self):
        """Initialize a fresh instance of the real MultimodalAssistantService for each test."""
        # Clear cached service instance if present
        clear_cache = getattr(multimodal_main.initialize_multimodal_service, "clear", None)
        if clear_cache:
            clear_cache()
        self.service = MultimodalAssistantService()

    # =========================================================================
    # SCENARIO A: TEXT-ONLY END-TO-END FLOW
    # =========================================================================

    def test_scenario_a_text_only_e2e_flow(self):
        """Verify full text-only pipeline from service request through reasoning to UI result."""
        session_id = "e2e_text_session_01"
        query_text = "Explain the self-attention mechanism in transformer models."

        # Execute through full service integration
        ui_result: MultimodalUIResult = self.service.process_interaction(
            session_id=session_id,
            query=query_text,
            image_bytes=None,
            file_name=None,
        )

        # 1. Verify result structure and contract
        self.assertIsInstance(ui_result, MultimodalUIResult)
        self.assertEqual(ui_result.modality, ModalityType.TEXT_ONLY)
        self.assertEqual(ui_result.turn_index, 1)
        self.assertTrue(ui_result.is_proceed)
        self.assertEqual(ui_result.fallback_decision.action, FallbackAction.PROCEED)

        # 2. Verify response content and groundedness
        response = ui_result.response
        self.assertIsInstance(response, MultimodalResponse)
        self.assertEqual(response.session_id, session_id)
        self.assertEqual(response.query, query_text)
        self.assertIsNotNone(response.answer)
        self.assertTrue(len(response.answer) > 0)
        self.assertTrue(response.grounded)
        self.assertEqual(len(response.visual_evidence), 0)

        # 3. Verify session state recorded turn properly
        session = self.service.get_or_create_session(session_id)
        self.assertEqual(session.get_turn_count(), 1)
        turn = session.turns[0]
        self.assertEqual(turn.user_query, query_text)
        self.assertEqual(turn.modality, ModalityType.TEXT_ONLY)

    # =========================================================================
    # SCENARIO B: IMAGE-ONLY END-TO-END FLOW
    # =========================================================================

    def test_scenario_b_image_only_e2e_flow(self):
        """Verify full image-only pipeline: ingestion -> visual extraction -> response -> UI result."""
        session_id = "e2e_image_session_01"
        file_name = "system_architecture.png"

        ui_result: MultimodalUIResult = self.service.process_interaction(
            session_id=session_id,
            query=None,
            image_bytes=self.valid_png_bytes,
            file_name=file_name,
        )

        # 1. Verify modality and pipeline execution
        self.assertIsInstance(ui_result, MultimodalUIResult)
        self.assertEqual(ui_result.modality, ModalityType.IMAGE_ONLY)
        self.assertEqual(ui_result.turn_index, 1)

        # 2. Verify response includes image-based analysis and visual evidence
        response = ui_result.response
        self.assertIsInstance(response, MultimodalResponse)
        self.assertEqual(response.session_id, session_id)
        self.assertIsNotNone(response.answer)
        self.assertGreater(len(response.answer), 0)
        self.assertGreater(len(response.visual_evidence), 0)

        # 3. Verify session turn contains visual metadata
        session = self.service.get_or_create_session(session_id)
        self.assertEqual(session.get_turn_count(), 1)
        turn = session.turns[0]
        self.assertEqual(turn.modality, ModalityType.IMAGE_ONLY)
        self.assertIsNotNone(turn.context_summary)
        self.assertEqual(turn.context_summary.image_metadata.get("format"), "PNG")

    # =========================================================================
    # SCENARIO C: TEXT + IMAGE JOINT END-TO-END FLOW (HAPPY PATH)
    # =========================================================================

    def test_scenario_c_text_and_image_joint_e2e_flow(self):
        """Verify full joint text+image pipeline: ingestion -> context -> reasoning -> validation -> UI."""
        session_id = "e2e_joint_session_01"
        query_text = "Analyze the components and flow depicted in this diagram."
        file_name = "pipeline_diagram.png"

        ui_result: MultimodalUIResult = self.service.process_interaction(
            session_id=session_id,
            query=query_text,
            image_bytes=self.valid_png_bytes,
            file_name=file_name,
        )

        # 1. Verify joint modality and PROCEED decision
        self.assertIsInstance(ui_result, MultimodalUIResult)
        self.assertEqual(ui_result.modality, ModalityType.TEXT_AND_IMAGE)
        self.assertEqual(ui_result.turn_index, 1)
        self.assertTrue(ui_result.is_proceed)
        self.assertEqual(ui_result.fallback_decision.action, FallbackAction.PROCEED)

        # 2. Verify evidence validation and confidence assessments
        if ui_result.evidence_validation:
            self.assertIn(
                ui_result.evidence_validation.status,
                [EvidenceStatus.VALID, EvidenceStatus.PARTIAL],
            )
        if ui_result.confidence_assessment:
            self.assertIn(
                ui_result.confidence_assessment.status,
                [ConfidenceStatus.AVAILABLE, ConfidenceStatus.MIXED],
            )

        # 3. Verify response and visual evidence collection
        response = ui_result.response
        self.assertEqual(response.session_id, session_id)
        self.assertEqual(response.query, query_text)
        self.assertIsNotNone(response.answer)
        self.assertTrue(response.grounded)
        self.assertGreater(len(response.visual_evidence), 0)

        # 4. Verify visual evidence item types
        for ev in response.visual_evidence:
            self.assertIsInstance(ev, VisualEvidenceItem)
            self.assertIsNotNone(ev.description)
            self.assertGreaterEqual(ev.confidence, 0.0)

    # =========================================================================
    # SCENARIO D: MULTI-TURN MULTIMODAL CONVERSATION FLOW
    # =========================================================================

    def test_scenario_d_multiturn_multimodal_flow(self):
        """Verify multi-turn session: Turn 1 (image) -> Turn 2 (follow-up query resolving prior visual context)."""
        session_id = "e2e_multiturn_session_01"

        # Turn 1: User uploads diagram with initial query
        turn1_result = self.service.process_interaction(
            session_id=session_id,
            query="Analyze this architecture diagram.",
            image_bytes=self.valid_png_bytes,
            file_name="arch_diagram.png",
        )
        self.assertEqual(turn1_result.turn_index, 1)
        self.assertEqual(turn1_result.modality, ModalityType.TEXT_AND_IMAGE)
        self.assertTrue(turn1_result.is_proceed)

        # Turn 2: Follow-up question referencing the diagram from Turn 1 (no new image uploaded)
        turn2_result = self.service.process_interaction(
            session_id=session_id,
            query="What does the primary component in that diagram do?",
            image_bytes=None,
            file_name=None,
        )

        # 1. Verify turn progression in the same session
        self.assertEqual(turn2_result.turn_index, 2)
        self.assertIsNotNone(turn2_result.response.answer)

        # 2. Verify follow-up resolver engaged and tracked conversational context
        session = self.service.get_or_create_session(session_id)
        self.assertEqual(session.get_turn_count(), 2)
        self.assertEqual(session.turns[0].turn_index, 0)
        self.assertEqual(session.turns[1].turn_index, 1)

        # 3. Verify turn 2 response is session-specific and preserved
        self.assertEqual(session.turns[1].user_query, "What does the primary component in that diagram do?")

    # =========================================================================
    # SCENARIO E: AMBIGUOUS INPUT FLOW & CLARIFICATION
    # =========================================================================

    def test_scenario_e_ambiguous_input_flow(self):
        """Verify ambiguous request triggers ambiguity detection and ASK_CLARIFICATION fallback."""
        session_id = "e2e_ambig_session_01"
        vague_query = "What is this?"

        ui_result = self.service.process_interaction(
            session_id=session_id,
            query=vague_query,
            image_bytes=None,
        )

        # 1. Verify fallback decision is ASK_CLARIFICATION
        self.assertIsInstance(ui_result, MultimodalUIResult)
        self.assertTrue(ui_result.is_clarification)
        self.assertEqual(ui_result.fallback_decision.action, FallbackAction.ASK_CLARIFICATION)
        self.assertFalse(ui_result.fallback_decision.should_proceed)

        # 2. Verify ambiguity assessment details
        self.assertIsNotNone(ui_result.ambiguity_assessment)
        self.assertTrue(ui_result.ambiguity_assessment.is_ambiguous)
        self.assertTrue(ui_result.ambiguity_assessment.requires_clarification)

        # 3. Verify safe clarification message is returned to the user
        response = ui_result.response
        self.assertIsNotNone(response.answer)
        self.assertIsNotNone(response.ambiguity)
        self.assertTrue(response.ambiguity.is_ambiguous)
        self.assertIsNotNone(response.warning_message)

    # =========================================================================
    # SCENARIO F: MISSING INFORMATION FLOW
    # =========================================================================

    def test_scenario_f_missing_information_flow(self):
        """Verify visual query without required image triggers REQUEST_MISSING_INFORMATION."""
        session_id = "e2e_missing_info_session_01"
        visual_query_without_image = "What is the label of the red box in this chart?"

        ui_result = self.service.process_interaction(
            session_id=session_id,
            query=visual_query_without_image,
            image_bytes=None,  # Intentionally missing
        )

        # 1. Verify fallback action is REQUEST_MISSING_INFORMATION
        self.assertIsInstance(ui_result, MultimodalUIResult)
        self.assertTrue(ui_result.is_missing_information)
        self.assertEqual(ui_result.fallback_decision.action, FallbackAction.REQUEST_MISSING_INFORMATION)
        self.assertFalse(ui_result.fallback_decision.should_proceed)

        # 2. Verify safe instruction to provide an image is surfaced
        self.assertIn("image", ui_result.response.answer.lower())
        self.assertFalse(ui_result.response.grounded)

    # =========================================================================
    # SCENARIO G: UNSUPPORTED CLAIM FLOW
    # =========================================================================

    def test_scenario_g_unsupported_claim_flow(self):
        """Verify that claims with invalid or unsupported evidence trigger DECLINE_UNSUPPORTED_CLAIM."""
        fallback_handler = MultimodalFallbackHandler()

        # Evaluate fallback directly with explicit unsupported claim
        decision = fallback_handler.evaluate(
            query="Confirm that this architecture uses Kubernetes version 1.30 without evidence.",
            unsupported_claim=True,
        )

        self.assertEqual(decision.action, FallbackAction.DECLINE_UNSUPPORTED_CLAIM)
        self.assertFalse(decision.should_proceed)
        self.assertIn("cannot", decision.safe_message.lower())

        # Convert to MultimodalResponse and verify safe contract
        resp = decision.to_multimodal_response(query="Confirm that this architecture uses Kubernetes version 1.30 without evidence.")
        self.assertFalse(resp.grounded)
        self.assertEqual(len(resp.visual_evidence), 0)
        self.assertIn("Fallback action triggered", resp.warning_message)

    # =========================================================================
    # SCENARIO H: SESSION ISOLATION & SESSION RESET
    # =========================================================================

    def test_scenario_h_session_isolation_and_reset(self):
        """Verify strict isolation between sessions and that clearing one session leaves others intact."""
        session_a = "user_alice_session"
        session_b = "user_bob_session"

        # Session A interactions
        self.service.process_interaction(session_id=session_a, query="Alice query 1")
        self.service.process_interaction(session_id=session_a, query="Alice query 2")

        # Session B interaction
        self.service.process_interaction(
            session_id=session_b,
            query="Bob query with image",
            image_bytes=self.valid_png_bytes,
            file_name="bob_chart.png",
        )

        sess_a = self.service.get_or_create_session(session_a)
        sess_b = self.service.get_or_create_session(session_b)

        # 1. Verify turn counts and isolation
        self.assertEqual(sess_a.get_turn_count(), 2)
        self.assertEqual(sess_b.get_turn_count(), 1)
        self.assertEqual(sess_a.turns[0].user_query, "Alice query 1")
        self.assertEqual(sess_b.turns[0].user_query, "Bob query with image")

        # 2. Reset Session A
        cleared = self.service.clear_session(session_a)
        self.assertTrue(cleared)

        # 3. Verify Session A is empty while Session B is completely preserved
        sess_a_new = self.service.get_or_create_session(session_a)
        self.assertEqual(sess_a_new.get_turn_count(), 0)

        sess_b_retained = self.service.get_or_create_session(session_b)
        self.assertEqual(sess_b_retained.get_turn_count(), 1)
        self.assertEqual(sess_b_retained.turns[0].user_query, "Bob query with image")

    # =========================================================================
    # SCENARIO I: RAW BYTE PROTECTION INVARIANT
    # =========================================================================

    def test_scenario_i_raw_byte_protection(self):
        """Verify raw image byte payloads are never exposed in responses, sessions, or dict exports."""
        session_id = "e2e_protection_session"

        res = self.service.process_interaction(
            session_id=session_id,
            query="Inspect the diagram layout",
            image_bytes=self.valid_png_bytes,
            file_name="sensitive_diagram.png",
        )

        # 1. Verify response object does not contain raw bytes
        resp_dict = res.response.to_dict()
        self.assertNotIn("data", resp_dict)
        self.assertNotIn(self.valid_png_bytes, str(resp_dict).encode())

        # 2. Verify UI result serialization does not contain raw bytes
        if res.fallback_decision:
            decision_dict = res.fallback_decision.to_dict()
            self.assertNotIn("data", decision_dict)
            self.assertNotIn(self.valid_png_bytes, str(decision_dict).encode())

        # 3. Verify session history serialization does not leak raw bytes
        session = self.service.get_or_create_session(session_id)
        session_dict = session.to_dict()
        for turn_data in session_dict.get("turns", []):
            summary = turn_data.get("context_summary")
            if summary:
                self.assertNotIn("data", summary)
                self.assertNotIn(self.valid_png_bytes, str(summary).encode())

    # =========================================================================
    # SCENARIO J: MULTI-FORMAT IMAGE SUPPORT (PNG, JPEG, WebP) & ERROR HANDLING
    # =========================================================================

    def test_scenario_j1_jpeg_image_support(self):
        """Verify successful end-to-end ingestion and processing of JPEG images."""
        res = self.service.process_interaction(
            session_id="e2e_jpeg_session",
            query="Analyze this JPEG image.",
            image_bytes=self.valid_jpg_bytes,
            file_name="photo.jpeg",
        )
        self.assertEqual(res.modality, ModalityType.TEXT_AND_IMAGE)
        self.assertTrue(res.is_proceed)
        self.assertIsNotNone(res.response.answer)

    def test_scenario_j2_webp_image_support(self):
        """Verify successful end-to-end ingestion and processing of WebP images."""
        res = self.service.process_interaction(
            session_id="e2e_webp_session",
            query="Analyze this WebP image.",
            image_bytes=self.valid_webp_bytes,
            file_name="graphic.webp",
        )
        self.assertEqual(res.modality, ModalityType.TEXT_AND_IMAGE)
        self.assertTrue(res.is_proceed)
        self.assertIsNotNone(res.response.answer)

    def test_scenario_j3_malformed_image_handling(self):
        """Verify malformed image bytes are safely caught during ingestion without crashing."""
        with self.assertRaises(ValueError) as ctx:
            self.service.process_interaction(
                session_id="e2e_malformed_session",
                query="Analyze this corrupt image.",
                image_bytes=self.malformed_bytes,
                file_name="corrupted.png",
            )
        self.assertIn("cannot identify image file", str(ctx.exception).lower())

    def test_scenario_j4_empty_input_handling(self):
        """Verify empty interaction (neither query nor image) raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.service.process_interaction(
                session_id="e2e_empty_session",
                query=None,
                image_bytes=None,
            )
        self.assertIn("requires at least a query string or an image upload", str(ctx.exception))

    # =========================================================================
    # SCENARIO K: STREAMLIT UI ADAPTATION LAYER INTEGRATION
    # =========================================================================

    @patch("streamlit.expander")
    @patch("streamlit.caption")
    @patch("streamlit.write")
    @patch("streamlit.markdown")
    @patch("streamlit.success")
    def test_scenario_k1_render_proceed_ui_state(self, mock_success, mock_md, mock_write, mock_caption, mock_exp):
        """Verify complete end-to-end rendering of a successful multimodal response in Streamlit UI."""
        mock_exp.return_value.__enter__.return_value = MagicMock()

        # Run real service interaction
        ui_result = self.service.process_interaction(
            session_id="e2e_render_proceed",
            query="Explain this architecture diagram.",
            image_bytes=self.valid_png_bytes,
            file_name="arch.png",
        )

        # Render through Streamlit UI adapter
        multimodal_main.render_multimodal_response(ui_result)

        # Verify UI components called
        mock_success.assert_called_once()
        self.assertIn("Grounded Analysis", mock_success.call_args[0][0])
        mock_write.assert_any_call(ui_result.response.answer)

    @patch("streamlit.expander")
    @patch("streamlit.caption")
    @patch("streamlit.write")
    @patch("streamlit.markdown")
    @patch("streamlit.info")
    def test_scenario_k2_render_missing_info_ui_state(self, mock_info, mock_md, mock_write, mock_caption, mock_exp):
        """Verify complete end-to-end rendering of a missing information response in Streamlit UI."""
        mock_exp.return_value.__enter__.return_value = MagicMock()

        # Run real service interaction triggering missing info
        ui_result = self.service.process_interaction(
            session_id="e2e_render_missing",
            query="What is the color of the second chart in this image?",
            image_bytes=None,
        )

        # Render through Streamlit UI adapter
        multimodal_main.render_multimodal_response(ui_result)

        # Verify info banner called
        mock_info.assert_called_once()
        self.assertIn("Additional Information Required", mock_info.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
