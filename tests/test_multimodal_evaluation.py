"""Multimodal AI Assistant Evaluation Suite (Phase 5 — Day 29 Step 3).

Comprehensive behavioral and safety evaluation of the completed Multimodal AI Assistant.
Measures and validates:
1. Multimodal response correctness (text-only, image-only, text+image, follow-up)
2. Evidence grounding (valid evidence, no-evidence, unsupported-claim refusal)
3. Confidence behavior (boundedness [0.0, 1.0], unavailable representation, deterministic aggregation)
4. Ambiguity handling (vague queries, unclear referents, clarification generation)
5. Missing-information handling (missing image, missing context, non-hallucination)
6. Safe fallback behavior (deterministic priority and safe message formulation)
7. Follow-up / context continuity (antecedent resolution, metadata retention)
8. Session isolation (independent turn states, clean session deletion)
9. Raw image-byte protection (serialization audit across all contracts)
10. Input / error robustness (multi-format support, malformed input rejection)
"""

import io
import os
import sys
import unittest
from PIL import Image

# Ensure project root and src/ are in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src.multimodal import (
    AmbiguityAssessmentResult,
    AmbiguityLevel,
    ConfidenceStatus,
    EvidenceStatus,
    FallbackAction,
    ModalityType,
    MultimodalAssistantService,
    MultimodalFallbackHandler,
    MultimodalRequest,
    MultimodalUIResult,
    VisualEvidenceItem,
    assess_multimodal_confidence,
    build_multimodal_context,
    ingest_image,
    validate_multimodal_evidence,
)
from src.multimodal.missing_information import (
    MissingInformationAssessmentResult,
    MissingInformationItem,
    MissingInformationSeverity,
    MissingInformationType,
)


class TestMultimodalEvaluation(unittest.TestCase):
    """Behavioral, grounding, safety, and robustness evaluation suite."""

    @classmethod
    def setUpClass(cls):
        """Prepare deterministic test images across all supported formats."""
        # Valid PNG
        png = Image.new("RGB", (160, 120), color=(30, 80, 160))
        buf_png = io.BytesIO()
        png.save(buf_png, format="PNG")
        cls.valid_png_bytes = buf_png.getvalue()

        # Valid JPEG
        jpg = Image.new("RGB", (240, 180), color=(180, 90, 40))
        buf_jpg = io.BytesIO()
        jpg.save(buf_jpg, format="JPEG")
        cls.valid_jpg_bytes = buf_jpg.getvalue()

        # Valid WebP
        webp = Image.new("RGB", (140, 100), color=(60, 160, 90))
        buf_webp = io.BytesIO()
        webp.save(buf_webp, format="WEBP")
        cls.valid_webp_bytes = buf_webp.getvalue()

        # Malformed Bytes
        cls.malformed_image_bytes = b"CORRUPTED_NON_IMAGE_DATA_0xDEADBEEF"

    def setUp(self):
        """Create a fresh MultimodalAssistantService for each evaluation test."""
        self.service = MultimodalAssistantService()

    # =========================================================================
    # 1. MULTIMODAL RESPONSE CORRECTNESS EVALUATION
    # =========================================================================

    def test_eval_01_text_only_response_correctness(self):
        """Evaluate behavioral correctness of text-only processing without visual requirements."""
        res = self.service.process_interaction(
            session_id="eval_text_sess",
            query="Summarize the core concept of cross-attention.",
        )
        self.assertIsInstance(res, MultimodalUIResult)
        self.assertEqual(res.modality, ModalityType.TEXT_ONLY)
        self.assertTrue(res.is_proceed)
        self.assertIsNotNone(res.response.answer)
        self.assertTrue(res.response.grounded)
        self.assertEqual(len(res.response.visual_evidence), 0)

    def test_eval_02_image_only_response_correctness(self):
        """Evaluate behavioral correctness of image-only analysis."""
        res = self.service.process_interaction(
            session_id="eval_img_sess",
            query=None,
            image_bytes=self.valid_png_bytes,
            file_name="benchmark_chart.png",
        )
        self.assertEqual(res.modality, ModalityType.IMAGE_ONLY)
        self.assertIsNotNone(res.response.answer)
        self.assertGreater(len(res.response.visual_evidence), 0)
        self.assertTrue(res.response.grounded)

    def test_eval_03_text_and_image_response_correctness(self):
        """Evaluate behavioral correctness of joint cross-modal reasoning."""
        res = self.service.process_interaction(
            session_id="eval_joint_sess",
            query="Explain the layout and visual structure of this image.",
            image_bytes=self.valid_png_bytes,
            file_name="system_model.png",
        )
        self.assertEqual(res.modality, ModalityType.TEXT_AND_IMAGE)
        self.assertTrue(res.is_proceed)
        self.assertIsNotNone(res.response.answer)
        self.assertGreater(len(res.response.visual_evidence), 0)
        self.assertTrue(res.response.grounded)

    # =========================================================================
    # 2. EVIDENCE GROUNDING EVALUATION
    # =========================================================================

    def test_eval_04_evidence_grounding_fidelity(self):
        """Evaluate that evidence items are structurally represented and strictly validated."""
        valid_evidence = [
            VisualEvidenceItem(description="Primary header block", region_label="Header", confidence=0.92),
            VisualEvidenceItem(description="Data processing pipeline flow", region_label="Pipeline", confidence=0.88),
        ]
        val_res = validate_multimodal_evidence(evidence=valid_evidence)
        self.assertTrue(val_res.is_valid)
        self.assertEqual(val_res.status, EvidenceStatus.VALID)
        self.assertEqual(val_res.valid_items_count, 2)
        self.assertEqual(val_res.invalid_items_count, 0)

    def test_eval_05_no_evidence_and_invalid_evidence_detection(self):
        """Evaluate that missing or invalid evidence items are not promoted to valid evidence."""
        # Empty evidence
        empty_val = validate_multimodal_evidence(evidence=[])
        self.assertFalse(empty_val.is_valid)
        self.assertEqual(empty_val.status, EvidenceStatus.NO_EVIDENCE)
        self.assertEqual(empty_val.valid_items_count, 0)

        # Malformed item
        invalid_evidence = [{"description": "", "confidence": -0.5}]
        inv_val = validate_multimodal_evidence(evidence=invalid_evidence)
        self.assertFalse(inv_val.is_valid)
        self.assertEqual(inv_val.invalid_items_count, 1)

    # =========================================================================
    # 3. CONFIDENCE BEHAVIOR EVALUATION
    # =========================================================================

    def test_eval_06_confidence_bounds_and_deterministic_aggregation(self):
        """Evaluate confidence scoring, boundedness within [0.0, 1.0], and deterministic aggregation."""
        evidence = [
            VisualEvidenceItem(description="Node A", confidence=0.85),
            VisualEvidenceItem(description="Node B", confidence=0.95),
        ]
        conf_res = assess_multimodal_confidence(evidence=evidence)
        self.assertEqual(conf_res.status, ConfidenceStatus.AVAILABLE)
        self.assertTrue(conf_res.is_confident)
        self.assertGreaterEqual(conf_res.aggregate_confidence, 0.0)
        self.assertLessEqual(conf_res.aggregate_confidence, 1.0)
        self.assertAlmostEqual(conf_res.aggregate_confidence, 0.90, places=2)
        self.assertEqual(conf_res.min_confidence, 0.85)
        self.assertEqual(conf_res.max_confidence, 0.95)

    def test_eval_07_missing_confidence_is_unavailable_not_invented(self):
        """Evaluate that absent confidence is classified as UNAVAILABLE without fabricating values."""
        conf_res = assess_multimodal_confidence(evidence=[])
        self.assertEqual(conf_res.status, ConfidenceStatus.UNAVAILABLE)
        self.assertIsNone(conf_res.aggregate_confidence)
        self.assertFalse(conf_res.is_confident)

    # =========================================================================
    # 4. AMBIGUITY HANDLING EVALUATION
    # =========================================================================

    def test_eval_08_ambiguity_detection_and_clarification_refusal_to_guess(self):
        """Evaluate that vague queries trigger ASK_CLARIFICATION without silently guessing."""
        vague_queries = [
            "What is this?",
            "Explain that",
            "What about it?",
        ]
        for query in vague_queries:
            res = self.service.process_interaction(
                session_id=f"eval_ambig_{abs(hash(query))}",
                query=query,
            )
            self.assertTrue(res.is_clarification, f"Expected clarification for '{query}'")
            self.assertEqual(res.fallback_decision.action, FallbackAction.ASK_CLARIFICATION)
            self.assertFalse(res.fallback_decision.should_proceed)
            self.assertIsNotNone(res.response.ambiguity)
            self.assertTrue(res.response.ambiguity.is_ambiguous)

    # =========================================================================
    # 5. MISSING INFORMATION EVALUATION
    # =========================================================================

    def test_eval_09_missing_information_detection_blocks_unsupported_answering(self):
        """Evaluate that absent required visual context triggers REQUEST_MISSING_INFORMATION."""
        res = self.service.process_interaction(
            session_id="eval_missing_sess",
            query="What is written in the bottom right corner of this chart?",
            image_bytes=None,
        )
        self.assertTrue(res.is_missing_information)
        self.assertEqual(res.fallback_decision.action, FallbackAction.REQUEST_MISSING_INFORMATION)
        self.assertFalse(res.fallback_decision.should_proceed)
        self.assertIn("image", res.response.answer.lower())
        self.assertFalse(res.response.grounded)

    # =========================================================================
    # 6. SAFE FALLBACK BEHAVIOR & PRIORITY EVALUATION
    # =========================================================================

    def test_eval_10_safe_fallback_priority_hierarchy(self):
        """Evaluate deterministic fallback hierarchy: Missing Info > Clarification > Unsupported Claim > Proceed."""
        handler = MultimodalFallbackHandler()

        # 1. Missing Info takes precedence
        missing_res = MissingInformationAssessmentResult(
            has_missing_information=True,
            severity=MissingInformationSeverity.CRITICAL,
            missing_items=[
                MissingInformationItem(
                    item_type=MissingInformationType.MISSING_IMAGE,
                    description="Missing image artifact",
                    why_required="Image required",
                    severity=MissingInformationSeverity.CRITICAL,
                )
            ],
            blocks_answering=True,
            available_summary={"has_image": False},
            reasons=["Image required for visual inquiry"],
        )
        ambig_res = AmbiguityAssessmentResult(
            is_ambiguous=True,
            ambiguity_level=AmbiguityLevel.HIGH,
            requires_clarification=True,
        )
        d1 = handler.evaluate(
            query="Inspect the diagram",
            missing_info_assessment=missing_res,
            ambiguity_assessment=ambig_res,
        )
        self.assertEqual(d1.action, FallbackAction.REQUEST_MISSING_INFORMATION)

        # 2. Clarification takes precedence over unsupported claim
        d2 = handler.evaluate(
            query="What is this?",
            ambiguity_assessment=ambig_res,
            unsupported_claim=True,
        )
        self.assertEqual(d2.action, FallbackAction.ASK_CLARIFICATION)

        # 3. Unsupported claim declines safely when not ambiguous
        d3 = handler.evaluate(
            query="Verify that accuracy is 100%",
            unsupported_claim=True,
        )
        self.assertEqual(d3.action, FallbackAction.DECLINE_UNSUPPORTED_CLAIM)
        self.assertFalse(d3.should_proceed)

    # =========================================================================
    # 7. FOLLOW-UP / CONTEXT CONTINUITY EVALUATION
    # =========================================================================

    def test_eval_11_multiturn_followup_and_turn_ordering(self):
        """Evaluate multi-turn context carryover, antecedent resolution, and turn monotonicity."""
        session_id = "eval_followup_continuity"

        # Turn 1
        t1 = self.service.process_interaction(
            session_id=session_id,
            query="Analyze this neural network architecture diagram.",
            image_bytes=self.valid_png_bytes,
            file_name="nn_arch.png",
        )
        self.assertEqual(t1.turn_index, 1)

        # Turn 2
        t2 = self.service.process_interaction(
            session_id=session_id,
            query="What are its key components?",
        )
        self.assertEqual(t2.turn_index, 2)
        self.assertIsNotNone(t2.response.answer)

        # Verify session state continuity
        session = self.service.get_or_create_session(session_id)
        self.assertEqual(session.get_turn_count(), 2)
        self.assertEqual(session.turns[0].turn_index, 0)
        self.assertEqual(session.turns[1].turn_index, 1)
        self.assertLessEqual(session.turns[0].created_at, session.turns[1].created_at)

    # =========================================================================
    # 8. SESSION ISOLATION EVALUATION
    # =========================================================================

    def test_eval_12_session_isolation_and_independent_clear(self):
        """Evaluate that distinct sessions do not leak context and clear operates independently."""
        sess_1 = "tenant_alpha"
        sess_2 = "tenant_beta"

        self.service.process_interaction(session_id=sess_1, query="Alpha turn 1")
        self.service.process_interaction(session_id=sess_1, query="Alpha turn 2")
        self.service.process_interaction(session_id=sess_2, query="Beta turn 1")

        s1 = self.service.get_or_create_session(sess_1)
        s2 = self.service.get_or_create_session(sess_2)

        self.assertEqual(s1.get_turn_count(), 2)
        self.assertEqual(s2.get_turn_count(), 1)
        self.assertEqual(s1.turns[0].user_query, "Alpha turn 1")
        self.assertEqual(s2.turns[0].user_query, "Beta turn 1")

        # Reset session 1 only
        self.service.clear_session(sess_1)
        s1_reset = self.service.get_or_create_session(sess_1)
        s2_retained = self.service.get_or_create_session(sess_2)

        self.assertEqual(s1_reset.get_turn_count(), 0)
        self.assertEqual(s2_retained.get_turn_count(), 1)
        self.assertEqual(s2_retained.turns[0].user_query, "Beta turn 1")

    # =========================================================================
    # 9. RAW IMAGE BYTE PROTECTION EVALUATION
    # =========================================================================

    def test_eval_13_raw_byte_protection_audit(self):
        """Perform strict programmatic audit across all data contracts to guarantee zero byte leakage."""
        artifact = ingest_image(self.valid_png_bytes, file_name="secret_diagram.png")
        req = MultimodalRequest(
            query="Examine diagram",
            images=[artifact],
            session_id="audit_bytes_session",
        )
        build_multimodal_context(request=req, artifact=artifact)

        res = self.service.process_interaction(
            session_id="audit_bytes_session",
            query="Examine diagram",
            image_bytes=self.valid_png_bytes,
            file_name="secret_diagram.png",
        )

        # 1. Inspect ImageArtifact.to_dict()
        art_dict = artifact.to_dict()
        self.assertNotIn("data", art_dict)
        self.assertEqual(art_dict["data_size_bytes"], len(self.valid_png_bytes))

        # 2. Inspect MultimodalResponse.to_dict()
        resp_dict = res.response.to_dict()
        self.assertNotIn("data", resp_dict)
        self.assertNotIn(self.valid_png_bytes, str(resp_dict).encode())

        # 3. Inspect FallbackDecision.to_dict()
        dec_dict = res.fallback_decision.to_dict()
        self.assertNotIn("data", dec_dict)
        self.assertNotIn(self.valid_png_bytes, str(dec_dict).encode())

        # 4. Inspect ConversationTurn & ContextSummary
        session = self.service.get_or_create_session("audit_bytes_session")
        turn_dict = session.turns[0].to_dict()
        self.assertNotIn(self.valid_png_bytes, str(turn_dict).encode())

    # =========================================================================
    # 10. INPUT & ERROR ROBUSTNESS EVALUATION
    # =========================================================================

    def test_eval_14_supported_image_formats(self):
        """Evaluate that all supported formats (PNG, JPEG, WebP) are accepted and processed."""
        # PNG
        res_png = self.service.process_interaction(
            session_id="fmt_png",
            query="Describe PNG",
            image_bytes=self.valid_png_bytes,
            file_name="chart.png",
        )
        self.assertTrue(res_png.is_proceed)

        # JPEG
        res_jpg = self.service.process_interaction(
            session_id="fmt_jpg",
            query="Describe JPEG",
            image_bytes=self.valid_jpg_bytes,
            file_name="photo.jpg",
        )
        self.assertTrue(res_jpg.is_proceed)

        # WebP
        res_webp = self.service.process_interaction(
            session_id="fmt_webp",
            query="Describe WebP",
            image_bytes=self.valid_webp_bytes,
            file_name="graphic.webp",
        )
        self.assertTrue(res_webp.is_proceed)

    def test_eval_15_malformed_and_empty_inputs_fail_safely(self):
        """Evaluate that malformed images, unsupported formats, and empty inputs are safely rejected."""
        # Malformed image bytes
        with self.assertRaises(ValueError) as ctx1:
            self.service.process_interaction(
                session_id="err_malformed",
                query="Analyze corrupt image",
                image_bytes=self.malformed_image_bytes,
                file_name="broken.png",
            )
        self.assertIn("cannot identify image file", str(ctx1.exception).lower())

        # Empty request (neither query nor image)
        with self.assertRaises(ValueError) as ctx2:
            self.service.process_interaction(
                session_id="err_empty",
                query=None,
                image_bytes=None,
            )
        self.assertIn("requires at least a query string or an image upload", str(ctx2.exception))


if __name__ == "__main__":
    unittest.main()
