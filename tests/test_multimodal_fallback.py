"""Unit tests for Multimodal Safe Fallback Behavior (Phase 5 — Day 28 Step 5).

Tests deterministic decision-making and safe fallback response generation across
ambiguous, missing, low-confidence, conflicting, and unsupported multimodal inputs.
Verifies strict non-hallucination, raw byte protection, contract compliance,
and priority order.
"""

import json
import unittest

from src.multimodal.ambiguity import (
    AmbiguityAssessmentResult,
    AmbiguityLevel,
    AmbiguityType,
)
from src.multimodal.confidence import (
    ConfidenceAssessmentResult,
    ConfidenceStatus,
)
from src.multimodal.context import (
    MultimodalContext,
    TextualContext,
    UnifiedEvidenceItem,
    VisualContext,
)
from src.multimodal.evidence_validator import (
    EvidenceItemValidation,
    EvidenceStatus,
    EvidenceValidationResult,
)
from src.multimodal.fallback import (
    FallbackAction,
    FallbackDecision,
    MultimodalFallbackHandler,
    evaluate_safe_fallback,
)
from src.multimodal.missing_information import (
    MissingInformationAssessmentResult,
    MissingInformationItem,
    MissingInformationSeverity,
    MissingInformationType,
)
from src.multimodal.models import (
    ImageArtifact,
    ModalityType,
    MultimodalRequest,
    MultimodalResponse,
    VisualEvidenceItem,
)


class TestMultimodalFallback(unittest.TestCase):
    """Test suite for deterministic safe fallback behavior."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.handler = MultimodalFallbackHandler()
        self.dummy_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x02\x00\x00\x00\x90\x91h6"
        self.dummy_artifact = ImageArtifact(
            data=self.dummy_bytes,
            mime_type="image/png",
            width=64,
            height=64,
            format="png",
        )

    def test_reliable_evidence_proceed(self) -> None:
        """1. Verify that sufficient and reliable evidence produces PROCEED decision."""
        ev_item = VisualEvidenceItem(
            description="Clear architectural diagram showing Load Balancer routing to API Gateway",
            confidence=0.96,
        )
        validation_res = EvidenceValidationResult(
            status=EvidenceStatus.VALID,
            is_valid=True,
            total_items=1,
            valid_items_count=1,
            invalid_items_count=0,
            valid_evidence=[ev_item],
        )
        conf_res = ConfidenceAssessmentResult(
            status=ConfidenceStatus.AVAILABLE,
            is_confident=True,
            aggregate_confidence=0.96,
            min_confidence=0.96,
            max_confidence=0.96,
            threshold=0.70,
            total_items=1,
            valid_confidence_count=1,
            unavailable_confidence_count=0,
            invalid_confidence_count=0,
        )

        decision = self.handler.evaluate(
            query="Explain the architecture flow",
            evidence_items=[ev_item],
            evidence_validation=validation_res,
            confidence_assessment=conf_res,
        )

        self.assertEqual(decision.action, FallbackAction.PROCEED)
        self.assertTrue(decision.should_proceed)
        self.assertIn("sufficient", decision.safe_message.lower())

    def test_ambiguous_input_ask_clarification(self) -> None:
        """2. Verify that ambiguous input triggers ASK_CLARIFICATION."""
        ambiguity_res = AmbiguityAssessmentResult(
            is_ambiguous=True,
            ambiguity_level=AmbiguityLevel.MEDIUM,
            ambiguity_types=[AmbiguityType.MULTIPLE_VISUAL_TARGETS],
            reasons=["Query refers to generic 'block', but multiple plausible visual targets exist."],
            affected_references=["the block"],
            candidate_interpretations=["Block A", "Block B"],
            requires_clarification=True,
            suggested_clarification="The request could refer to Block A or Block B. Please specify which one you mean.",
        )

        decision = self.handler.evaluate(
            query="What does the block do?",
            ambiguity_assessment=ambiguity_res,
        )

        self.assertEqual(decision.action, FallbackAction.ASK_CLARIFICATION)
        self.assertFalse(decision.should_proceed)
        self.assertIsNotNone(decision.clarification_needed)
        self.assertIn("Block A or Block B", decision.safe_message)

    def test_missing_image_information_request_missing_info(self) -> None:
        """3. Verify that missing image/info triggers REQUEST_MISSING_INFORMATION."""
        missing_res = MissingInformationAssessmentResult(
            has_missing_information=True,
            severity=MissingInformationSeverity.CRITICAL,
            missing_items=[
                MissingInformationItem(
                    item_type=MissingInformationType.MISSING_IMAGE,
                    description="User query requires image analysis, but no image was provided.",
                    why_required="Image is needed to view visual content.",
                    affected_reference="image",
                    severity=MissingInformationSeverity.CRITICAL,
                )
            ],
            blocks_answering=True,
            available_summary={"has_image": False},
            reasons=["Image is absent."],
        )

        decision = self.handler.evaluate(
            query="What does this diagram show in the attached image?",
            missing_info_assessment=missing_res,
        )

        self.assertEqual(decision.action, FallbackAction.REQUEST_MISSING_INFORMATION)
        self.assertFalse(decision.should_proceed)
        self.assertIn("image", decision.safe_message.lower())
        self.assertGreater(len(decision.missing_requirements), 0)

    def test_unsupported_claim_decline(self) -> None:
        """4. Verify that explicitly unsupported claim triggers DECLINE_UNSUPPORTED_CLAIM."""
        decision = self.handler.evaluate(
            query="Confirm that this architecture uses Kubernetes version 1.30",
            unsupported_claim=True,
        )

        self.assertEqual(decision.action, FallbackAction.DECLINE_UNSUPPORTED_CLAIM)
        self.assertFalse(decision.should_proceed)
        self.assertIn("cannot", decision.safe_message.lower())

    def test_low_confidence_evidence_safe_limited_response(self) -> None:
        """5. Verify that low-confidence evidence triggers SAFE_LIMITED_RESPONSE."""
        ev_item = VisualEvidenceItem(
            description="Blurry label potentially reading 'Cache'",
            confidence=0.35,
        )
        conf_res = ConfidenceAssessmentResult(
            status=ConfidenceStatus.AVAILABLE,
            is_confident=False,  # below 0.70 threshold
            aggregate_confidence=0.35,
            min_confidence=0.35,
            max_confidence=0.35,
            threshold=0.70,
            total_items=1,
            valid_confidence_count=1,
            unavailable_confidence_count=0,
            invalid_confidence_count=0,
        )

        decision = self.handler.evaluate(
            query="What is the label on the small box?",
            evidence_items=[ev_item],
            confidence_assessment=conf_res,
        )

        self.assertEqual(decision.action, FallbackAction.SAFE_LIMITED_RESPONSE)
        self.assertFalse(decision.should_proceed)
        self.assertIn("limited confidence", decision.safe_message.lower())
        self.assertGreater(len(decision.limitations_or_qualifications), 0)

    def test_multiple_failure_conditions_priority_order(self) -> None:
        """6. Verify deterministic priority: Missing Info > Ambiguity > Low Confidence."""
        # Setup: Query has missing image (Priority 1), ambiguity (Priority 2), AND low confidence (Priority 4)
        missing_res = MissingInformationAssessmentResult(
            has_missing_information=True,
            severity=MissingInformationSeverity.CRITICAL,
            missing_items=[
                MissingInformationItem(
                    item_type=MissingInformationType.MISSING_IMAGE,
                    description="No image provided.",
                    why_required="Image needed.",
                    severity=MissingInformationSeverity.CRITICAL,
                )
            ],
            blocks_answering=True,
            available_summary={},
            reasons=["Missing image."],
        )
        ambiguity_res = AmbiguityAssessmentResult(
            is_ambiguous=True,
            ambiguity_level=AmbiguityLevel.HIGH,
            requires_clarification=True,
        )
        conf_res = ConfidenceAssessmentResult(
            status=ConfidenceStatus.AVAILABLE,
            is_confident=False,
            aggregate_confidence=0.20,
            min_confidence=0.20,
            max_confidence=0.20,
            threshold=0.70,
            total_items=1,
            valid_confidence_count=1,
            unavailable_confidence_count=0,
            invalid_confidence_count=0,
        )

        decision = self.handler.evaluate(
            query="Analyze this",
            missing_info_assessment=missing_res,
            ambiguity_assessment=ambiguity_res,
            confidence_assessment=conf_res,
        )

        # Priority 1 must win
        self.assertEqual(decision.action, FallbackAction.REQUEST_MISSING_INFORMATION)
        self.assertEqual(decision.metadata.get("priority_level"), 1)

        # Now test without missing info: Ambiguity (Priority 2) must win over Low Confidence (Priority 4)
        decision2 = self.handler.evaluate(
            query="Analyze this",
            missing_info_assessment=None,
            ambiguity_assessment=ambiguity_res,
            confidence_assessment=conf_res,
        )
        self.assertEqual(decision2.action, FallbackAction.ASK_CLARIFICATION)
        self.assertEqual(decision2.metadata.get("priority_level"), 2)

    def test_no_evidence_safe_fallback(self) -> None:
        """7. Verify that zero evidence available triggers safe DECLINE_UNSUPPORTED_CLAIM."""
        validation_res = EvidenceValidationResult(
            status=EvidenceStatus.NO_EVIDENCE,
            is_valid=False,
            total_items=0,
            valid_items_count=0,
            invalid_items_count=0,
        )

        decision = self.handler.evaluate(
            query="What is the data flow between microservices?",
            evidence_validation=validation_res,
        )

        self.assertEqual(decision.action, FallbackAction.DECLINE_UNSUPPORTED_CLAIM)
        self.assertFalse(decision.should_proceed)
        self.assertIn("no evidence", decision.safe_message.lower())

    def test_conflicting_evidence_safe_fallback(self) -> None:
        """8. Verify that conflicting evidence triggers safe DECLINE_UNSUPPORTED_CLAIM."""
        ambiguity_res = AmbiguityAssessmentResult(
            is_ambiguous=True,
            ambiguity_level=AmbiguityLevel.HIGH,
            ambiguity_types=[AmbiguityType.CONFLICTING_EVIDENCE],
            reasons=["Visual OCR says 'Active' but visual diagram label says 'Standby'."],
            requires_clarification=False,
        )

        decision = self.handler.evaluate(
            query="Is the server active or standby?",
            ambiguity_assessment=ambiguity_res,
        )

        self.assertEqual(decision.action, FallbackAction.DECLINE_UNSUPPORTED_CLAIM)
        self.assertFalse(decision.should_proceed)
        self.assertIn("conflicting", decision.safe_message.lower())

    def test_deterministic_repeated_decisions(self) -> None:
        """9. Verify identical inputs produce strictly identical deterministic decisions."""
        kwargs = {
            "query": "In this image, describe the flow",
            "unsupported_claim": True,
        }

        d1 = self.handler.evaluate(**kwargs)
        d2 = self.handler.evaluate(**kwargs)

        self.assertEqual(d1.to_dict(), d2.to_dict())
        self.assertEqual(d1.action, d2.action)
        self.assertEqual(d1.safe_message, d2.safe_message)

    def test_json_serialization(self) -> None:
        """10. Verify clean JSON serialization of FallbackDecision."""
        decision = evaluate_safe_fallback(
            query="What is the source code of this block?",
        )

        res_dict = decision.to_dict()
        serialized = json.dumps(res_dict)
        deserialized = json.loads(serialized)

        self.assertIn("action", deserialized)
        self.assertIn("safe_message", deserialized)
        self.assertIsInstance(deserialized["missing_requirements"], list)

    def test_raw_image_bytes_never_exposed(self) -> None:
        """11. Verify raw image byte payloads are strictly excluded from all fallback outputs."""
        decision = self.handler.evaluate(
            query="Analyze this image",
            metadata={"raw_data": self.dummy_bytes, "nested": {"bytes": self.dummy_bytes}},
        )

        dict_str = json.dumps(decision.to_dict())
        self.assertNotIn(str(self.dummy_bytes), dict_str)
        self.assertNotIn(str(self.dummy_bytes), repr(decision))
        self.assertNotIn(str(self.dummy_bytes), str(decision))

    def test_no_hallucinated_evidence(self) -> None:
        """12. Verify that fallback does not invent evidence or convert uncertainty into certainty."""
        decision = evaluate_safe_fallback(
            query="What is the internal code implementation?",
        )

        # It must not invent an answer or fake evidence items
        self.assertNotEqual(decision.action, FallbackAction.PROCEED)
        self.assertFalse(decision.should_proceed)

    def test_existing_valid_requests_not_unnecessarily_blocked(self) -> None:
        """13. Verify that normal valid requests with good evidence are never blocked."""
        ev = VisualEvidenceItem(description="Fully grounded diagram observation", confidence=0.99)
        decision = evaluate_safe_fallback(
            query="Summarize the system layout",
            evidence_items=[ev],
            evidence_validation=EvidenceValidationResult(
                status=EvidenceStatus.VALID,
                is_valid=True,
                total_items=1,
                valid_items_count=1,
                invalid_items_count=0,
                valid_evidence=[ev],
            ),
            confidence_assessment=ConfidenceAssessmentResult(
                status=ConfidenceStatus.AVAILABLE,
                is_confident=True,
                aggregate_confidence=0.99,
                min_confidence=0.99,
                max_confidence=0.99,
                threshold=0.70,
                total_items=1,
                valid_confidence_count=1,
                unavailable_confidence_count=0,
                invalid_confidence_count=0,
            ),
            missing_info_assessment=MissingInformationAssessmentResult(
                has_missing_information=False,
                severity=MissingInformationSeverity.NONE,
                missing_items=[],
                blocks_answering=False,
                available_summary={"has_image": True},
                reasons=[],
            ),
            ambiguity_assessment=AmbiguityAssessmentResult(
                is_ambiguous=False,
                ambiguity_level=AmbiguityLevel.NONE,
                requires_clarification=False,
            ),
        )

        self.assertEqual(decision.action, FallbackAction.PROCEED)
        self.assertTrue(decision.should_proceed)

    def test_to_multimodal_response_contract_conversion(self) -> None:
        """14. Verify conversion of FallbackDecision to canonical MultimodalResponse."""
        decision = FallbackDecision(
            action=FallbackAction.ASK_CLARIFICATION,
            should_proceed=False,
            reason="Ambiguous referent",
            safe_message="Which module do you mean?",
            clarification_needed="Specify module A or B",
            missing_requirements=["module selection"],
        )

        response = decision.to_multimodal_response(
            query="Explain the module",
            session_id="session_123",
            modality=ModalityType.TEXT_AND_IMAGE,
        )

        self.assertIsInstance(response, MultimodalResponse)
        self.assertEqual(response.session_id, "session_123")
        self.assertEqual(response.modality, ModalityType.TEXT_AND_IMAGE)
        self.assertEqual(response.answer, "Which module do you mean?")
        self.assertTrue(response.ambiguity.is_ambiguous)
        self.assertFalse(response.grounded)
        self.assertIn("Fallback action triggered", response.warning_message)


if __name__ == "__main__":
    unittest.main()
