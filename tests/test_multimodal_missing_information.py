"""Unit tests for Multimodal Missing Information Assessment (Phase 5 — Day 28 Step 4).

Tests deterministic identification and structured assessment of missing information
required to reliably answer multimodal requests, strictly verifying zero raw byte
leakage, non-guessing behavior, and clear separation from ambiguity and invalid evidence.
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
from src.multimodal.context_retention import RetainedConversationContext, RetainedTurnView
from src.multimodal.conversation import ConversationTurn
from src.multimodal.evidence_validator import (
    EvidenceItemValidation,
    EvidenceStatus,
    EvidenceValidationResult,
)
from src.multimodal.missing_information import (
    MissingInformationAssessmentResult,
    MissingInformationItem,
    MissingInformationSeverity,
    MissingInformationType,
    MultimodalMissingInformationDetector,
    assess_multimodal_missing_information,
)
from src.multimodal.models import (
    ImageArtifact,
    ModalityType,
    MultimodalRequest,
    VisualEvidenceItem,
)


class TestMultimodalMissingInformation(unittest.TestCase):
    """Test suite for deterministic missing information assessment."""

    def setUp(self) -> None:
        """Set up standard fixtures."""
        self.detector = MultimodalMissingInformationDetector()
        self.dummy_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x02\x00\x00\x00\x90\x91h6"
        self.dummy_artifact = ImageArtifact(
            data=self.dummy_bytes,
            mime_type="image/png",
            width=64,
            height=64,
            format="png",
        )

    def test_complete_information_nothing_missing(self) -> None:
        """1. Verify complete information request returns no missing information."""
        result = self.detector.assess(
            query="What is the architecture of this system?",
            images=[self.dummy_artifact],
            visual_objects=["Load Balancer", "API Gateway", "Database"],
            visible_text=["Load Balancer", "API Gateway"],
            evidence_items=[
                VisualEvidenceItem(description="Load Balancer routing to API Gateway", confidence=0.95),
            ],
        )

        self.assertFalse(result.has_missing_information)
        self.assertEqual(result.severity, MissingInformationSeverity.NONE)
        self.assertEqual(len(result.missing_items), 0)
        self.assertFalse(result.blocks_answering)
        self.assertTrue(result.available_summary["has_image"])
        self.assertEqual(result.available_summary["detected_objects_count"], 3)

    def test_missing_image(self) -> None:
        """2. Verify detection when query explicitly requires an image but none is provided."""
        result = self.detector.assess(
            query="What does this diagram show in the attached image?",
            images=None,  # No image provided
        )

        self.assertTrue(result.has_missing_information)
        self.assertEqual(result.severity, MissingInformationSeverity.CRITICAL)
        self.assertTrue(result.blocks_answering)
        missing_types = [item.item_type for item in result.missing_items]
        self.assertIn(MissingInformationType.MISSING_IMAGE, missing_types)

    def test_missing_visual_evidence(self) -> None:
        """3. Verify detection when image is present but visual processing extracted zero evidence."""
        result = self.detector.assess(
            query="What components and elements are depicted here?",
            images=[self.dummy_artifact],
            visual_objects=[],
            visible_text=[],
            evidence_items=[],
        )

        self.assertTrue(result.has_missing_information)
        self.assertEqual(result.severity, MissingInformationSeverity.HIGH)
        self.assertTrue(result.blocks_answering)
        missing_types = [item.item_type for item in result.missing_items]
        self.assertIn(MissingInformationType.MISSING_VISUAL_EVIDENCE, missing_types)

    def test_missing_requested_property_detail(self) -> None:
        """4. Verify detection when user asks for fine-grained details not present in diagram."""
        result = self.detector.assess(
            query="What is the Python source code and function implementation inside this component?",
            images=[self.dummy_artifact],
            visual_objects=["Component A", "Component B"],
            visible_text=["Component A", "Component B"],
            evidence_items=[
                VisualEvidenceItem(description="Component A is connected to Component B", confidence=0.9)
            ],
        )

        self.assertTrue(result.has_missing_information)
        missing_types = [item.item_type for item in result.missing_items]
        self.assertIn(MissingInformationType.MISSING_PROPERTY_OR_DETAIL, missing_types)
        self.assertTrue(result.blocks_answering)

    def test_missing_conversational_context(self) -> None:
        """5. Verify detection when follow-up query depends on history that is absent."""
        result = self.detector.assess(
            query="What was the second option we discussed in your previous answer?",
            conversation_context=None,  # Zero conversation history
        )

        self.assertTrue(result.has_missing_information)
        self.assertEqual(result.severity, MissingInformationSeverity.HIGH)
        self.assertTrue(result.blocks_answering)
        missing_types = [item.item_type for item in result.missing_items]
        self.assertIn(MissingInformationType.MISSING_CONVERSATION_CONTEXT, missing_types)

    def test_missing_required_reference_target(self) -> None:
        """6. Verify detection when query asks about a specific entity absent from all context."""
        result = self.detector.assess(
            query="Explain how Component C processes inputs",
            images=[self.dummy_artifact],
            visual_objects=["Component A", "Component B"],
            visible_text=["Component A", "Component B"],
        )

        self.assertTrue(result.has_missing_information)
        missing_types = [item.item_type for item in result.missing_items]
        self.assertIn(MissingInformationType.MISSING_REFERENCE_TARGET, missing_types)
        affected_refs = [item.affected_reference for item in result.missing_items]
        self.assertIn("C", affected_refs)

    def test_multiple_missing_information_items(self) -> None:
        """7. Verify reporting of multiple missing items simultaneously."""
        # Query requires image AND asks about previous conversational decision
        result = self.detector.assess(
            query="In the diagram you mentioned earlier, what does it show?",
            images=None,
            conversation_context=None,
        )

        self.assertTrue(result.has_missing_information)
        self.assertGreaterEqual(len(result.missing_items), 2)
        missing_types = [item.item_type for item in result.missing_items]
        self.assertIn(MissingInformationType.MISSING_IMAGE, missing_types)
        self.assertIn(MissingInformationType.MISSING_CONVERSATION_CONTEXT, missing_types)
        self.assertTrue(result.blocks_answering)

    def test_distinguishing_missing_information_from_ambiguity(self) -> None:
        """8. Verify that generic referent with multiple present targets is NOT classified as missing info."""
        # Query refers to "the block". In image, "Block 1" and "Block 2" exist.
        # This is ambiguous, but the blocks ARE present in the image, so information is NOT missing.
        ambiguity_res = AmbiguityAssessmentResult(
            is_ambiguous=True,
            ambiguity_level=AmbiguityLevel.MEDIUM,
            ambiguity_types=[AmbiguityType.MULTIPLE_VISUAL_TARGETS],
            reasons=["Query refers to 'the block', but multiple plausible visual targets exist."],
            affected_references=["the block"],
            candidate_interpretations=["Block 1", "Block 2"],
            requires_clarification=True,
        )

        result = self.detector.assess(
            query="What does the block do?",
            images=[self.dummy_artifact],
            visual_objects=["Block 1", "Block 2"],
            visible_text=["Block 1", "Block 2"],
            ambiguity_assessment=ambiguity_res,
        )

        # It should NOT be flagged as missing reference target because candidates are present
        missing_types = [item.item_type for item in result.missing_items]
        self.assertNotIn(MissingInformationType.MISSING_REFERENCE_TARGET, missing_types)
        self.assertNotIn(MissingInformationType.MISSING_IMAGE, missing_types)

    def test_distinguishing_missing_information_from_invalid_evidence(self) -> None:
        """9. Verify that an evidence validation error alone does not create missing info when facts are present."""
        # Evidence validator found one malformed item, but the context has detected objects and valid items
        validation_res = EvidenceValidationResult(
            status=EvidenceStatus.PARTIAL,
            is_valid=False,
            total_items=2,
            valid_items_count=1,
            invalid_items_count=1,
            item_validations=[
                EvidenceItemValidation(item_index=0, is_valid=True, description="Valid observation"),
                EvidenceItemValidation(item_index=1, is_valid=False, error_messages=["Invalid confidence"]),
            ],
        )

        result = self.detector.assess(
            query="What is the architecture?",
            images=[self.dummy_artifact],
            visual_objects=["Module A", "Module B"],
            visible_text=["Module A", "Module B"],
            evidence_validation=validation_res,
        )

        self.assertFalse(result.has_missing_information)
        self.assertEqual(len(result.missing_items), 0)

    def test_deterministic_repeated_results(self) -> None:
        """10. Verify identical inputs produce strictly identical assessment outputs."""
        kwargs = {
            "query": "In this image, show the source code of Component X",
            "images": [self.dummy_artifact],
            "visual_objects": ["Module 1"],
            "visible_text": ["Module 1"],
        }

        res1 = self.detector.assess(**kwargs)
        res2 = self.detector.assess(**kwargs)

        self.assertEqual(res1.to_dict(), res2.to_dict())
        self.assertEqual(res1.has_missing_information, res2.has_missing_information)
        self.assertEqual(res1.severity, res2.severity)
        self.assertEqual(res1.blocks_answering, res2.blocks_answering)

    def test_json_serialization(self) -> None:
        """11. Verify clean JSON serialization of assessment result."""
        result = self.detector.assess(
            query="What did we say about the previous topic in the attached image?",
            images=None,
            conversation_context=None,
        )

        res_dict = result.to_dict()
        serialized = json.dumps(res_dict)
        deserialized = json.loads(serialized)

        self.assertTrue(deserialized["has_missing_information"])
        self.assertEqual(deserialized["severity"], "critical")
        self.assertIsInstance(deserialized["missing_items"], list)
        self.assertIsInstance(deserialized["available_summary"], dict)

    def test_raw_image_bytes_never_exposed(self) -> None:
        """12. Verify raw image byte payloads are strictly excluded from all outputs and strings."""
        result = self.detector.assess(
            query="Describe this image",
            images=[self.dummy_artifact],
            visual_objects=[],
            visible_text=[],
        )

        res_dict = result.to_dict()
        dict_str = json.dumps(res_dict)

        self.assertNotIn(str(self.dummy_bytes), dict_str)
        self.assertNotIn(str(self.dummy_bytes), repr(result))
        self.assertNotIn(str(self.dummy_bytes), str(result))

    def test_no_guessing_fabrication_of_missing_information(self) -> None:
        """13. Verify that assessment identifies missing requirements without fabricating answers."""
        result = assess_multimodal_missing_information(
            query="What is the exact price and release date of this system?",
            images=[self.dummy_artifact],
            visual_objects=["System Overview"],
            visible_text=["System Overview"],
        )

        self.assertTrue(result.has_missing_information)
        self.assertTrue(result.blocks_answering)
        # Verify it specifically diagnoses the missing properties rather than inventing them
        descriptions = [item.description for item in result.missing_items]
        self.assertTrue(any("pricing" in d for d in descriptions))
        self.assertTrue(any("date" in d for d in descriptions))


if __name__ == "__main__":
    unittest.main()
