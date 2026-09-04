"""Unit tests for Multimodal Ambiguity Detection (Phase 5 — Day 28 Step 3).
"""

import json
import os
import sys
import unittest

# Ensure src is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from multimodal.ambiguity import (
    AmbiguityAssessmentResult,
    AmbiguityLevel,
    AmbiguityType,
    MultimodalAmbiguityDetector,
    assess_multimodal_ambiguity,
)
from multimodal.confidence import (
    ConfidenceAssessmentResult,
    ConfidenceStatus,
)
from multimodal.context import (
    MultimodalContext,
    UnifiedEvidenceItem,
    build_multimodal_context,
)
from multimodal.context_retention import (
    RetainedConversationContext,
    RetainedTurnView,
)
from multimodal.evidence_validator import (
    EvidenceStatus,
    EvidenceValidationResult,
)
from multimodal.models import (
    AmbiguityDetails,
    ImageArtifact,
    ModalityType,
    MultimodalResponse,
    VisualEvidenceItem,
)
from multimodal.vision import StructuredVisualOutput


class TestMultimodalAmbiguityDetection(unittest.TestCase):
    """Tests for Day 28 Step 3 ambiguity detection and assessment component."""

    def setUp(self):
        self.detector = MultimodalAmbiguityDetector()

        # Dummy byte buffer for artifact testing
        self.dummy_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 40
        self.artifact = ImageArtifact(
            data=self.dummy_bytes,
            mime_type="image/png",
            width=640,
            height=480,
            format="PNG",
            file_name="pipeline.png",
        )

    def test_clearly_unambiguous_request(self):
        """1. Verify clearly unambiguous, well-grounded requests return no ambiguity."""
        query = "What is the computational complexity of transformer self-attention?"
        detected_objects = ["transformer_block"]

        result = self.detector.assess_ambiguity(
            query=query,
            detected_objects=detected_objects,
        )

        self.assertFalse(result.is_ambiguous)
        self.assertEqual(result.ambiguity_level, AmbiguityLevel.NONE)
        self.assertEqual(len(result.ambiguity_types), 0)
        self.assertEqual(len(result.reasons), 0)
        self.assertFalse(result.requires_clarification)
        self.assertIsNone(result.suggested_clarification)

    def test_vague_user_request(self):
        """2. Verify detection of vague or underspecified queries (e.g. 'Analyze this', 'What is this?')."""
        vague_queries = [
            "What is this?",
            "Analyze this",
            "Explain",
            "What about it?",
            "This stuff?",
        ]

        for q in vague_queries:
            result = self.detector.assess_ambiguity(query=q)
            self.assertTrue(result.is_ambiguous, f"Failed for vague query: {q}")
            self.assertIn(
                result.ambiguity_level,
                (AmbiguityLevel.MEDIUM, AmbiguityLevel.HIGH),
            )
            self.assertTrue(
                AmbiguityType.VAGUE_QUERY in result.ambiguity_types or AmbiguityType.INSUFFICIENT_SPECIFICITY in result.ambiguity_types
            )
            self.assertTrue(result.requires_clarification)
            self.assertIsNotNone(result.suggested_clarification)

    def test_ambiguous_visual_reference(self):
        """3. Verify detection when query references visual elements not detected in the image."""
        query = "What does the green module do?"
        # Image only contains blue and red components
        detected_objects = ["blue component", "red decoder"]

        result = self.detector.assess_ambiguity(
            query=query,
            detected_objects=detected_objects,
        )

        self.assertTrue(result.is_ambiguous)
        self.assertIn(AmbiguityType.UNCLEAR_VISUAL_REFERENCE, result.ambiguity_types)
        self.assertTrue(any("green module" in r.lower() for r in result.reasons))

    def test_multiple_plausible_visual_targets(self):
        """4. Verify detection when query refers to generic target and image has multiple matching items."""
        query = "What does the component do?"
        # Image contains multiple distinct components
        detected_objects = ["encoder component", "decoder component", "filter component"]

        result = self.detector.assess_ambiguity(
            query=query,
            detected_objects=detected_objects,
        )

        self.assertTrue(result.is_ambiguous)
        self.assertEqual(result.ambiguity_level, AmbiguityLevel.MEDIUM)
        self.assertIn(AmbiguityType.MULTIPLE_VISUAL_TARGETS, result.ambiguity_types)
        self.assertTrue(result.requires_clarification)
        self.assertGreaterEqual(len(result.candidate_interpretations), 2)
        self.assertIn("encoder component", result.candidate_interpretations)
        self.assertIn("decoder component", result.candidate_interpretations)
        self.assertIn("Which specific target are you referring to", result.suggested_clarification)

    def test_conflicting_evidence(self):
        """5. Verify detection of conflicting or contradictory evidence items."""
        evidence_items = [
            VisualEvidenceItem(description="Target component is present and active", confidence=0.9),
            VisualEvidenceItem(description="Target component is absent and disabled", confidence=0.85),
        ]

        result = self.detector.assess_ambiguity(
            query="Analyze component state",
            evidence=evidence_items,
        )

        self.assertTrue(result.is_ambiguous)
        self.assertEqual(result.ambiguity_level, AmbiguityLevel.HIGH)
        self.assertIn(AmbiguityType.CONFLICTING_EVIDENCE, result.ambiguity_types)
        self.assertTrue(result.requires_clarification)
        self.assertIn("conflicting", result.suggested_clarification.lower())

    def test_ambiguous_conversational_reference(self):
        """6. Verify detection when anaphoric pronoun refers to history with multiple candidate subjects."""
        # Setup retained history that discussed two distinct entities
        turn1 = RetainedTurnView(
            turn_index=0,
            modality=ModalityType.TEXT_AND_IMAGE,
            user_query="Inspect the neural encoder",
            assistant_answer="Encoder active",
            detected_objects=["neural_encoder"],
        )
        turn2 = RetainedTurnView(
            turn_index=1,
            modality=ModalityType.TEXT_AND_IMAGE,
            user_query="Now check the memory buffer",
            assistant_answer="Buffer active",
            detected_objects=["memory_buffer"],
        )
        history = RetainedConversationContext(
            session_id="sess_multi_hist",
            retained_turns=[turn1, turn2],
            total_session_turns=2,
            has_history=True,
        )

        # Follow-up with ambiguous pronoun "How does it work?"
        result = self.detector.assess_ambiguity(
            query="How does it work?",
            conversation_context=history,
        )

        self.assertTrue(result.is_ambiguous)
        self.assertTrue(
            AmbiguityType.AMBIGUOUS_DISCOURSE in result.ambiguity_types or AmbiguityType.VAGUE_QUERY in result.ambiguity_types
        )
        self.assertGreaterEqual(len(result.candidate_interpretations), 2)
        self.assertIn("neural_encoder", result.candidate_interpretations)
        self.assertIn("memory_buffer", result.candidate_interpretations)

    def test_multiple_ambiguity_reasons(self):
        """7. Verify reporting of multiple simultaneous ambiguity reasons triggering HIGH severity."""
        query = "Analyze this thing"  # Vague query + generic visual reference
        detected_objects = ["block A", "block B", "block C"]  # Multiple visual targets

        result = self.detector.assess_ambiguity(
            query=query,
            detected_objects=detected_objects,
        )

        self.assertTrue(result.is_ambiguous)
        self.assertEqual(result.ambiguity_level, AmbiguityLevel.HIGH)
        self.assertGreaterEqual(len(result.ambiguity_types), 2)
        self.assertIn(AmbiguityType.VAGUE_QUERY, result.ambiguity_types)
        self.assertIn(AmbiguityType.MULTIPLE_VISUAL_TARGETS, result.ambiguity_types)

    def test_deterministic_repeated_results(self):
        """8. Verify identical inputs produce strictly identical deterministic ambiguity reports."""
        query = "Explain the block"
        detected_objects = ["block 1", "block 2"]

        res1 = self.detector.assess_ambiguity(query=query, detected_objects=detected_objects)
        res2 = self.detector.assess_ambiguity(query=query, detected_objects=detected_objects)

        self.assertEqual(res1.is_ambiguous, res2.is_ambiguous)
        self.assertEqual(res1.ambiguity_level, res2.ambiguity_level)
        self.assertEqual(res1.reasons, res2.reasons)
        self.assertEqual(res1.suggested_clarification, res2.suggested_clarification)
        self.assertEqual(res1.to_dict(), res2.to_dict())

    def test_json_serialization_and_details_conversion(self):
        """9. Verify clean JSON serialization and conversion to Day 24 AmbiguityDetails."""
        query = "What does the block do?"
        detected_objects = ["encoder block", "decoder block"]

        result = self.detector.assess_ambiguity(query=query, detected_objects=detected_objects)

        # 1. Serialization to dict
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertTrue(d["is_ambiguous"])
        self.assertEqual(d["ambiguity_level"], "medium")

        # Must be valid JSON string
        json_str = json.dumps(d)
        self.assertIn('"is_ambiguous": true', json_str)
        self.assertIn('"ambiguity_level": "medium"', json_str)

        # 2. Conversion to canonical AmbiguityDetails
        details = result.to_ambiguity_details()
        self.assertIsInstance(details, AmbiguityDetails)
        self.assertTrue(details.is_ambiguous)
        self.assertIsNotNone(details.clarification_question)
        self.assertGreater(len(details.missing_aspects), 0)

    def test_raw_image_bytes_never_exposed(self):
        """10. Verify raw image byte payloads are never exposed in ambiguity assessment outputs."""
        vis_out = StructuredVisualOutput(
            scene_description="Technical schematic",
            detected_objects=["block A", "block B"],
        )
        ctx = build_multimodal_context(
            query="Explain the block",
            artifact=self.artifact,
            visual_output=vis_out,
        )

        result = self.detector.assess_ambiguity(query="Explain the block", context=ctx)
        self.assertTrue(result.is_ambiguous)

        d = result.to_dict()
        d_str = str(d)

        self.assertNotIn("b'\\x89PNG", d_str)
        self.assertNotIn("b'\\x00", d_str)

    def test_no_guessing_when_ambiguity_exists(self):
        """11. Verify that when multiple candidates exist, neither is arbitrarily chosen as definitive."""
        query = "What does the layer do?"
        detected_objects = ["convolutional layer", "pooling layer", "dense layer"]

        result = self.detector.assess_ambiguity(query=query, detected_objects=detected_objects)

        self.assertTrue(result.is_ambiguous)
        self.assertTrue(result.requires_clarification)
        # All candidate interpretations are explicitly recorded, not arbitrarily pruned to one
        self.assertEqual(len(result.candidate_interpretations), 3)
        self.assertIn("convolutional layer", result.candidate_interpretations)
        self.assertIn("pooling layer", result.candidate_interpretations)
        self.assertIn("dense layer", result.candidate_interpretations)


if __name__ == "__main__":
    unittest.main()
