"""Unit tests for Multimodal Context Retention Layer (Phase 5 — Day 27 Step 2).
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
    RetainedTurnView,
    retrieve_conversation_context,
)
from multimodal.conversation import (
    MultimodalConversationSession,
    MultimodalSessionManager,
)
from multimodal.models import (
    ImageArtifact,
    ModalityType,
    MultimodalResponse,
    VisualEvidenceItem,
)
from multimodal.vision import StructuredVisualOutput


class TestMultimodalContextRetention(unittest.TestCase):
    """Tests for Day 27 Step 2 context retention and retrieval capabilities."""

    def setUp(self):
        self.retriever = MultimodalContextRetriever(default_window_size=3)

        # Create sample responses
        self.resp_text = MultimodalResponse(
            query="Explain attention mechanisms",
            answer="Attention computes dynamic weighted alignments between token representations.",
            modality=ModalityType.TEXT_ONLY,
            session_id="sess_retention",
        )
        self.resp_vis = MultimodalResponse(
            query="Inspect the diagram",
            answer="The diagram illustrates a multi-stage encoder-decoder pipeline with residual links.",
            modality=ModalityType.TEXT_AND_IMAGE,
            session_id="sess_retention",
        )

        # Create sample image artifact
        self.dummy_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 60
        self.artifact = ImageArtifact(
            data=self.dummy_bytes,
            mime_type="image/png",
            width=640,
            height=480,
            format="PNG",
            file_name="pipeline_schematic.png",
        )

        self.visual_output = StructuredVisualOutput(
            scene_description="Technical flowchart depicting data pipeline",
            detected_objects=["arrow", "encoder_block", "decoder_block"],
            visible_text=["INPUT", "ENCODER", "DECODER", "OUTPUT"],
            visual_attributes={"primary_color": "teal", "complexity": "medium"},
        )

        self.context = build_multimodal_context(
            query="Inspect the diagram",
            artifact=self.artifact,
            visual_output=self.visual_output,
            session_id="sess_retention",
        )

    def test_retrieving_recent_turns(self):
        """1. Verify retrieving recent turns returns the latest conversational interactions."""
        session = MultimodalConversationSession(session_id="sess_recent", max_history_turns=5)
        session.add_turn(query="Q1", response=self.resp_text)
        session.add_turn(query="Q2", response=self.resp_vis, context=self.context)

        retained = self.retriever.retrieve_context(session)
        self.assertIsInstance(retained, RetainedConversationContext)
        self.assertTrue(retained.has_history)
        self.assertEqual(retained.turn_count, 2)
        self.assertEqual(retained.session_id, "sess_recent")

        last = retained.get_last_turn()
        self.assertIsNotNone(last)
        self.assertEqual(last.user_query, "Q2")
        self.assertEqual(last.scene_description, "Technical flowchart depicting data pipeline")

    def test_configurable_context_window(self):
        """2. Verify window_size restricts the number of retrieved turns appropriately."""
        session = MultimodalConversationSession(session_id="sess_window", max_history_turns=6)
        for i in range(5):
            resp = MultimodalResponse(
                query=f"Query {i}",
                answer=f"Answer {i}",
                modality=ModalityType.TEXT_ONLY,
                session_id="sess_window",
            )
            session.add_turn(query=f"Query {i}", response=resp)

        # Retrieve with window_size = 2
        retained_2 = self.retriever.retrieve_context(session, window_size=2)
        self.assertEqual(retained_2.turn_count, 2)
        self.assertEqual([t.user_query for t in retained_2.retained_turns], ["Query 3", "Query 4"])

        # Retrieve with window_size = 4
        retained_4 = self.retriever.retrieve_context(session, window_size=4)
        self.assertEqual(retained_4.turn_count, 4)
        self.assertEqual(
            [t.user_query for t in retained_4.retained_turns],
            ["Query 1", "Query 2", "Query 3", "Query 4"],
        )

    def test_chronological_ordering(self):
        """3. Verify retained turns are strictly ordered from oldest to newest."""
        session = MultimodalConversationSession(session_id="sess_order", max_history_turns=5)
        session.add_turn(query="First question", response=self.resp_text)
        session.add_turn(query="Second question", response=self.resp_vis, context=self.context)
        session.add_turn(query="Third question", response=self.resp_text)

        retained = self.retriever.retrieve_context(session, window_size=3)
        indices = [t.turn_index for t in retained.retained_turns]
        self.assertEqual(indices, [0, 1, 2])
        self.assertEqual(retained.retained_turns[0].user_query, "First question")
        self.assertEqual(retained.retained_turns[1].user_query, "Second question")
        self.assertEqual(retained.retained_turns[2].user_query, "Third question")

    def test_empty_session_behavior(self):
        """4. Verify safe behavior when retrieving from an empty or non-existent session."""
        empty_session = MultimodalConversationSession(session_id="sess_empty")
        retained = self.retriever.retrieve_context(empty_session)

        self.assertFalse(retained.has_history)
        self.assertEqual(retained.turn_count, 0)
        self.assertEqual(retained.retained_turns, [])
        self.assertIsNone(retained.get_last_turn())
        self.assertIn("No previous conversation history", retained.get_dialogue_history_text())

        # Passing None
        none_retained = self.retriever.retrieve_context(None)
        self.assertFalse(none_retained.has_history)
        self.assertEqual(none_retained.turn_count, 0)

    def test_multimodal_context_preservation(self):
        """5. Verify preservation of visual descriptions, detected entities, and metadata."""
        session = MultimodalConversationSession(session_id="sess_mm_pres")
        session.add_turn(query="Inspect pipeline", response=self.resp_vis, context=self.context)

        retained = self.retriever.retrieve_context(session)
        turn = retained.retained_turns[0]

        self.assertEqual(turn.scene_description, "Technical flowchart depicting data pipeline")
        self.assertIn("encoder_block", turn.detected_objects)
        self.assertIn("INPUT", turn.visible_text)
        self.assertEqual(turn.visual_attributes.get("primary_color"), "teal")
        self.assertEqual(turn.image_metadata.get("format"), "PNG")
        self.assertEqual(turn.image_metadata.get("width"), 640)

        # Aggregate helper check
        all_objs = retained.get_all_detected_objects()
        self.assertEqual(all_objs, ["arrow", "encoder_block", "decoder_block"])

        # Last visual summary helper check
        vis_summary = retained.get_last_visual_summary()
        self.assertIsNotNone(vis_summary)
        self.assertEqual(vis_summary["scene_description"], "Technical flowchart depicting data pipeline")

    def test_session_isolation(self):
        """6. Verify context from session A never leaks into session B."""
        manager = MultimodalSessionManager(default_max_history_turns=5)
        sess_a = manager.get_or_create_session("tenant_a")
        sess_b = manager.get_or_create_session("tenant_b")

        sess_a.add_turn(query="Secret A query", response=self.resp_text)

        ctx_a = self.retriever.retrieve_context_from_manager(manager, "tenant_a")
        ctx_b = self.retriever.retrieve_context_from_manager(manager, "tenant_b")

        self.assertTrue(ctx_a.has_history)
        self.assertEqual(ctx_a.turn_count, 1)
        self.assertEqual(ctx_a.retained_turns[0].user_query, "Secret A query")

        self.assertFalse(ctx_b.has_history)
        self.assertEqual(ctx_b.turn_count, 0)
        self.assertNotIn("Secret A query", ctx_b.get_dialogue_history_text())

    def test_no_raw_image_bytes_in_retained_context(self):
        """7. Verify zero raw image byte blobs exist in the retained context representation."""
        session = MultimodalConversationSession(session_id="sess_no_bytes")
        session.add_turn(query="Analyze bytes", response=self.resp_vis, context=self.context)

        retained = self.retriever.retrieve_context(session)
        d = retained.to_dict()
        d_str = str(d)

        self.assertNotIn("data", d["retained_turns"][0]["image_metadata"])
        self.assertNotIn("b'\\x89PNG", d_str)
        self.assertNotIn("b'\\x00", d_str)

    def test_deterministic_context_serialization(self):
        """8. Verify deterministic dictionary serialization and formatting across repeated calls."""
        session = MultimodalConversationSession(session_id="sess_det")
        session.add_turn(query="Turn 1", response=self.resp_text)
        session.add_turn(query="Turn 2", response=self.resp_vis, context=self.context)

        retained1 = self.retriever.retrieve_context(session, window_size=2)
        retained2 = self.retriever.retrieve_context(session, window_size=2)

        self.assertEqual(retained1.to_dict(), retained2.to_dict())
        self.assertEqual(retained1.get_dialogue_history_text(), retained2.get_dialogue_history_text())

    def test_context_window_respecting_session_maximum_history(self):
        """9. Verify requested window_size exceeding available turns or max history clamps safely."""
        session = MultimodalConversationSession(session_id="sess_bounds", max_history_turns=3)
        for i in range(5):
            resp = MultimodalResponse(
                query=f"Q{i}",
                answer=f"A{i}",
                modality=ModalityType.TEXT_ONLY,
                session_id="sess_bounds",
            )
            session.add_turn(query=f"Q{i}", response=resp)

        # Session has 3 retained turns (FIFO pruned from 5)
        # Request window of 10 -> should safely return all 3 available turns without crashing
        retained = self.retriever.retrieve_context(session, window_size=10)
        self.assertEqual(retained.turn_count, 3)
        self.assertEqual([t.user_query for t in retained.retained_turns], ["Q2", "Q3", "Q4"])

        # Function interface check
        func_retained = retrieve_conversation_context(session, window_size=2)
        self.assertEqual(func_retained.turn_count, 2)
        self.assertEqual([t.user_query for t in func_retained.retained_turns], ["Q3", "Q4"])


if __name__ == "__main__":
    unittest.main()
