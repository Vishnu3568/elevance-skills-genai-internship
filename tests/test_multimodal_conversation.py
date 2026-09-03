"""Unit tests for Multimodal Conversation State and Session Architecture (Phase 5 — Day 27 Step 1).
"""

import os
import sys
import unittest

# Ensure src is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from multimodal.context import (
    MultimodalContext,
    build_multimodal_context,
)
from multimodal.conversation import (
    ConversationTurn,
    MultimodalContextSummary,
    MultimodalConversationSession,
    MultimodalSessionManager,
    validate_conversation_turn,
)
from multimodal.models import (
    ImageArtifact,
    ModalityType,
    MultimodalResponse,
    VisualEvidenceItem,
)
from multimodal.vision import StructuredVisualOutput


class TestMultimodalConversationState(unittest.TestCase):
    """Tests for Day 27 Step 1 conversation session state and isolation."""

    def setUp(self):
        # Sample response
        self.sample_resp_1 = MultimodalResponse(
            query="What is the architecture?",
            answer="The architecture consists of transformer encoder and decoder blocks.",
            modality=ModalityType.TEXT_ONLY,
            session_id="sess_alpha",
            grounded=True,
        )
        self.sample_resp_2 = MultimodalResponse(
            query="Can you explain the attention layer?",
            answer="Multi-head attention projects queries, keys, and values into parallel subspaces.",
            modality=ModalityType.TEXT_ONLY,
            session_id="sess_alpha",
            grounded=True,
        )

        # Sample dummy image artifact
        self.dummy_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 50
        self.artifact = ImageArtifact(
            data=self.dummy_bytes,
            mime_type="image/png",
            width=320,
            height=240,
            format="PNG",
            file_name="diagram.png",
        )

    def test_new_session_creation(self):
        """1. Verify new session creation with default and custom configurations."""
        session = MultimodalConversationSession(session_id="session_01")
        self.assertEqual(session.session_id, "session_01")
        self.assertEqual(session.max_history_turns, 5)
        self.assertEqual(session.get_turn_count(), 0)
        self.assertEqual(session.total_turns_count, 0)
        self.assertIsNone(session.get_last_turn())

        custom_session = MultimodalConversationSession(session_id="session_custom", max_history_turns=10)
        self.assertEqual(custom_session.max_history_turns, 10)

    def test_adding_conversation_turn(self):
        """2. Verify adding a single turn records query, response, and turn index."""
        session = MultimodalConversationSession(session_id="sess_alpha")
        turn = session.add_turn(
            query="What is the architecture?",
            response=self.sample_resp_1,
        )

        self.assertIsInstance(turn, ConversationTurn)
        self.assertEqual(turn.turn_index, 0)
        self.assertEqual(turn.session_id, "sess_alpha")
        self.assertEqual(turn.user_query, "What is the architecture?")
        self.assertEqual(turn.assistant_response.answer, self.sample_resp_1.answer)
        self.assertEqual(turn.modality, ModalityType.TEXT_ONLY)
        self.assertEqual(session.get_turn_count(), 1)
        self.assertEqual(session.total_turns_count, 1)
        self.assertEqual(session.get_last_turn(), turn)

    def test_multiple_turns_preserve_order(self):
        """3. Verify multiple turns maintain strictly ordered sequence."""
        session = MultimodalConversationSession(session_id="sess_ordered")

        turn1 = session.add_turn(query="Question 1", response=self.sample_resp_1)
        turn2 = session.add_turn(query="Question 2", response=self.sample_resp_2)

        resp3 = MultimodalResponse(
            query="Question 3",
            answer="Third response answer",
            modality=ModalityType.TEXT_ONLY,
            session_id="sess_ordered",
        )
        turn3 = session.add_turn(query="Question 3", response=resp3)

        turns = session.get_turns()
        self.assertEqual(len(turns), 3)
        self.assertEqual(turns[0].turn_index, 0)
        self.assertEqual(turns[1].turn_index, 1)
        self.assertEqual(turns[2].turn_index, 2)
        self.assertEqual(turns[0].user_query, "Question 1")
        self.assertEqual(turns[1].user_query, "Question 2")
        self.assertEqual(turns[2].user_query, "Question 3")
        self.assertEqual(session.get_last_turn().turn_index, 2)

    def test_maximum_history_enforcement(self):
        """4. Verify bounded FIFO history pruning drops oldest turns while preserving total count."""
        session = MultimodalConversationSession(session_id="sess_bounded", max_history_turns=3)

        for i in range(5):
            resp = MultimodalResponse(
                query=f"Query {i}",
                answer=f"Answer {i}",
                modality=ModalityType.TEXT_ONLY,
                session_id="sess_bounded",
            )
            session.add_turn(query=f"Query {i}", response=resp)

        # Retained turns should be bounded to 3 (newest: 2, 3, 4)
        self.assertEqual(session.get_turn_count(), 3)
        self.assertEqual(session.total_turns_count, 5)

        retained_queries = [t.user_query for t in session.get_turns()]
        self.assertEqual(retained_queries, ["Query 2", "Query 3", "Query 4"])
        self.assertEqual(session.get_last_turn().user_query, "Query 4")

    def test_session_isolation(self):
        """5. Verify separate session instances and manager prevent cross-session leakage."""
        manager = MultimodalSessionManager(default_max_history_turns=4)

        session_a = manager.get_or_create_session("client_alice")
        session_b = manager.get_or_create_session("client_bob")

        session_a.add_turn(query="Alice question", response=self.sample_resp_1)

        self.assertEqual(session_a.get_turn_count(), 1)
        self.assertEqual(session_b.get_turn_count(), 0)

        session_b.add_turn(query="Bob question", response=self.sample_resp_2)

        self.assertEqual(session_a.get_turn_count(), 1)
        self.assertEqual(session_b.get_turn_count(), 1)
        self.assertEqual(session_a.get_turns()[0].user_query, "Alice question")
        self.assertEqual(session_b.get_turns()[0].user_query, "Bob question")

        self.assertTrue(manager.has_session("client_alice"))
        self.assertTrue(manager.has_session("client_bob"))
        self.assertFalse(manager.has_session("client_charlie"))

        manager.delete_session("client_alice")
        self.assertFalse(manager.has_session("client_alice"))
        self.assertTrue(manager.has_session("client_bob"))

    def test_empty_and_invalid_session_handling(self):
        """6. Verify appropriate validation errors on invalid session construction and turn addition."""
        # 1. Empty or non-string session_id
        with self.assertRaises(ValueError):
            MultimodalConversationSession(session_id="")
        with self.assertRaises(ValueError):
            MultimodalConversationSession(session_id="   ")
        with self.assertRaises(TypeError):
            MultimodalConversationSession(session_id=123)  # type: ignore

        # 2. Non-positive max_history_turns
        with self.assertRaises(ValueError):
            MultimodalConversationSession(session_id="valid", max_history_turns=0)
        with self.assertRaises(ValueError):
            MultimodalConversationSession(session_id="valid", max_history_turns=-3)
        with self.assertRaises(TypeError):
            MultimodalConversationSession(session_id="valid", max_history_turns="five")  # type: ignore

        # 3. Invalid turn parameters
        session = MultimodalConversationSession(session_id="valid_sess")
        with self.assertRaises(TypeError):
            session.add_turn(query=123, response=self.sample_resp_1)  # type: ignore
        with self.assertRaises(TypeError):
            session.add_turn(query="Query", response="not_a_response")  # type: ignore

    def test_serialization_and_inspection(self):
        """7. Verify serialization of conversation state and human-readable inspection."""
        session = MultimodalConversationSession(session_id="inspect_sess", max_history_turns=5)
        session.add_turn(query="Inspect test query", response=self.sample_resp_1)

        d = session.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["session_id"], "inspect_sess")
        self.assertEqual(d["retained_turn_count"], 1)
        self.assertEqual(d["total_turn_count"], 1)
        self.assertEqual(len(d["turns"]), 1)
        self.assertEqual(d["turns"][0]["user_query"], "Inspect test query")

        summary = session.get_history_summary()
        self.assertIn("inspect_sess", summary)
        self.assertIn("Inspect test query", summary)

    def test_no_raw_image_bytes_in_history(self):
        """8. Verify raw image byte payloads are not duplicated into conversation history."""
        # Build a multimodal context containing visual artifact and structured output
        vis_out = StructuredVisualOutput(
            scene_description="Technical flowchart depicting data pipeline",
            detected_objects=["arrow", "process_box"],
            visible_text=["INPUT", "OUTPUT"],
            visual_attributes={"primary_color": "teal"},
        )
        ctx = build_multimodal_context(
            query="Analyze data flow",
            artifact=self.artifact,
            visual_output=vis_out,
            session_id="sess_img_hist",
        )

        joint_resp = MultimodalResponse(
            query="Analyze data flow",
            answer="The flowchart indicates sequential data processing from input to output.",
            modality=ModalityType.TEXT_AND_IMAGE,
            visual_evidence=[VisualEvidenceItem(description="Data processing arrow", region_label="Center")],
            session_id="sess_img_hist",
        )

        session = MultimodalConversationSession(session_id="sess_img_hist")
        turn = session.add_turn(
            query="Analyze data flow",
            response=joint_resp,
            context=ctx,
        )

        self.assertIsNotNone(turn.context_summary)
        self.assertEqual(turn.context_summary.scene_description, "Technical flowchart depicting data pipeline")
        self.assertIn("arrow", turn.context_summary.detected_objects)

        # Inspect serialized dictionary
        d = session.to_dict()
        serialized_str = str(d)

        self.assertNotIn("data", d["turns"][0]["context_summary"]["image_metadata"])
        self.assertNotIn("b'\\x89PNG", serialized_str)
        self.assertNotIn("b'\\x00", serialized_str)
        self.assertEqual(d["turns"][0]["context_summary"]["image_metadata"]["format"], "PNG")
        self.assertEqual(d["turns"][0]["context_summary"]["image_metadata"]["width"], 320)


if __name__ == "__main__":
    unittest.main()
