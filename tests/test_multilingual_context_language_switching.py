"""Comprehensive Unit and Integration Tests for Multilingual Context Retention and Language Switching (Phase 6 Day 33).

Validates:
1. Multi-turn same-language conversational context retention.
2. Cross-turn pronoun and ellipsis reference resolution across EN, ES, FR, DE, HI.
3. Multi-turn language switching across conversational turns (EN -> ES, EN -> HI, ES -> DE, FR -> EN).
4. Context-aware cross-lingual retrieval and reasoning synthesis.
5. Strict session isolation and session reset lifecycle.
6. Bounded turn memory management.
7. Backward compatibility for stateless / single-turn requests.
"""

import unittest
from typing import Any, Dict, List

from src.multilingual.context import MultilingualContextResolver
from src.multilingual.detector import LanguageDetector
from src.multilingual.intents import MultilingualIntentClassifier
from src.multilingual.models import (
    MultilingualConversationSession,
    MultilingualConversationTurn,
    MultilingualIntent,
    MultilingualResponse,
    MultilingualTextRequest,
    SupportedLanguage,
)
from src.multilingual.reasoning import MultilingualReasoner
from src.multilingual.retrieval import CrossLingualRetriever
from src.multilingual.service import MultilingualService


class TestMultilingualContextAndLanguageSwitching(unittest.TestCase):
    """Test suite verifying context retention, reference resolution, and language switching."""

    def setUp(self) -> None:
        """Initialize mock-isolated and production-compatible service instances."""
        self.detector = LanguageDetector()
        self.intent_classifier = MultilingualIntentClassifier(language_detector=self.detector)

        # Mock vector store search function for deterministic test isolation
        def mock_search_fn(query: str, top_k: int) -> List[Dict[str, Any]]:
            q_lower = query.lower()
            if "refund" in q_lower or "money back" in q_lower:
                return [
                    {
                        "page_content": "We offer a 100% money-back guarantee within the refund policy terms.",
                        "metadata": {"topic": "refund_policy"},
                    }
                ]
            elif "prerequisite" in q_lower or "4gb" in q_lower or "laptop" in q_lower:
                return [
                    {
                        "page_content": "Prerequisites: Basic computer usage, laptop with 4GB RAM, no prior coding required.",
                        "metadata": {"topic": "prerequisites"},
                    }
                ]
            elif "duration" in q_lower or "lifetime" in q_lower or "syllabus" in q_lower:
                return [
                    {
                        "page_content": "Course duration is 6 months with lifetime access to materials and curriculum.",
                        "metadata": {"topic": "course_details"},
                    }
                ]
            elif "job" in q_lower or "internship" in q_lower or "career" in q_lower:
                return [
                    {
                        "page_content": "We provide virtual internship support, resume reviews, and placement assistance.",
                        "metadata": {"topic": "career_assistance"},
                    }
                ]
            return [
                {
                    "page_content": "General information regarding courses, bootcamps, and training programs.",
                    "metadata": {"topic": "general_inquiry"},
                }
            ]

        self.retriever = CrossLingualRetriever(custom_search_fn=mock_search_fn)
        self.reasoner = MultilingualReasoner(llm=False)
        self.service = MultilingualService(
            detector=self.detector,
            intent_classifier=self.intent_classifier,
            retriever=self.retriever,
            reasoner=self.reasoner,
            max_session_turns=5,
        )

    # -------------------------------------------------------------------------
    # A. Model Invariant & Contract Tests
    # -------------------------------------------------------------------------

    def test_01_conversation_turn_model_validation(self) -> None:
        """Verify MultilingualConversationTurn validation rules and serialization."""
        turn = MultilingualConversationTurn(
            turn_id=1,
            query="What are the prerequisites?",
            resolved_query="What are the prerequisites? (context: course)",
            language="en",
            intent=MultilingualIntent.PREREQUISITES.value,
            final_answer="Laptop with 4GB RAM is required.",
            topic="course prerequisites",
            is_grounded=True,
            timestamp="2026-09-13T10:00:00Z",
        )
        d = turn.to_dict()
        self.assertEqual(d["turn_id"], 1)
        self.assertEqual(d["language"], "en")
        self.assertEqual(d["intent"], "prerequisites")
        self.assertTrue(d["is_grounded"])

        # Invalid field types
        with self.assertRaises(ValueError):
            MultilingualConversationTurn(
                turn_id=-1,
                query="valid",
                resolved_query="valid",
                language="en",
                intent="prerequisites",
                final_answer="valid",
            )
        with self.assertRaises(ValueError):
            MultilingualConversationTurn(
                turn_id=1,
                query="   ",
                resolved_query="valid",
                language="en",
                intent="prerequisites",
                final_answer="valid",
            )

    def test_02_conversation_session_bounded_history(self) -> None:
        """Verify that MultilingualConversationSession enforces bounded turn history."""
        session = MultilingualConversationSession(session_id="sess-bound-test", max_turns=3)
        self.assertEqual(session.session_id, "sess-bound-test")
        self.assertEqual(session.max_turns, 3)
        self.assertEqual(len(session.turns), 0)

        # Add 5 turns to a session capped at 3
        for i in range(1, 6):
            session.add_turn(
                MultilingualConversationTurn(
                    turn_id=i,
                    query=f"Question {i}",
                    resolved_query=f"Question {i}",
                    language="en",
                    intent=MultilingualIntent.COURSE_DETAILS.value,
                    final_answer=f"Answer {i}",
                )
            )

        self.assertEqual(len(session.turns), 3)
        self.assertEqual(session.turns[0].turn_id, 3)
        self.assertEqual(session.turns[-1].turn_id, 5)
        self.assertEqual(session.get_last_turn().turn_id, 5)  # type: ignore

    # -------------------------------------------------------------------------
    # B. Context Retention (Same Language Multi-Turn)
    # -------------------------------------------------------------------------

    def test_03_same_language_english_multi_turn_retention(self) -> None:
        """Verify Turn 1 entity establishes context for Turn 2 follow-up in English."""
        session_id = "sess-en-followup"

        # Turn 1: Ask about course details
        req1 = MultilingualTextRequest(
            text="Tell me about the Data Science course details and syllabus.",
            session_id=session_id,
        )
        resp1 = self.service.answer_query(req1)
        self.assertEqual(resp1.language, "en")
        self.assertEqual(resp1.intent, MultilingualIntent.COURSE_DETAILS.value)

        # Turn 2: Follow-up using pronoun "its"
        req2 = MultilingualTextRequest(
            text="What are its prerequisites?",
            session_id=session_id,
        )
        proc2 = self.service.process_request(req2)
        self.assertTrue(proc2["is_followup"])
        self.assertIn("course details", proc2["resolved_query"].lower())
        self.assertEqual(proc2["intent"]["intent"], MultilingualIntent.PREREQUISITES.value)
        self.assertTrue(proc2["is_grounded"])

        # Check session records 2 turns
        sess = self.service.get_session(session_id)
        self.assertIsNotNone(sess)
        self.assertEqual(len(sess.turns), 2)  # type: ignore

    def test_04_same_language_spanish_multi_turn_retention(self) -> None:
        """Verify Turn 1 establishes context for Turn 2 follow-up in Spanish."""
        session_id = "sess-es-followup"

        # Turn 1: Course information in Spanish
        req1 = MultilingualTextRequest(
            text="¿Cuál es la información sobre el curso de ciencia de datos?",
            session_id=session_id,
        )
        resp1 = self.service.answer_query(req1)
        self.assertEqual(resp1.language, "es")

        # Turn 2: Follow-up with Spanish pronoun "sus"
        req2 = MultilingualTextRequest(
            text="¿Cuáles son sus requisitos previos?",
            session_id=session_id,
        )
        proc2 = self.service.process_request(req2)
        self.assertEqual(proc2["effective_language"], "es")
        self.assertTrue(proc2["is_followup"])
        self.assertEqual(proc2["intent"]["intent"], MultilingualIntent.PREREQUISITES.value)

    # -------------------------------------------------------------------------
    # C. Language Switching Between Turns
    # -------------------------------------------------------------------------

    def test_05_language_switching_english_to_hindi(self) -> None:
        """Verify Turn 1 English -> Turn 2 Hindi preserves context and responds in Hindi."""
        session_id = "sess-en-to-hi"

        # Turn 1: English course inquiry
        req1 = MultilingualTextRequest(
            text="I want information about the data science training course.",
            session_id=session_id,
        )
        resp1 = self.service.answer_query(req1)
        self.assertEqual(resp1.language, "en")
        self.assertEqual(resp1.intent, MultilingualIntent.GENERAL_INQUIRY.value)

        # Turn 2: Switch to Hindi follow-up asking for duration
        req2 = MultilingualTextRequest(
            text="इसकी अवधि कितनी है?",
            session_id=session_id,
        )
        proc2 = self.service.process_request(req2)
        self.assertEqual(proc2["effective_language"], "hi")
        self.assertTrue(proc2["is_followup"])
        self.assertEqual(proc2["intent"]["intent"], MultilingualIntent.COURSE_DETAILS.value)

        resp2 = self.service.answer_query(req2)
        self.assertEqual(resp2.language, "hi")
        self.assertTrue(resp2.is_grounded)
        self.assertIn("पाठ्यक्रम विवरण", resp2.final_answer)

    def test_06_language_switching_english_to_spanish(self) -> None:
        """Verify Turn 1 English -> Turn 2 Spanish preserves context and responds in Spanish."""
        session_id = "sess-en-to-es"

        # Turn 1: English
        req1 = MultilingualTextRequest(
            text="Can you explain the refund policy for the program?",
            session_id=session_id,
        )
        resp1 = self.service.answer_query(req1)
        self.assertEqual(resp1.language, "en")
        self.assertEqual(resp1.intent, MultilingualIntent.REFUND_POLICY.value)

        # Turn 2: Spanish follow-up asking for prerequisites
        req2 = MultilingualTextRequest(
            text="¿Y cuáles son los requisitos previos?",
            session_id=session_id,
        )
        resp2 = self.service.answer_query(req2)
        self.assertEqual(resp2.language, "es")
        self.assertEqual(resp2.intent, MultilingualIntent.PREREQUISITES.value)
        self.assertIn("Requisitos previos", resp2.final_answer)

    def test_07_language_switching_spanish_to_german(self) -> None:
        """Verify Turn 1 Spanish -> Turn 2 German preserves context and responds in German."""
        session_id = "sess-es-to-de"

        # Turn 1: Spanish
        req1 = MultilingualTextRequest(
            text="¿Ofrecen asistencia para conseguir empleo o prácticas?",
            session_id=session_id,
        )
        resp1 = self.service.answer_query(req1)
        self.assertEqual(resp1.language, "es")
        self.assertEqual(resp1.intent, MultilingualIntent.CAREER_ASSISTANCE.value)

        # Turn 2: German follow-up
        req2 = MultilingualTextRequest(
            text="Wie lange dauert das Programm?",
            session_id=session_id,
        )
        resp2 = self.service.answer_query(req2)
        self.assertEqual(resp2.language, "de")
        self.assertEqual(resp2.intent, MultilingualIntent.COURSE_DETAILS.value)
        self.assertIn("Kursdetails", resp2.final_answer)

    def test_08_language_switching_french_to_english(self) -> None:
        """Verify Turn 1 French -> Turn 2 English preserves context and responds in English."""
        session_id = "sess-fr-to-en"

        # Turn 1: French
        req1 = MultilingualTextRequest(
            text="Quels sont les prérequis pour suivre cette formation?",
            session_id=session_id,
        )
        resp1 = self.service.answer_query(req1)
        self.assertEqual(resp1.language, "fr")
        self.assertEqual(resp1.intent, MultilingualIntent.PREREQUISITES.value)

        # Turn 2: English follow-up
        req2 = MultilingualTextRequest(
            text="And how can I contact support on Discord?",
            session_id=session_id,
        )
        resp2 = self.service.answer_query(req2)
        self.assertEqual(resp2.language, "en")
        self.assertEqual(resp2.intent, MultilingualIntent.SUPPORT_CONTACT.value)

    # -------------------------------------------------------------------------
    # D. Session Isolation & Reset
    # -------------------------------------------------------------------------

    def test_09_session_isolation_between_distinct_sessions(self) -> None:
        """Verify that distinct session IDs maintain completely isolated histories."""
        # Session A: Inquires about refund policy
        req_a = MultilingualTextRequest(
            text="What is your refund policy?",
            session_id="session-user-A",
        )
        self.service.answer_query(req_a)

        # Session B: Inquires about career placement
        req_b = MultilingualTextRequest(
            text="Do you provide job assistance?",
            session_id="session-user-B",
        )
        self.service.answer_query(req_b)

        # Check turn counts and isolated topics
        sess_a = self.service.get_session("session-user-A")
        sess_b = self.service.get_session("session-user-B")
        self.assertIsNotNone(sess_a)
        self.assertIsNotNone(sess_b)
        self.assertEqual(sess_a.turns[0].intent, MultilingualIntent.REFUND_POLICY.value)  # type: ignore
        self.assertEqual(sess_b.turns[0].intent, MultilingualIntent.CAREER_ASSISTANCE.value)  # type: ignore

    def test_10_session_clear_and_reset_all_lifecycle(self) -> None:
        """Verify that clearing single session or resetting all sessions cleans history."""
        self.service.answer_query(MultilingualTextRequest(text="Hello!", session_id="sess-1"))
        self.service.answer_query(MultilingualTextRequest(text="Hello!", session_id="sess-2"))

        self.assertIsNotNone(self.service.get_session("sess-1"))
        self.assertIsNotNone(self.service.get_session("sess-2"))

        # Clear sess-1
        self.service.clear_session("sess-1")
        self.assertIsNone(self.service.get_session("sess-1"))
        self.assertIsNone(self.service.get_session_language("sess-1"))
        self.assertIsNotNone(self.service.get_session("sess-2"))

        # Reset all
        self.service.reset_all_sessions()
        self.assertIsNone(self.service.get_session("sess-2"))
        self.assertEqual(len(self.service._sessions), 0)
        self.assertEqual(len(self.service._session_languages), 0)

    # -------------------------------------------------------------------------
    # E. Backward Compatibility & Stateless Execution
    # -------------------------------------------------------------------------

    def test_11_stateless_request_without_session_id_compatibility(self) -> None:
        """Verify that single-turn queries without session_id operate safely and statelessly."""
        req = MultilingualTextRequest(text="¿Cómo funciona el reembolso?")
        resp = self.service.answer_query(req)
        self.assertEqual(resp.language, "es")
        self.assertEqual(resp.intent, MultilingualIntent.REFUND_POLICY.value)
        self.assertTrue(resp.is_grounded)
        self.assertEqual(len(self.service._sessions), 0)

    def test_12_context_aware_reasoner_with_custom_llm_adapter(self) -> None:
        """Verify that when an LLM adapter is configured, conversation history is passed into prompt."""
        class CapturingMockLLM:
            def __init__(self) -> None:
                self.last_prompt: str = ""

            def invoke(self, prompt: str) -> str:
                self.last_prompt = prompt
                return "Mock LLM contextual response."

        mock_llm = CapturingMockLLM()
        custom_reasoner = MultilingualReasoner(llm=mock_llm)
        service = MultilingualService(
            detector=self.detector,
            intent_classifier=self.intent_classifier,
            retriever=self.retriever,
            reasoner=custom_reasoner,
        )

        # Turn 1
        service.answer_query(MultilingualTextRequest(text="Tell me about the course.", session_id="llm-sess"))
        # Turn 2
        service.answer_query(MultilingualTextRequest(text="What are its prerequisites?", session_id="llm-sess"))

        self.assertIn("CONVERSATION HISTORY:", mock_llm.last_prompt)
        self.assertIn("Tell me about the course.", mock_llm.last_prompt)

    # -------------------------------------------------------------------------
    # F. Multi-Hop Language Switching & Edge Cases
    # -------------------------------------------------------------------------

    def test_13_multi_turn_consecutive_language_switching_en_es_de_hi(self) -> None:
        """Verify seamless 4-turn language switching chain: EN -> ES -> DE -> HI in single session."""
        session_id = "sess-multi-hop"

        # Turn 1: EN
        r1 = self.service.answer_query(MultilingualTextRequest(text="What is the course syllabus and duration?", session_id=session_id))
        self.assertEqual(r1.language, "en")
        self.assertEqual(r1.intent, MultilingualIntent.COURSE_DETAILS.value)

        # Turn 2: ES
        r2 = self.service.answer_query(MultilingualTextRequest(text="¿Cuáles son sus requisitos?", session_id=session_id))
        self.assertEqual(r2.language, "es")
        self.assertEqual(r2.intent, MultilingualIntent.PREREQUISITES.value)

        # Turn 3: DE
        r3 = self.service.answer_query(MultilingualTextRequest(text="Bieten Sie Karrierehilfe und Berufsunterstützung an?", session_id=session_id))
        self.assertEqual(r3.language, "de")
        self.assertEqual(r3.intent, MultilingualIntent.CAREER_ASSISTANCE.value)

        # Turn 4: HI
        r4 = self.service.answer_query(MultilingualTextRequest(text="रिफंड नीति क्या है?", session_id=session_id))
        self.assertEqual(r4.language, "hi")
        self.assertEqual(r4.intent, MultilingualIntent.REFUND_POLICY.value)

        sess = self.service.get_session(session_id)
        self.assertIsNotNone(sess)
        self.assertEqual(len(sess.turns), 4)  # type: ignore
        self.assertEqual(sess.turns[0].language, "en")  # type: ignore
        self.assertEqual(sess.turns[1].language, "es")  # type: ignore
        self.assertEqual(sess.turns[2].language, "de")  # type: ignore
        self.assertEqual(sess.turns[3].language, "hi")  # type: ignore

    def test_14_nonexistent_or_empty_session_id(self) -> None:
        """Verify that requests without session_id or with new session_ids work safely."""
        r1 = self.service.answer_query(MultilingualTextRequest(text="Hello!"))
        self.assertEqual(r1.intent, MultilingualIntent.GREETING.value)
        self.assertIsNone(self.service.get_session("non-existent"))

    def test_15_context_resolver_pronoun_matching_across_languages(self) -> None:
        """Direct unit verification of MultilingualContextResolver across supported languages."""
        session = MultilingualConversationSession(session_id="dummy-sess")
        session.add_turn(
            MultilingualConversationTurn(
                turn_id=1,
                query="Tell me about the course",
                resolved_query="Tell me about the course",
                language="en",
                intent=MultilingualIntent.COURSE_DETAILS.value,
                final_answer="Details.",
                topic="course details",
            )
        )

        # English pronoun
        self.assertTrue(MultilingualContextResolver.is_followup_query("What are its prerequisites?", "en", session))
        # Spanish pronoun
        self.assertTrue(MultilingualContextResolver.is_followup_query("¿Cuáles son sus requisitos?", "es", session))
        # French pronoun
        self.assertTrue(MultilingualContextResolver.is_followup_query("Quelle est sa durée?", "fr", session))
        # German pronoun
        self.assertTrue(MultilingualContextResolver.is_followup_query("Was sind seine Voraussetzungen?", "de", session))
        # Hindi pronoun
        self.assertTrue(MultilingualContextResolver.is_followup_query("इसकी अवधि क्या है?", "hi", session))
        # Standalone without pronoun
        resolved, topic, is_followup = MultilingualContextResolver.resolve_context("What are its prerequisites?", "en", session)
        self.assertTrue(is_followup)
        self.assertEqual(topic, "course details")
        self.assertIn("course details", resolved)

    def test_16_concurrent_session_isolation_three_users(self) -> None:
        """Verify strict isolation across 3 simultaneous multi-turn user sessions."""
        self.service.answer_query(MultilingualTextRequest(text="I want course info", session_id="user-1"))
        self.service.answer_query(MultilingualTextRequest(text="¿Cómo funciona el reembolso?", session_id="user-2"))
        self.service.answer_query(MultilingualTextRequest(text="Bieten Sie Praktika an?", session_id="user-3"))

        # Follow-ups
        r1 = self.service.answer_query(MultilingualTextRequest(text="What are its prerequisites?", session_id="user-1"))
        r2 = self.service.answer_query(MultilingualTextRequest(text="¿Y en cuánto tiempo se procesa?", session_id="user-2"))
        r3 = self.service.answer_query(MultilingualTextRequest(text="Und wie kann ich mich bewerben?", session_id="user-3"))

        self.assertEqual(r1.language, "en")
        self.assertEqual(r2.language, "es")
        self.assertEqual(r3.language, "de")

        s1 = self.service.get_session("user-1")
        s2 = self.service.get_session("user-2")
        s3 = self.service.get_session("user-3")
        self.assertEqual(len(s1.turns), 2)  # type: ignore
        self.assertEqual(len(s2.turns), 2)  # type: ignore
        self.assertEqual(len(s3.turns), 2)  # type: ignore


if __name__ == "__main__":
    unittest.main()

