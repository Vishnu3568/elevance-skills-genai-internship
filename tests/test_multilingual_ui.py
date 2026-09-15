"""Unit and Component Tests for Multilingual Streamlit UI (Phase 6 — Day 35).

Tests:
1. Module exports and functions in multilingual_main.py.
2. Service initialization via cached factory.
3. Request model construction and validation.
4. Auto-detect mode vs Forced language modes.
5. Session ID continuity across multi-turn queries.
6. Session reset mechanics and service state clearing.
7. Dynamic language switching within active UI sessions.
8. Mixed-language / Hinglish input processing.
9. Ambiguity and localized clarification prompt rendering.
10. Grounded answer vs fallback rendering safety.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import multilingual_main
from src.multilingual import (
    CrossLingualRetriever,
    LanguageDetector,
    MultilingualContextResolver,
    MultilingualIntent,
    MultilingualIntentClassifier,
    MultilingualReasoner,
    MultilingualResponse,
    MultilingualService,
    MultilingualTextRequest,
    SupportedLanguage,
)


class TestMultilingualUI(unittest.TestCase):
    """Unit test suite for the Multilingual Streamlit UI and service integration."""

    def setUp(self) -> None:
        """Clear cached resources and initialize deterministic service before each test."""
        clear_cache = getattr(multilingual_main.initialize_multilingual_service, "clear", None)
        if clear_cache:
            clear_cache()
        self.detector = LanguageDetector()
        self.intent_classifier = MultilingualIntentClassifier(language_detector=self.detector)

        self.mock_docs = [
            {
                "page_content": "The Data Science bootcamp costs $1,200. Installment options are available.",
                "metadata": {"topic": "fees", "intent": "PAYMENT_PRICING"},
            },
            {
                "page_content": "We offer a 100% refund policy within the first 14 days of enrollment.",
                "metadata": {"topic": "refund", "intent": "REFUND_POLICY"},
            },
            {
                "page_content": "Prerequisites: Basic computer skills. No prior coding experience required. 4GB RAM laptop needed.",
                "metadata": {"topic": "prerequisites", "intent": "PREREQUISITES"},
            },
        ]

        def mock_search_fn(query: str, top_k: int = 3):
            q_lower = query.lower()
            results = []
            for doc in self.mock_docs:
                content_lower = doc["page_content"].lower()
                topic_lower = doc["metadata"]["topic"].lower()
                if topic_lower in q_lower or any(w in content_lower for w in q_lower.split() if len(w) > 3):
                    results.append(doc)
            return results[:top_k] if results else []

        self.retriever = CrossLingualRetriever(custom_search_fn=mock_search_fn)
        self.reasoner = MultilingualReasoner(llm="offline")
        self.context_resolver = MultilingualContextResolver()
        self.service = MultilingualService(
            detector=self.detector,
            intent_classifier=self.intent_classifier,
            retriever=self.retriever,
            reasoner=self.reasoner,
            context_resolver=self.context_resolver,
        )

    # -------------------------------------------------------------------------
    # 1. Module Exports & Symbols
    # -------------------------------------------------------------------------
    def test_01_module_exports(self) -> None:
        """Verify multilingual_main exports required functions and configurations."""
        self.assertTrue(hasattr(multilingual_main, "initialize_multilingual_service"))
        self.assertTrue(hasattr(multilingual_main, "render_multilingual_response"))
        self.assertTrue(hasattr(multilingual_main, "main"))
        self.assertTrue(hasattr(multilingual_main, "LANGUAGE_OPTIONS"))
        self.assertIn("Auto-Detect (Adaptive)", multilingual_main.LANGUAGE_OPTIONS)
        self.assertIn("Spanish (es)", multilingual_main.LANGUAGE_OPTIONS)
        self.assertIn("Hindi (hi)", multilingual_main.LANGUAGE_OPTIONS)

    # -------------------------------------------------------------------------
    # 2. Service Initialization Factory
    # -------------------------------------------------------------------------
    def test_02_service_initialization(self) -> None:
        """Verify initialize_multilingual_service returns an active MultilingualService."""
        service, error = multilingual_main.initialize_multilingual_service()
        self.assertIsNotNone(service)
        self.assertIsNone(error)
        self.assertIsInstance(service, MultilingualService)

    # -------------------------------------------------------------------------
    # 3. Request Model Construction
    # -------------------------------------------------------------------------
    def test_03_request_model_construction(self) -> None:
        """Verify valid MultilingualTextRequest construction from UI parameters."""
        req = MultilingualTextRequest(
            text="¿Cuáles son los requisitos previos?",
            forced_language="es",
            session_id="test-session-123",
        )
        self.assertEqual(req.text, "¿Cuáles son los requisitos previos?")
        self.assertEqual(req.forced_language, "es")
        self.assertEqual(req.session_id, "test-session-123")

    # -------------------------------------------------------------------------
    # 4. Auto-Detect vs Forced Language Modes
    # -------------------------------------------------------------------------
    def test_04_auto_detect_vs_forced_language(self) -> None:
        """Verify Auto-Detect uses detected language while Forced Language overrides it."""
        # Auto-detect mode (forced_language=None)
        req_auto = MultilingualTextRequest(
            text="Quelle est la durée de la formation?",
            forced_language=None,
        )
        res_auto = self.service.process_request(req_auto)
        self.assertEqual(res_auto["effective_language"], SupportedLanguage.FRENCH.value)
        self.assertFalse(res_auto["is_forced"])

        # Forced mode (forced_language="es")
        req_forced = MultilingualTextRequest(
            text="Quelle est la durée de la formation?",
            forced_language=SupportedLanguage.SPANISH.value,
        )
        res_forced = self.service.process_request(req_forced)
        self.assertEqual(res_forced["effective_language"], SupportedLanguage.SPANISH.value)
        self.assertTrue(res_forced["is_forced"])

    # -------------------------------------------------------------------------
    # 5. Session ID Continuity
    # -------------------------------------------------------------------------
    def test_05_session_id_continuity(self) -> None:
        """Verify session ID maintains multi-turn conversation history."""
        session_id = "ui-test-session-456"

        req1 = MultilingualTextRequest(
            text="Tell me about the Data Science bootcamp.",
            session_id=session_id,
        )
        self.service.process_request(req1)

        req2 = MultilingualTextRequest(
            text="What are its prerequisites?",
            session_id=session_id,
        )
        res2 = self.service.process_request(req2)
        self.assertTrue(res2["is_followup"])

        session = self.service.get_session(session_id)
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(len(session.turns), 2)

    # -------------------------------------------------------------------------
    # 6. Session Reset Mechanics
    # -------------------------------------------------------------------------
    def test_06_session_reset(self) -> None:
        """Verify clearing session removes history and isolates new turns."""
        session_id = "ui-test-session-reset"

        # Create 2 turns
        self.service.process_request(MultilingualTextRequest(text="Hello", session_id=session_id))
        self.service.process_request(MultilingualTextRequest(text="Course info", session_id=session_id))
        self.assertIsNotNone(self.service.get_session(session_id))

        # Clear session
        self.service.clear_session(session_id)
        self.assertIsNone(self.service.get_session(session_id))

    # -------------------------------------------------------------------------
    # 7. Language Switching in UI Session
    # -------------------------------------------------------------------------
    def test_07_language_switching_in_session(self) -> None:
        """Verify seamless language switching across consecutive turns in a UI session."""
        session_id = "ui-test-lang-switch"

        # Turn 1: English
        r1 = self.service.process_request(
            MultilingualTextRequest(text="Tell me about Machine Learning course.", session_id=session_id)
        )
        self.assertEqual(r1["effective_language"], SupportedLanguage.ENGLISH.value)

        # Turn 2: Hindi follow-up
        r2 = self.service.process_request(
            MultilingualTextRequest(text="इसकी फीस कितनी है?", session_id=session_id)
        )
        self.assertEqual(r2["effective_language"], SupportedLanguage.HINDI.value)
        self.assertTrue(r2["is_followup"])

        # Turn 3: Spanish follow-up
        r3 = self.service.process_request(
            MultilingualTextRequest(text="¿Y cuáles son los requisitos previos?", session_id=session_id)
        )
        self.assertEqual(r3["effective_language"], SupportedLanguage.SPANISH.value)
        self.assertTrue(r3["is_followup"])

    # -------------------------------------------------------------------------
    # 8. Mixed-Language & Hinglish Input
    # -------------------------------------------------------------------------
    def test_08_mixed_language_and_hinglish_input(self) -> None:
        """Verify mixed-language and Hinglish queries pass correctly to service."""
        req = MultilingualTextRequest(text="Python course की fees कितनी है?")
        res = self.service.process_request(req)
        self.assertEqual(res["effective_language"], SupportedLanguage.HINDI.value)
        self.assertEqual(res["intent"]["intent"], MultilingualIntent.PAYMENT_PRICING.value)
        self.assertTrue(res["detection"]["is_mixed_language"])

    # -------------------------------------------------------------------------
    # 9. Ambiguity & Clarification Handling
    # -------------------------------------------------------------------------
    def test_09_ambiguity_and_clarification_handling(self) -> None:
        """Verify ambiguous dual-intent queries produce ambiguity metadata."""
        req = MultilingualTextRequest(text="What are the course fees and refund policy?")
        res = self.service.process_request(req)
        self.assertTrue(res["intent"]["is_ambiguous"])
        self.assertGreaterEqual(len(res["intent"]["competing_intents"]), 2)

    # -------------------------------------------------------------------------
    # 10. Grounded vs Fallback Response Payloads
    # -------------------------------------------------------------------------
    def test_10_grounded_vs_fallback_rendering_safety(self) -> None:
        """Verify grounded answers and fallback notices format safely."""
        # Grounded query
        req_grounded = MultilingualTextRequest(text="What is the refund policy?")
        res_grounded = self.service.process_request(req_grounded)
        self.assertTrue(res_grounded["is_grounded"])
        self.assertIn("refund", res_grounded["final_answer"].lower())

        # Unsupported query (fallback)
        req_unknown = MultilingualTextRequest(text="What is the quantum telekinesis schedule?")
        res_unknown = self.service.process_request(req_unknown)
        self.assertFalse(res_unknown["is_grounded"])
        self.assertIn("not have enough information", res_unknown["final_answer"].lower())


if __name__ == "__main__":
    unittest.main()
