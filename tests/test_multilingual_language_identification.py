"""Unit and Integration Tests for Multilingual Architecture & Language Identification (Phase 6 Day 30).

Tests statistical language detection, script analysis, boundary condition handling,
confidence scoring, data contracts, and service layer integration.
"""

import unittest

from src.multilingual.detector import LanguageDetector
from src.multilingual.models import (
    LanguageCandidate,
    LanguageIdentificationResult,
    MultilingualTextRequest,
    SupportedLanguage,
)
from src.multilingual.service import MultilingualService


class TestMultilingualLanguageIdentification(unittest.TestCase):
    """Test suite for Day 30 Multilingual Architecture & Language Identification."""

    def setUp(self) -> None:
        """Initialize detector and service instances for test isolation."""
        self.detector = LanguageDetector(confidence_threshold=0.50)
        self.service = MultilingualService(detector=self.detector)

    # -------------------------------------------------------------------------
    # 1. Supported Language Identification
    # -------------------------------------------------------------------------

    def test_01_detect_english_text(self) -> None:
        """Verify accurate identification of English text."""
        text = "What are the main benefits of using vector embeddings for semantic search?"
        result = self.detector.detect(text)

        self.assertEqual(result.language, SupportedLanguage.ENGLISH.value)
        self.assertTrue(result.is_supported)
        self.assertTrue(result.is_reliable)
        self.assertEqual(result.language_name, "English")
        self.assertGreaterEqual(result.confidence, 0.50)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertEqual(result.detected_script, "Latin")

    def test_02_detect_spanish_text(self) -> None:
        """Verify accurate identification of Spanish text."""
        text = "¿Cuáles son los beneficios de utilizar bases de datos vectoriales en este sistema?"
        result = self.detector.detect(text)

        self.assertEqual(result.language, SupportedLanguage.SPANISH.value)
        self.assertTrue(result.is_supported)
        self.assertTrue(result.is_reliable)
        self.assertEqual(result.language_name, "Spanish")
        self.assertGreaterEqual(result.confidence, 0.50)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertEqual(result.detected_script, "Latin")

    def test_03_detect_french_text(self) -> None:
        """Verify accurate identification of French text."""
        text = "Bonjour, pouvez-vous m'expliquer comment fonctionne la recherche sémantique dans ce projet?"
        result = self.detector.detect(text)

        self.assertEqual(result.language, SupportedLanguage.FRENCH.value)
        self.assertTrue(result.is_supported)
        self.assertTrue(result.is_reliable)
        self.assertEqual(result.language_name, "French")
        self.assertGreaterEqual(result.confidence, 0.50)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertEqual(result.detected_script, "Latin")

    def test_04_detect_german_text(self) -> None:
        """Verify accurate identification of German text."""
        text = "Guten Tag, welche Vorteile bietet die semantische Suche für unser Unternehmen?"
        result = self.detector.detect(text)

        self.assertEqual(result.language, SupportedLanguage.GERMAN.value)
        self.assertTrue(result.is_supported)
        self.assertTrue(result.is_reliable)
        self.assertEqual(result.language_name, "German")
        self.assertGreaterEqual(result.confidence, 0.50)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertEqual(result.detected_script, "Latin")

    def test_05_detect_hindi_text(self) -> None:
        """Verify accurate identification of Hindi (Devanagari script) text."""
        text = "नमस्ते, क्या आप मुझे इस प्रणाली के बारे में जानकारी दे सकते हैं?"
        result = self.detector.detect(text)

        self.assertEqual(result.language, SupportedLanguage.HINDI.value)
        self.assertTrue(result.is_supported)
        self.assertTrue(result.is_reliable)
        self.assertEqual(result.language_name, "Hindi")
        self.assertGreaterEqual(result.confidence, 0.70)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertEqual(result.detected_script, "Devanagari")

    # -------------------------------------------------------------------------
    # 2. Edge Cases, Non-alphabetic & Ambiguous Inputs
    # -------------------------------------------------------------------------

    def test_06_empty_and_whitespace_input(self) -> None:
        """Verify that empty and whitespace inputs are safely handled as UNKNOWN."""
        for empty_val in ["", "   ", "\t\n  "]:
            result = self.detector.detect(empty_val)
            self.assertEqual(result.language, SupportedLanguage.UNKNOWN.value)
            self.assertFalse(result.is_supported)
            self.assertFalse(result.is_reliable)
            self.assertEqual(result.confidence, 0.0)

    def test_07_none_and_non_string_input(self) -> None:
        """Verify that None or non-string inputs do not raise unhandled exceptions."""
        # type: ignore
        result = self.detector.detect(None)  # type: ignore
        self.assertEqual(result.language, SupportedLanguage.UNKNOWN.value)
        self.assertEqual(result.confidence, 0.0)
        self.assertFalse(result.is_reliable)

    def test_08_digits_and_symbols_only(self) -> None:
        """Verify that inputs with only numbers or symbols yield UNKNOWN."""
        symbols_text = "12345 67890 #$%^& *()_+ ~`"
        result = self.detector.detect(symbols_text)

        self.assertEqual(result.language, SupportedLanguage.UNKNOWN.value)
        self.assertFalse(result.is_supported)
        self.assertFalse(result.is_reliable)
        self.assertEqual(result.confidence, 0.0)

    def test_09_confidence_bounds_invariant(self) -> None:
        """Verify that all confidence scores across any query strictly satisfy 0 <= c <= 1."""
        test_queries = [
            "Hello world",
            "¿Cómo estás?",
            "C'est la vie",
            "Das ist ein Test",
            "यह एक परीक्षण है",
            "xyz abc 123",
            "???",
        ]
        for q in test_queries:
            res = self.detector.detect(q)
            self.assertGreaterEqual(res.confidence, 0.0)
            self.assertLessEqual(res.confidence, 1.0)
            for cand in res.candidates:
                self.assertGreaterEqual(cand.confidence, 0.0)
                self.assertLessEqual(cand.confidence, 1.0)

    # -------------------------------------------------------------------------
    # 3. Model Contracts & Serializability
    # -------------------------------------------------------------------------

    def test_10_candidate_ranking_order(self) -> None:
        """Verify that candidates are sorted in descending order of confidence."""
        text = "Bonjour le monde, ceci est un message de test."
        result = self.detector.detect(text)

        if len(result.candidates) > 1:
            for i in range(len(result.candidates) - 1):
                self.assertGreaterEqual(
                    result.candidates[i].confidence,
                    result.candidates[i + 1].confidence,
                )

    def test_11_model_serialization(self) -> None:
        """Verify to_dict serialization for all multilingual data contracts."""
        cand = LanguageCandidate(language="en", confidence=0.95, language_name="English")
        cand_dict = cand.to_dict()
        self.assertEqual(cand_dict["language"], "en")
        self.assertEqual(cand_dict["confidence"], 0.95)

        res = LanguageIdentificationResult(
            text="Hello",
            language="en",
            confidence=0.90,
            is_supported=True,
            language_name="English",
            candidates=[cand],
            is_reliable=True,
            detected_script="Latin",
            metadata={"source": "unit_test"},
        )
        res_dict = res.to_dict()
        self.assertEqual(res_dict["language"], "en")
        self.assertEqual(res_dict["text"], "Hello")
        self.assertEqual(len(res_dict["candidates"]), 1)
        self.assertEqual(res_dict["metadata"]["source"], "unit_test")

        req = MultilingualTextRequest(text="Bonjour", forced_language="fr", session_id="sess-1")
        req_dict = req.to_dict()
        self.assertEqual(req_dict["text"], "Bonjour")
        self.assertEqual(req_dict["forced_language"], "fr")
        self.assertEqual(req_dict["session_id"], "sess-1")

    def test_12_invalid_model_parameters(self) -> None:
        """Verify that dataclass invariants raise appropriate validation exceptions."""
        with self.assertRaises(ValueError):
            LanguageCandidate(language="", confidence=0.5, language_name="English")

        with self.assertRaises(ValueError):
            LanguageCandidate(language="en", confidence=1.5, language_name="English")

        with self.assertRaises(TypeError):
            LanguageIdentificationResult(
                text="test",
                language=123,  # type: ignore
                confidence=0.5,
                is_supported=True,
                language_name="English",
            )

        with self.assertRaises(ValueError):
            MultilingualTextRequest(text="   ")

    # -------------------------------------------------------------------------
    # 4. Service Layer Integration & Session Management
    # -------------------------------------------------------------------------

    def test_13_service_process_request_automatic(self) -> None:
        """Verify that MultilingualService detects language and tracks session state."""
        req = MultilingualTextRequest(
            text="¿Cómo puedo restablecer mi contraseña?",
            session_id="session-es-1",
        )
        processed = self.service.process_request(req)

        self.assertEqual(processed["effective_language"], "es")
        self.assertEqual(processed["language_name"], "Spanish")
        self.assertFalse(processed["is_forced"])
        self.assertEqual(self.service.get_session_language("session-es-1"), "es")

    def test_14_service_forced_language_override(self) -> None:
        """Verify that explicit forced_language override takes precedence."""
        req = MultilingualTextRequest(
            text="Hello world",
            forced_language="de",
            session_id="session-forced-1",
        )
        processed = self.service.process_request(req)

        self.assertEqual(processed["effective_language"], "de")
        self.assertEqual(processed["language_name"], "German")
        self.assertTrue(processed["is_forced"])
        self.assertEqual(self.service.get_session_language("session-forced-1"), "de")

    def test_15_service_session_isolation_and_clear(self) -> None:
        """Verify that multiple sessions maintain isolated language states and clear cleanly."""
        req1 = MultilingualTextRequest(text="Bonjour", session_id="user-session-fr")
        req2 = MultilingualTextRequest(text="Guten Tag", session_id="user-session-de")

        self.service.process_request(req1)
        self.service.process_request(req2)

        self.assertEqual(self.service.get_session_language("user-session-fr"), "fr")
        self.assertEqual(self.service.get_session_language("user-session-de"), "de")

        # Clear session 1
        self.service.clear_session("user-session-fr")
        self.assertIsNone(self.service.get_session_language("user-session-fr"))
        self.assertEqual(self.service.get_session_language("user-session-de"), "de")

        # Reset all
        self.service.reset_all_sessions()
        self.assertIsNone(self.service.get_session_language("user-session-de"))


if __name__ == "__main__":
    unittest.main()
