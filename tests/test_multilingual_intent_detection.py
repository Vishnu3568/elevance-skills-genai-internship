"""Unit and Integration Tests for Multilingual Intent Detection (Phase 6 Day 31).

Tests intent classification across English, Spanish, French, German, and Hindi,
confidence calibration, edge cases, model contracts, and MultilingualService integration.
"""

import unittest

from src.multilingual.detector import LanguageDetector
from src.multilingual.intents import MultilingualIntentClassifier
from src.multilingual.models import (
    MultilingualIntent,
    MultilingualIntentResult,
    MultilingualTextRequest,
    SupportedLanguage,
)
from src.multilingual.service import MultilingualService


class TestMultilingualIntentDetection(unittest.TestCase):
    """Test suite for Day 31 Multilingual Intent Detection."""

    def setUp(self) -> None:
        """Initialize classifier and service instances for testing."""
        self.detector = LanguageDetector(confidence_threshold=0.50)
        self.classifier = MultilingualIntentClassifier(language_detector=self.detector)
        self.service = MultilingualService(detector=self.detector, intent_classifier=self.classifier)

    # -------------------------------------------------------------------------
    # 1. English (en) Intent Detection
    # -------------------------------------------------------------------------

    def test_01_english_refund_intent(self) -> None:
        """Verify English refund query intent classification."""
        text = "What is the course refund policy if I don't like it?"
        res = self.classifier.classify(text, language="en")

        self.assertEqual(res.intent, MultilingualIntent.REFUND_POLICY.value)
        self.assertTrue(res.is_recognized)
        self.assertGreaterEqual(res.confidence, 0.50)
        self.assertLessEqual(res.confidence, 1.0)
        self.assertEqual(res.language, "en")
        self.assertTrue(any("refund" in kw.lower() for kw in res.matched_keywords))

    def test_02_english_prerequisites_intent(self) -> None:
        """Verify English prerequisites query intent classification."""
        text = "Is there any prerequisite for a beginner with no coding experience?"
        res = self.classifier.classify(text, language="en")

        self.assertEqual(res.intent, MultilingualIntent.PREREQUISITES.value)
        self.assertTrue(res.is_recognized)
        self.assertGreaterEqual(res.confidence, 0.50)

    def test_03_english_career_intent(self) -> None:
        """Verify English career / job assistance query intent classification."""
        text = "Do you provide any job assistance or virtual internship after this course?"
        res = self.classifier.classify(text, language="en")

        self.assertEqual(res.intent, MultilingualIntent.CAREER_ASSISTANCE.value)
        self.assertTrue(res.is_recognized)
        self.assertGreaterEqual(res.confidence, 0.50)

    # -------------------------------------------------------------------------
    # 2. Spanish (es) Intent Detection
    # -------------------------------------------------------------------------

    def test_04_spanish_refund_intent(self) -> None:
        """Verify Spanish refund query intent classification."""
        text = "¿Cómo puedo solicitar un reembolso o devolución de mi dinero?"
        res = self.classifier.classify(text, language="es")

        self.assertEqual(res.intent, MultilingualIntent.REFUND_POLICY.value)
        self.assertTrue(res.is_recognized)
        self.assertEqual(res.language, "es")
        self.assertGreaterEqual(res.confidence, 0.50)

    def test_05_spanish_prerequisites_intent(self) -> None:
        """Verify Spanish prerequisites query intent classification."""
        text = "¿Cuáles son los requisitos previos para un principiante sin programación previa?"
        res = self.classifier.classify(text, language="es")

        self.assertEqual(res.intent, MultilingualIntent.PREREQUISITES.value)
        self.assertTrue(res.is_recognized)
        self.assertGreaterEqual(res.confidence, 0.50)

    def test_06_spanish_payment_pricing_intent(self) -> None:
        """Verify Spanish EMI/payment query intent classification."""
        text = "¿Tienen opciones de pago en cuotas o plazos mensuales?"
        res = self.classifier.classify(text, language="es")

        self.assertEqual(res.intent, MultilingualIntent.PAYMENT_PRICING.value)
        self.assertTrue(res.is_recognized)
        self.assertGreaterEqual(res.confidence, 0.50)

    # -------------------------------------------------------------------------
    # 3. French (fr) Intent Detection
    # -------------------------------------------------------------------------

    def test_07_french_greeting_intent(self) -> None:
        """Verify French greeting query intent classification."""
        text = "Bonjour à toute l'équipe!"
        res = self.classifier.classify(text, language="fr")

        self.assertEqual(res.intent, MultilingualIntent.GREETING.value)
        self.assertTrue(res.is_recognized)
        self.assertEqual(res.language, "fr")

    def test_08_french_course_details_intent(self) -> None:
        """Verify French course details query intent classification."""
        text = "Quelle est la durée de cette formation et le programme des cours?"
        res = self.classifier.classify(text, language="fr")

        self.assertEqual(res.intent, MultilingualIntent.COURSE_DETAILS.value)
        self.assertTrue(res.is_recognized)
        self.assertGreaterEqual(res.confidence, 0.50)

    def test_09_french_technical_support_intent(self) -> None:
        """Verify French technical support query intent classification."""
        text = "J'ai une erreur dans la formule excel, mon code ne fonctionne pas."
        res = self.classifier.classify(text, language="fr")

        self.assertEqual(res.intent, MultilingualIntent.TECHNICAL_SUPPORT.value)
        self.assertTrue(res.is_recognized)
        self.assertGreaterEqual(res.confidence, 0.50)

    # -------------------------------------------------------------------------
    # 4. German (de) Intent Detection
    # -------------------------------------------------------------------------

    def test_10_german_refund_intent(self) -> None:
        """Verify German refund query intent classification."""
        text = "Wie funktioniert die Rückerstattung, wenn ich den Kurs stornieren möchte?"
        res = self.classifier.classify(text, language="de")

        self.assertEqual(res.intent, MultilingualIntent.REFUND_POLICY.value)
        self.assertTrue(res.is_recognized)
        self.assertEqual(res.language, "de")
        self.assertGreaterEqual(res.confidence, 0.50)

    def test_11_german_career_assistance_intent(self) -> None:
        """Verify German career assistance query intent classification."""
        text = "Bieten Sie Unterstützung beim Lebenslauf und bei der Stellenvermittlung an?"
        res = self.classifier.classify(text, language="de")

        self.assertEqual(res.intent, MultilingualIntent.CAREER_ASSISTANCE.value)
        self.assertTrue(res.is_recognized)
        self.assertGreaterEqual(res.confidence, 0.50)

    def test_12_german_support_contact_intent(self) -> None:
        """Verify German instructor support contact query intent classification."""
        text = "Wie kann ich die Dozenten kontaktieren oder dem Discord Server beitreten?"
        res = self.classifier.classify(text, language="de")

        self.assertEqual(res.intent, MultilingualIntent.SUPPORT_CONTACT.value)
        self.assertTrue(res.is_recognized)
        self.assertGreaterEqual(res.confidence, 0.50)

    # -------------------------------------------------------------------------
    # 5. Hindi (hi) Intent Detection
    # -------------------------------------------------------------------------

    def test_13_hindi_greeting_intent(self) -> None:
        """Verify Hindi greeting query intent classification."""
        text = "नमस्ते, मुझे कुछ जानकारी चाहिए।"
        res = self.classifier.classify(text, language="hi")

        self.assertEqual(res.intent, MultilingualIntent.GREETING.value)
        self.assertTrue(res.is_recognized)
        self.assertEqual(res.language, "hi")

    def test_14_hindi_refund_intent(self) -> None:
        """Verify Hindi refund query intent classification."""
        text = "क्या मुझे कोर्स का रिफंड या पैसे वापस मिल सकते हैं?"
        res = self.classifier.classify(text, language="hi")

        self.assertEqual(res.intent, MultilingualIntent.REFUND_POLICY.value)
        self.assertTrue(res.is_recognized)
        self.assertEqual(res.language, "hi")
        self.assertGreaterEqual(res.confidence, 0.50)

    def test_15_hindi_prerequisites_and_career_intent(self) -> None:
        """Verify Hindi prerequisites and job assistance queries."""
        text1 = "क्या शुरुआती बिना कोडिंग अनुभव के यह कोर्स कर सकते हैं?"
        res1 = self.classifier.classify(text1, language="hi")
        self.assertEqual(res1.intent, MultilingualIntent.PREREQUISITES.value)

        text2 = "क्या इस कोर्स के बाद नौकरी और इंटर्नशिप में मदद मिलेगी?"
        res2 = self.classifier.classify(text2, language="hi")
        self.assertEqual(res2.intent, MultilingualIntent.CAREER_ASSISTANCE.value)

    # -------------------------------------------------------------------------
    # 6. Edge Cases & Safety Invariants
    # -------------------------------------------------------------------------

    def test_16_empty_and_whitespace_input(self) -> None:
        """Verify that empty and whitespace inputs yield UNKNOWN intent safely."""
        for empty_val in ["", "   ", "\n\t  "]:
            res = self.classifier.classify(empty_val)
            self.assertEqual(res.intent, MultilingualIntent.UNKNOWN.value)
            self.assertFalse(res.is_recognized)
            self.assertEqual(res.confidence, 0.0)

    def test_17_none_and_non_string_input(self) -> None:
        """Verify that None or non-string inputs yield UNKNOWN without unhandled exceptions."""
        res = self.classifier.classify(None)  # type: ignore
        self.assertEqual(res.intent, MultilingualIntent.UNKNOWN.value)
        self.assertFalse(res.is_recognized)
        self.assertEqual(res.confidence, 0.0)

    def test_18_symbols_and_numbers_only(self) -> None:
        """Verify that digits/symbols without alphabetic text yield UNKNOWN."""
        res = self.classifier.classify("12345 67890 !@#$%^&*")
        self.assertEqual(res.intent, MultilingualIntent.UNKNOWN.value)
        self.assertFalse(res.is_recognized)
        self.assertEqual(res.confidence, 0.0)

    def test_19_general_inquiry_fallback(self) -> None:
        """Verify that general conversational text without specialized keywords defaults to GENERAL_INQUIRY."""
        text = "I am interested in learning more about your educational philosophy."
        res = self.classifier.classify(text, language="en")

        self.assertEqual(res.intent, MultilingualIntent.GENERAL_INQUIRY.value)
        self.assertTrue(res.is_recognized)
        self.assertGreaterEqual(res.confidence, 0.50)

    def test_20_confidence_bounds_invariant(self) -> None:
        """Verify that all confidence scores strictly satisfy 0.0 <= confidence <= 1.0."""
        test_queries = [
            "Hello there",
            "¿Puedo obtener un reembolso?",
            "Je veux annuler ma formation",
            "Gibt es Ratenzahlung?",
            "नमस्ते, मुझे मदद चाहिए",
            "Random unclassified phrase 123",
            "$$$ !!!",
        ]
        for q in test_queries:
            res = self.classifier.classify(q)
            self.assertGreaterEqual(res.confidence, 0.0)
            self.assertLessEqual(res.confidence, 1.0)

    # -------------------------------------------------------------------------
    # 7. Model Contracts & Service Integration
    # -------------------------------------------------------------------------

    def test_21_model_serialization_and_validation(self) -> None:
        """Verify MultilingualIntentResult to_dict serialization and invariant checks."""
        res = MultilingualIntentResult(
            text="How long is the course?",
            intent="course_details",
            confidence=0.88,
            language="en",
            is_recognized=True,
            matched_keywords=["how long"],
            metadata={"test": "ok"},
        )
        d = res.to_dict()
        self.assertEqual(d["intent"], "course_details")
        self.assertEqual(d["confidence"], 0.88)
        self.assertEqual(d["language"], "en")
        self.assertTrue(d["is_recognized"])
        self.assertEqual(d["matched_keywords"], ["how long"])

        with self.assertRaises(ValueError):
            MultilingualIntentResult(text="a", intent="b", confidence=1.5, language="en")

    def test_22_service_process_request_with_intent(self) -> None:
        """Verify MultilingualService.process_request exposes both language and intent."""
        req = MultilingualTextRequest(
            text="¿Puedo obtener un reembolso por el curso?",
            session_id="session-intent-es-1",
        )
        processed = self.service.process_request(req)

        self.assertIn("effective_language", processed)
        self.assertEqual(processed["effective_language"], "es")
        self.assertIn("detection", processed)
        self.assertIn("intent", processed)
        self.assertEqual(processed["intent"]["intent"], MultilingualIntent.REFUND_POLICY.value)
        self.assertTrue(processed["intent"]["is_recognized"])


if __name__ == "__main__":
    unittest.main()
