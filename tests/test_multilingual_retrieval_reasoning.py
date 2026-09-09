"""Unit and Integration Tests for Cross-Lingual Retrieval and Grounded Reasoning (Phase 6 Day 32).

Tests cross-lingual query alignment, retrieval against English knowledge bases,
grounded multilingual answer generation across English, Spanish, French, German, and Hindi,
evidence tracking, offline mode resilience, and backward compatibility.
"""

import unittest
from typing import Any, Dict, List

from src.multilingual.detector import LanguageDetector
from src.multilingual.intents import MultilingualIntentClassifier
from src.multilingual.models import (
    CrossLingualRetrievalResult,
    MultilingualIntent,
    MultilingualIntentResult,
    MultilingualResponse,
    MultilingualTextRequest,
    SupportedLanguage,
)
from src.multilingual.reasoning import MultilingualReasoner
from src.multilingual.retrieval import CrossLingualRetriever
from src.multilingual.service import MultilingualService


def mock_search_fn(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Deterministic mock search function simulating FAISS retrieval without filesystem dependency."""
    query_lower = query.lower()
    if "refund" in query_lower:
        return [{
            "page_content": "prompt: What if I don't like this bootcamp?\nresponse: As promised we will give you a 100% refund based on the guidelines (Please refer to our course refund policy before enrolling).",
            "metadata": {"source": "knowledge_base.csv", "row": 10},
        }]
    elif "prerequisite" in query_lower or "beginner" in query_lower:
        return [{
            "page_content": "prompt: Is there any prerequisite for taking this bootcamp ?\nresponse: Our bootcamp is specifically designed for beginners with no prior experience in this field. The only prerequisite is that you need to have a functional laptop with at least 4GB ram, an internet connection, and a thrill to learn data science.",
            "metadata": {"source": "knowledge_base.csv", "row": 4},
        }]
    elif "job" in query_lower or "internship" in query_lower or "resume" in query_lower:
        return [{
            "page_content": "prompt: Do you provide any job assistance?\nresponse: Yes, We help you with resume and interview preparation along with that we help you in building online credibility, and based on requirements we refer candidates to potential recruiters.",
            "metadata": {"source": "knowledge_base.csv", "row": 15},
        }]
    elif "duration" in query_lower or "schedule" in query_lower:
        return [{
            "page_content": "prompt: What is the duration of this bootcamp? How long will it last?\nresponse: You can complete all courses in 3 months if you dedicate 2-3 hours per day.",
            "metadata": {"source": "knowledge_base.csv", "row": 12},
        }]
    elif "contact" in query_lower or "discord" in query_lower:
        return [{
            "page_content": "prompt: How can I contact the instructors for any doubts/support?\nresponse: You can join our active discord community which is a dedicated platform to discuss & clear your doubts with fellow learners & mentors.",
            "metadata": {"source": "knowledge_base.csv", "row": 9},
        }]
    elif "emi" in query_lower or "payment" in query_lower or "cost" in query_lower:
        return [{
            "page_content": "prompt: Do we have an EMI option?\nresponse: No, but we have multiple affordable pricing tiers.",
            "metadata": {"source": "knowledge_base.csv", "row": 17},
        }]
    return []


class TestMultilingualRetrievalReasoning(unittest.TestCase):
    """Test suite for Day 32 Cross-Lingual Retrieval + Reasoning."""

    def setUp(self) -> None:
        """Initialize service stack with injected mock retriever and offline reasoner."""
        self.detector = LanguageDetector(confidence_threshold=0.50)
        self.intent_classifier = MultilingualIntentClassifier(language_detector=self.detector)
        self.retriever = CrossLingualRetriever(custom_search_fn=mock_search_fn)
        self.reasoner = MultilingualReasoner(llm="offline")  # Explicit deterministic offline synthesizer
        self.service = MultilingualService(
            detector=self.detector,
            intent_classifier=self.intent_classifier,
            retriever=self.retriever,
            reasoner=self.reasoner,
        )

    # -------------------------------------------------------------------------
    # A. English Compatibility
    # -------------------------------------------------------------------------

    def test_01_english_retrieval_and_reasoning(self) -> None:
        """Verify English query passes through cross-lingual retrieval and produces grounded response."""
        req = MultilingualTextRequest(text="What is the refund policy for this course?")
        resp = self.service.answer_query(req)

        self.assertEqual(resp.language, "en")
        self.assertEqual(resp.intent, MultilingualIntent.REFUND_POLICY.value)
        self.assertTrue(resp.is_grounded)
        self.assertGreaterEqual(resp.confidence_score, 0.70)
        self.assertTrue(len(resp.source_documents) > 0)
        self.assertIn("100% refund", resp.final_answer)

    # -------------------------------------------------------------------------
    # B. Spanish Cross-Lingual Retrieval & Reasoning
    # -------------------------------------------------------------------------

    def test_02_spanish_refund_cross_lingual(self) -> None:
        """Verify Spanish refund query aligns to English search and returns Spanish grounded response."""
        req = MultilingualTextRequest(text="¿Cómo puedo obtener un reembolso si no me gusta el curso?")
        resp = self.service.answer_query(req)

        self.assertEqual(resp.language, "es")
        self.assertEqual(resp.intent, MultilingualIntent.REFUND_POLICY.value)
        self.assertIn("refund", resp.aligned_query.lower())
        self.assertTrue(resp.is_grounded)
        self.assertIn("reembolso", resp.final_answer.lower())
        self.assertTrue(len(resp.source_documents) > 0)

    def test_03_spanish_prerequisites_cross_lingual(self) -> None:
        """Verify Spanish prerequisites query aligns to English vector retrieval."""
        req = MultilingualTextRequest(text="¿Cuáles son los requisitos previos para principiantes?")
        resp = self.service.answer_query(req)

        self.assertEqual(resp.language, "es")
        self.assertEqual(resp.intent, MultilingualIntent.PREREQUISITES.value)
        self.assertIn("prerequisites", resp.aligned_query.lower())
        self.assertTrue(resp.is_grounded)
        self.assertIn("requisitos", resp.final_answer.lower())

    # -------------------------------------------------------------------------
    # C. French Cross-Lingual Retrieval & Reasoning
    # -------------------------------------------------------------------------

    def test_04_french_job_assistance_cross_lingual(self) -> None:
        """Verify French career query aligns to English retrieval and returns French answer."""
        req = MultilingualTextRequest(text="Proposez-vous une aide à l'emploi et un stage après la formation?")
        resp = self.service.answer_query(req)

        self.assertEqual(resp.language, "fr")
        self.assertEqual(resp.intent, MultilingualIntent.CAREER_ASSISTANCE.value)
        self.assertIn("job assistance", resp.aligned_query.lower())
        self.assertTrue(resp.is_grounded)
        self.assertIn("emploi", resp.final_answer.lower())

    def test_05_french_duration_cross_lingual(self) -> None:
        """Verify French course duration query aligns to English retrieval."""
        req = MultilingualTextRequest(text="Quelle est la durée de cette formation?")
        resp = self.service.answer_query(req)

        self.assertEqual(resp.language, "fr")
        self.assertEqual(resp.intent, MultilingualIntent.COURSE_DETAILS.value)
        self.assertIn("duration", resp.aligned_query.lower())
        self.assertTrue(resp.is_grounded)
        self.assertIn("formation", resp.final_answer.lower())

    # -------------------------------------------------------------------------
    # D. German Cross-Lingual Retrieval & Reasoning
    # -------------------------------------------------------------------------

    def test_06_german_support_cross_lingual(self) -> None:
        """Verify German instructor support query aligns to English retrieval."""
        req = MultilingualTextRequest(text="Wie kann ich den Dozenten kontaktieren oder Fragen im Discord stellen?")
        resp = self.service.answer_query(req)

        self.assertEqual(resp.language, "de")
        self.assertEqual(resp.intent, MultilingualIntent.SUPPORT_CONTACT.value)
        self.assertIn("discord", resp.aligned_query.lower())
        self.assertTrue(resp.is_grounded)
        self.assertIn("discord", resp.final_answer.lower())

    def test_07_german_refund_cross_lingual(self) -> None:
        """Verify German refund query produces grounded German answer."""
        req = MultilingualTextRequest(text="Gibt es eine Geld-zurück-Garantie bei Stornierung?")
        resp = self.service.answer_query(req)

        self.assertEqual(resp.language, "de")
        self.assertEqual(resp.intent, MultilingualIntent.REFUND_POLICY.value)
        self.assertIn("refund", resp.aligned_query.lower())
        self.assertTrue(resp.is_grounded)
        self.assertIn("rückerstattung", resp.final_answer.lower())

    # -------------------------------------------------------------------------
    # E. Hindi Cross-Lingual Retrieval & Reasoning
    # -------------------------------------------------------------------------

    def test_08_hindi_refund_cross_lingual(self) -> None:
        """Verify Hindi refund query aligns to English retrieval and returns Hindi answer."""
        req = MultilingualTextRequest(text="क्या मुझे कोर्स पसंद न आने पर रिफंड और पैसे वापस मिलेंगे?")
        resp = self.service.answer_query(req)

        self.assertEqual(resp.language, "hi")
        self.assertEqual(resp.intent, MultilingualIntent.REFUND_POLICY.value)
        self.assertIn("refund", resp.aligned_query.lower())
        self.assertTrue(resp.is_grounded)
        self.assertIn("रिफंड", resp.final_answer)

    def test_09_hindi_prerequisites_cross_lingual(self) -> None:
        """Verify Hindi prerequisites query aligns to English retrieval."""
        req = MultilingualTextRequest(text="क्या बिना प्रोग्रामिंग अनुभव वाले शुरुआती लोग यह कोर्स कर सकते हैं?")
        resp = self.service.answer_query(req)

        self.assertEqual(resp.language, "hi")
        self.assertEqual(resp.intent, MultilingualIntent.PREREQUISITES.value)
        self.assertIn("prerequisites", resp.aligned_query.lower())
        self.assertTrue(resp.is_grounded)
        self.assertIn("योग्यता", resp.final_answer)

    # -------------------------------------------------------------------------
    # F. Conversational Greeting & Out-of-Domain / Insufficient Evidence
    # -------------------------------------------------------------------------

    def test_10_multilingual_greetings(self) -> None:
        """Verify greetings in all 5 languages return localized conversational openings."""
        greetings = {
            "en": "Hello there!",
            "es": "¡Hola, buenos días!",
            "fr": "Bonjour tout le monde!",
            "de": "Guten Tag!",
            "hi": "नमस्ते!",
        }
        for lang, text in greetings.items():
            req = MultilingualTextRequest(text=text)
            resp = self.service.answer_query(req)
            self.assertEqual(resp.language, lang)
            self.assertEqual(resp.intent, MultilingualIntent.GREETING.value)
            self.assertTrue(resp.is_grounded)
            self.assertGreaterEqual(resp.confidence_score, 0.90)

    def test_11_insufficient_evidence_safe_fallback(self) -> None:
        """Verify queries with no matching evidence return safe ungrounded refusals."""
        # Query about quantum astrophysics (completely unrelated to customer bootcamp)
        retriever_no_match = CrossLingualRetriever(custom_search_fn=lambda q, k: [])
        service_empty = MultilingualService(
            detector=self.detector,
            intent_classifier=self.intent_classifier,
            retriever=retriever_no_match,
            reasoner=self.reasoner,
        )

        req_es = MultilingualTextRequest(text="¿Cómo funciona la astrofísica cuántica en el espacio?")
        resp_es = service_empty.answer_query(req_es)

        self.assertEqual(resp_es.language, "es")
        self.assertFalse(resp_es.is_grounded)
        self.assertEqual(resp_es.confidence_score, 0.0)
        self.assertIn("no dispongo de suficiente información", resp_es.final_answer.lower())

    # -------------------------------------------------------------------------
    # G. LLM Mock Adapter Execution
    # -------------------------------------------------------------------------

    def test_12_mock_llm_adapter_synthesis(self) -> None:
        """Verify that when an LLM adapter is provided, reasoner calls it properly."""
        class MockLLM:
            def invoke(self, prompt: str) -> str:
                return "Respuesta simulada generada por LLM."

        custom_reasoner = MultilingualReasoner(llm=MockLLM())
        custom_service = MultilingualService(
            detector=self.detector,
            intent_classifier=self.intent_classifier,
            retriever=self.retriever,
            reasoner=custom_reasoner,
        )

        req = MultilingualTextRequest(text="¿Cómo puedo solicitar un reembolso?")
        resp = custom_service.answer_query(req)

        self.assertEqual(resp.final_answer, "Respuesta simulada generada por LLM.")
        self.assertEqual(resp.metadata.get("reasoning_type"), "llm_grounded_synthesis")
        self.assertTrue(resp.is_grounded)


    # -------------------------------------------------------------------------
    # H. Original Query Preservation & Contract Serialization
    # -------------------------------------------------------------------------

    def test_13_original_query_preservation(self) -> None:
        """Verify original multilingual query is strictly preserved and never mutated."""
        original = "¿Cómo puedo solicitar un reembolso?"
        req = MultilingualTextRequest(text=original)
        resp = self.service.answer_query(req)

        self.assertEqual(resp.query, original)
        self.assertNotEqual(resp.aligned_query, original)  # Aligned query is in English
        self.assertIn("refund", resp.aligned_query.lower())

    def test_14_model_serialization(self) -> None:
        """Verify CrossLingualRetrievalResult and MultilingualResponse serialization."""
        ret_res = CrossLingualRetrievalResult(
            original_query="Hola",
            detected_language="es",
            intent="greeting",
            aligned_query="hello",
            retrieved_documents=[{"page_content": "doc1"}],
            evidence_text="doc1",
            is_evidence_found=True,
            confidence=0.90,
            metadata={"test": "1"},
        )
        d_ret = ret_res.to_dict()
        self.assertEqual(d_ret["original_query"], "Hola")
        self.assertEqual(d_ret["detected_language"], "es")
        self.assertEqual(d_ret["confidence"], 0.90)

        resp_obj = MultilingualResponse(
            query="Hola",
            language="es",
            intent="greeting",
            aligned_query="hello",
            final_answer="¡Hola!",
            raw_answer="Hello",
            is_grounded=True,
            confidence_score=0.95,
            source_documents=[{"page_content": "doc1"}],
            metadata={"meta": "val"},
        )
        d_resp = resp_obj.to_dict()
        self.assertEqual(d_resp["query"], "Hola")
        self.assertEqual(d_resp["final_answer"], "¡Hola!")
        self.assertTrue(d_resp["is_grounded"])

    def test_15_service_process_request_backward_compatibility(self) -> None:
        """Verify process_request dictionary output retains Day 30 and Day 31 keys alongside Day 32."""
        req = MultilingualTextRequest(text="Bonjour, avez-vous des cours d'IA?")
        processed = self.service.process_request(req)

        # Day 30 keys
        self.assertIn("text", processed)
        self.assertIn("effective_language", processed)
        self.assertIn("language_name", processed)
        self.assertIn("is_forced", processed)
        self.assertIn("detection", processed)

        # Day 31 keys
        self.assertIn("intent", processed)

        # Day 32 keys
        self.assertIn("retrieval", processed)
        self.assertIn("response", processed)
        self.assertIn("final_answer", processed)
        self.assertIn("is_grounded", processed)


if __name__ == "__main__":
    unittest.main()
