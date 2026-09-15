"""Multilingual AI Assistant Evaluation and Benchmark Suite (Phase 6 — Day 35).

Comprehensive behavioral, safety, and performance evaluation of the completed Multilingual Chatbot.
Evaluates:
1. Language Identification Accuracy (5 Supported Languages + Script Mixing + Hinglish)
2. Intent Classification Precision across 8 Canonical Customer Service Domains
3. Cross-Lingual Knowledge Retrieval & Evidence Alignment
4. Grounded Multilingual Reasoning & Safe Out-of-Domain Fallback
5. Multi-Turn Context Retention & Demonstrative Reference Resolution
6. Dynamic Language Switching & Response Language Fidelity
7. Intra-Utterance Code-Switching & Romanized Hinglish Comprehension
8. Competing Intent Detection & Localized Clarification Prompting
9. Session Isolation & State Boundaries
10. Deterministic Metric Aggregation & Engineering Benchmark Validation
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src.multilingual import (
    CrossLingualRetrievalResult,
    CrossLingualRetriever,
    LanguageCandidate,
    LanguageDetector,
    LanguageIdentificationResult,
    MultilingualContextResolver,
    MultilingualConversationSession,
    MultilingualConversationTurn,
    MultilingualIntent,
    MultilingualIntentClassifier,
    MultilingualIntentResult,
    MultilingualReasoner,
    MultilingualResponse,
    MultilingualService,
    MultilingualTextRequest,
    SupportedLanguage,
)


class TestMultilingualEvaluation(unittest.TestCase):
    """End-to-end evaluation suite assessing all Phase 6 multilingual requirements."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize shared components with deterministic offline reasoner."""
        cls.detector = LanguageDetector()
        cls.intent_classifier = MultilingualIntentClassifier(language_detector=cls.detector)
        cls.context_resolver = MultilingualContextResolver()

        # Deterministic mock knowledge base for evaluation isolation
        cls.eval_kb_docs = [
            {
                "page_content": "The Data Science bootcamp costs $1,200 with 3-month and 6-month EMI installment options available.",
                "metadata": {"topic": "fees", "intent": "PAYMENT_PRICING"},
            },
            {
                "page_content": "We offer a 100% refund policy within the first 14 days of enrollment. After 14 days, no refund is provided.",
                "metadata": {"topic": "refund", "intent": "REFUND_POLICY"},
            },
            {
                "page_content": "Prerequisites: Basic computer literacy. No prior coding experience is needed. Minimum 4GB RAM laptop required.",
                "metadata": {"topic": "prerequisites", "intent": "PREREQUISITES"},
            },
            {
                "page_content": "Course details: The curriculum covers Python, SQL, Machine Learning, Power BI, and practical capstone projects with lifetime access.",
                "metadata": {"topic": "course_details", "intent": "COURSE_DETAILS"},
            },
            {
                "page_content": "Career assistance: We provide resume reviews, mock interview sessions, and direct portfolio referral to hiring partner recruiters.",
                "metadata": {"topic": "career", "intent": "CAREER_ASSISTANCE"},
            },
            {
                "page_content": "Support contact: Join our 24/7 Discord developer community to get help from dedicated mentors and instructors.",
                "metadata": {"topic": "support", "intent": "SUPPORT_CONTACT"},
            },
            {
                "page_content": "Technical support: For Excel spill errors, Power BI on Mac installation, or virtual machines, contact technical support desk.",
                "metadata": {"topic": "technical", "intent": "TECHNICAL_SUPPORT"},
            },
        ]

        def eval_search_fn(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
            q_lower = query.lower()
            matched = []
            for d in cls.eval_kb_docs:
                topic_kw = d["metadata"]["topic"].lower()
                content_kw = d["page_content"].lower()
                if topic_kw in q_lower or any(word in content_kw for word in q_lower.split() if len(word) > 3):
                    matched.append(d)
            return matched[:top_k] if matched else []

        cls.retriever = CrossLingualRetriever(custom_search_fn=eval_search_fn)
        cls.reasoner = MultilingualReasoner(llm="offline")
        cls.service = MultilingualService(
            detector=cls.detector,
            intent_classifier=cls.intent_classifier,
            retriever=cls.retriever,
            reasoner=cls.reasoner,
            context_resolver=cls.context_resolver,
        )

    # -------------------------------------------------------------------------
    # 1. Language Identification Accuracy
    # -------------------------------------------------------------------------
    def test_01_language_identification_accuracy(self) -> None:
        """Evaluate identification accuracy across all 5 supported languages and code-switching."""
        eval_cases: List[Tuple[str, str, bool]] = [
            # English
            ("What are the prerequisites for the Data Science course?", "en", False),
            ("How much does the bootcamp cost and is there an installment plan?", "en", False),
            # Spanish
            ("¿Cuáles son los requisitos previos del curso?", "es", False),
            ("¿Ofrecen una política de reembolso completo?", "es", False),
            # French
            ("Quelle est la durée de la formation et le programme?", "fr", False),
            ("Comment puis-je contacter les instructeurs sur Discord?", "fr", False),
            # German
            ("Welche Voraussetzungen sind für die Teilnahme erforderlich?", "de", False),
            ("Wie funktioniert die Rückerstattungsrichtlinie?", "de", False),
            # Hindi Devanagari
            ("इस कोर्स की फीस कितनी है और क्या किस्त उपलब्ध है?", "hi", False),
            ("क्या मुझे प्लेसमेंट सहायता मिलेगी?", "hi", False),
            # Code-switching / Hinglish
            ("Python course की fees कितनी है?", "hi", True),
            ("Is course ke prerequisites kya hain?", "hi", True),
            ("What are los prerequisites del course?", "en", True),
        ]

        success_count = 0
        for text, expected_lang, expected_mixed in eval_cases:
            det = self.detector.detect(text)
            self.assertEqual(det.language, expected_lang, f"Language mismatch for: '{text}'")
            if expected_mixed:
                self.assertTrue(det.is_mixed_language, f"Expected mixed-language for: '{text}'")
            success_count += 1

        accuracy = success_count / len(eval_cases)
        self.assertEqual(accuracy, 1.0, "Language identification accuracy should be 100% on evaluation set.")

    # -------------------------------------------------------------------------
    # 2. Intent Classification Precision
    # -------------------------------------------------------------------------
    def test_02_intent_classification_precision(self) -> None:
        """Evaluate intent recognition precision across all 8 canonical customer domains."""
        eval_intents: List[Tuple[str, Optional[str], str]] = [
            ("Hello, good morning!", "en", MultilingualIntent.GREETING.value),
            ("How much does the course cost?", "en", MultilingualIntent.PAYMENT_PRICING.value),
            ("What is your refund policy?", "en", MultilingualIntent.REFUND_POLICY.value),
            ("Do I need programming background to join?", "en", MultilingualIntent.PREREQUISITES.value),
            ("What topics and curriculum are covered?", "en", MultilingualIntent.COURSE_DETAILS.value),
            ("Do you provide job placement and resume assistance?", "en", MultilingualIntent.CAREER_ASSISTANCE.value),
            ("How can I contact instructors or join Discord?", "en", MultilingualIntent.SUPPORT_CONTACT.value),
            ("I am getting an Excel formula spill error on my laptop", "en", MultilingualIntent.TECHNICAL_SUPPORT.value),
            # Multilingual & Hinglish intents
            ("¿Cuánto cuesta la formación y opciones de cuotas?", "es", MultilingualIntent.PAYMENT_PRICING.value),
            ("Comment fonctionne le remboursement?", "fr", MultilingualIntent.REFUND_POLICY.value),
            ("Welche Voraussetzungen werden benötigt?", "de", MultilingualIntent.PREREQUISITES.value),
            ("इस कोर्स की फीस कितनी है?", "hi", MultilingualIntent.PAYMENT_PRICING.value),
            ("course ke prerequisites kya hain?", "hi", MultilingualIntent.PREREQUISITES.value),
        ]

        correct_count = 0
        for text, lang, expected_intent in eval_intents:
            res = self.intent_classifier.classify(text, language=lang)
            self.assertEqual(res.intent, expected_intent, f"Intent mismatch for '{text}'")
            self.assertTrue(res.is_recognized)
            correct_count += 1

        precision = correct_count / len(eval_intents)
        self.assertEqual(precision, 1.0, "Intent precision must be 100% on evaluation benchmark.")

    # -------------------------------------------------------------------------
    # 3. Cross-Lingual Retrieval Alignment
    # -------------------------------------------------------------------------
    def test_03_cross_lingual_retrieval_alignment(self) -> None:
        """Evaluate that non-English queries retrieve relevant English evidence."""
        queries: List[Tuple[str, str, str]] = [
            ("¿Cuál es la política de reembolso?", "es", "refund"),
            ("Quels sont les prérequis techniques?", "fr", "prerequisites"),
            ("Wie viel kostet der Kurs und welche Ratenzahlung gibt es?", "de", "fees"),
            ("इस कोर्स के लिए क्या योग्यता चाहिए?", "hi", "prerequisites"),
        ]

        for query, lang, expected_keyword in queries:
            intent_res = self.intent_classifier.classify(query, language=lang)
            ret_res = self.retriever.retrieve(query, language=lang, intent_result=intent_res)
            self.assertTrue(ret_res.is_evidence_found)
            self.assertGreaterEqual(len(ret_res.retrieved_documents), 1)
            self.assertIn(expected_keyword, ret_res.aligned_query.lower())

    # -------------------------------------------------------------------------
    # 4. Multi-Turn Context Continuity & Entity Resolution
    # -------------------------------------------------------------------------
    def test_04_multiturn_context_continuity(self) -> None:
        """Evaluate entity tracking and demonstrative pronoun resolution across turns."""
        session_id = "eval-context-session-001"

        # Turn 1: Establish topic
        req1 = MultilingualTextRequest(
            text="Tell me about the Data Science bootcamp curriculum.",
            session_id=session_id,
        )
        resp1 = self.service.answer_query(req1)
        self.assertEqual(resp1.intent, MultilingualIntent.COURSE_DETAILS.value)

        # Turn 2: Follow-up using English demonstrative "its"
        req2 = MultilingualTextRequest(
            text="What are its prerequisites?",
            session_id=session_id,
        )
        proc2 = self.service.process_request(req2)
        self.assertTrue(proc2["is_followup"])
        self.assertEqual(proc2["intent"]["intent"], MultilingualIntent.PREREQUISITES.value)
        self.assertIn("data science", proc2["resolved_query"].lower())

        # Turn 3: Follow-up in Hinglish
        req3 = MultilingualTextRequest(
            text="Is course ki fees kitni hai?",
            session_id=session_id,
        )
        proc3 = self.service.process_request(req3)
        self.assertTrue(proc3["is_followup"])
        self.assertEqual(proc3["intent"]["intent"], MultilingualIntent.PAYMENT_PRICING.value)
        self.assertEqual(proc3["effective_language"], SupportedLanguage.HINDI.value)

    # -------------------------------------------------------------------------
    # 5. Dynamic Language Switching Fidelity
    # -------------------------------------------------------------------------
    def test_05_dynamic_language_switching_fidelity(self) -> None:
        """Evaluate seamless transitions across French, German, Spanish, and English."""
        session_id = "eval-lang-switch-002"

        # Turn 1: French
        r1 = self.service.answer_query(
            MultilingualTextRequest(text="Quels sont les prérequis pour la formation?", session_id=session_id)
        )
        self.assertEqual(r1.language, SupportedLanguage.FRENCH.value)

        # Turn 2: German switch
        r2 = self.service.answer_query(
            MultilingualTextRequest(text="Und wie funktioniert die Rückerstattung?", session_id=session_id)
        )
        self.assertEqual(r2.language, SupportedLanguage.GERMAN.value)

        # Turn 3: Spanish switch
        r3 = self.service.answer_query(
            MultilingualTextRequest(text="¿Tienen asistencia para encontrar empleo?", session_id=session_id)
        )
        self.assertEqual(r3.language, SupportedLanguage.SPANISH.value)

        # Turn 4: English switch
        r4 = self.service.answer_query(
            MultilingualTextRequest(text="Can I pay in monthly installments?", session_id=session_id)
        )
        self.assertEqual(r4.language, SupportedLanguage.ENGLISH.value)

    # -------------------------------------------------------------------------
    # 6. Competing Intent Ambiguity & Localized Clarification
    # -------------------------------------------------------------------------
    def test_06_ambiguity_and_clarification_evaluation(self) -> None:
        """Evaluate dual-intent ambiguity detection and localized clarification prompting."""
        # Dual-intent ambiguous query
        query = "What are the course fees and refund policy?"
        intent_res = self.intent_classifier.classify(query, language="en")
        self.assertTrue(intent_res.is_ambiguous)
        self.assertIn(MultilingualIntent.PAYMENT_PRICING.value, intent_res.competing_intents)
        self.assertIn(MultilingualIntent.REFUND_POLICY.value, intent_res.competing_intents)

        # Underspecified ambiguous query without evidence produces clarification
        empty_retrieval = CrossLingualRetrievalResult(
            original_query="tell me details",
            detected_language="es",
            intent=MultilingualIntent.GENERAL_INQUIRY.value,
            aligned_query="general",
            retrieved_documents=[],
            evidence_text="",
            is_evidence_found=False,
            confidence=0.0,
            metadata={"is_ambiguous": True},
        )
        resp = self.reasoner.reason(empty_retrieval, target_language="es")
        self.assertTrue(resp.is_ambiguous)
        self.assertIsNotNone(resp.clarification_prompt)
        self.assertIn("aclarar", resp.clarification_prompt or "")

    # -------------------------------------------------------------------------
    # 7. Grounding vs Safe Fallback Protection
    # -------------------------------------------------------------------------
    def test_07_grounding_vs_safe_fallback(self) -> None:
        """Evaluate that ungroundable queries safely refuse without hallucination."""
        req = MultilingualTextRequest(text="Can you explain quantum astrophysics in deep space?")
        resp = self.service.answer_query(req)
        # Should gracefully return ungrounded fallback message
        self.assertFalse(resp.is_grounded)
        self.assertEqual(resp.confidence_score, 0.0)
        self.assertIn("not have enough information", resp.final_answer.lower())

    # -------------------------------------------------------------------------
    # 8. Session Isolation & State Boundaries
    # -------------------------------------------------------------------------
    def test_08_session_isolation(self) -> None:
        """Evaluate strict isolation between separate concurrent conversation sessions."""
        sess_1 = "eval-user-alice"
        sess_2 = "eval-user-bob"

        # Session 1: Python bootcamp
        self.service.process_request(
            MultilingualTextRequest(text="Tell me about Python bootcamp.", session_id=sess_1)
        )

        # Session 2: Machine Learning course
        self.service.process_request(
            MultilingualTextRequest(text="Tell me about Machine Learning program.", session_id=sess_2)
        )

        # Follow-up in Session 1
        p1 = self.service.process_request(
            MultilingualTextRequest(text="What are its prerequisites?", session_id=sess_1)
        )
        self.assertIn("Python", p1["resolved_query"])
        self.assertNotIn("Machine Learning", p1["resolved_query"])

        # Follow-up in Session 2
        p2 = self.service.process_request(
            MultilingualTextRequest(text="What are its prerequisites?", session_id=sess_2)
        )
        self.assertIn("Machine Learning", p2["resolved_query"])
        self.assertNotIn("Python", p2["resolved_query"])


if __name__ == "__main__":
    unittest.main()
