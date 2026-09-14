"""Tests for Multilingual Mixed-Language and Ambiguity Handling (Phase 6 Day 34).

Covers:
1. Pure single-language input remains intact.
2. English + Hindi Devanagari script mixed queries (code-switching).
3. English + Romanized Hindi (Hinglish) queries.
4. Mixed-language intent classification across supported domains.
5. Spanish/English and French/English intra-utterance mixed queries.
6. Competing/ambiguous intent detection with multi-intent extraction.
7. Dominant single-intent queries correctly avoiding false-positive ambiguity.
8. Underspecified queries safely handled with clarification prompts.
9. Multi-turn mixed-language follow-up using conversational context.
10. Language switching combined with mixed-language follow-ups.
11. Multi-session isolation and boundary safety.
12. Stateless query processing backward compatibility.
13. Grounded fallback behavior when evidence is insufficient.
14. Backward compatibility with existing single-language contracts and data models.
"""

import unittest

from src.multilingual.context import MultilingualContextResolver
from src.multilingual.detector import LanguageDetector
from src.multilingual.intents import MultilingualIntentClassifier
from src.multilingual.models import (
    CrossLingualRetrievalResult,
    LanguageCandidate,
    LanguageIdentificationResult,
    MultilingualConversationSession,
    MultilingualConversationTurn,
    MultilingualIntent,
    MultilingualIntentResult,
    MultilingualResponse,
    MultilingualTextRequest,
    SupportedLanguage,
)
from src.multilingual.reasoning import MultilingualReasoner
from src.multilingual.retrieval import CrossLingualRetriever
from src.multilingual.service import MultilingualService


class TestMultilingualMixedLanguageAmbiguity(unittest.TestCase):
    """Test suite for Day 34 mixed-language queries, Hinglish, and ambiguity handling."""

    def setUp(self) -> None:
        """Initialize components with isolated mock data sources."""
        self.detector = LanguageDetector()
        self.classifier = MultilingualIntentClassifier(language_detector=self.detector)

        # Mock retrieval engine for deterministic unit testing
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
                if topic_lower in q_lower or any(w in content_lower for w in q_lower.split()):
                    results.append(doc)
            return results[:top_k] if results else self.mock_docs[:1]

        self.retriever = CrossLingualRetriever(custom_search_fn=mock_search_fn)
        self.reasoner = MultilingualReasoner(llm="offline")
        self.context_resolver = MultilingualContextResolver()
        self.service = MultilingualService(
            detector=self.detector,
            intent_classifier=self.classifier,
            retriever=self.retriever,
            reasoner=self.reasoner,
            context_resolver=self.context_resolver,
        )

    # -------------------------------------------------------------------------
    # 1. Pure single-language input
    # -------------------------------------------------------------------------
    def test_pure_single_language_remains_unchanged(self) -> None:
        """Verify pure English, Hindi, Spanish, French, German queries are unaffected."""
        res_en = self.detector.detect("What are the prerequisites for this course?")
        self.assertEqual(res_en.language, SupportedLanguage.ENGLISH.value)
        self.assertFalse(res_en.is_mixed_language)

        res_hi = self.detector.detect("इस कोर्स की फीस कितनी है?")
        self.assertEqual(res_hi.language, SupportedLanguage.HINDI.value)
        self.assertFalse(res_hi.is_mixed_language)

        res_es = self.detector.detect("¿Cuáles son los requisitos previos del curso?")
        self.assertEqual(res_es.language, SupportedLanguage.SPANISH.value)
        self.assertFalse(res_es.is_mixed_language)

    # -------------------------------------------------------------------------
    # 2. English + Hindi Devanagari script mixed query
    # -------------------------------------------------------------------------
    def test_english_hindi_script_code_switching(self) -> None:
        """Verify intra-utterance code-switching between English words and Hindi Devanagari."""
        query = "Python course की fees कितनी है?"
        det = self.detector.detect(query)
        self.assertEqual(det.language, SupportedLanguage.HINDI.value)
        self.assertTrue(det.is_mixed_language)
        self.assertEqual(det.secondary_language, SupportedLanguage.ENGLISH.value)

        # End-to-end service test
        req = MultilingualTextRequest(text=query)
        resp = self.service.answer_query(req)
        self.assertEqual(resp.language, SupportedLanguage.HINDI.value)
        self.assertEqual(resp.intent, MultilingualIntent.PAYMENT_PRICING.value)
        self.assertTrue(resp.is_grounded)

    # -------------------------------------------------------------------------
    # 3. English + Romanized Hindi (Hinglish) queries
    # -------------------------------------------------------------------------
    def test_hinglish_romanized_query_detection(self) -> None:
        """Verify Romanized Hindi/Hinglish query is identified as Hindi."""
        query = "Is course ke prerequisites kya hain?"
        det = self.detector.detect(query)
        self.assertEqual(det.language, SupportedLanguage.HINDI.value)
        self.assertTrue(det.is_mixed_language)

        # Test another Hinglish variation
        query_2 = "kya mujhe certificate milega is course mein?"
        det_2 = self.detector.detect(query_2)
        self.assertEqual(det_2.language, SupportedLanguage.HINDI.value)

    # -------------------------------------------------------------------------
    # 4. Mixed-language intent detection
    # -------------------------------------------------------------------------
    def test_mixed_language_intent_classification(self) -> None:
        """Verify intent classifier resolves Hinglish and mixed-language tokens."""
        # Pricing intent
        res1 = self.classifier.classify("course ki fees kya hai?")
        self.assertEqual(res1.intent, MultilingualIntent.PAYMENT_PRICING.value)

        # Refund policy intent
        res2 = self.classifier.classify("refund policy kya hai?")
        self.assertEqual(res2.intent, MultilingualIntent.REFUND_POLICY.value)

        # Prerequisites intent
        res3 = self.classifier.classify("course ke prerequisites kya hain?")
        self.assertEqual(res3.intent, MultilingualIntent.PREREQUISITES.value)

        # Career assistance intent
        res4 = self.classifier.classify("kya mujhe job placement assistance milegi?")
        self.assertEqual(res4.intent, MultilingualIntent.CAREER_ASSISTANCE.value)

    # -------------------------------------------------------------------------
    # 5. Spanish/English and French/English mixed queries
    # -------------------------------------------------------------------------
    def test_spanish_and_french_mixed_queries(self) -> None:
        """Verify European mixed-language queries are correctly detected and handled."""
        # Spanish + English
        es_query = "What are los prerequisites del course?"
        det_es = self.detector.detect(es_query)
        self.assertTrue(det_es.is_mixed_language)
        # Primary or secondary is Spanish
        self.assertTrue(
            det_es.language == SupportedLanguage.SPANISH.value
            or det_es.secondary_language == SupportedLanguage.SPANISH.value
        )

        # French + English
        fr_query = "What is la durée of the course?"
        det_fr = self.detector.detect(fr_query)
        self.assertTrue(det_fr.is_mixed_language)
        self.assertTrue(
            det_fr.language == SupportedLanguage.FRENCH.value
            or det_fr.secondary_language == SupportedLanguage.FRENCH.value
        )

    # -------------------------------------------------------------------------
    # 6. Competing/ambiguous intent detection
    # -------------------------------------------------------------------------
    def test_competing_intents_query(self) -> None:
        """Verify queries asking about two distinct topics are flagged as ambiguous."""
        multi_intent_query = "What are the course fees and refund policy?"
        intent_res = self.classifier.classify(multi_intent_query, language="en")
        self.assertTrue(intent_res.is_ambiguous)
        self.assertGreaterEqual(len(intent_res.competing_intents), 2)
        self.assertIn(MultilingualIntent.PAYMENT_PRICING.value, intent_res.competing_intents)
        self.assertIn(MultilingualIntent.REFUND_POLICY.value, intent_res.competing_intents)

        # Hinglish dual intent
        hinglish_multi = "Fees kitni hai aur refund policy kya hai?"
        intent_res_hi = self.classifier.classify(hinglish_multi, language="hi")
        self.assertTrue(intent_res_hi.is_ambiguous)
        self.assertIn(MultilingualIntent.PAYMENT_PRICING.value, intent_res_hi.competing_intents)
        self.assertIn(MultilingualIntent.REFUND_POLICY.value, intent_res_hi.competing_intents)

    # -------------------------------------------------------------------------
    # 7. Clearly dominant single-intent is NOT marked ambiguous
    # -------------------------------------------------------------------------
    def test_dominant_single_intent_not_ambiguous(self) -> None:
        """Verify focused single-topic queries avoid false-positive ambiguity flags."""
        single_query = "How much does the data science bootcamp cost?"
        intent_res = self.classifier.classify(single_query, language="en")
        self.assertEqual(intent_res.intent, MultilingualIntent.PAYMENT_PRICING.value)
        self.assertFalse(intent_res.is_ambiguous)
        self.assertEqual(intent_res.competing_intents, [MultilingualIntent.PAYMENT_PRICING.value])

    # -------------------------------------------------------------------------
    # 8. Underspecified queries safely handled
    # -------------------------------------------------------------------------
    def test_underspecified_query_clarification(self) -> None:
        """Verify ambiguous queries without evidence produce localized clarification prompts."""
        # Create empty retrieval result with is_ambiguous=True
        empty_retrieval = CrossLingualRetrievalResult(
            original_query="tell me about everything",
            detected_language="hi",
            intent=MultilingualIntent.GENERAL_INQUIRY.value,
            aligned_query="general overview",
            retrieved_documents=[],
            evidence_text="",
            is_evidence_found=False,
            confidence=0.0,
            metadata={"is_ambiguous": True},
        )
        resp = self.reasoner.reason(empty_retrieval, target_language="hi")
        self.assertTrue(resp.is_ambiguous)
        self.assertIsNotNone(resp.clarification_prompt)
        self.assertIn("स्पष्ट", resp.clarification_prompt or "")

    # -------------------------------------------------------------------------
    # 9. Mixed-language follow-up using Day 33 context
    # -------------------------------------------------------------------------
    def test_mixed_language_followup_with_context(self) -> None:
        """Verify Hinglish follow-up query resolves entity topic from previous English turn."""
        session_id = "day34_sess_followup"

        # Turn 1: English topic establishment
        req1 = MultilingualTextRequest(
            text="Tell me about the Data Science course.",
            session_id=session_id,
        )
        resp1 = self.service.answer_query(req1)
        self.assertEqual(resp1.language, SupportedLanguage.ENGLISH.value)

        # Turn 2: Hinglish follow-up referencing the course
        req2 = MultilingualTextRequest(
            text="Is course ki fees kitni hai?",
            session_id=session_id,
        )
        resp2 = self.service.answer_query(req2)
        self.assertEqual(resp2.language, SupportedLanguage.HINDI.value)
        self.assertEqual(resp2.intent, MultilingualIntent.PAYMENT_PRICING.value)
        self.assertTrue(resp2.is_grounded)

        # Verify context resolution enriched the query with Data Science
        session = self.service.get_session(session_id)
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(len(session.turns), 2)
        self.assertIn("Data Science", session.turns[1].resolved_query)

    # -------------------------------------------------------------------------
    # 10. Language switching + mixed-language follow-up
    # -------------------------------------------------------------------------
    def test_language_switching_with_mixed_followup(self) -> None:
        """Verify seamless switching across French, English, and Hinglish in a session."""
        session_id = "day34_sess_switch"

        # Turn 1: French
        req1 = MultilingualTextRequest(
            text="Quels sont les prérequis pour la formation Machine Learning?",
            session_id=session_id,
        )
        resp1 = self.service.answer_query(req1)
        self.assertEqual(resp1.language, SupportedLanguage.FRENCH.value)

        # Turn 2: Mixed English/Hindi follow-up
        req2 = MultilingualTextRequest(
            text="Is machine learning course ki fees kya hai?",
            session_id=session_id,
        )
        resp2 = self.service.answer_query(req2)
        self.assertEqual(resp2.language, SupportedLanguage.HINDI.value)
        self.assertEqual(resp2.intent, MultilingualIntent.PAYMENT_PRICING.value)

        # Turn 3: Switch to Spanish
        req3 = MultilingualTextRequest(
            text="¿Tienen opciones de pago a plazos?",
            session_id=session_id,
        )
        resp3 = self.service.answer_query(req3)
        self.assertEqual(resp3.language, SupportedLanguage.SPANISH.value)

    # -------------------------------------------------------------------------
    # 11. Multi-session isolation
    # -------------------------------------------------------------------------
    def test_multi_session_isolation(self) -> None:
        """Verify session contexts and mixed-language preferences remain strictly isolated."""
        sess_a = "sess_user_alpha"
        sess_b = "sess_user_beta"

        # Session A: established on Python
        self.service.answer_query(
            MultilingualTextRequest(text="Tell me about Python bootcamp", session_id=sess_a)
        )

        # Session B: established on Machine Learning
        self.service.answer_query(
            MultilingualTextRequest(text="Tell me about Machine Learning program", session_id=sess_b)
        )

        # Follow-up on Session A in Hinglish
        res_a = self.service.process_request(
            MultilingualTextRequest(text="Is course ke prerequisites kya hain?", session_id=sess_a)
        )
        self.assertIn("Python", res_a["resolved_query"])
        self.assertNotIn("Machine Learning", res_a["resolved_query"])

        # Follow-up on Session B in Hinglish
        res_b = self.service.process_request(
            MultilingualTextRequest(text="Is course ke prerequisites kya hain?", session_id=sess_b)
        )
        self.assertIn("Machine Learning", res_b["resolved_query"])
        self.assertNotIn("Python", res_b["resolved_query"])

    # -------------------------------------------------------------------------
    # 12. Stateless query processing backward compatibility
    # -------------------------------------------------------------------------
    def test_stateless_query_processing(self) -> None:
        """Verify requests without session_id function identically and statelessly."""
        req = MultilingualTextRequest(text="Python course ki fees kitni hai?")
        resp = self.service.answer_query(req)
        self.assertEqual(resp.language, SupportedLanguage.HINDI.value)
        self.assertEqual(resp.intent, MultilingualIntent.PAYMENT_PRICING.value)
        self.assertIsNone(req.session_id)

    # -------------------------------------------------------------------------
    # 13. Grounded fallback when evidence is insufficient
    # -------------------------------------------------------------------------
    def test_grounded_fallback_safety(self) -> None:
        """Verify unknown or ungroundable queries safely fall back without hallucination."""
        # Query that produces no relevant matches
        def empty_search(query: str, top_k: int = 3):
            return []

        empty_retriever = CrossLingualRetriever(custom_search_fn=empty_search)
        svc = MultilingualService(
            detector=self.detector,
            intent_classifier=self.classifier,
            retriever=empty_retriever,
            reasoner=self.reasoner,
        )

        req = MultilingualTextRequest(text="What is the quantum teleportation course schedule?")
        resp = svc.answer_query(req)
        self.assertFalse(resp.is_grounded)
        self.assertEqual(resp.confidence_score, 0.0)
        self.assertIn("not have enough information", resp.final_answer)

    # -------------------------------------------------------------------------
    # 14. Backward compatibility with existing models
    # -------------------------------------------------------------------------
    def test_model_contracts_backward_compatibility(self) -> None:
        """Verify LanguageIdentificationResult, MultilingualIntentResult, MultilingualResponse defaults."""
        # 1. LanguageIdentificationResult
        lir = LanguageIdentificationResult(
            text="test query",
            language="en",
            language_name="English",
            confidence=0.9,
            is_supported=True,
            is_reliable=True,
        )
        self.assertFalse(lir.is_mixed_language)
        self.assertIsNone(lir.secondary_language)

        # 2. MultilingualIntentResult
        mir = MultilingualIntentResult(
            text="test query",
            language="en",
            intent=MultilingualIntent.COURSE_DETAILS.value,
            confidence=0.8,
            is_recognized=True,
        )
        self.assertFalse(mir.is_ambiguous)
        self.assertEqual(mir.competing_intents, [])

        # 3. MultilingualResponse
        mr = MultilingualResponse(
            query="test",
            language="en",
            intent="COURSE_DETAILS",
            aligned_query="test",
            final_answer="answer",
            raw_answer="answer",
            is_grounded=True,
            confidence_score=0.9,
        )
        self.assertFalse(mr.is_ambiguous)
        self.assertIsNone(mr.clarification_prompt)


if __name__ == "__main__":
    unittest.main()
