"""Multilingual Service and Coordination Layer for Multilingual Chatbot (Phase 6).

Provides an integration boundary for incoming requests, coordinating language detection,
session language state management, and request routing.
"""

from typing import Any, Dict, Optional

from src.multilingual.detector import LanguageDetector
from src.multilingual.intents import MultilingualIntentClassifier
from src.multilingual.models import (
    CrossLingualRetrievalResult,
    LanguageIdentificationResult,
    MultilingualIntentResult,
    MultilingualResponse,
    MultilingualTextRequest,
    SupportedLanguage,
)
from src.multilingual.reasoning import MultilingualReasoner
from src.multilingual.retrieval import CrossLingualRetriever


class MultilingualService:
    """Service layer managing multilingual interactions, language identification, intent detection, and cross-lingual RAG."""

    def __init__(
        self,
        detector: Optional[LanguageDetector] = None,
        intent_classifier: Optional[MultilingualIntentClassifier] = None,
        retriever: Optional[CrossLingualRetriever] = None,
        reasoner: Optional[MultilingualReasoner] = None,
        default_language: str = SupportedLanguage.ENGLISH.value,
    ) -> None:
        """Initialize the multilingual service.

        Args:
            detector (Optional[LanguageDetector]): Injected or default language detector.
            intent_classifier (Optional[MultilingualIntentClassifier]): Injected or default intent classifier.
            retriever (Optional[CrossLingualRetriever]): Injected or default cross-lingual retriever.
            reasoner (Optional[MultilingualReasoner]): Injected or default multilingual reasoner.
            default_language (str): Fallback language if identification is unreliable/unknown.
        """
        self.detector = detector if detector is not None else LanguageDetector()
        self.intent_classifier = (
            intent_classifier
            if intent_classifier is not None
            else MultilingualIntentClassifier(language_detector=self.detector)
        )
        self.retriever = retriever if retriever is not None else CrossLingualRetriever()
        self.reasoner = reasoner if reasoner is not None else MultilingualReasoner()
        self.default_language = default_language
        self._session_languages: Dict[str, str] = {}

    def identify_language(self, text: str) -> LanguageIdentificationResult:
        """Perform language identification on raw query text.

        Args:
            text (str): Query text.

        Returns:
            LanguageIdentificationResult: Structured identification result.
        """
        return self.detector.detect(text)

    def classify_intent(
        self,
        text: str,
        language: Optional[str] = None,
    ) -> MultilingualIntentResult:
        """Classify user query into canonical customer-service domain intent.

        Args:
            text (str): Query text.
            language (Optional[str]): Language code if identified.

        Returns:
            MultilingualIntentResult: Structured intent result.
        """
        return self.intent_classifier.classify(text, language=language)

    def retrieve_evidence(
        self,
        query: str,
        language: str,
        intent_result: Optional[MultilingualIntentResult] = None,
        top_k: int = 3,
    ) -> CrossLingualRetrievalResult:
        """Retrieve English evidence for a multilingual query using intent-aligned representations.

        Args:
            query (str): User query.
            language (str): Detected or specified language.
            intent_result (Optional[MultilingualIntentResult]): Intent classification result.
            top_k (int): Number of top documents to retrieve.

        Returns:
            CrossLingualRetrievalResult: Structured retrieval evidence.
        """
        return self.retriever.retrieve(
            query=query,
            language=language,
            intent_result=intent_result,
            top_k=top_k,
        )

    def generate_response(
        self,
        retrieval_result: CrossLingualRetrievalResult,
        target_language: Optional[str] = None,
    ) -> MultilingualResponse:
        """Generate a grounded response in the user's language based on retrieved evidence.

        Args:
            retrieval_result (CrossLingualRetrievalResult): Retrieval outcome.
            target_language (Optional[str]): Target language for synthesis.

        Returns:
            MultilingualResponse: Grounded response payload.
        """
        return self.reasoner.reason(
            retrieval_result=retrieval_result,
            target_language=target_language,
        )

    def answer_query(self, request: MultilingualTextRequest) -> MultilingualResponse:
        """End-to-end processing of a multilingual request returning a structured MultilingualResponse.

        Args:
            request (MultilingualTextRequest): Validated text request.

        Returns:
            MultilingualResponse: Grounded response object.
        """
        processed = self.process_request(request)
        return MultilingualResponse(
            query=request.text,
            language=processed["effective_language"],
            intent=processed["intent"]["intent"],
            aligned_query=processed["retrieval"]["aligned_query"],
            final_answer=processed["final_answer"],
            raw_answer=processed["response"]["raw_answer"],
            is_grounded=processed["is_grounded"],
            confidence_score=processed["response"]["confidence_score"],
            source_documents=processed["retrieval"]["retrieved_documents"],
            metadata=processed["response"]["metadata"],
        )

    def process_request(self, request: MultilingualTextRequest) -> Dict[str, Any]:
        """Process an incoming multilingual text request through detection, intent, retrieval, and reasoning.

        Args:
            request (MultilingualTextRequest): Validated text request.

        Returns:
            Dict[str, Any]: Complete request execution payload with all stage metadata.
        """
        if not isinstance(request, MultilingualTextRequest):
            raise TypeError(f"Expected MultilingualTextRequest, got {type(request).__name__}")

        # 1. Language Identification
        if request.forced_language:
            effective_lang = request.forced_language
            detection_result = self.identify_language(request.text)
            is_forced = True
        else:
            detection_result = self.identify_language(request.text)
            if detection_result.is_supported and detection_result.is_reliable:
                effective_lang = detection_result.language
            else:
                session_lang = (
                    self._session_languages.get(request.session_id)
                    if request.session_id
                    else None
                )
                effective_lang = session_lang or self.default_language
            is_forced = False

        # 2. Intent Classification
        intent_result = self.classify_intent(request.text, language=effective_lang)

        # 3. Cross-Lingual Retrieval
        retrieval_result = self.retrieve_evidence(
            query=request.text,
            language=effective_lang,
            intent_result=intent_result,
        )

        # 4. Grounded Multilingual Reasoning
        response_obj = self.generate_response(
            retrieval_result=retrieval_result,
            target_language=effective_lang,
        )

        # Update session language state if session_id is present
        if request.session_id:
            self._session_languages[request.session_id] = effective_lang

        return {
            "text": request.text,
            "effective_language": effective_lang,
            "language_name": SupportedLanguage.get_language_name(effective_lang),
            "is_forced": is_forced,
            "session_id": request.session_id,
            "detection": detection_result.to_dict(),
            "intent": intent_result.to_dict(),
            "retrieval": retrieval_result.to_dict(),
            "response": response_obj.to_dict(),
            "final_answer": response_obj.final_answer,
            "is_grounded": response_obj.is_grounded,
        }


    def get_session_language(self, session_id: str) -> Optional[str]:
        """Retrieve current tracked language preference for a session."""
        return self._session_languages.get(session_id)

    def clear_session(self, session_id: str) -> None:
        """Clear language preferences for a session."""
        self._session_languages.pop(session_id, None)

    def reset_all_sessions(self) -> None:
        """Clear all session states."""
        self._session_languages.clear()
