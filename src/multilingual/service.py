"""Multilingual Service and Coordination Layer for Multilingual Chatbot (Phase 6).

Provides an integration boundary for incoming requests, coordinating language detection,
session language state management, multi-turn context retention, and request routing.
"""

from typing import Any, Dict, List, Optional

from src.multilingual.context import MultilingualContextResolver
from src.multilingual.detector import LanguageDetector
from src.multilingual.intents import MultilingualIntentClassifier
from src.multilingual.models import (
    CrossLingualRetrievalResult,
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


class MultilingualService:
    """Service layer managing multilingual interactions, language identification, intent detection, and cross-lingual RAG."""

    def __init__(
        self,
        detector: Optional[LanguageDetector] = None,
        intent_classifier: Optional[MultilingualIntentClassifier] = None,
        retriever: Optional[CrossLingualRetriever] = None,
        reasoner: Optional[MultilingualReasoner] = None,
        context_resolver: Optional[MultilingualContextResolver] = None,
        default_language: str = SupportedLanguage.ENGLISH.value,
        max_session_turns: int = 10,
    ) -> None:
        """Initialize the multilingual service.

        Args:
            detector (Optional[LanguageDetector]): Injected or default language detector.
            intent_classifier (Optional[MultilingualIntentClassifier]): Injected or default intent classifier.
            retriever (Optional[CrossLingualRetriever]): Injected or default cross-lingual retriever.
            reasoner (Optional[MultilingualReasoner]): Injected or default multilingual reasoner.
            context_resolver (Optional[MultilingualContextResolver]): Injected or default context resolver.
            default_language (str): Fallback language if identification is unreliable/unknown.
            max_session_turns (int): Maximum turns retained per conversation session.
        """
        self.detector = detector if detector is not None else LanguageDetector()
        self.intent_classifier = (
            intent_classifier
            if intent_classifier is not None
            else MultilingualIntentClassifier(language_detector=self.detector)
        )
        self.retriever = retriever if retriever is not None else CrossLingualRetriever()
        self.reasoner = reasoner if reasoner is not None else MultilingualReasoner()
        self.context_resolver = context_resolver if context_resolver is not None else MultilingualContextResolver()
        self.default_language = default_language
        self.max_session_turns = max_session_turns
        self._session_languages: Dict[str, str] = {}
        self._sessions: Dict[str, MultilingualConversationSession] = {}

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
        context_topic: Optional[str] = None,
    ) -> CrossLingualRetrievalResult:
        """Retrieve English evidence for a multilingual query using intent-aligned representations.

        Args:
            query (str): User query.
            language (str): Detected or specified language.
            intent_result (Optional[MultilingualIntentResult]): Intent classification result.
            top_k (int): Number of top documents to retrieve.
            context_topic (Optional[str]): Optional conversation context topic for follow-ups.

        Returns:
            CrossLingualRetrievalResult: Structured retrieval evidence.
        """
        return self.retriever.retrieve(
            query=query,
            language=language,
            intent_result=intent_result,
            top_k=top_k,
            context_topic=context_topic,
        )

    def generate_response(
        self,
        retrieval_result: CrossLingualRetrievalResult,
        target_language: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> MultilingualResponse:
        """Generate a grounded response in the user's language based on retrieved evidence.

        Args:
            retrieval_result (CrossLingualRetrievalResult): Retrieval outcome.
            target_language (Optional[str]): Target language for synthesis.
            conversation_history (Optional[List[Dict[str, Any]]]): Prior conversational turns.

        Returns:
            MultilingualResponse: Grounded response payload.
        """
        return self.reasoner.reason(
            retrieval_result=retrieval_result,
            target_language=target_language,
            conversation_history=conversation_history,
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

        # Retrieve or initialize conversation session if session_id is provided
        session: Optional[MultilingualConversationSession] = None
        if request.session_id:
            if request.session_id not in self._sessions:
                self._sessions[request.session_id] = MultilingualConversationSession(
                    session_id=request.session_id,
                    max_turns=self.max_session_turns,
                )
            session = self._sessions[request.session_id]

        # 1. Language Identification (with seamless multi-turn language switching support)
        if request.forced_language:
            effective_lang = request.forced_language
            detection_result = self.identify_language(request.text)
            is_forced = True
        else:
            detection_result = self.identify_language(request.text)
            if detection_result.is_supported and detection_result.is_reliable:
                effective_lang = detection_result.language
            elif detection_result.is_supported and detection_result.language != SupportedLanguage.UNKNOWN.value and detection_result.confidence >= 0.35:
                effective_lang = detection_result.language
            else:
                session_lang = (
                    session.active_language
                    if session and session.active_language
                    else (self._session_languages.get(request.session_id) if request.session_id else None)
                )
                if session_lang:
                    effective_lang = session_lang
                elif detection_result.is_supported and detection_result.language != SupportedLanguage.UNKNOWN.value:
                    effective_lang = detection_result.language
                else:
                    effective_lang = self.default_language
            is_forced = False

        # 2. Context Resolution (Reference & Pronoun resolution from prior turns)
        resolved_query, context_topic, is_followup = self.context_resolver.resolve_context(
            text=request.text,
            language=effective_lang,
            session=session,
        )

        # 3. Intent Classification (prioritize active utterance intent with fallback to context)
        intent_result = self.classify_intent(request.text, language=effective_lang)
        if (not intent_result.is_recognized or intent_result.intent == MultilingualIntent.UNKNOWN.value) and is_followup:
            resolved_intent_result = self.classify_intent(resolved_query, language=effective_lang)
            if resolved_intent_result.is_recognized and resolved_intent_result.intent != MultilingualIntent.UNKNOWN.value:
                intent_result = resolved_intent_result

        # 4. Cross-Lingual Retrieval (context-enriched English alignment)
        retrieval_result = self.retrieve_evidence(
            query=request.text,
            language=effective_lang,
            intent_result=intent_result,
            context_topic=context_topic,
        )

        # 5. Grounded Multilingual Reasoning
        conv_history = [t.to_dict() for t in session.turns] if session else None
        response_obj = self.generate_response(
            retrieval_result=retrieval_result,
            target_language=effective_lang,
            conversation_history=conv_history,
        )

        # 6. Record Turn in Session History
        if session is not None:
            turn = MultilingualConversationTurn(
                turn_id=len(session.turns) + 1,
                query=request.text,
                resolved_query=resolved_query,
                language=effective_lang,
                intent=intent_result.intent,
                final_answer=response_obj.final_answer,
                topic=context_topic,
                is_grounded=response_obj.is_grounded,
            )
            session.add_turn(turn)
            if request.session_id:
                self._session_languages[request.session_id] = effective_lang

        return {
            "text": request.text,
            "resolved_query": resolved_query,
            "is_followup": is_followup,
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

    def get_session(self, session_id: str) -> Optional[MultilingualConversationSession]:
        """Retrieve full conversation session object if present."""
        return self._sessions.get(session_id)

    def get_session_language(self, session_id: str) -> Optional[str]:
        """Retrieve current tracked language preference for a session."""
        session = self._sessions.get(session_id)
        if session and session.active_language:
            return session.active_language
        return self._session_languages.get(session_id)

    def clear_session(self, session_id: str) -> None:
        """Clear session state and language preferences for a session."""
        self._session_languages.pop(session_id, None)
        self._sessions.pop(session_id, None)

    def reset_all_sessions(self) -> None:
        """Clear all session states."""
        self._session_languages.clear()
        self._sessions.clear()
