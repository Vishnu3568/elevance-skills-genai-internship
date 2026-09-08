"""Multilingual Service and Coordination Layer for Multilingual Chatbot (Phase 6).

Provides an integration boundary for incoming requests, coordinating language detection,
session language state management, and request routing.
"""

from typing import Any, Dict, Optional

from src.multilingual.detector import LanguageDetector
from src.multilingual.intents import MultilingualIntentClassifier
from src.multilingual.models import (
    LanguageIdentificationResult,
    MultilingualIntentResult,
    MultilingualTextRequest,
    SupportedLanguage,
)


class MultilingualService:
    """Service layer managing multilingual interactions, language identification, and intent detection."""

    def __init__(
        self,
        detector: Optional[LanguageDetector] = None,
        intent_classifier: Optional[MultilingualIntentClassifier] = None,
        default_language: str = SupportedLanguage.ENGLISH.value,
    ) -> None:
        """Initialize the multilingual service.

        Args:
            detector (Optional[LanguageDetector]): Injected or default language detector.
            intent_classifier (Optional[MultilingualIntentClassifier]): Injected or default intent classifier.
            default_language (str): Fallback language if identification is unreliable/unknown.
        """
        self.detector = detector if detector is not None else LanguageDetector()
        self.intent_classifier = (
            intent_classifier
            if intent_classifier is not None
            else MultilingualIntentClassifier(language_detector=self.detector)
        )
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

    def process_request(self, request: MultilingualTextRequest) -> Dict[str, Any]:
        """Process an incoming multilingual text request and determine language and intent.

        Args:
            request (MultilingualTextRequest): Validated text request.

        Returns:
            Dict[str, Any]: Processed request context with language and intent metadata.
        """
        if not isinstance(request, MultilingualTextRequest):
            raise TypeError(f"Expected MultilingualTextRequest, got {type(request).__name__}")

        # If forced language is provided, honor it
        if request.forced_language:
            effective_lang = request.forced_language
            detection_result = self.identify_language(request.text)
            is_forced = True
        else:
            detection_result = self.identify_language(request.text)
            if detection_result.is_supported and detection_result.is_reliable:
                effective_lang = detection_result.language
            else:
                # Use session language preference or default
                session_lang = (
                    self._session_languages.get(request.session_id)
                    if request.session_id
                    else None
                )
                effective_lang = session_lang or self.default_language
            is_forced = False

        # Classify intent using effective language
        intent_result = self.classify_intent(request.text, language=effective_lang)

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
