"""Multilingual Conversational Context Resolver (Phase 6 Day 33).

Provides reference resolution and context enrichment across English, Spanish, French,
German, and Hindi for multi-turn conversational follow-ups and language switching.
"""

import re
from typing import Dict, List, Optional, Tuple

from src.multilingual.models import (
    MultilingualConversationSession,
    MultilingualConversationTurn,
    MultilingualIntent,
    SupportedLanguage,
)


# -----------------------------------------------------------------------------
# Follow-up Patterns and Pronoun Signals Across Supported Languages
# -----------------------------------------------------------------------------

PRONOUN_PATTERNS: Dict[str, re.Pattern[str]] = {
    SupportedLanguage.ENGLISH.value: re.compile(
        r"\b(it|its|this|that|these|those|the course|the bootcamp|the program|the policy)\b",
        re.IGNORECASE,
    ),
    SupportedLanguage.SPANISH.value: re.compile(
        r"\b(su|sus|este|esta|esto|estos|estas|el curso|el programa|la política|del mismo|de este)\b",
        re.IGNORECASE,
    ),
    SupportedLanguage.FRENCH.value: re.compile(
        r"\b(sa|son|ses|ce|cet|cette|ces|le cours|la formation|le programme|la politique|de celui-ci)\b",
        re.IGNORECASE,
    ),
    SupportedLanguage.GERMAN.value: re.compile(
        r"\b(sein|seine|seinem|seinen|ihr|ihre|ihrem|dieser|dieses|diesem|der kurs|das programm|die richtlinie)\b",
        re.IGNORECASE,
    ),
    SupportedLanguage.HINDI.value: re.compile(
        r"(इसका|इसकी|इसके|यह|ये|उसका|उसकी|उसके|वह|कोर्स|बूटकैंप|कार्यक्रम|नीति)",
        re.IGNORECASE,
    ),
}

ELLIPSIS_OR_FOLLOWUP_STARTS: Dict[str, Tuple[str, ...]] = {
    SupportedLanguage.ENGLISH.value: (
        "what about", "how about", "and ", "what are its", "what is its", "how long",
        "how much", "can i", "is there", "tell me about its", "prerequisites", "duration", "cost", "fees",
    ),
    SupportedLanguage.SPANISH.value: (
        "¿y ", "y ", "¿qué tal", "¿cuánto dura", "¿cuánto cuesta", "¿cuáles son sus",
        "requisitos", "duración", "precio", "costo", "cuotas",
    ),
    SupportedLanguage.FRENCH.value: (
        "et ", "qu'en est-il", "combien coûte", "quelle est sa", "durée", "prérequis",
        "tarif", "prix", "conditions",
    ),
    SupportedLanguage.GERMAN.value: (
        "und ", "was ist mit", "wie lange dauert", "was kostet", "wie viel", "welche",
        "dauer", "voraussetzungen", "kosten", "gebühren",
    ),
    SupportedLanguage.HINDI.value: (
        "और ", "क्या ", "कितनी ", "कितना ", "इसकी ", "इसके ", "अवधि", "फीस", "योग्यता", "शर्तें",
    ),
}

INTENT_TOPIC_NAMES: Dict[str, str] = {
    MultilingualIntent.COURSE_DETAILS.value: "course details curriculum",
    MultilingualIntent.PREREQUISITES.value: "course prerequisites laptop 4GB ram",
    MultilingualIntent.REFUND_POLICY.value: "refund policy money back guarantee",
    MultilingualIntent.CAREER_ASSISTANCE.value: "job placement assistance internship",
    MultilingualIntent.PAYMENT_PRICING.value: "payment options EMI pricing",
    MultilingualIntent.SUPPORT_CONTACT.value: "discord community support contact",
    MultilingualIntent.TECHNICAL_SUPPORT.value: "technical support software installation",
    MultilingualIntent.GENERAL_INQUIRY.value: "data science AI bootcamp training",
    MultilingualIntent.GREETING.value: "course inquiry",
}


class MultilingualContextResolver:
    """Resolves conversational references and enriches queries for multi-turn follow-ups."""

    @classmethod
    def is_followup_query(
        cls,
        text: str,
        language: str,
        session: Optional[MultilingualConversationSession] = None,
    ) -> bool:
        """Determine whether a query is a follow-up that relies on prior conversation context."""
        if not session or not session.turns:
            return False

        q_clean = text.strip().lower()

        # Check pronoun match
        pat = PRONOUN_PATTERNS.get(language) or PRONOUN_PATTERNS[SupportedLanguage.ENGLISH.value]
        if pat.search(q_clean):
            return True

        # Check ellipsis / short follow-up starts
        starts = ELLIPSIS_OR_FOLLOWUP_STARTS.get(language) or ELLIPSIS_OR_FOLLOWUP_STARTS[SupportedLanguage.ENGLISH.value]
        if any(q_clean.startswith(prefix) for prefix in starts):
            return True

        # Check ultra-short query length (< 5 words without explicit standalone entity)
        words = q_clean.split()
        if len(words) <= 4 and session.get_last_turn() is not None:
            return True

        return False

    @classmethod
    def resolve_context(
        cls,
        text: str,
        language: str,
        session: Optional[MultilingualConversationSession] = None,
    ) -> Tuple[str, Optional[str], bool]:
        """Resolve conversational context and return (resolved_query, topic, is_followup).

        Args:
            text (str): Incoming user query.
            language (str): Detected or specified language.
            session (Optional[MultilingualConversationSession]): Active session if any.

        Returns:
            Tuple[str, Optional[str], bool]: (resolved_query, topic, is_followup).
        """
        if not session or not session.turns:
            return text, None, False

        last_turn = session.get_last_turn()
        if last_turn is None:
            return text, None, False

        is_followup = cls.is_followup_query(text, language, session)
        if not is_followup:
            return text, None, False

        # Extract topic from prior turn
        prior_topic = last_turn.topic or INTENT_TOPIC_NAMES.get(last_turn.intent, "data science bootcamp")

        # Create resolved query incorporating previous context topic
        resolved_text = f"{text} (context: {prior_topic})"
        return resolved_text, prior_topic, True
