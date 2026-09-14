"""Data models and contracts for Multilingual Chatbot (Phase 6).

Defines canonical representations for supported languages, language identification
results, candidates, and multilingual query inputs.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


# -----------------------------------------------------------------------------
# Supported Languages Enumeration
# -----------------------------------------------------------------------------

class SupportedLanguage(str, Enum):
    """Enumeration of officially supported languages in Phase 6."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    HINDI = "hi"
    UNKNOWN = "unknown"

    @classmethod
    def get_supported_codes(cls) -> Set[str]:
        """Return the set of valid ISO 639-1 language codes (excluding unknown)."""
        return {cls.ENGLISH.value, cls.SPANISH.value, cls.FRENCH.value, cls.GERMAN.value, cls.HINDI.value}

    @classmethod
    def get_language_name(cls, code: str) -> str:
        """Return human-readable language name for an ISO code."""
        names = {
            cls.ENGLISH.value: "English",
            cls.SPANISH.value: "Spanish",
            cls.FRENCH.value: "French",
            cls.GERMAN.value: "German",
            cls.HINDI.value: "Hindi",
            cls.UNKNOWN.value: "Unknown",
        }
        return names.get(code.lower(), "Unknown")


# -----------------------------------------------------------------------------
# Language Candidate Model
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class LanguageCandidate:
    """Represents a scored language candidate from the identification engine."""

    language: str
    confidence: float
    language_name: str

    def __post_init__(self) -> None:
        """Validate candidate fields and confidence bounds."""
        if not isinstance(self.language, str) or not self.language.strip():
            raise ValueError("Candidate language code must be a non-empty string.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence score {self.confidence} must be within [0.0, 1.0].")

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary representation of the language candidate."""
        return {
            "language": self.language,
            "confidence": round(self.confidence, 4),
            "language_name": self.language_name,
        }


# -----------------------------------------------------------------------------
# Language Identification Result Model
# -----------------------------------------------------------------------------

@dataclass
class LanguageIdentificationResult:
    """Canonical contract for the outcome of language identification."""

    text: str
    language: str
    confidence: float
    is_supported: bool
    language_name: str
    candidates: List[LanguageCandidate] = field(default_factory=list)
    is_reliable: bool = True
    detected_script: Optional[str] = None
    is_mixed_language: bool = False
    secondary_language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate invariants of the identification result."""
        if not isinstance(self.text, str):
            raise TypeError(f"text must be str, got {type(self.text).__name__}")
        if not isinstance(self.language, str):
            raise TypeError(f"language must be str, got {type(self.language).__name__}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence score {self.confidence} must be within [0.0, 1.0].")
        if not isinstance(self.is_supported, bool):
            raise TypeError("is_supported must be a boolean.")
        if not isinstance(self.is_reliable, bool):
            raise TypeError("is_reliable must be a boolean.")
        if not isinstance(self.is_mixed_language, bool):
            raise TypeError("is_mixed_language must be a boolean.")

    def to_dict(self) -> Dict[str, Any]:
        """Return a clean dictionary representation for logging and serialization."""
        return {
            "text": self.text,
            "language": self.language,
            "confidence": round(self.confidence, 4),
            "is_supported": self.is_supported,
            "language_name": self.language_name,
            "is_reliable": self.is_reliable,
            "detected_script": self.detected_script,
            "is_mixed_language": self.is_mixed_language,
            "secondary_language": self.secondary_language,
            "candidates": [c.to_dict() for c in self.candidates],
            "metadata": dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# Multilingual Input Request Contract
# -----------------------------------------------------------------------------

@dataclass
class MultilingualTextRequest:
    """Contract for incoming multilingual text requests."""

    text: str
    forced_language: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate multilingual text request."""
        if not isinstance(self.text, str):
            raise TypeError(f"Request text must be str, got {type(self.text).__name__}")
        if not self.text.strip():
            raise ValueError("Request text cannot be empty or whitespace-only.")
        if self.forced_language is not None:
            if not isinstance(self.forced_language, str) or not self.forced_language.strip():
                raise ValueError("forced_language if provided must be a non-empty string.")

    def to_dict(self) -> Dict[str, Any]:
        """Return serialized request dictionary."""
        return {
            "text": self.text,
            "forced_language": self.forced_language,
            "session_id": self.session_id,
            "metadata": dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# Multilingual Intent Enumeration and Result Contract
# -----------------------------------------------------------------------------

class MultilingualIntent(str, Enum):
    """Enumeration of canonical customer-service domain intents in Phase 6."""

    GREETING = "greeting"
    REFUND_POLICY = "refund_policy"
    PREREQUISITES = "prerequisites"
    COURSE_DETAILS = "course_details"
    CAREER_ASSISTANCE = "career_assistance"
    SUPPORT_CONTACT = "support_contact"
    PAYMENT_PRICING = "payment_pricing"
    TECHNICAL_SUPPORT = "technical_support"
    GENERAL_INQUIRY = "general_inquiry"
    UNKNOWN = "unknown"

    @classmethod
    def get_valid_intents(cls) -> Set[str]:
        """Return all valid intent string values excluding unknown."""
        return {
            cls.GREETING.value,
            cls.REFUND_POLICY.value,
            cls.PREREQUISITES.value,
            cls.COURSE_DETAILS.value,
            cls.CAREER_ASSISTANCE.value,
            cls.SUPPORT_CONTACT.value,
            cls.PAYMENT_PRICING.value,
            cls.TECHNICAL_SUPPORT.value,
            cls.GENERAL_INQUIRY.value,
        }


@dataclass
class MultilingualIntentResult:
    """Canonical contract for the outcome of multilingual intent detection."""

    text: str
    intent: str
    confidence: float
    language: str
    is_recognized: bool = True
    matched_keywords: List[str] = field(default_factory=list)
    is_ambiguous: bool = False
    competing_intents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate invariants of the intent detection result."""
        if not isinstance(self.text, str):
            raise TypeError(f"text must be str, got {type(self.text).__name__}")
        if not isinstance(self.intent, str):
            raise TypeError(f"intent must be str, got {type(self.intent).__name__}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence score {self.confidence} must be within [0.0, 1.0].")
        if not isinstance(self.language, str):
            raise TypeError(f"language must be str, got {type(self.language).__name__}")
        if not isinstance(self.is_recognized, bool):
            raise TypeError("is_recognized must be a boolean.")
        if not isinstance(self.is_ambiguous, bool):
            raise TypeError("is_ambiguous must be a boolean.")

    def to_dict(self) -> Dict[str, Any]:
        """Return a clean dictionary representation for logging and serialization."""
        return {
            "text": self.text,
            "intent": self.intent,
            "confidence": round(self.confidence, 4),
            "language": self.language,
            "is_recognized": self.is_recognized,
            "is_ambiguous": self.is_ambiguous,
            "competing_intents": list(self.competing_intents),
            "matched_keywords": list(self.matched_keywords),
            "metadata": dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# Cross-Lingual Retrieval and Multilingual Response Contracts (Day 32)
# -----------------------------------------------------------------------------

@dataclass
class CrossLingualRetrievalResult:
    """Represents the outcome of cross-lingual query alignment and knowledge retrieval."""

    original_query: str
    detected_language: str
    intent: str
    aligned_query: str
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    evidence_text: str = ""
    is_evidence_found: bool = False
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate retrieval result invariants."""
        if not isinstance(self.original_query, str):
            raise TypeError("original_query must be str.")
        if not isinstance(self.detected_language, str):
            raise TypeError("detected_language must be str.")
        if not isinstance(self.intent, str):
            raise TypeError("intent must be str.")
        if not isinstance(self.aligned_query, str):
            raise TypeError("aligned_query must be str.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence {self.confidence} must be in [0.0, 1.0].")

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary representation of the retrieval result."""
        return {
            "original_query": self.original_query,
            "detected_language": self.detected_language,
            "intent": self.intent,
            "aligned_query": self.aligned_query,
            "retrieved_documents": [dict(d) for d in self.retrieved_documents],
            "evidence_text": self.evidence_text,
            "is_evidence_found": self.is_evidence_found,
            "confidence": round(self.confidence, 4),
            "metadata": dict(self.metadata),
        }


@dataclass
class MultilingualResponse:
    """Canonical contract for grounded multilingual answer generation and reasoning."""

    query: str
    language: str
    intent: str
    aligned_query: str
    final_answer: str
    raw_answer: str
    is_grounded: bool = True
    confidence_score: float = 0.0
    is_ambiguous: bool = False
    clarification_prompt: Optional[str] = None
    source_documents: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate response invariants."""
        if not isinstance(self.query, str):
            raise TypeError("query must be str.")
        if not isinstance(self.language, str):
            raise TypeError("language must be str.")
        if not isinstance(self.intent, str):
            raise TypeError("intent must be str.")
        if not isinstance(self.aligned_query, str):
            raise TypeError("aligned_query must be str.")
        if not isinstance(self.final_answer, str):
            raise TypeError("final_answer must be str.")
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(f"Confidence score {self.confidence_score} must be in [0.0, 1.0].")
        if not isinstance(self.is_ambiguous, bool):
            raise TypeError("is_ambiguous must be a boolean.")

    def to_dict(self) -> Dict[str, Any]:
        """Return serialized response dictionary."""
        return {
            "query": self.query,
            "language": self.language,
            "intent": self.intent,
            "aligned_query": self.aligned_query,
            "final_answer": self.final_answer,
            "raw_answer": self.raw_answer,
            "is_grounded": self.is_grounded,
            "confidence_score": round(self.confidence_score, 4),
            "is_ambiguous": self.is_ambiguous,
            "clarification_prompt": self.clarification_prompt,
            "source_documents": [dict(d) for d in self.source_documents],
            "metadata": dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# Conversation & Session State Models (Day 33)
# -----------------------------------------------------------------------------

@dataclass
class MultilingualConversationTurn:
    """Represents a single conversational turn in a multilingual session."""

    turn_id: int
    query: str
    resolved_query: str
    language: str
    intent: str
    final_answer: str
    topic: Optional[str] = None
    is_grounded: bool = True
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate turn invariants."""
        if not isinstance(self.turn_id, int) or self.turn_id < 0:
            raise ValueError("turn_id must be a non-negative integer.")
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must be a non-empty string.")
        if not isinstance(self.language, str) or not self.language.strip():
            raise ValueError("language must be a non-empty string.")
        if not isinstance(self.intent, str) or not self.intent.strip():
            raise ValueError("intent must be a non-empty string.")
        if not isinstance(self.final_answer, str):
            raise TypeError("final_answer must be str.")

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary representation of the conversation turn."""
        return {
            "turn_id": self.turn_id,
            "query": self.query,
            "resolved_query": self.resolved_query,
            "language": self.language,
            "intent": self.intent,
            "final_answer": self.final_answer,
            "topic": self.topic,
            "is_grounded": self.is_grounded,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


@dataclass
class MultilingualConversationSession:
    """Bounded conversation session tracking multi-turn dialogue and language preferences."""

    session_id: str
    max_turns: int = 10
    active_language: Optional[str] = None
    turns: List[MultilingualConversationTurn] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate session invariants."""
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        if not isinstance(self.max_turns, int) or self.max_turns <= 0:
            raise ValueError("max_turns must be a positive integer.")

    def add_turn(self, turn: MultilingualConversationTurn) -> None:
        """Add a completed turn and bound the history."""
        if not isinstance(turn, MultilingualConversationTurn):
            raise TypeError(f"Expected MultilingualConversationTurn, got {type(turn).__name__}")
        self.turns.append(turn)
        self.active_language = turn.language
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_last_turn(self) -> Optional[MultilingualConversationTurn]:
        """Return the most recent conversation turn if any exists."""
        return self.turns[-1] if self.turns else None

    def get_history(self, limit: Optional[int] = None) -> List[MultilingualConversationTurn]:
        """Return chronological turn history, optionally limited."""
        if limit is not None and limit > 0:
            return list(self.turns[-limit:])
        return list(self.turns)

    def clear(self) -> None:
        """Reset conversation turns and session language."""
        self.turns.clear()
        self.active_language = None
        self.metadata.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Return serialized session state."""
        return {
            "session_id": self.session_id,
            "max_turns": self.max_turns,
            "active_language": self.active_language,
            "turn_count": len(self.turns),
            "turns": [t.to_dict() for t in self.turns],
            "metadata": dict(self.metadata),
        }
