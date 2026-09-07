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
