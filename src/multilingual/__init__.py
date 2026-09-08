"""Multilingual AI Assistant Package (Phase 6).

Exports core models, language identification engines, and multilingual service components.
"""

from src.multilingual.detector import (
    LanguageDetector,
)
from src.multilingual.intents import (
    MultilingualIntentClassifier,
)
from src.multilingual.models import (
    LanguageCandidate,
    LanguageIdentificationResult,
    MultilingualIntent,
    MultilingualIntentResult,
    MultilingualTextRequest,
    SupportedLanguage,
)
from src.multilingual.service import (
    MultilingualService,
)

__all__ = [
    "SupportedLanguage",
    "LanguageCandidate",
    "LanguageIdentificationResult",
    "MultilingualTextRequest",
    "MultilingualIntent",
    "MultilingualIntentResult",
    "LanguageDetector",
    "MultilingualIntentClassifier",
    "MultilingualService",
]

