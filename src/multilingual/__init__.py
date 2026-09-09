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
    CrossLingualRetrievalResult,
    LanguageCandidate,
    LanguageIdentificationResult,
    MultilingualIntent,
    MultilingualIntentResult,
    MultilingualResponse,
    MultilingualTextRequest,
    SupportedLanguage,
)
from src.multilingual.reasoning import (
    MultilingualReasoner,
)
from src.multilingual.retrieval import (
    CrossLingualRetriever,
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
    "CrossLingualRetrievalResult",
    "MultilingualResponse",
    "LanguageDetector",
    "MultilingualIntentClassifier",
    "CrossLingualRetriever",
    "MultilingualReasoner",
    "MultilingualService",
]
