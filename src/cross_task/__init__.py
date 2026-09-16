"""Cross-Task Integration Package (Phase 6 — Day 36).

Provides a clean orchestration layer integrating Tasks 1–6 (Customer Support,
Sentiment Analysis, Medical Q&A, Scientific Literature, Multimodal AI, and
Multilingual Cross-Lingual capabilities) into a unified architecture.
"""

from .models import (
    CitationItem,
    ConfidenceTier,
    DomainType,
    EvidenceItem,
    UnifiedRequest,
    UnifiedResponse,
)
from .adapters import (
    adapt_chatbot_response,
    adapt_medical_response,
    adapt_multilingual_response,
    adapt_multimodal_response,
    adapt_scientific_response,
)
from .router import CrossTaskRouter, RoutingDecision
from .orchestrator import CrossTaskOrchestrator
from .service import CrossTaskService

__all__ = [
    "CitationItem",
    "ConfidenceTier",
    "DomainType",
    "EvidenceItem",
    "UnifiedRequest",
    "UnifiedResponse",
    "RoutingDecision",
    "CrossTaskRouter",
    "CrossTaskOrchestrator",
    "CrossTaskService",
    "adapt_chatbot_response",
    "adapt_medical_response",
    "adapt_scientific_response",
    "adapt_multimodal_response",
    "adapt_multilingual_response",
]
