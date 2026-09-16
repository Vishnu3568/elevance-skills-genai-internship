"""Data Models and Contracts for Cross-Task Integration (Day 36).

Defines the canonical UnifiedRequest, UnifiedResponse, DomainType, and supporting
evidence and citation data structures for cross-task coordination.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from src.multimodal.models import ImageArtifact
except ImportError:
    from multimodal.models import ImageArtifact  # type: ignore


class DomainType(str, Enum):
    """Supported specialist business domains in the ElevanceSkills platform."""
    CUSTOMER_SUPPORT = "customer_support"
    MEDICAL = "medical"
    SCIENTIFIC = "scientific"
    MULTIMODAL = "multimodal"
    GENERAL_CHITCHAT = "general_chitchat"


class ConfidenceTier(str, Enum):
    """Standardized deterministic confidence rating tiers."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass
class EvidenceItem:
    """Represents a discrete verified piece of evidence (text, visual, or clinical)."""
    description: str
    source: str = "text"
    confidence: float = 1.0
    region_label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evidence item to dictionary."""
        return asdict(self)


@dataclass
class CitationItem:
    """Represents a formal bibliographic, medical, or corpus citation."""
    title: str
    source_id: str
    url: Optional[str] = None
    authors: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize citation to dictionary."""
        return asdict(self)


@dataclass
class UnifiedRequest:
    """Canonical incoming cross-task request contract."""
    query: str
    image: Optional[ImageArtifact] = None
    session_id: str = "default_session"
    language_hint: Optional[str] = None
    domain_override: Optional[str] = None
    temperature: float = 0.1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate request invariants."""
        if not isinstance(self.query, str):
            raise TypeError(f"query must be a string, got {type(self.query).__name__}")
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        if self.domain_override is not None:
            if not isinstance(self.domain_override, str):
                raise TypeError("domain_override must be a string or None.")
            valid_domains = [d.value for d in DomainType]
            if self.domain_override.lower() not in valid_domains:
                raise ValueError(
                    f"Invalid domain_override '{self.domain_override}'. Supported: {valid_domains}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize request to dictionary."""
        return {
            "query": self.query,
            "has_image": self.image is not None,
            "session_id": self.session_id,
            "language_hint": self.language_hint,
            "domain_override": self.domain_override,
            "temperature": self.temperature,
            "metadata": dict(self.metadata),
        }


@dataclass
class UnifiedResponse:
    """Standardized cross-task response contract across all specialist tasks."""
    query: str
    final_text_response: str
    domain: str
    detected_language: str = "en"
    detected_language_name: str = "English"
    confidence_score: float = 1.0
    confidence_tier: str = "HIGH"
    evidence: List[EvidenceItem] = field(default_factory=list)
    citations: List[CitationItem] = field(default_factory=list)
    is_grounded: bool = True
    sentiment: Optional[str] = None
    warning_message: Optional[str] = None
    execution_time_ms: float = 0.0
    session_id: str = "default_session"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate response invariants."""
        if not isinstance(self.query, str):
            raise TypeError("query must be a string.")
        if not isinstance(self.final_text_response, str):
            raise TypeError("final_text_response must be a string.")
        if not isinstance(self.domain, str):
            raise TypeError("domain must be a string.")
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(f"confidence_score ({self.confidence_score}) must be in [0.0, 1.0].")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize response to dictionary."""
        return {
            "query": self.query,
            "final_text_response": self.final_text_response,
            "domain": self.domain,
            "detected_language": self.detected_language,
            "detected_language_name": self.detected_language_name,
            "confidence_score": round(self.confidence_score, 4),
            "confidence_tier": self.confidence_tier,
            "evidence": [e.to_dict() for e in self.evidence],
            "citations": [c.to_dict() for c in self.citations],
            "is_grounded": self.is_grounded,
            "sentiment": self.sentiment,
            "warning_message": self.warning_message,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "session_id": self.session_id,
            "metadata": dict(self.metadata),
        }
