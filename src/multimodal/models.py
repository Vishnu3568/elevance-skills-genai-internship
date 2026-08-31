"""Data contracts, models, and validations for Multimodal AI Assistant (Phase 5).

Defines canonical representations for image artifacts, multimodal requests,
visual evidence items, ambiguity diagnoses, and multimodal responses.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union


# -----------------------------------------------------------------------------
# Constants & Contract Invariants
# -----------------------------------------------------------------------------

SUPPORTED_MIME_TYPES: Set[str] = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

MAX_IMAGE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
MIN_IMAGE_DIMENSION: int = 10                  # 10 pixels
MAX_IMAGE_DIMENSION: int = 4096                # 4096 pixels


# -----------------------------------------------------------------------------
# Modality Enumeration
# -----------------------------------------------------------------------------

class ModalityType(str, Enum):
    """Enumeration of supported interaction modalities."""

    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    TEXT_AND_IMAGE = "text_and_image"


# -----------------------------------------------------------------------------
# Image Artifact Model
# -----------------------------------------------------------------------------

@dataclass
class ImageArtifact:
    """Canonical data model representing a validated in-memory image artifact."""

    data: bytes
    mime_type: str
    width: int
    height: int
    format: str
    base64_str: Optional[str] = None
    file_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate artifact invariants immediately upon construction."""
        validate_image_artifact(self)

    def to_dict(self) -> Dict[str, Any]:
        """Return a clean dictionary representation (excluding raw bytes for telemetry/logging)."""
        d = asdict(self)
        d["data_size_bytes"] = len(self.data)
        d.pop("data", None)
        return d


def validate_image_artifact(artifact: ImageArtifact) -> None:
    """Validate that an ImageArtifact instance satisfies all contract invariants.

    Args:
        artifact (ImageArtifact): The image artifact to validate.

    Raises:
        TypeError: If fields have incorrect types.
        ValueError: If fields contain invalid values or violate bounds.
    """
    if not isinstance(artifact, ImageArtifact):
        raise TypeError(f"Expected ImageArtifact instance, got {type(artifact).__name__}")

    # 1. Validate data (bytes)
    if not isinstance(artifact.data, bytes):
        raise TypeError(f"data must be bytes, got {type(artifact.data).__name__}")
    if len(artifact.data) == 0:
        raise ValueError("Image data bytes cannot be empty.")
    if len(artifact.data) > MAX_IMAGE_SIZE_BYTES:
        raise ValueError(
            f"Image size ({len(artifact.data)} bytes) exceeds maximum permitted limit of "
            f"{MAX_IMAGE_SIZE_BYTES} bytes (10 MB)."
        )

    # 2. Validate mime_type
    if not isinstance(artifact.mime_type, str):
        raise TypeError(f"mime_type must be a string, got {type(artifact.mime_type).__name__}")
    normalized_mime = artifact.mime_type.strip().lower()
    if normalized_mime not in SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"Unsupported MIME type '{artifact.mime_type}'. Supported types: "
            f"{sorted(list(SUPPORTED_MIME_TYPES))}"
        )

    # 3. Validate dimensions
    if not isinstance(artifact.width, int) or isinstance(artifact.width, bool):
        raise TypeError(f"width must be an integer, got {type(artifact.width).__name__}")
    if not isinstance(artifact.height, int) or isinstance(artifact.height, bool):
        raise TypeError(f"height must be an integer, got {type(artifact.height).__name__}")

    if artifact.width < MIN_IMAGE_DIMENSION or artifact.width > MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Image width ({artifact.width}px) out of bounds [{MIN_IMAGE_DIMENSION}, {MAX_IMAGE_DIMENSION}]."
        )
    if artifact.height < MIN_IMAGE_DIMENSION or artifact.height > MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Image height ({artifact.height}px) out of bounds [{MIN_IMAGE_DIMENSION}, {MAX_IMAGE_DIMENSION}]."
        )

    # 4. Validate format
    if not isinstance(artifact.format, str):
        raise TypeError(f"format must be a string, got {type(artifact.format).__name__}")
    if not artifact.format.strip():
        raise ValueError("format cannot be empty or whitespace-only.")

    # 5. Optional fields type checks
    if artifact.base64_str is not None and not isinstance(artifact.base64_str, str):
        raise TypeError(f"base64_str must be a string or None, got {type(artifact.base64_str).__name__}")
    if artifact.file_name is not None and not isinstance(artifact.file_name, str):
        raise TypeError(f"file_name must be a string or None, got {type(artifact.file_name).__name__}")


# -----------------------------------------------------------------------------
# Multimodal Request Model
# -----------------------------------------------------------------------------

@dataclass
class MultimodalRequest:
    """Canonical data model representing an incoming multimodal user request."""

    query: Optional[str] = None
    images: List[ImageArtifact] = field(default_factory=list)
    session_id: str = "default_multimodal_session"
    temperature: float = 0.1
    max_output_tokens: Optional[int] = None
    metadata_filters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate request invariants immediately upon construction."""
        validate_multimodal_request(self)

    @property
    def modality(self) -> ModalityType:
        """Deterministically infer modality type based on present fields."""
        has_query = bool(self.query and self.query.strip())
        has_images = bool(self.images and len(self.images) > 0)

        if has_query and has_images:
            return ModalityType.TEXT_AND_IMAGE
        elif has_images:
            return ModalityType.IMAGE_ONLY
        elif has_query:
            return ModalityType.TEXT_ONLY
        raise ValueError("Invalid request: both query and images are missing.")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize request metadata to dictionary."""
        return {
            "query": self.query,
            "image_count": len(self.images),
            "modality": self.modality.value,
            "session_id": self.session_id,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "metadata_filters": dict(self.metadata_filters),
        }


def validate_multimodal_request(request: MultimodalRequest) -> None:
    """Validate that a MultimodalRequest instance satisfies all contract invariants.

    Args:
        request (MultimodalRequest): The request instance to validate.

    Raises:
        TypeError: If fields have incorrect types.
        ValueError: If fields contain invalid values or empty inputs.
    """
    if not isinstance(request, MultimodalRequest):
        raise TypeError(f"Expected MultimodalRequest instance, got {type(request).__name__}")

    # 1. Validate query
    if request.query is not None and not isinstance(request.query, str):
        raise TypeError(f"query must be a string or None, got {type(request.query).__name__}")

    clean_query = request.query.strip() if isinstance(request.query, str) else ""

    # 2. Validate images
    if not isinstance(request.images, list):
        raise TypeError(f"images must be a list of ImageArtifact instances, got {type(request.images).__name__}")

    for idx, img in enumerate(request.images):
        if not isinstance(img, ImageArtifact):
            raise TypeError(f"Image at index {idx} must be an ImageArtifact instance, got {type(img).__name__}")

    # 3. Invariant: At least one of query or images must be non-empty
    if not clean_query and len(request.images) == 0:
        raise ValueError("MultimodalRequest must contain either a non-empty query or at least one image.")

    # 4. Validate session_id
    if not isinstance(request.session_id, str):
        raise TypeError(f"session_id must be a string, got {type(request.session_id).__name__}")
    if not request.session_id.strip():
        raise ValueError("session_id cannot be empty or whitespace-only.")

    # 5. Validate temperature
    if not isinstance(request.temperature, (int, float)) or isinstance(request.temperature, bool):
        raise TypeError(f"temperature must be a float, got {type(request.temperature).__name__}")
    if request.temperature < 0.0 or request.temperature > 2.0:
        raise ValueError(f"temperature ({request.temperature}) must be within [0.0, 2.0].")

    # 6. Validate max_output_tokens
    if request.max_output_tokens is not None:
        if not isinstance(request.max_output_tokens, int) or isinstance(request.max_output_tokens, bool):
            raise TypeError(f"max_output_tokens must be an int or None, got {type(request.max_output_tokens).__name__}")
        if request.max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be positive, got {request.max_output_tokens}.")

    # 7. Validate metadata_filters
    if not isinstance(request.metadata_filters, dict):
        raise TypeError(f"metadata_filters must be a dict, got {type(request.metadata_filters).__name__}")


# -----------------------------------------------------------------------------
# Visual Evidence Item Model
# -----------------------------------------------------------------------------

@dataclass
class VisualEvidenceItem:
    """Represents a discrete piece of evidence derived from visual analysis."""

    description: str
    region_label: Optional[str] = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        """Validate visual evidence invariants."""
        validate_visual_evidence_item(self)

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary representation."""
        return asdict(self)


def validate_visual_evidence_item(item: VisualEvidenceItem) -> None:
    """Validate visual evidence item schema."""
    if not isinstance(item, VisualEvidenceItem):
        raise TypeError(f"Expected VisualEvidenceItem instance, got {type(item).__name__}")

    if not isinstance(item.description, str):
        raise TypeError(f"description must be a string, got {type(item.description).__name__}")
    if not item.description.strip():
        raise ValueError("description cannot be empty or whitespace-only.")

    if item.region_label is not None and not isinstance(item.region_label, str):
        raise TypeError(f"region_label must be a string or None, got {type(item.region_label).__name__}")

    if not isinstance(item.confidence, (int, float)) or isinstance(item.confidence, bool):
        raise TypeError(f"confidence must be a float, got {type(item.confidence).__name__}")
    if item.confidence < 0.0 or item.confidence > 1.0:
        raise ValueError(f"confidence ({item.confidence}) must be in range [0.0, 1.0].")


# -----------------------------------------------------------------------------
# Ambiguity Details Model
# -----------------------------------------------------------------------------

@dataclass
class AmbiguityDetails:
    """Contains diagnostic details when a multimodal query or visual context is ambiguous."""

    is_ambiguous: bool = False
    clarification_question: Optional[str] = None
    missing_aspects: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate ambiguity details schema."""
        validate_ambiguity_details(self)

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary representation."""
        return asdict(self)


def validate_ambiguity_details(details: AmbiguityDetails) -> None:
    """Validate ambiguity details schema."""
    if not isinstance(details, AmbiguityDetails):
        raise TypeError(f"Expected AmbiguityDetails instance, got {type(details).__name__}")

    if not isinstance(details.is_ambiguous, bool):
        raise TypeError(f"is_ambiguous must be a boolean, got {type(details.is_ambiguous).__name__}")

    if details.clarification_question is not None:
        if not isinstance(details.clarification_question, str):
            raise TypeError(
                f"clarification_question must be a string or None, got {type(details.clarification_question).__name__}"
            )
        if details.is_ambiguous and not details.clarification_question.strip():
            raise ValueError("clarification_question cannot be empty when is_ambiguous=True.")

    if not isinstance(details.missing_aspects, list):
        raise TypeError(f"missing_aspects must be a list of strings, got {type(details.missing_aspects).__name__}")

    for idx, aspect in enumerate(details.missing_aspects):
        if not isinstance(aspect, str):
            raise TypeError(f"missing_aspects[{idx}] must be a string, got {type(aspect).__name__}")


# -----------------------------------------------------------------------------
# Multimodal Response Model
# -----------------------------------------------------------------------------

@dataclass
class MultimodalResponse:
    """Canonical data model representing the structured response from the multimodal service."""

    query: str
    answer: str
    modality: ModalityType
    ambiguity: AmbiguityDetails = field(default_factory=AmbiguityDetails)
    visual_evidence: List[VisualEvidenceItem] = field(default_factory=list)
    grounded: bool = True
    warning_message: Optional[str] = None
    image_metadata: Optional[Dict[str, Any]] = None
    formatted_markdown: str = ""
    session_id: str = "default_multimodal_session"

    def __post_init__(self) -> None:
        """Validate response invariants."""
        validate_multimodal_response(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize multimodal response to dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "modality": self.modality.value if isinstance(self.modality, ModalityType) else str(self.modality),
            "ambiguity": self.ambiguity.to_dict() if hasattr(self.ambiguity, "to_dict") else asdict(self.ambiguity),
            "visual_evidence": [ve.to_dict() for ve in self.visual_evidence],
            "grounded": self.grounded,
            "warning_message": self.warning_message,
            "image_metadata": dict(self.image_metadata) if self.image_metadata else None,
            "formatted_markdown": self.formatted_markdown,
            "session_id": self.session_id,
        }


def validate_multimodal_response(response: MultimodalResponse) -> None:
    """Validate that a MultimodalResponse instance satisfies all schema invariants."""
    if not isinstance(response, MultimodalResponse):
        raise TypeError(f"Expected MultimodalResponse instance, got {type(response).__name__}")

    if not isinstance(response.query, str):
        raise TypeError(f"query must be a string, got {type(response.query).__name__}")

    if not isinstance(response.answer, str):
        raise TypeError(f"answer must be a string, got {type(response.answer).__name__}")

    if not isinstance(response.modality, ModalityType):
        raise TypeError(f"modality must be a ModalityType enum, got {type(response.modality).__name__}")

    if not isinstance(response.ambiguity, AmbiguityDetails):
        raise TypeError(f"ambiguity must be an AmbiguityDetails instance, got {type(response.ambiguity).__name__}")

    if not isinstance(response.visual_evidence, list):
        raise TypeError(f"visual_evidence must be a list, got {type(response.visual_evidence).__name__}")

    for idx, item in enumerate(response.visual_evidence):
        if not isinstance(item, VisualEvidenceItem):
            raise TypeError(f"visual_evidence[{idx}] must be a VisualEvidenceItem instance, got {type(item).__name__}")

    if not isinstance(response.grounded, bool):
        raise TypeError(f"grounded must be a boolean, got {type(response.grounded).__name__}")

    if response.warning_message is not None and not isinstance(response.warning_message, str):
        raise TypeError(f"warning_message must be a string or None, got {type(response.warning_message).__name__}")

    if response.image_metadata is not None and not isinstance(response.image_metadata, dict):
        raise TypeError(f"image_metadata must be a dict or None, got {type(response.image_metadata).__name__}")

    if not isinstance(response.formatted_markdown, str):
        raise TypeError(f"formatted_markdown must be a string, got {type(response.formatted_markdown).__name__}")

    if not isinstance(response.session_id, str) or not response.session_id.strip():
        raise ValueError("session_id must be a non-empty string.")
