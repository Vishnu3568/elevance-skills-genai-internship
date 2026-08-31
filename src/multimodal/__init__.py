"""Multimodal AI Assistant Package (Phase 5).

Exposes core contract models, validations, invariants, and the central
orchestration pipeline for multimodal interactions.
"""

from .models import (
    AmbiguityDetails,
    ImageArtifact,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_SIZE_BYTES,
    MIN_IMAGE_DIMENSION,
    ModalityType,
    MultimodalRequest,
    MultimodalResponse,
    SUPPORTED_MIME_TYPES,
    VisualEvidenceItem,
    validate_ambiguity_details,
    validate_image_artifact,
    validate_multimodal_request,
    validate_multimodal_response,
    validate_visual_evidence_item,
)
from .orchestrator import (
    AmbiguityDetectorHook,
    GroundingValidatorHook,
    ImageUnderstandingHook,
    MultimodalOrchestrator,
    ReasoningEngineHook,
)

__all__ = [
    "AmbiguityDetails",
    "AmbiguityDetectorHook",
    "GroundingValidatorHook",
    "ImageArtifact",
    "ImageUnderstandingHook",
    "MAX_IMAGE_DIMENSION",
    "MAX_IMAGE_SIZE_BYTES",
    "MIN_IMAGE_DIMENSION",
    "ModalityType",
    "MultimodalOrchestrator",
    "MultimodalRequest",
    "MultimodalResponse",
    "ReasoningEngineHook",
    "SUPPORTED_MIME_TYPES",
    "VisualEvidenceItem",
    "validate_ambiguity_details",
    "validate_image_artifact",
    "validate_multimodal_request",
    "validate_multimodal_response",
    "validate_visual_evidence_item",
]
