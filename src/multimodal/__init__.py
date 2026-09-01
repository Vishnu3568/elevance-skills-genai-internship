"""Multimodal AI Assistant Package (Phase 5).

Exposes core contract models, validations, invariants, image ingestion,
image preprocessing, vision model abstractions, and the central
orchestration pipeline for multimodal interactions.
"""

from .ingestion import (
    FORMAT_TO_MIME,
    ImageIngestionService,
    ingest_image,
)
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
from .preprocessing import (
    ImagePreprocessor,
    PreprocessedImage,
    preprocess_image,
)
from .vision import (
    DeterministicMockVisionProvider,
    GeminiVisionProvider,
    StructuredVisualOutput,
    VisionModelProvider,
    VisionService,
)

__all__ = [
    "AmbiguityDetails",
    "AmbiguityDetectorHook",
    "DeterministicMockVisionProvider",
    "FORMAT_TO_MIME",
    "GeminiVisionProvider",
    "GroundingValidatorHook",
    "ImageArtifact",
    "ImageIngestionService",
    "ImagePreprocessor",
    "ImageUnderstandingHook",
    "MAX_IMAGE_DIMENSION",
    "MAX_IMAGE_SIZE_BYTES",
    "MIN_IMAGE_DIMENSION",
    "ModalityType",
    "MultimodalOrchestrator",
    "MultimodalRequest",
    "MultimodalResponse",
    "PreprocessedImage",
    "ReasoningEngineHook",
    "SUPPORTED_MIME_TYPES",
    "StructuredVisualOutput",
    "VisionModelProvider",
    "VisionService",
    "VisualEvidenceItem",
    "ingest_image",
    "preprocess_image",
    "validate_ambiguity_details",
    "validate_image_artifact",
    "validate_multimodal_request",
    "validate_multimodal_response",
    "validate_visual_evidence_item",
]
