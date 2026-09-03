"""Multimodal AI Assistant Package (Phase 5).

Exposes core contract models, validations, invariants, image ingestion,
image preprocessing, vision model abstractions, unified multimodal context representation,
cross-modal reasoning engines, evidence-aware response generation, and the central orchestration pipeline.
"""

from .context import (
    EvidenceProvenance,
    MultimodalContext,
    TextualContext,
    UnifiedEvidenceItem,
    VisualContext,
    build_multimodal_context,
    validate_multimodal_context,
)
from .context_retention import (
    MultimodalContextRetriever,
    RetainedConversationContext,
    RetainedTurnView,
    retrieve_conversation_context,
)
from .conversation import (
    ConversationTurn,
    MultimodalContextSummary,
    MultimodalConversationSession,
    MultimodalSessionManager,
    validate_conversation_turn,
)
from .followup import (
    FollowUpResolution,
    MultimodalFollowUpResolver,
    resolve_multimodal_followup,
)
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
    create_multimodal_pipeline,
)
from .preprocessing import (
    ImagePreprocessor,
    PreprocessedImage,
    preprocess_image,
)
from .reasoning import (
    DeterministicReasoningEngine,
    MultimodalReasoningEngine,
    MultimodalReasoningResult,
    MultimodalReasoningService,
    ReasoningEngine,
    ReasoningResult,
)
from .response_generator import (
    MultimodalResponseGenerator,
    generate_multimodal_response,
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
    "ConversationTurn",
    "DeterministicMockVisionProvider",
    "DeterministicReasoningEngine",
    "EvidenceProvenance",
    "FORMAT_TO_MIME",
    "FollowUpResolution",
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
    "MultimodalContext",
    "MultimodalContextRetriever",
    "MultimodalContextSummary",
    "MultimodalConversationSession",
    "MultimodalFollowUpResolver",
    "MultimodalOrchestrator",
    "MultimodalReasoningEngine",
    "MultimodalReasoningResult",
    "MultimodalReasoningService",
    "MultimodalRequest",
    "MultimodalResponse",
    "MultimodalResponseGenerator",
    "MultimodalSessionManager",
    "PreprocessedImage",
    "ReasoningEngine",
    "ReasoningEngineHook",
    "ReasoningResult",
    "RetainedConversationContext",
    "RetainedTurnView",
    "SUPPORTED_MIME_TYPES",
    "StructuredVisualOutput",
    "TextualContext",
    "UnifiedEvidenceItem",
    "VisionModelProvider",
    "VisionService",
    "VisualContext",
    "VisualEvidenceItem",
    "build_multimodal_context",
    "create_multimodal_pipeline",
    "generate_multimodal_response",
    "ingest_image",
    "preprocess_image",
    "resolve_multimodal_followup",
    "retrieve_conversation_context",
    "validate_ambiguity_details",
    "validate_conversation_turn",
    "validate_image_artifact",
    "validate_multimodal_context",
    "validate_multimodal_request",
    "validate_multimodal_response",
    "validate_visual_evidence_item",
]
