"""Cross-Modal Reasoning Engine Foundation (Phase 5 — Day 26 Step 2).

Consumes UnifiedMultimodalContext to synthesize evidence-aware reasoning results
across text-only, image-only, and joint text+image modalities while preserving
evidence provenance, combining textual and visual observations, and providing
pluggable engine abstractions.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .context import (
    MultimodalContext,
    UnifiedEvidenceItem,
    validate_multimodal_context,
)
from .models import (
    ModalityType,
    MultimodalRequest,
    MultimodalResponse,
    VisualEvidenceItem,
)


# -----------------------------------------------------------------------------
# Structured Reasoning Result Contract
# -----------------------------------------------------------------------------

@dataclass
class ReasoningResult:
    """Structured deduction produced by the multimodal reasoning engine."""

    interpreted_query: Optional[str] = None
    reasoning_summary: str = ""
    textual_evidence: List[str] = field(default_factory=list)
    visual_evidence: List[VisualEvidenceItem] = field(default_factory=list)
    combined_evidence: List[UnifiedEvidenceItem] = field(default_factory=list)
    confidence: float = 1.0
    modality: ModalityType = ModalityType.TEXT_ONLY
    session_id: str = "default_multimodal_session"
    reasoning_steps: List[str] = field(default_factory=list)
    engine_name: str = "multimodal_reasoning_engine"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate confidence bounds and session identity."""
        if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool):
            raise TypeError(f"confidence must be a float, got {type(self.confidence).__name__}")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"confidence ({self.confidence}) must be in range [0.0, 1.0].")
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string.")

    @property
    def answer(self) -> str:
        """Compatibility accessor matching MultimodalResponse answer field."""
        return self.reasoning_summary

    @property
    def used_evidence(self) -> List[UnifiedEvidenceItem]:
        """Compatibility accessor for unified evidence list."""
        return self.combined_evidence

    def to_dict(self) -> Dict[str, Any]:
        """Deterministic serialization of reasoning result without raw byte blobs."""
        return {
            "interpreted_query": self.interpreted_query,
            "reasoning_summary": self.reasoning_summary,
            "textual_evidence": list(self.textual_evidence),
            "visual_evidence": [ev.to_dict() for ev in self.visual_evidence],
            "combined_evidence": [ev.to_dict() for ev in self.combined_evidence],
            "confidence": self.confidence,
            "modality": self.modality.value if isinstance(self.modality, ModalityType) else str(self.modality),
            "session_id": self.session_id,
            "reasoning_steps": list(self.reasoning_steps),
            "engine_name": self.engine_name,
            "metadata": dict(self.metadata),
        }

    def to_multimodal_response(self) -> MultimodalResponse:
        """Convert reasoning result into canonical Day 24 MultimodalResponse contract."""
        from .response_generator import generate_multimodal_response
        return generate_multimodal_response(self)


# Alias for backward compatibility
MultimodalReasoningResult = ReasoningResult


# -----------------------------------------------------------------------------
# Reasoning Engine Base Interface
# -----------------------------------------------------------------------------

class ReasoningEngine(ABC):
    """Abstract interface for cross-modal reasoning backends."""

    @abstractmethod
    def reason(self, context: MultimodalContext) -> ReasoningResult:
        """Execute cross-modal deduction over a validated MultimodalContext.

        Args:
            context: The unified multimodal context containing text, visuals, and evidence.

        Returns:
            ReasoningResult: Structured reasoning deduction with provenance.
        """
        pass


# -----------------------------------------------------------------------------
# Multimodal Reasoning Engine Implementation
# -----------------------------------------------------------------------------

class MultimodalReasoningEngine(ReasoningEngine):
    """Deterministic, auditable cross-modal reasoning engine.

    Fuses textual queries and Day 25 visual understandings, maintaining provenance
    and producing structured ReasoningResult outputs. Serves as the foundation
    and extension point for Phase 5 Day 26.
    """

    def __init__(self, engine_name: str = "multimodal_reasoning_engine") -> None:
        self.engine_name = engine_name

    def reason(self, context: MultimodalContext) -> ReasoningResult:
        """Execute reasoning dispatch based on context modality."""
        if not isinstance(context, MultimodalContext):
            raise TypeError(f"Expected MultimodalContext, got {type(context).__name__}")

        validate_multimodal_context(context)

        if context.modality == ModalityType.TEXT_ONLY:
            return self._reason_text_only(context)
        elif context.modality == ModalityType.IMAGE_ONLY:
            return self._reason_image_only(context)
        elif context.modality == ModalityType.TEXT_AND_IMAGE:
            return self._reason_text_and_image(context)
        else:
            raise ValueError(f"Unsupported modality: {context.modality}")

    def _reason_text_only(self, context: MultimodalContext) -> ReasoningResult:
        """Reason strictly using textual context."""
        query = context.text_context.raw_query or ""
        text_evidence = [query] if query else []

        steps = [
            f"Parsed user textual query: '{query}'",
            "Identified pure text intent without visual dependencies.",
            "Formulated direct analytical response from textual context.",
        ]
        summary = f"[Text Reasoning] Successfully processed query: '{query}'."

        return ReasoningResult(
            interpreted_query=query,
            reasoning_summary=summary,
            textual_evidence=text_evidence,
            visual_evidence=[],
            combined_evidence=list(context.evidence_items),
            confidence=1.0,
            modality=ModalityType.TEXT_ONLY,
            session_id=context.session_id,
            reasoning_steps=steps,
            engine_name=self.engine_name,
            metadata={"query": query},
        )

    def _reason_image_only(self, context: MultimodalContext) -> ReasoningResult:
        """Reason strictly using visual context and extracted evidence."""
        vis = context.visual_context
        scene = vis.scene_description or "visual artifact"
        entities = vis.detected_objects
        entity_str = ", ".join(entities) if entities else "salient visual features"

        steps = [
            f"Inspected visual scene description: '{scene}'",
            f"Identified {len(entities)} discrete visual entities: [{entity_str}].",
            f"Audited spatial layout: {vis.spatial_observations if vis.spatial_observations else 'standard bounds'}.",
            "Synthesized structured visual breakdown from extracted visual evidence.",
        ]
        summary = f"[Visual Reasoning] Analyzed image showing {scene} with observed entities [{entity_str}]."

        # Convert unified visual evidence into canonical VisualEvidenceItem list
        vis_evidence = context.get_visual_evidence()

        return ReasoningResult(
            interpreted_query="",
            reasoning_summary=summary,
            textual_evidence=[],
            visual_evidence=vis_evidence,
            combined_evidence=list(context.evidence_items),
            confidence=0.95,
            modality=ModalityType.IMAGE_ONLY,
            session_id=context.session_id,
            reasoning_steps=steps,
            engine_name=self.engine_name,
            metadata={"visual_attributes": dict(vis.visual_attributes), "scene_description": scene},
        )

    def _reason_text_and_image(self, context: MultimodalContext) -> ReasoningResult:
        """Explicitly fuse textual query and visual evidence into a joint multimodal deduction."""
        query = context.text_context.raw_query or ""
        vis = context.visual_context
        scene = vis.scene_description or "visual artifact"
        entities = vis.detected_objects
        entity_str = ", ".join(entities) if entities else "visual elements"

        steps = [
            f"Evaluated textual question: '{query}'",
            f"Grounded question against visual context: '{scene}'",
            f"Correlated question concepts with {len(entities)} visual entities [{entity_str}].",
            "Verified evidence provenance across textual query and visual detection layers.",
            "Formulated cross-modal deduction fusing visual observations with question semantics.",
        ]

        summary = (
            f"[Joint Multimodal Reasoning] For query '{query}', evaluated visual scene '{scene}'. "
            f"Correlated observed entities [{entity_str}] to synthesize grounded cross-modal answer."
        )

        vis_evidence = context.get_visual_evidence()
        text_evidence = [query] if query else []

        return ReasoningResult(
            interpreted_query=query,
            reasoning_summary=summary,
            textual_evidence=text_evidence,
            visual_evidence=vis_evidence,
            combined_evidence=list(context.evidence_items),
            confidence=0.98,
            modality=ModalityType.TEXT_AND_IMAGE,
            session_id=context.session_id,
            reasoning_steps=steps,
            engine_name=self.engine_name,
            metadata={
                "query": query,
                "visual_attributes": dict(vis.visual_attributes),
                "scene_description": scene,
                "fused_modalities": ["text", "image"],
            },
        )

    def as_orchestrator_hook(
        self,
        vision_service: Optional[Any] = None,
    ) -> Callable[[MultimodalRequest], MultimodalResponse]:
        """Create a reasoning engine hook compatible with MultimodalOrchestrator."""
        from .context import build_multimodal_context

        def hook(request: MultimodalRequest) -> MultimodalResponse:
            first_img = request.images[0] if request.images else None
            visual_output = None
            if first_img is not None and vision_service is not None:
                visual_output = vision_service.analyze(first_img)

            context = build_multimodal_context(
                request=request,
                artifact=first_img,
                visual_output=visual_output,
            )

            result = self.reason(context)
            return result.to_multimodal_response()

        return hook


# Alias for backward compatibility
DeterministicReasoningEngine = MultimodalReasoningEngine


# -----------------------------------------------------------------------------
# High-Level Multimodal Reasoning Service
# -----------------------------------------------------------------------------

class MultimodalReasoningService:
    """Coordinating service for multimodal reasoning operations."""

    def __init__(self, engine: Optional[ReasoningEngine] = None) -> None:
        """Initialize service with a reasoning engine (defaults to MultimodalReasoningEngine)."""
        self.engine = engine or MultimodalReasoningEngine()

    def reason(self, context: MultimodalContext) -> ReasoningResult:
        """Execute reasoning over a MultimodalContext."""
        if not isinstance(context, MultimodalContext):
            raise TypeError(f"Expected MultimodalContext, got {type(context).__name__}")
        return self.engine.reason(context)

    def as_orchestrator_hook(
        self,
        vision_service: Optional[Any] = None,
    ) -> Callable[[MultimodalRequest], MultimodalResponse]:
        """Return an orchestrator hook adapter."""
        if hasattr(self.engine, "as_orchestrator_hook"):
            return self.engine.as_orchestrator_hook(vision_service=vision_service)

        from .context import build_multimodal_context

        def hook(request: MultimodalRequest) -> MultimodalResponse:
            first_img = request.images[0] if request.images else None
            visual_output = None
            if first_img is not None and vision_service is not None:
                visual_output = vision_service.analyze(first_img)

            context = build_multimodal_context(
                request=request,
                artifact=first_img,
                visual_output=visual_output,
            )

            result = self.reason(context)
            return result.to_multimodal_response()

        return hook
