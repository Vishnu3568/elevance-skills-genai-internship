"""Unified Multimodal Context Representation (Phase 5 — Day 26 Step 1).

Provides a typed, validated internal context container fusing text queries,
Day 25 visual understanding outputs, metadata, and evidence provenance
into an unified structure ready for downstream cross-modal reasoning.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from .models import (
    ImageArtifact,
    ModalityType,
    MultimodalRequest,
    VisualEvidenceItem,
)
from .preprocessing import PreprocessedImage
from .vision import StructuredVisualOutput


# -----------------------------------------------------------------------------
# Evidence Provenance Model
# -----------------------------------------------------------------------------

@dataclass
class EvidenceProvenance:
    """Tracks origin, modality, provider, and source identity of evidence."""

    source_type: str  # e.g. "user_query", "visual_inspection", "ocr_text", "object_detection"
    modality: ModalityType
    source_id: Optional[str] = None  # e.g. file_name, session_id, or artifact hash
    provider: Optional[str] = None   # e.g. "user", "deterministic_mock", "gemini-2.5-flash"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize provenance to dictionary."""
        return {
            "source_type": self.source_type,
            "modality": self.modality.value if isinstance(self.modality, ModalityType) else str(self.modality),
            "source_id": self.source_id,
            "provider": self.provider,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# Unified Evidence Item Model
# -----------------------------------------------------------------------------

@dataclass
class UnifiedEvidenceItem:
    """Discrete evidence item binding observations with explicit provenance."""

    description: str
    modality: ModalityType
    region_label: Optional[str] = None
    confidence: float = 1.0
    source_type: str = "visual"
    provenance: Optional[EvidenceProvenance] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate evidence invariants."""
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string.")
        if not isinstance(self.modality, ModalityType):
            raise TypeError(f"modality must be a ModalityType, got {type(self.modality).__name__}")
        if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool):
            raise TypeError(f"confidence must be a float, got {type(self.confidence).__name__}")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"confidence ({self.confidence}) must be in range [0.0, 1.0].")

    def to_visual_evidence_item(self) -> VisualEvidenceItem:
        """Convert to canonical Day 24 VisualEvidenceItem."""
        return VisualEvidenceItem(
            description=self.description,
            region_label=self.region_label,
            confidence=self.confidence,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evidence item with provenance to dictionary."""
        return {
            "description": self.description,
            "modality": self.modality.value if isinstance(self.modality, ModalityType) else str(self.modality),
            "region_label": self.region_label,
            "confidence": self.confidence,
            "source_type": self.source_type,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "metadata": dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# Sub-Context Containers
# -----------------------------------------------------------------------------

@dataclass
class TextualContext:
    """Encapsulates textual query information and entity annotations."""

    raw_query: Optional[str] = None
    normalized_query: Optional[str] = None
    extracted_entities: List[str] = field(default_factory=list)
    provenance: Optional[EvidenceProvenance] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_text(self) -> bool:
        """Check if non-empty query text is present."""
        return bool(self.raw_query and self.raw_query.strip())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize textual context to dictionary."""
        return {
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
            "extracted_entities": list(self.extracted_entities),
            "has_text": self.has_text,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "metadata": dict(self.metadata),
        }


@dataclass
class VisualContext:
    """Encapsulates Day 25 visual understanding results and image properties."""

    artifact: Optional[ImageArtifact] = None
    preprocessed: Optional[PreprocessedImage] = None
    visual_output: Optional[StructuredVisualOutput] = None
    scene_description: Optional[str] = None
    detected_objects: List[str] = field(default_factory=list)
    visible_text: List[str] = field(default_factory=list)
    spatial_observations: List[str] = field(default_factory=list)
    visual_attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: Optional[EvidenceProvenance] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_visuals(self) -> bool:
        """Check if visual data or findings are present."""
        return bool(
            self.artifact is not None
            or self.preprocessed is not None
            or self.visual_output is not None
            or (self.scene_description and self.scene_description.strip())
            or len(self.detected_objects) > 0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize visual context to dictionary (excluding raw byte arrays)."""
        return {
            "has_visuals": self.has_visuals,
            "scene_description": self.scene_description,
            "detected_objects": list(self.detected_objects),
            "visible_text": list(self.visible_text),
            "spatial_observations": list(self.spatial_observations),
            "visual_attributes": dict(self.visual_attributes),
            "artifact_summary": self.artifact.to_dict() if self.artifact else None,
            "preprocessed_summary": self.preprocessed.to_dict() if self.preprocessed else None,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "metadata": dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# Unified Multimodal Context Contract
# -----------------------------------------------------------------------------

@dataclass
class MultimodalContext:
    """Unified internal representation fusing text, visuals, metadata, and evidence.

    Serves as the central input container for the Day 26 cross-modal reasoning engine.
    """

    session_id: str
    modality: ModalityType
    text_context: TextualContext = field(default_factory=TextualContext)
    visual_context: VisualContext = field(default_factory=VisualContext)
    evidence_items: List[UnifiedEvidenceItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        """Validate unified context invariants and modality consistency."""
        validate_multimodal_context(self)

    def add_evidence(self, item: UnifiedEvidenceItem) -> None:
        """Add a unified evidence item."""
        if not isinstance(item, UnifiedEvidenceItem):
            raise TypeError(f"Expected UnifiedEvidenceItem, got {type(item).__name__}")
        self.evidence_items.append(item)

    def get_visual_evidence(self) -> List[VisualEvidenceItem]:
        """Return all visual evidence items in canonical Day 24 format."""
        items: List[VisualEvidenceItem] = []
        for ev in self.evidence_items:
            if ev.modality in (ModalityType.IMAGE_ONLY, ModalityType.TEXT_AND_IMAGE):
                items.append(ev.to_visual_evidence_item())
        return items

    def get_summary_description(self) -> str:
        """Generate a concise textual overview of the fused multimodal state."""
        parts: List[str] = [f"Session: {self.session_id}", f"Modality: {self.modality.value}"]
        if self.text_context.has_text:
            parts.append(f"Query: '{self.text_context.raw_query}'")
        if self.visual_context.has_visuals:
            desc = self.visual_context.scene_description or "Visual features available"
            parts.append(f"Visual: {desc}")
        parts.append(f"Evidence Count: {len(self.evidence_items)}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Deterministic serialization of multimodal context without raw binary blobs."""
        return {
            "session_id": self.session_id,
            "modality": self.modality.value if isinstance(self.modality, ModalityType) else str(self.modality),
            "created_at": self.created_at,
            "text_context": self.text_context.to_dict(),
            "visual_context": self.visual_context.to_dict(),
            "evidence_items": [item.to_dict() for item in self.evidence_items],
            "metadata": dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# Validation Function
# -----------------------------------------------------------------------------

def validate_multimodal_context(context: MultimodalContext) -> None:
    """Validate that a MultimodalContext adheres to all contract rules.

    Args:
        context: The MultimodalContext instance to validate.

    Raises:
        TypeError: If fields have improper types.
        ValueError: If fields violate modality consistency or contain empty inputs.
    """
    if not isinstance(context, MultimodalContext):
        raise TypeError(f"Expected MultimodalContext instance, got {type(context).__name__}")

    # 1. Validate session_id
    if not isinstance(context.session_id, str):
        raise TypeError(f"session_id must be a string, got {type(context.session_id).__name__}")
    if not context.session_id.strip():
        raise ValueError("session_id cannot be empty or whitespace-only.")

    # 2. Validate modality
    if not isinstance(context.modality, ModalityType):
        raise TypeError(f"modality must be a ModalityType, got {type(context.modality).__name__}")

    # 3. Invariant: Context cannot be entirely empty
    has_text = context.text_context.has_text
    has_visuals = context.visual_context.has_visuals
    has_evidence = len(context.evidence_items) > 0

    if not has_text and not has_visuals and not has_evidence:
        raise ValueError(
            "MultimodalContext cannot be empty: must contain at least text query, visual data, or evidence."
        )

    # 4. Modality consistency checks
    if context.modality == ModalityType.TEXT_ONLY:
        if not has_text:
            raise ValueError("Modality is TEXT_ONLY, but text_context has no query text.")
        if has_visuals:
            raise ValueError("Modality is TEXT_ONLY, but visual_context contains visual data.")

    elif context.modality == ModalityType.IMAGE_ONLY:
        if not has_visuals:
            raise ValueError("Modality is IMAGE_ONLY, but visual_context has no visual data.")
        if has_text:
            raise ValueError("Modality is IMAGE_ONLY, but text_context contains query text.")

    elif context.modality == ModalityType.TEXT_AND_IMAGE:
        if not has_text:
            raise ValueError("Modality is TEXT_AND_IMAGE, but text_context has no query text.")
        if not has_visuals:
            raise ValueError("Modality is TEXT_AND_IMAGE, but visual_context has no visual data.")


# -----------------------------------------------------------------------------
# Factory / Builder
# -----------------------------------------------------------------------------

def build_multimodal_context(
    query: Optional[str] = None,
    artifact: Optional[ImageArtifact] = None,
    preprocessed: Optional[PreprocessedImage] = None,
    visual_output: Optional[StructuredVisualOutput] = None,
    request: Optional[MultimodalRequest] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MultimodalContext:
    """Build a validated MultimodalContext fusing inputs and Day 25 visual understandings.

    Args:
        query: User text query.
        artifact: Raw ImageArtifact.
        preprocessed: Normalized PreprocessedImage.
        visual_output: Structured visual findings from Day 25 vision layer.
        request: Optional MultimodalRequest (extracts query, session_id, and images if provided).
        session_id: Identifier for conversational session.
        metadata: Additional session or execution metadata.

    Returns:
        MultimodalContext: Validated, unified context container.
    """
    # 1. Resolve query and session_id
    resolved_query = query
    resolved_session_id = session_id or "default_multimodal_session"
    resolved_artifact = artifact
    resolved_preprocessed = preprocessed
    resolved_visual_output = visual_output
    resolved_metadata = dict(metadata or {})

    if request is not None:
        if resolved_query is None:
            resolved_query = request.query
        if session_id is None:
            resolved_session_id = request.session_id
        if resolved_artifact is None and len(request.images) > 0:
            resolved_artifact = request.images[0]
        if request.metadata_filters:
            resolved_metadata.setdefault("filters", dict(request.metadata_filters))

    clean_query = resolved_query.strip() if isinstance(resolved_query, str) and resolved_query.strip() else None

    # 2. Build TextualContext
    text_prov: Optional[EvidenceProvenance] = None
    if clean_query:
        text_prov = EvidenceProvenance(
            source_type="user_query",
            modality=ModalityType.TEXT_ONLY,
            source_id=resolved_session_id,
            provider="user",
            confidence=1.0,
        )

    text_ctx = TextualContext(
        raw_query=clean_query,
        normalized_query=clean_query.lower() if clean_query else None,
        provenance=text_prov,
    )

    # 3. Build VisualContext
    vis_prov: Optional[EvidenceProvenance] = None
    scene_desc: Optional[str] = None
    detected_objs: List[str] = []
    vis_text: List[str] = []
    spatial_obs: List[str] = []
    vis_attrs: Dict[str, Any] = {}

    if resolved_visual_output is not None:
        scene_desc = resolved_visual_output.scene_description
        detected_objs = list(resolved_visual_output.detected_objects)
        vis_text = list(resolved_visual_output.visible_text)
        spatial_obs = list(resolved_visual_output.spatial_observations)
        vis_attrs = dict(resolved_visual_output.visual_attributes)
        vis_prov = EvidenceProvenance(
            source_type="visual_inspection",
            modality=ModalityType.IMAGE_ONLY,
            source_id=resolved_artifact.file_name if resolved_artifact else None,
            provider=resolved_visual_output.provider_name,
            confidence=resolved_visual_output.confidence,
            metadata=dict(resolved_visual_output.metadata),
        )
    elif resolved_artifact is not None:
        scene_desc = f"Image artifact {resolved_artifact.format} ({resolved_artifact.width}x{resolved_artifact.height}px)"
        vis_attrs = {
            "format": resolved_artifact.format,
            "width": resolved_artifact.width,
            "height": resolved_artifact.height,
            "mime_type": resolved_artifact.mime_type,
        }
        vis_prov = EvidenceProvenance(
            source_type="visual_artifact",
            modality=ModalityType.IMAGE_ONLY,
            source_id=resolved_artifact.file_name,
            provider="ingestion",
            confidence=1.0,
        )

    vis_ctx = VisualContext(
        artifact=resolved_artifact,
        preprocessed=resolved_preprocessed,
        visual_output=resolved_visual_output,
        scene_description=scene_desc,
        detected_objects=detected_objs,
        visible_text=vis_text,
        spatial_observations=spatial_obs,
        visual_attributes=vis_attrs,
        provenance=vis_prov,
    )

    # 4. Resolve Modality
    has_text = text_ctx.has_text
    has_vis = vis_ctx.has_visuals

    if has_text and has_vis:
        detected_modality = ModalityType.TEXT_AND_IMAGE
    elif has_vis:
        detected_modality = ModalityType.IMAGE_ONLY
    elif has_text:
        detected_modality = ModalityType.TEXT_ONLY
    else:
        raise ValueError("Cannot build MultimodalContext: both text and visual inputs are empty.")

    # 5. Populate Evidence Items with Provenance
    evidence: List[UnifiedEvidenceItem] = []
    if resolved_visual_output is not None:
        for item in resolved_visual_output.to_visual_evidence_items():
            evidence.append(
                UnifiedEvidenceItem(
                    description=item.description,
                    modality=detected_modality,
                    region_label=item.region_label,
                    confidence=item.confidence,
                    source_type="visual",
                    provenance=vis_prov,
                )
            )
    elif resolved_artifact is not None:
        evidence.append(
            UnifiedEvidenceItem(
                description=f"Observed {resolved_artifact.format} image of size {resolved_artifact.width}x{resolved_artifact.height}",
                modality=detected_modality,
                region_label="Global Canvas",
                confidence=1.0,
                source_type="visual",
                provenance=vis_prov,
            )
        )

    return MultimodalContext(
        session_id=resolved_session_id,
        modality=detected_modality,
        text_context=text_ctx,
        visual_context=vis_ctx,
        evidence_items=evidence,
        metadata=resolved_metadata,
    )
