"""Multimodal Missing Information Assessment (Phase 5 — Day 28 Step 4).

Provides deterministic detection and structured representation of required but
unavailable information in multimodal requests, contexts, and evidence.
Explicitly identifies missing images, missing visual evidence, missing requested
properties/details, missing conversational history, and missing reference targets
without guessing, fabricating answers, or confusing missing info with ambiguity.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .ambiguity import AmbiguityAssessmentResult, AmbiguityType
from .confidence import ConfidenceAssessmentResult, ConfidenceStatus
from .context import MultimodalContext, UnifiedEvidenceItem
from .context_retention import RetainedConversationContext
from .conversation import ConversationTurn
from .evidence_validator import EvidenceStatus, EvidenceValidationResult
from .models import ImageArtifact, ModalityType, MultimodalRequest, VisualEvidenceItem


# -----------------------------------------------------------------------------
# Classification Enumerations
# -----------------------------------------------------------------------------

class MissingInformationSeverity(str, Enum):
    """Severity classification of missing information."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MissingInformationType(str, Enum):
    """Categorical classification of missing information type."""

    MISSING_IMAGE = "missing_image"
    MISSING_VISUAL_EVIDENCE = "missing_visual_evidence"
    MISSING_PROPERTY_OR_DETAIL = "missing_property_or_detail"
    MISSING_CONVERSATION_CONTEXT = "missing_conversation_context"
    MISSING_REFERENCE_TARGET = "missing_reference_target"
    MISSING_TEXTUAL_EVIDENCE = "missing_textual_evidence"
    INSUFFICIENT_CONTEXT = "insufficient_context"


# -----------------------------------------------------------------------------
# Regex Patterns for Missing Information Detection
# -----------------------------------------------------------------------------

# Explicit visual reference patterns requiring an image
IMAGE_REQUIRED_PATTERNS = [
    re.compile(r"\b(?:in|on|from|of|at)\s+(?:(?:the|this|that)\s+)?(?:attached\s+)?(?:image|picture|photo|screenshot|diagram|chart|flowchart|figure|schematic|drawing)\b", re.IGNORECASE),
    re.compile(r"\b(?:attached\s+)?(?:image|picture|photo|screenshot|diagram|chart|flowchart|figure|schematic)\b", re.IGNORECASE),
    re.compile(r"\b(?:what\s+does\s+(?:this|the)\s+(?:image|diagram|chart|picture|figure)\s+(?:show|depict|illustrate|represent|mean))\b", re.IGNORECASE),
    re.compile(r"\b(?:look\s+at\s+(?:this|the)\s+(?:image|diagram|photo|figure))\b", re.IGNORECASE),
    re.compile(r"\b(?:what\s+color\s+is|what\s+is\s+the\s+color\s+of)\b", re.IGNORECASE),
    re.compile(r"\b(?:read|transcribe|what\s+is\s+written\s+in)\s+(?:the|this)\s+(?:image|diagram|screenshot|photo)\b", re.IGNORECASE),
    re.compile(r"\b(?:where\s+in\s+the\s+image|where\s+in\s+the\s+diagram)\b", re.IGNORECASE),
]

# Conversational references requiring prior turns
CONVERSATIONAL_REQUIRED_PATTERNS = [
    re.compile(r"\b(?:the\s+previous\s+(?:answer|response|topic|question|point|message|conversation))\b", re.IGNORECASE),
    re.compile(r"\b(?:what\s+(?:did\s+you|did\s+we)\s+(?:say|mention|discuss|explain|decide)(?:\s+earlier|\s+before)?)\b", re.IGNORECASE),
    re.compile(r"\b(?:you\s+mentioned(?:\s+earlier|\s+before)?)\b", re.IGNORECASE),
    re.compile(r"\b(?:the\s+second\s+(?:option|alternative|choice|step)|the\s+first\s+(?:option|step))\b", re.IGNORECASE),
    re.compile(r"\b(?:why\s+did\s+that\s+fail|why\s+did\s+we\s+choose\s+that)\b", re.IGNORECASE),
]

# Fine-grained property/detail patterns that cannot be answered from high-level diagram boxes
PROPERTY_DETAIL_PATTERNS = [
    (re.compile(r"\b(?:source\s+code|python\s+code|implementation\s+code|code\s+implementation|internal\s+code|internal\s+implementation|function\s+implementation|code\s+inside)\b", re.IGNORECASE), "source_code", "implementation source code"),
    (re.compile(r"\b(?:exact\s+price|pricing\s+tiers?|cost\s+breakdown|subscription\s+cost)\b", re.IGNORECASE), "pricing", "pricing and cost breakdown"),
    (re.compile(r"\b(?:creation\s+date|timestamp|when\s+was\s+it\s+(?:built|created|deployed)|release\s+date)\b", re.IGNORECASE), "date_time", "exact creation date or timestamp"),
    (re.compile(r"\b(?:author\s+contact|phone\s+number|email\s+address)\b", re.IGNORECASE), "contact_info", "author contact information"),
    (re.compile(r"\b(?:internal\s+algorithm|underlying\s+mathematical\s+proof|algorithmic\s+complexity)\b", re.IGNORECASE), "internal_algorithm", "internal algorithm or mathematical proof"),
]

# Named target extraction patterns
NAMED_TARGET_AFTER_NOUN = re.compile(
    r"\b(?:component|module|block|box|layer|node|unit|service)\s+([A-Za-z0-9_\-]+)\b",
    re.IGNORECASE,
)
NAMED_TARGET_BEFORE_NOUN = re.compile(
    r"\b([A-Za-z0-9_\-]+)\s+(?:component|module|block|box|layer|node|unit|service)\b",
    re.IGNORECASE,
)

# Common modifier words, verbs, and descriptive adjectives that are not target names
EXCLUDED_TARGET_MODIFIERS = {
    "this", "that", "the", "a", "an", "each", "every", "which", "how", "what",
    "why", "any", "some", "our", "its", "their", "your", "my", "previous", "next",
    "first", "second", "third", "one", "two", "three", "main", "other", "another",
    "do", "does", "did", "is", "are", "was", "were", "have", "has", "had",
    "mean", "show", "work", "run", "act", "be", "look", "seem", "call", "use",
    "small", "large", "big", "little", "tiny", "upper", "lower", "top", "bottom",
    "left", "right", "center", "middle", "central", "outer", "inner",
    "red", "green", "blue", "yellow", "black", "white", "gray", "dark", "light",
}


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class MissingInformationItem:
    """Individual record of a required but unavailable piece of information."""

    item_type: MissingInformationType
    description: str
    why_required: str
    affected_reference: Optional[str] = None
    severity: MissingInformationSeverity = MissingInformationSeverity.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize missing info item to clean dictionary without raw bytes."""
        return {
            "item_type": self.item_type.value if isinstance(self.item_type, MissingInformationType) else str(self.item_type),
            "description": self.description,
            "why_required": self.why_required,
            "affected_reference": self.affected_reference,
            "severity": self.severity.value if isinstance(self.severity, MissingInformationSeverity) else str(self.severity),
            "metadata": _clean_metadata_dict(self.metadata),
        }


@dataclass
class MissingInformationAssessmentResult:
    """Complete, structured assessment of missing information."""

    has_missing_information: bool
    severity: MissingInformationSeverity
    missing_items: List[MissingInformationItem]
    blocks_answering: bool
    available_summary: Dict[str, Any]
    reasons: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete assessment result to JSON-safe dictionary."""
        return {
            "has_missing_information": self.has_missing_information,
            "severity": self.severity.value if isinstance(self.severity, MissingInformationSeverity) else str(self.severity),
            "missing_items": [item.to_dict() for item in self.missing_items],
            "blocks_answering": self.blocks_answering,
            "available_summary": dict(self.available_summary),
            "reasons": list(self.reasons),
            "metadata": _clean_metadata_dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# Missing Information Detector Implementation
# -----------------------------------------------------------------------------

class MultimodalMissingInformationDetector:
    """Deterministic assessor of missing information for multimodal requests."""

    def assess(
        self,
        query: Optional[str] = None,
        request: Optional[MultimodalRequest] = None,
        context: Optional[MultimodalContext] = None,
        images: Optional[List[ImageArtifact]] = None,
        visual_objects: Optional[List[str]] = None,
        visible_text: Optional[List[str]] = None,
        evidence_items: Optional[List[Any]] = None,
        conversation_context: Optional[Union[RetainedConversationContext, List[ConversationTurn]]] = None,
        evidence_validation: Optional[EvidenceValidationResult] = None,
        confidence_assessment: Optional[ConfidenceAssessmentResult] = None,
        ambiguity_assessment: Optional[AmbiguityAssessmentResult] = None,
    ) -> MissingInformationAssessmentResult:
        """Perform deterministic missing information assessment.

        Args:
            query: Optional query string.
            request: Optional MultimodalRequest.
            context: Optional MultimodalContext.
            images: Optional list of ImageArtifacts.
            visual_objects: Optional explicit list of detected visual objects.
            visible_text: Optional explicit list of OCR text lines.
            evidence_items: Optional list of evidence items.
            conversation_context: Optional retained history or list of turns.
            evidence_validation: Optional EvidenceValidationResult from Step 1.
            confidence_assessment: Optional ConfidenceAssessmentResult from Step 2.
            ambiguity_assessment: Optional AmbiguityAssessmentResult from Step 3.

        Returns:
            MissingInformationAssessmentResult detailing all missing requirements.
        """
        # 1. Resolve normalized query text
        clean_query = ""
        if query and isinstance(query, str) and query.strip():
            clean_query = query.strip()
        elif request and request.query and request.query.strip():
            clean_query = request.query.strip()
        elif context and context.textual and context.textual.raw_query:
            clean_query = context.textual.raw_query.strip()

        # 2. Resolve image availability
        has_image = False
        image_count = 0
        if images and len(images) > 0:
            has_image = True
            image_count = len(images)
        elif request and request.images and len(request.images) > 0:
            has_image = True
            image_count = len(request.images)
        elif context and context.visual and context.visual.has_image:
            has_image = True
            image_count = 1

        # 3. Resolve visual elements (objects, text, scene description)
        v_objects: List[str] = []
        if visual_objects is not None:
            v_objects = [str(o).strip() for o in visual_objects if str(o).strip()]
        elif context and context.visual and context.visual.detected_objects:
            v_objects = [str(o).strip() for o in context.visual.detected_objects if str(o).strip()]

        v_text: List[str] = []
        if visible_text is not None:
            v_text = [str(t).strip() for t in visible_text if str(t).strip()]
        elif context and context.visual and context.visual.visible_text:
            v_text = [str(t).strip() for t in context.visual.visible_text if str(t).strip()]

        scene_desc = ""
        if context and context.visual and context.visual.scene_description:
            scene_desc = context.visual.scene_description.strip()

        # 4. Resolve evidence items and validity
        raw_evidence: List[Any] = []
        if evidence_items is not None:
            raw_evidence = list(evidence_items)
        elif context and context.evidence_items:
            raw_evidence = list(context.evidence_items)

        evidence_count = len(raw_evidence)
        valid_evidence_count = evidence_count
        if evidence_validation is not None:
            valid_evidence_count = getattr(
                evidence_validation, "valid_items_count", getattr(evidence_validation, "valid_count", evidence_count)
            )

        # 5. Resolve conversation history
        turns_count = 0
        has_conversation = False
        if conversation_context is not None:
            if isinstance(conversation_context, RetainedConversationContext):
                turns_count = len(conversation_context.recent_turns)
                has_conversation = turns_count > 0
            elif isinstance(conversation_context, list):
                turns_count = len(conversation_context)
                has_conversation = turns_count > 0

        # Construct available summary
        available_summary = {
            "has_query": bool(clean_query),
            "has_image": has_image,
            "image_count": image_count,
            "has_detected_objects": len(v_objects) > 0,
            "detected_objects_count": len(v_objects),
            "has_visible_text": len(v_text) > 0,
            "visible_text_count": len(v_text),
            "has_scene_description": bool(scene_desc),
            "evidence_count": evidence_count,
            "valid_evidence_count": valid_evidence_count,
            "has_conversation_history": has_conversation,
            "conversation_turns_count": turns_count,
        }

        missing_items: List[MissingInformationItem] = []
        reasons: List[str] = []

        # ---------------------------------------------------------------------
        # Detection Rule 1: Missing Image
        # ---------------------------------------------------------------------
        # Query explicitly requires visual inspection or references an image,
        # but no image is available.
        if clean_query and not has_image:
            requires_image = any(pat.search(clean_query) for pat in IMAGE_REQUIRED_PATTERNS)
            if requires_image:
                item = MissingInformationItem(
                    item_type=MissingInformationType.MISSING_IMAGE,
                    description="User request explicitly requires image analysis, but no image was provided.",
                    why_required="An image is strictly required to inspect visual contents, layout, or depicted entities.",
                    affected_reference="image",
                    severity=MissingInformationSeverity.CRITICAL,
                )
                missing_items.append(item)
                reasons.append("Missing required image artifact for visual request.")

        # ---------------------------------------------------------------------
        # Detection Rule 2: Missing Visual Evidence
        # ---------------------------------------------------------------------
        # Image is present, but visual understanding returned zero detected objects,
        # zero visible text, and no scene description, while the query requires element inspection.
        if has_image and clean_query:
            query_asks_for_elements = bool(
                re.search(r"\b(?:what\s+elements|what\s+objects|what\s+components|list\s+(?:the\s+)?(?:items|objects|components)|read\s+the\s+text)\b", clean_query, re.IGNORECASE)
            )
            has_any_visual_evidence = bool(v_objects or v_text or scene_desc)
            if query_asks_for_elements and not has_any_visual_evidence:
                item = MissingInformationItem(
                    item_type=MissingInformationType.MISSING_VISUAL_EVIDENCE,
                    description="Image provided but visual extraction yielded no detected objects, visible text, or scene description.",
                    why_required="Visual evidence is needed to identify or enumerate elements shown in the image.",
                    affected_reference="visual_evidence",
                    severity=MissingInformationSeverity.HIGH,
                )
                missing_items.append(item)
                reasons.append("Visual evidence is absent from the provided image context.")

        # ---------------------------------------------------------------------
        # Detection Rule 3: Missing Conversational Context
        # ---------------------------------------------------------------------
        # Query references previous conversation, past turns, or decisions,
        # but conversational history is empty.
        if clean_query and not has_conversation:
            requires_conversation = any(pat.search(clean_query) for pat in CONVERSATIONAL_REQUIRED_PATTERNS)
            if requires_conversation:
                item = MissingInformationItem(
                    item_type=MissingInformationType.MISSING_CONVERSATION_CONTEXT,
                    description="Query references previous turns or conversational context, but no conversation history exists.",
                    why_required="Prior conversational turn context is required to resolve conversational anaphora and historical decisions.",
                    affected_reference="conversation_history",
                    severity=MissingInformationSeverity.HIGH,
                )
                missing_items.append(item)
                reasons.append("Required conversational context is absent.")

        # ---------------------------------------------------------------------
        # Detection Rule 4: Missing Required Reference Target
        # ---------------------------------------------------------------------
        # User query references a specific named target (e.g. "Component C", "Module 3")
        # that is completely absent from visual objects, visible text, and evidence.
        # CRITICAL: We distinguish this from ambiguity. If multiple plausible targets
        # match a generic noun, that is ambiguity, NOT missing reference target.
        if clean_query:
            candidate_targets: Set[str] = set()
            for m in NAMED_TARGET_AFTER_NOUN.finditer(clean_query):
                cand = m.group(1).strip()
                if cand.lower() not in EXCLUDED_TARGET_MODIFIERS:
                    candidate_targets.add(cand)

            for m in NAMED_TARGET_BEFORE_NOUN.finditer(clean_query):
                cand = m.group(1).strip()
                if cand.lower() not in EXCLUDED_TARGET_MODIFIERS:
                    candidate_targets.add(cand)

            known_sources = v_objects + v_text + ([scene_desc] if scene_desc else [])

            for target_clean in sorted(candidate_targets):
                target_lower = target_clean.lower()

                # Check if this specific target exists as an exact token or word-bounded match
                target_found = False
                for src in known_sources:
                    tokens = [t.lower().strip(".,;:()[]{}'\"") for t in src.split()]
                    if target_lower in tokens:
                        target_found = True
                        break
                    if re.search(rf"\b{re.escape(target_lower)}\b", src.lower()):
                        target_found = True
                        break

                if not target_found:
                    # Check that this is not an entity present under candidate interpretations
                    is_candidate_present = False
                    if ambiguity_assessment and ambiguity_assessment.candidate_interpretations:
                        is_candidate_present = any(target_lower in cand.lower() for cand in ambiguity_assessment.candidate_interpretations)

                    if not is_candidate_present:
                        item = MissingInformationItem(
                            item_type=MissingInformationType.MISSING_REFERENCE_TARGET,
                            description=f"Query refers to specific target '{target_clean}', which is not found in visual objects, visible text, or context.",
                            why_required=f"Observations or evidence specifically regarding '{target_clean}' are needed to answer.",
                            affected_reference=target_clean,
                            severity=MissingInformationSeverity.HIGH,
                        )
                        missing_items.append(item)
                        reasons.append(f"Target entity '{target_clean}' is missing from available context.")

        # ---------------------------------------------------------------------
        # Detection Rule 5: Missing Requested Property or Detail
        # ---------------------------------------------------------------------
        # Query asks for specific fine-grained details (source code, pricing, timestamps, internal algorithms)
        # that are not provided in the diagram or evidence.
        if clean_query:
            all_known_details = " ".join(v_objects + v_text + [scene_desc]).lower()
            for pattern, detail_key, detail_label in PROPERTY_DETAIL_PATTERNS:
                if pattern.search(clean_query):
                    # Check if detail exists in evidence or known text
                    detail_present = (
                        detail_key in all_known_details or
                        any(term in all_known_details for term in detail_label.split())
                    )
                    if not detail_present:
                        item = MissingInformationItem(
                            item_type=MissingInformationType.MISSING_PROPERTY_OR_DETAIL,
                            description=f"Query asks for {detail_label}, but available evidence contains only high-level visual observations.",
                            why_required=f"Accurate response requires {detail_label}, which is not present in the multimodal evidence.",
                            affected_reference=detail_key,
                            severity=MissingInformationSeverity.HIGH,
                        )
                        missing_items.append(item)
                        reasons.append(f"Requested detail ({detail_label}) is missing from evidence.")

        # ---------------------------------------------------------------------
        # Determine Severity and Blocking Status
        # ---------------------------------------------------------------------
        has_missing = len(missing_items) > 0
        blocks_answering = False
        overall_severity = MissingInformationSeverity.NONE

        if has_missing:
            severities = [item.severity for item in missing_items]
            if MissingInformationSeverity.CRITICAL in severities:
                overall_severity = MissingInformationSeverity.CRITICAL
                blocks_answering = True
            elif MissingInformationSeverity.HIGH in severities:
                overall_severity = MissingInformationSeverity.HIGH
                blocks_answering = True
            elif MissingInformationSeverity.MEDIUM in severities:
                overall_severity = MissingInformationSeverity.MEDIUM
                # Medium blocks answering if there are multiple medium items
                blocks_answering = len(missing_items) > 1
            else:
                overall_severity = MissingInformationSeverity.LOW
                blocks_answering = False

        # Diagnostic metadata
        metadata = {
            "missing_item_count": len(missing_items),
            "missing_types": [item.item_type.value for item in missing_items],
            "has_ambiguity_input": ambiguity_assessment is not None,
            "has_confidence_input": confidence_assessment is not None,
            "has_evidence_validation_input": evidence_validation is not None,
        }

        return MissingInformationAssessmentResult(
            has_missing_information=has_missing,
            severity=overall_severity,
            missing_items=missing_items,
            blocks_answering=blocks_answering,
            available_summary=available_summary,
            reasons=reasons,
            metadata=metadata,
        )


# -----------------------------------------------------------------------------
# Functional Helper
# -----------------------------------------------------------------------------

def assess_multimodal_missing_information(
    query: Optional[str] = None,
    request: Optional[MultimodalRequest] = None,
    context: Optional[MultimodalContext] = None,
    images: Optional[List[ImageArtifact]] = None,
    visual_objects: Optional[List[str]] = None,
    visible_text: Optional[List[str]] = None,
    evidence_items: Optional[List[Any]] = None,
    conversation_context: Optional[Union[RetainedConversationContext, List[ConversationTurn]]] = None,
    evidence_validation: Optional[EvidenceValidationResult] = None,
    confidence_assessment: Optional[ConfidenceAssessmentResult] = None,
    ambiguity_assessment: Optional[AmbiguityAssessmentResult] = None,
) -> MissingInformationAssessmentResult:
    """Functional convenience wrapper for multimodal missing information assessment.

    Args:
        query: Optional query string.
        request: Optional MultimodalRequest.
        context: Optional MultimodalContext.
        images: Optional list of ImageArtifacts.
        visual_objects: Optional explicit list of detected visual objects.
        visible_text: Optional explicit list of OCR text lines.
        evidence_items: Optional list of evidence items.
        conversation_context: Optional retained history or list of turns.
        evidence_validation: Optional EvidenceValidationResult from Step 1.
        confidence_assessment: Optional ConfidenceAssessmentResult from Step 2.
        ambiguity_assessment: Optional AmbiguityAssessmentResult from Step 3.

    Returns:
        MissingInformationAssessmentResult detailing all missing requirements.
    """
    detector = MultimodalMissingInformationDetector()
    return detector.assess(
        query=query,
        request=request,
        context=context,
        images=images,
        visual_objects=visual_objects,
        visible_text=visible_text,
        evidence_items=evidence_items,
        conversation_context=conversation_context,
        evidence_validation=evidence_validation,
        confidence_assessment=confidence_assessment,
        ambiguity_assessment=ambiguity_assessment,
    )


def _clean_metadata_dict(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensure metadata contains no raw bytes or un-serializable objects."""
    if not d or not isinstance(d, dict):
        return {}

    cleaned: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (bytes, bytearray)):
            cleaned[k] = f"<bytes: {len(v)}>"
        elif isinstance(v, dict):
            cleaned[k] = _clean_metadata_dict(v)
        elif isinstance(v, list):
            cleaned[k] = [
                f"<bytes: {len(elem)}>" if isinstance(elem, (bytes, bytearray))
                else _clean_metadata_dict(elem) if isinstance(elem, dict)
                else elem
                for elem in v
            ]
        elif isinstance(v, Enum):
            cleaned[k] = v.value
        else:
            cleaned[k] = v
    return cleaned
