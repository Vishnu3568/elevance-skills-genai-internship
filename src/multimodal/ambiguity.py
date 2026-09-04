"""Multimodal Ambiguity Detection and Assessment (Phase 5 — Day 28 Step 3).

Provides deterministic ambiguity detection across multimodal queries, visual contexts,
and evidence collections. Explicitly identifies vague wording, unclear visual references,
multiple plausible targets, conflicting evidence, and ambiguous conversational referents
without guessing or silently selecting one interpretation.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .confidence import ConfidenceAssessmentResult, ConfidenceStatus
from .context import MultimodalContext, UnifiedEvidenceItem
from .context_retention import RetainedConversationContext
from .evidence_validator import EvidenceStatus, EvidenceValidationResult
from .models import AmbiguityDetails, ModalityType, VisualEvidenceItem


# -----------------------------------------------------------------------------
# Ambiguity Classification Enumerations
# -----------------------------------------------------------------------------

class AmbiguityLevel(str, Enum):
    """Severity classification of detected ambiguity."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AmbiguityType(str, Enum):
    """Categorical classification of ambiguity source."""

    VAGUE_QUERY = "vague_query"
    UNCLEAR_VISUAL_REFERENCE = "unclear_visual_reference"
    MULTIPLE_VISUAL_TARGETS = "multiple_visual_targets"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    AMBIGUOUS_DISCOURSE = "ambiguous_discourse"
    INSUFFICIENT_SPECIFICITY = "insufficient_specificity"


# -----------------------------------------------------------------------------
# Regex Patterns for Ambiguity Detection
# -----------------------------------------------------------------------------

# Generic / vague query patterns
VAGUE_QUERY_PATTERNS = [
    re.compile(r"^(?:what|how|why|explain|describe|analyze|check|tell me|look at)\s+(?:is\s+)?(?:this|that|it|these|those)(?:\s+(?:thing|stuff|item|object|part))?\??$", re.IGNORECASE),
    re.compile(r"^(?:analyze|explain|describe|check|summarize)\??$", re.IGNORECASE),
    re.compile(r"^(?:what about it|what does it mean|what is going on)\??$", re.IGNORECASE),
    re.compile(r"^(?:this stuff|this thing|the stuff|the thing)\??$", re.IGNORECASE),
]

# Underspecified visual category referents (generic nouns without qualifiers)
GENERIC_VISUAL_NOUNS = {
    "component", "element", "block", "box", "node", "module",
    "part", "section", "layer", "arrow", "line", "shape", "item", "diagram",
    "thing", "object",
}

GENERIC_VISUAL_REFERENTS = re.compile(
    rf"\b(?:the|this|that)\s+({'|'.join(GENERIC_VISUAL_NOUNS)})\b",
    re.IGNORECASE,
)


# -----------------------------------------------------------------------------
# Structured Ambiguity Assessment Report
# -----------------------------------------------------------------------------

@dataclass
class AmbiguityAssessmentResult:
    """Detailed diagnostic report evaluating multimodal ambiguity."""

    is_ambiguous: bool
    ambiguity_level: AmbiguityLevel
    ambiguity_types: List[AmbiguityType] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    affected_references: List[str] = field(default_factory=list)
    candidate_interpretations: List[str] = field(default_factory=list)
    requires_clarification: bool = False
    suggested_clarification: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ambiguity assessment to a clean JSON-safe dictionary."""
        return {
            "is_ambiguous": self.is_ambiguous,
            "ambiguity_level": self.ambiguity_level.value,
            "ambiguity_types": [at.value for at in self.ambiguity_types],
            "reasons": list(self.reasons),
            "affected_references": list(self.affected_references),
            "candidate_interpretations": list(self.candidate_interpretations),
            "requires_clarification": self.requires_clarification,
            "suggested_clarification": self.suggested_clarification,
            "metadata": _clean_metadata_dict(self.metadata),
        }

    def to_ambiguity_details(self) -> AmbiguityDetails:
        """Convert to canonical Day 24 AmbiguityDetails contract."""
        return AmbiguityDetails(
            is_ambiguous=self.is_ambiguous,
            clarification_question=self.suggested_clarification,
            missing_aspects=list(self.reasons),
        )


# -----------------------------------------------------------------------------
# Metadata Cleansing Helper (Zero Raw Byte Leakage)
# -----------------------------------------------------------------------------

def _clean_metadata_dict(data: Any) -> Any:
    """Recursively sanitize metadata dictionaries to purge raw bytes or image buffers."""
    if isinstance(data, (bytes, bytearray)):
        return f"<bytes: {len(data)}>"
    if isinstance(data, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in data.items():
            if k in ("data", "base64_str", "artifact_data"):
                continue
            cleaned[k] = _clean_metadata_dict(v)
        return cleaned
    if isinstance(data, (list, tuple)):
        return [_clean_metadata_dict(v) for v in data]
    return data


# -----------------------------------------------------------------------------
# Multimodal Ambiguity Detector Engine
# -----------------------------------------------------------------------------

class MultimodalAmbiguityDetector:
    """Evaluates multimodal requests and context for ambiguity and underspecification."""

    def __init__(self, high_severity_threshold: int = 2) -> None:
        """Initialize detector.

        Args:
            high_severity_threshold: Number of distinct ambiguity types triggering HIGH level.
        """
        self.high_severity_threshold = high_severity_threshold

    def assess_ambiguity(
        self,
        query: Optional[str] = None,
        context: Optional[MultimodalContext] = None,
        evidence: Optional[Union[List[Any], EvidenceValidationResult]] = None,
        confidence: Optional[Union[ConfidenceAssessmentResult, float]] = None,
        conversation_context: Optional[RetainedConversationContext] = None,
        detected_objects: Optional[List[str]] = None,
    ) -> AmbiguityAssessmentResult:
        """Assess ambiguity across query wording, visual context, evidence, and history.

        Args:
            query: The user query string to inspect.
            context: Optional MultimodalContext containing visual outputs and evidence.
            evidence: Optional evidence collection or EvidenceValidationResult.
            confidence: Optional ConfidenceAssessmentResult or float score.
            conversation_context: Optional RetainedConversationContext from dialogue history.
            detected_objects: Optional explicit list of detected visual entities.

        Returns:
            AmbiguityAssessmentResult: Comprehensive, deterministic ambiguity evaluation.
        """
        reasons: List[str] = []
        ambiguity_types: List[AmbiguityType] = []
        affected_refs: List[str] = []
        candidates: List[str] = []
        metadata: Dict[str, Any] = {}

        clean_query = query.strip() if query and isinstance(query, str) else ""

        # Extract visual entities from available containers
        visual_objects: List[str] = []
        if detected_objects:
            visual_objects.extend(detected_objects)
        elif context is not None and context.visual_context:
            visual_objects.extend(context.visual_context.detected_objects)

        # ---------------------------------------------------------------------
        # 1. Detect Vague / Underspecified User Query
        # ---------------------------------------------------------------------
        if clean_query:
            for pattern in VAGUE_QUERY_PATTERNS:
                if pattern.match(clean_query):
                    ambiguity_types.append(AmbiguityType.VAGUE_QUERY)
                    reasons.append(
                        f"Query '{clean_query}' is overly vague and lacks specific entity or task focus."
                    )
                    affected_refs.append(clean_query)
                    break

            # Short word count check without visual grounding
            words = clean_query.split()
            if len(words) <= 2 and not any(p.match(clean_query) for p in VAGUE_QUERY_PATTERNS):
                if clean_query.lower() in ("analyze", "explain", "why", "how", "what", "check"):
                    ambiguity_types.append(AmbiguityType.INSUFFICIENT_SPECIFICITY)
                    reasons.append(f"Query '{clean_query}' is a single-word command lacking context.")
                    affected_refs.append(clean_query)

        # ---------------------------------------------------------------------
        # 2. Detect Multiple Plausible Visual Targets
        # ---------------------------------------------------------------------
        # e.g. User asks "What does the component do?", but visual context has multiple components
        if clean_query and visual_objects:
            match = GENERIC_VISUAL_REFERENTS.search(clean_query)
            if match:
                ref_noun = match.group(1).lower()
                # Find all detected objects matching or belonging to this generic category
                if ref_noun in ("thing", "object", "item"):
                    matching_targets = list(visual_objects)
                else:
                    matching_targets = [
                        obj for obj in visual_objects
                        if ref_noun in obj.lower() or any(term in obj.lower() for term in ("block", "layer", "module", "box", "node", "unit"))
                    ]

                if len(matching_targets) > 1:
                    ambiguity_types.append(AmbiguityType.MULTIPLE_VISUAL_TARGETS)
                    reasons.append(
                        f"Query refers to generic '{match.group(0)}', but multiple plausible visual targets exist: {matching_targets}."
                    )
                    affected_refs.append(match.group(0))
                    # Explicitly preserve candidate interpretations without guessing
                    candidates.extend(matching_targets)

        # ---------------------------------------------------------------------
        # 3. Detect Unclear Visual References (Referent Not in Visual Context)
        # ---------------------------------------------------------------------
        if clean_query and visual_objects:
            # Check if query references a specific qualified visual entity (e.g. "the green module", "the upper circle")
            qualified_pattern = re.compile(
                r"\b(?:the|this)\s+([a-zA-Z0-9_\-]+\s+(?:component|module|block|box|layer|unit|shape))\b",
                re.IGNORECASE,
            )
            for m in qualified_pattern.finditer(clean_query):
                spec_ref = m.group(1).lower()
                # Check if this specified reference matches any detected object
                if not any(spec_ref in obj.lower() for obj in visual_objects):
                    ambiguity_types.append(AmbiguityType.UNCLEAR_VISUAL_REFERENCE)
                    reasons.append(
                        f"Referenced visual element '{m.group(0)}' was not clearly detected in the image (detected: {visual_objects})."
                    )
                    affected_refs.append(m.group(0))

        # ---------------------------------------------------------------------
        # 4. Detect Conflicting Evidence
        # ---------------------------------------------------------------------
        evidence_items: List[Any] = []
        if isinstance(evidence, EvidenceValidationResult):
            evidence_items = list(evidence.valid_evidence) + list(evidence.invalid_evidence)
        elif isinstance(evidence, list):
            evidence_items = list(evidence)
        elif context is not None:
            evidence_items = list(context.evidence_items)

        if len(evidence_items) >= 2:
            # Check for direct contradictions (e.g. contradictory boolean, opposing attributes, or opposite polarity)
            descriptions = []
            for ev in evidence_items:
                if hasattr(ev, "description"):
                    descriptions.append(ev.description.lower())
                elif isinstance(ev, dict) and "description" in ev:
                    descriptions.append(str(ev["description"]).lower())

            # Check opposing assertions
            has_positive = any(re.search(r"\b(active|present|normal|correct|valid|high|positive)\b", d) for d in descriptions)
            has_negative = any(re.search(r"\b(inactive|absent|abnormal|incorrect|invalid|low|negative|failed)\b", d) for d in descriptions)

            # Check conflicting classifications on the same subject
            conflict_detected = False
            for i in range(len(descriptions)):
                for j in range(i + 1, len(descriptions)):
                    d1, d2 = descriptions[i], descriptions[j]
                    if ("is present" in d1 and "is absent" in d2) or ("positive" in d1 and "negative" in d2):
                        conflict_detected = True
                        break
                    if ("healthy" in d1 and "defective" in d2) or ("enabled" in d1 and "disabled" in d2):
                        conflict_detected = True
                        break

            if conflict_detected:
                ambiguity_types.append(AmbiguityType.CONFLICTING_EVIDENCE)
                reasons.append(
                    "Evidence collection contains directly contradictory or conflicting findings."
                )
                affected_refs.extend(descriptions[:2])

        # ---------------------------------------------------------------------
        # 5. Detect Ambiguous Conversational Discourse References
        # ---------------------------------------------------------------------
        if clean_query and conversation_context is not None and conversation_context.has_history:
            # If query uses anaphoric pronoun ("it", "this", "that") and recent turns discussed multiple subjects
            if re.search(r"\b(it|this|that)\b", clean_query, re.IGNORECASE):
                # Check recent turns
                prior_queries = [t.user_query for t in conversation_context.retained_turns if t.user_query]
                prior_objects = conversation_context.get_all_detected_objects()

                # If multiple distinct entities were previously discussed
                if len(prior_objects) > 1 and not any(obj.lower() in clean_query.lower() for obj in prior_objects):
                    if AmbiguityType.MULTIPLE_VISUAL_TARGETS not in ambiguity_types:
                        ambiguity_types.append(AmbiguityType.AMBIGUOUS_DISCOURSE)
                        reasons.append(
                            f"Pronoun in query '{clean_query}' is ambiguous across multiple previously discussed entities: {prior_objects}."
                        )
                        affected_refs.append(clean_query)
                        candidates.extend(prior_objects)

        # ---------------------------------------------------------------------
        # Aggregate Severity & Clarification Synthesis
        # ---------------------------------------------------------------------
        unique_types = list(dict.fromkeys(ambiguity_types))
        is_ambiguous = len(unique_types) > 0

        if not is_ambiguous:
            level = AmbiguityLevel.NONE
            requires_clarification = False
            clarification_q = None
        elif len(unique_types) >= self.high_severity_threshold or AmbiguityType.CONFLICTING_EVIDENCE in unique_types:
            level = AmbiguityLevel.HIGH
            requires_clarification = True
        elif AmbiguityType.MULTIPLE_VISUAL_TARGETS in unique_types or AmbiguityType.VAGUE_QUERY in unique_types:
            level = AmbiguityLevel.MEDIUM
            requires_clarification = True
        else:
            level = AmbiguityLevel.LOW
            requires_clarification = False

        # Formulate deterministic clarification question without guessing
        clarification_q = None
        if requires_clarification:
            if candidates:
                cand_str = ", ".join(f"'{c}'" for c in candidates[:3])
                clarification_q = f"Which specific target are you referring to ({cand_str})?"
            elif AmbiguityType.VAGUE_QUERY in unique_types or AmbiguityType.INSUFFICIENT_SPECIFICITY in unique_types:
                clarification_q = "Could you please specify which element or question you would like me to analyze?"
            elif AmbiguityType.CONFLICTING_EVIDENCE in unique_types:
                clarification_q = "The available evidence contains conflicting observations. Could you clarify which aspect you want to focus on?"
            elif AmbiguityType.UNCLEAR_VISUAL_REFERENCE in unique_types and affected_refs:
                clarification_q = f"Could you clarify the location or label of {affected_refs[0]} in the image?"
            else:
                clarification_q = "Could you please provide more specific details about your request?"

        metadata["detected_types_count"] = len(unique_types)
        metadata["visual_objects_checked"] = visual_objects

        return AmbiguityAssessmentResult(
            is_ambiguous=is_ambiguous,
            ambiguity_level=level,
            ambiguity_types=unique_types,
            reasons=reasons,
            affected_references=list(dict.fromkeys(affected_refs)),
            candidate_interpretations=list(dict.fromkeys(candidates)),
            requires_clarification=requires_clarification,
            suggested_clarification=clarification_q,
            metadata=metadata,
        )


# -----------------------------------------------------------------------------
# Convenience Functional Interface
# -----------------------------------------------------------------------------

def assess_multimodal_ambiguity(
    query: Optional[str] = None,
    context: Optional[MultimodalContext] = None,
    evidence: Optional[Union[List[Any], EvidenceValidationResult]] = None,
    confidence: Optional[Union[ConfidenceAssessmentResult, float]] = None,
    conversation_context: Optional[RetainedConversationContext] = None,
    detected_objects: Optional[List[str]] = None,
) -> AmbiguityAssessmentResult:
    """Assess multimodal ambiguity with sensible defaults.

    Args:
        query: User query string.
        context: Optional MultimodalContext.
        evidence: Optional evidence collection or validation result.
        confidence: Optional confidence evaluation.
        conversation_context: Optional conversation context.
        detected_objects: Optional list of detected visual entities.

    Returns:
        AmbiguityAssessmentResult: Deterministic ambiguity diagnostic report.
    """
    detector = MultimodalAmbiguityDetector()
    return detector.assess_ambiguity(
        query=query,
        context=context,
        evidence=evidence,
        confidence=confidence,
        conversation_context=conversation_context,
        detected_objects=detected_objects,
    )
