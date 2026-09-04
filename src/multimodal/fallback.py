"""Safe Fallback Behavior Component (Phase 5 — Day 28 Step 5).

Provides deterministic decision making and safe fallback response handling for
unreliable, ambiguous, conflicting, or insufficiently supported multimodal requests.
Consumes structured assessments from Evidence Validation, Confidence Assessment,
Ambiguity Detection, and Missing Information Assessment to produce safe, non-hallucinated
interventions according to an explicit deterministic priority.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .ambiguity import (
    AmbiguityAssessmentResult,
    AmbiguityType,
    assess_multimodal_ambiguity,
)
from .confidence import (
    ConfidenceAssessmentResult,
    ConfidenceStatus,
    assess_multimodal_confidence,
)
from .context import MultimodalContext, UnifiedEvidenceItem
from .evidence_validator import (
    EvidenceStatus,
    EvidenceValidationResult,
    validate_multimodal_evidence,
)
from .missing_information import (
    MissingInformationAssessmentResult,
    MissingInformationType,
    assess_multimodal_missing_information,
)
from .models import (
    AmbiguityDetails,
    ModalityType,
    MultimodalRequest,
    MultimodalResponse,
    VisualEvidenceItem,
)


# -----------------------------------------------------------------------------
# Fallback Action Enumeration
# -----------------------------------------------------------------------------

class FallbackAction(str, Enum):
    """Categorical actions determining how the system handles the request."""

    PROCEED = "proceed"
    ASK_CLARIFICATION = "ask_clarification"
    REQUEST_MISSING_INFORMATION = "request_missing_information"
    DECLINE_UNSUPPORTED_CLAIM = "decline_unsupported_claim"
    SAFE_LIMITED_RESPONSE = "safe_limited_response"


# -----------------------------------------------------------------------------
# Structured Fallback Decision
# -----------------------------------------------------------------------------

@dataclass
class FallbackDecision:
    """Structured decision output produced by the fallback handler."""

    action: FallbackAction
    should_proceed: bool
    reason: str
    safe_message: str
    clarification_needed: Optional[str] = None
    missing_requirements: List[str] = field(default_factory=list)
    limitations_or_qualifications: List[str] = field(default_factory=list)
    evidence_status: Optional[str] = None
    confidence_status: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize fallback decision to clean JSON-safe dictionary without raw bytes."""
        return {
            "action": self.action.value if isinstance(self.action, FallbackAction) else str(self.action),
            "should_proceed": self.should_proceed,
            "reason": self.reason,
            "safe_message": self.safe_message,
            "clarification_needed": self.clarification_needed,
            "missing_requirements": list(self.missing_requirements),
            "limitations_or_qualifications": list(self.limitations_or_qualifications),
            "evidence_status": self.evidence_status,
            "confidence_status": self.confidence_status,
            "metadata": _clean_metadata_dict(self.metadata),
        }

    def to_multimodal_response(
        self,
        query: str,
        session_id: str = "default_multimodal_session",
        modality: ModalityType = ModalityType.TEXT_ONLY,
        evidence_items: Optional[List[VisualEvidenceItem]] = None,
    ) -> MultimodalResponse:
        """Convert fallback decision to a canonical Day 24 MultimodalResponse contract.

        Args:
            query: The originating user query.
            session_id: Session identifier.
            modality: Modality of the interaction.
            evidence_items: Optional supported visual evidence items to attach.

        Returns:
            MultimodalResponse with safe text, ambiguity details, and warning message.
        """
        clarif_q = self.clarification_needed or "Please clarify your request."
        ambiguity_details = AmbiguityDetails(
            is_ambiguous=(self.action == FallbackAction.ASK_CLARIFICATION),
            clarification_question=clarif_q if self.action == FallbackAction.ASK_CLARIFICATION else None,
            missing_aspects=list(self.missing_requirements),
        )

        valid_evidence: List[VisualEvidenceItem] = []
        if evidence_items and self.action != FallbackAction.DECLINE_UNSUPPORTED_CLAIM:
            for item in evidence_items:
                if isinstance(item, VisualEvidenceItem):
                    valid_evidence.append(item)

        is_grounded = (self.action == FallbackAction.PROCEED)

        warning: Optional[str] = None
        if self.action != FallbackAction.PROCEED:
            warning = f"Fallback action triggered: {self.action.value}. {self.reason}"

        return MultimodalResponse(
            query=query,
            answer=self.safe_message,
            modality=modality,
            ambiguity=ambiguity_details,
            visual_evidence=valid_evidence,
            grounded=is_grounded,
            warning_message=warning,
            formatted_markdown=f"### Response\n\n{self.safe_message}",
            session_id=session_id,
        )


# -----------------------------------------------------------------------------
# Multimodal Fallback Handler Implementation
# -----------------------------------------------------------------------------

class MultimodalFallbackHandler:
    """Deterministic assessor and generator for safe multimodal fallbacks."""

    def evaluate(
        self,
        query: Optional[str] = None,
        context: Optional[MultimodalContext] = None,
        request: Optional[MultimodalRequest] = None,
        evidence_items: Optional[List[Any]] = None,
        evidence_validation: Optional[EvidenceValidationResult] = None,
        confidence_assessment: Optional[ConfidenceAssessmentResult] = None,
        ambiguity_assessment: Optional[AmbiguityAssessmentResult] = None,
        missing_info_assessment: Optional[MissingInformationAssessmentResult] = None,
        unsupported_claim: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FallbackDecision:
        """Evaluate multimodal inputs and determine the safe fallback action.

        Deterministic Decision Priority:
        1. REQUEST_MISSING_INFORMATION (Critical/blocking missing info)
        2. ASK_CLARIFICATION (Ambiguous inputs requiring user clarification)
        3. DECLINE_UNSUPPORTED_CLAIM (Unsupported claims, contradictory evidence, or zero valid evidence)
        4. SAFE_LIMITED_RESPONSE (Low-confidence evidence or partial observations)
        5. PROCEED (Sufficient, valid, reliable evidence)

        Args:
            query: Optional query string.
            context: Optional MultimodalContext.
            request: Optional MultimodalRequest.
            evidence_items: Optional raw or validated evidence items.
            evidence_validation: Step 1 validation result.
            confidence_assessment: Step 2 confidence result.
            ambiguity_assessment: Step 3 ambiguity result.
            missing_info_assessment: Step 4 missing information result.
            unsupported_claim: Explicit flag indicating claim cannot be substantiated.
            metadata: Optional user/caller metadata.

        Returns:
            FallbackDecision describing the safe action, reason, and message.
        """
        custom_metadata = dict(metadata) if metadata else {}
        clean_query = ""
        if query and isinstance(query, str) and query.strip():
            clean_query = query.strip()
        elif request and request.query and request.query.strip():
            clean_query = request.query.strip()
        elif context and context.textual and context.textual.raw_query:
            clean_query = context.textual.raw_query.strip()

        # Auto-compute missing information if not provided
        if missing_info_assessment is None and (clean_query or context or request):
            missing_info_assessment = assess_multimodal_missing_information(
                query=clean_query,
                request=request,
                context=context,
                evidence_items=evidence_items,
                evidence_validation=evidence_validation,
                confidence_assessment=confidence_assessment,
                ambiguity_assessment=ambiguity_assessment,
            )

        # Auto-compute ambiguity if not provided
        if ambiguity_assessment is None and (clean_query or context or request):
            ambiguity_assessment = assess_multimodal_ambiguity(
                query=clean_query,
                context=context,
                evidence=evidence_validation or evidence_items,
                confidence=confidence_assessment,
            )

        # Auto-compute evidence validation if not provided
        if evidence_validation is None and (evidence_items is not None or (context and context.evidence_items)):
            evidence_to_validate = evidence_items if evidence_items is not None else context
            evidence_validation = validate_multimodal_evidence(evidence=evidence_to_validate)

        # Auto-compute confidence assessment if not provided
        if confidence_assessment is None and (evidence_items is not None or (context and context.evidence_items)):
            evidence_to_assess = evidence_items if evidence_items is not None else context
            confidence_assessment = assess_multimodal_confidence(evidence=evidence_to_assess)

        ev_status_val = evidence_validation.status.value if evidence_validation else None
        conf_status_val = confidence_assessment.status.value if confidence_assessment else None

        def _make_meta(decision_meta: Dict[str, Any]) -> Dict[str, Any]:
            m = dict(custom_metadata)
            m.update(decision_meta)
            return _clean_metadata_dict(m)

        # ---------------------------------------------------------------------
        # PRIORITY 1: Missing Required Information
        # ---------------------------------------------------------------------
        if missing_info_assessment and missing_info_assessment.has_missing_information and missing_info_assessment.blocks_answering:
            missing_descriptions = [item.description for item in missing_info_assessment.missing_items]
            missing_types = [item.item_type for item in missing_info_assessment.missing_items]

            if MissingInformationType.MISSING_IMAGE in missing_types:
                safe_msg = (
                    "To analyze or inspect the requested visual elements, an image artifact is required. "
                    "Please provide an image with your request."
                )
            elif MissingInformationType.MISSING_CONVERSATION_CONTEXT in missing_types:
                safe_msg = (
                    "This request refers to previous conversational context or earlier decisions, "
                    "but no conversation history is available in this session."
                )
            else:
                items_str = "; ".join(missing_descriptions)
                safe_msg = f"Cannot answer definitively because required information is missing: {items_str}."

            return FallbackDecision(
                action=FallbackAction.REQUEST_MISSING_INFORMATION,
                should_proceed=False,
                reason="Required information is absent, preventing reliable answering.",
                safe_message=safe_msg,
                missing_requirements=missing_descriptions,
                evidence_status=ev_status_val,
                confidence_status=conf_status_val,
                metadata=_make_meta({
                    "priority_level": 1,
                    "missing_types": [mt.value for mt in missing_types],
                }),
            )

        # ---------------------------------------------------------------------
        # PRIORITY 2: Ambiguous Inputs Requiring Clarification
        # ---------------------------------------------------------------------
        if ambiguity_assessment and ambiguity_assessment.is_ambiguous and ambiguity_assessment.requires_clarification:
            clarification_text = ambiguity_assessment.suggested_clarification
            if not clarification_text:
                if ambiguity_assessment.candidate_interpretations:
                    cands_str = ", ".join(ambiguity_assessment.candidate_interpretations)
                    clarification_text = (
                        f"The request could refer to multiple elements ({cands_str}). "
                        f"Please specify which one you are referring to."
                    )
                else:
                    clarification_text = (
                        "The request is underspecified or ambiguous. "
                        "Please provide more specific details about the component or query."
                    )

            return FallbackDecision(
                action=FallbackAction.ASK_CLARIFICATION,
                should_proceed=False,
                reason="Request contains ambiguous or underspecified references that require user clarification.",
                safe_message=clarification_text,
                clarification_needed=clarification_text,
                missing_requirements=list(ambiguity_assessment.affected_references),
                evidence_status=ev_status_val,
                confidence_status=conf_status_val,
                metadata=_make_meta({
                    "priority_level": 2,
                    "ambiguity_types": [at.value for at in ambiguity_assessment.ambiguity_types],
                    "candidate_interpretations": list(ambiguity_assessment.candidate_interpretations),
                }),
            )

        # ---------------------------------------------------------------------
        # PRIORITY 3: Unsupported Claim / Conflicting Evidence / Zero Evidence
        # ---------------------------------------------------------------------
        # 3a. Explicit unsupported claim
        if unsupported_claim:
            return FallbackDecision(
                action=FallbackAction.DECLINE_UNSUPPORTED_CLAIM,
                should_proceed=False,
                reason="The requested claim cannot be supported by the verified multimodal evidence.",
                safe_message="I cannot confirm or support this claim based on the available multimodal evidence.",
                evidence_status=ev_status_val,
                confidence_status=conf_status_val,
                metadata=_make_meta({"priority_level": 3, "trigger": "unsupported_claim_flag"}),
            )

        # 3b. Conflicting evidence
        if ambiguity_assessment and AmbiguityType.CONFLICTING_EVIDENCE in ambiguity_assessment.ambiguity_types:
            return FallbackDecision(
                action=FallbackAction.DECLINE_UNSUPPORTED_CLAIM,
                should_proceed=False,
                reason="Available evidence contains contradictory observations, preventing a definitive claim.",
                safe_message=(
                    "I cannot provide a definitive answer because the available evidence contains "
                    "conflicting or contradictory observations."
                ),
                evidence_status=ev_status_val,
                confidence_status=conf_status_val,
                metadata=_make_meta({"priority_level": 3, "trigger": "conflicting_evidence"}),
            )

        # 3c. Zero evidence available
        if evidence_validation and evidence_validation.status == EvidenceStatus.NO_EVIDENCE:
            return FallbackDecision(
                action=FallbackAction.DECLINE_UNSUPPORTED_CLAIM,
                should_proceed=False,
                reason="No visual or contextual evidence is available to answer this request.",
                safe_message="No evidence is available in the provided context to substantiate this query.",
                evidence_status=ev_status_val,
                confidence_status=conf_status_val,
                metadata=_make_meta({"priority_level": 3, "trigger": "no_evidence"}),
            )

        # 3d. All evidence items invalid
        if evidence_validation and evidence_validation.total_items > 0 and evidence_validation.valid_items_count == 0:
            return FallbackDecision(
                action=FallbackAction.DECLINE_UNSUPPORTED_CLAIM,
                should_proceed=False,
                reason="All provided evidence items failed validation and cannot be safely utilized.",
                safe_message="The provided evidence is invalid or malformed, preventing reliable verification.",
                evidence_status=ev_status_val,
                confidence_status=conf_status_val,
                metadata=_make_meta({"priority_level": 3, "trigger": "all_evidence_invalid"}),
            )

        # ---------------------------------------------------------------------
        # PRIORITY 4: Low-Confidence Evidence / Safe Limited Response
        # ---------------------------------------------------------------------
        # If evidence exists but confidence is low (below threshold)
        is_conf = getattr(confidence_assessment, "is_confident", None)
        is_rel = getattr(confidence_assessment, "is_reliable", None)
        has_low_conf = (is_conf is False) or (is_rel is False)
        if confidence_assessment and has_low_conf:
            score_disp = (
                f"{confidence_assessment.aggregate_confidence:.2f}"
                if confidence_assessment.aggregate_confidence is not None
                else "uncalibrated"
            )
            qualifications = [
                f"Evidence confidence ({score_disp}) is below reliability threshold ({confidence_assessment.threshold:.2f})."
            ]
            safe_msg = (
                f"Based on available observations with limited confidence ({score_disp}): "
                "Preliminary findings are available, but additional higher-resolution evidence "
                "or verification is recommended before relying on these conclusions."
            )

            return FallbackDecision(
                action=FallbackAction.SAFE_LIMITED_RESPONSE,
                should_proceed=False,
                reason="Evidence confidence is below the reliability threshold; limited qualification provided.",
                safe_message=safe_msg,
                limitations_or_qualifications=qualifications,
                evidence_status=ev_status_val,
                confidence_status=conf_status_val,
                metadata=_make_meta({
                    "priority_level": 4,
                    "confidence_score": confidence_assessment.aggregate_confidence,
                    "threshold": confidence_assessment.threshold,
                }),
            )

        # If evidence validation is partial (some valid, some invalid)
        if evidence_validation and evidence_validation.status == EvidenceStatus.PARTIAL:
            qualifications = [
                f"{evidence_validation.invalid_items_count} of {evidence_validation.total_items} evidence items failed validation and were excluded."
            ]
            safe_msg = (
                "Observations are limited to verified evidence items; "
                "unsupported or unverified details have been excluded."
            )

            return FallbackDecision(
                action=FallbackAction.SAFE_LIMITED_RESPONSE,
                should_proceed=False,
                reason="Partial evidence validation outcome; unverified items excluded.",
                safe_message=safe_msg,
                limitations_or_qualifications=qualifications,
                evidence_status=ev_status_val,
                confidence_status=conf_status_val,
                metadata=_make_meta({"priority_level": 4, "trigger": "partial_evidence"}),
            )

        # ---------------------------------------------------------------------
        # PRIORITY 5: PROCEED
        # ---------------------------------------------------------------------
        return FallbackDecision(
            action=FallbackAction.PROCEED,
            should_proceed=True,
            reason="Verified evidence and context are sufficient and reliable to answer the request.",
            safe_message="Verified evidence is sufficient to generate a grounded response.",
            evidence_status=ev_status_val,
            confidence_status=conf_status_val,
            metadata=_make_meta({"priority_level": 5}),
        )


# -----------------------------------------------------------------------------
# Functional Helper
# -----------------------------------------------------------------------------

def evaluate_safe_fallback(
    query: Optional[str] = None,
    context: Optional[MultimodalContext] = None,
    request: Optional[MultimodalRequest] = None,
    evidence_items: Optional[List[Any]] = None,
    evidence_validation: Optional[EvidenceValidationResult] = None,
    confidence_assessment: Optional[ConfidenceAssessmentResult] = None,
    ambiguity_assessment: Optional[AmbiguityAssessmentResult] = None,
    missing_info_assessment: Optional[MissingInformationAssessmentResult] = None,
    unsupported_claim: bool = False,
) -> FallbackDecision:
    """Evaluate multimodal inputs and determine safe fallback action.

    Args:
        query: Optional query string.
        context: Optional MultimodalContext.
        request: Optional MultimodalRequest.
        evidence_items: Optional raw or validated evidence items.
        evidence_validation: Step 1 validation result.
        confidence_assessment: Step 2 confidence result.
        ambiguity_assessment: Step 3 ambiguity result.
        missing_info_assessment: Step 4 missing information result.
        unsupported_claim: Explicit flag indicating claim cannot be substantiated.

    Returns:
        FallbackDecision describing the safe action, reason, and message.
    """
    handler = MultimodalFallbackHandler()
    return handler.evaluate(
        query=query,
        context=context,
        request=request,
        evidence_items=evidence_items,
        evidence_validation=evidence_validation,
        confidence_assessment=confidence_assessment,
        ambiguity_assessment=ambiguity_assessment,
        missing_info_assessment=missing_info_assessment,
        unsupported_claim=unsupported_claim,
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
