"""Multimodal Assistant High-Level Service (Phase 5 — Day 29).

Orchestrates user interactions, session management, image ingestion, follow-up query
resolution, cross-modal reasoning, evidence validation, confidence assessment,
ambiguity detection, missing information detection, and safe fallback behavior
for presentation in Streamlit.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .ambiguity import (
    AmbiguityAssessmentResult,
    MultimodalAmbiguityDetector,
)
from .confidence import (
    ConfidenceAssessmentResult,
    MultimodalConfidenceAssessor,
)
from .context import (
    build_multimodal_context,
)
from .conversation import (
    MultimodalConversationSession,
    MultimodalSessionManager,
)
from .evidence_validator import (
    EvidenceValidationResult,
    MultimodalEvidenceValidator,
)
from .fallback import (
    FallbackAction,
    FallbackDecision,
    MultimodalFallbackHandler,
)
from .followup import (
    FollowUpResolution,
    MultimodalFollowUpResolver,
)
from .ingestion import ingest_image
from .missing_information import (
    MissingInformationAssessmentResult,
    MultimodalMissingInformationDetector,
    assess_multimodal_missing_information,
)
from .models import (
    ImageArtifact,
    ModalityType,
    MultimodalRequest,
    MultimodalResponse,
)
from .orchestrator import (
    create_multimodal_pipeline,
)
from .reasoning import (
    MultimodalReasoningEngine,
)
from .response_generator import MultimodalResponseGenerator
from .vision import VisionService


# -----------------------------------------------------------------------------
# Structured Multimodal UI Result Contract
# -----------------------------------------------------------------------------

@dataclass
class MultimodalUIResult:
    """Consolidated result contract returned to the Streamlit UI."""

    response: MultimodalResponse
    fallback_decision: FallbackDecision
    evidence_validation: Optional[EvidenceValidationResult] = None
    confidence_assessment: Optional[ConfidenceAssessmentResult] = None
    ambiguity_assessment: Optional[AmbiguityAssessmentResult] = None
    missing_info_assessment: Optional[MissingInformationAssessmentResult] = None
    followup_resolution: Optional[FollowUpResolution] = None
    turn_index: int = 0
    modality: ModalityType = ModalityType.TEXT_ONLY
    error: Optional[str] = None

    @property
    def is_clarification(self) -> bool:
        """Check if the result represents a clarification request."""
        return self.fallback_decision.action == FallbackAction.ASK_CLARIFICATION

    @property
    def is_missing_information(self) -> bool:
        """Check if the result represents a missing information request."""
        return self.fallback_decision.action == FallbackAction.REQUEST_MISSING_INFORMATION

    @property
    def is_unsupported(self) -> bool:
        """Check if the result declined an unsupported claim."""
        return self.fallback_decision.action == FallbackAction.DECLINE_UNSUPPORTED_CLAIM

    @property
    def is_safe_limited(self) -> bool:
        """Check if the result is a safe limited response."""
        return self.fallback_decision.action == FallbackAction.SAFE_LIMITED_RESPONSE

    @property
    def is_proceed(self) -> bool:
        """Check if the result proceeded normally."""
        return self.fallback_decision.action == FallbackAction.PROCEED


# -----------------------------------------------------------------------------
# Multimodal Assistant Service Implementation
# -----------------------------------------------------------------------------

class MultimodalAssistantService:
    """Coordinates backend multimodal services for Streamlit UI interaction."""

    def __init__(
        self,
        session_manager: Optional[MultimodalSessionManager] = None,
        vision_service: Optional[VisionService] = None,
        reasoning_engine: Optional[MultimodalReasoningEngine] = None,
        response_generator: Optional[MultimodalResponseGenerator] = None,
        evidence_validator: Optional[MultimodalEvidenceValidator] = None,
        confidence_assessor: Optional[MultimodalConfidenceAssessor] = None,
        ambiguity_detector: Optional[MultimodalAmbiguityDetector] = None,
        missing_info_detector: Optional[MultimodalMissingInformationDetector] = None,
        fallback_handler: Optional[MultimodalFallbackHandler] = None,
        followup_resolver: Optional[MultimodalFollowUpResolver] = None,
    ) -> None:
        """Initialize all multimodal subsystem components."""
        self.session_manager = session_manager or MultimodalSessionManager()
        self.vision_service = vision_service or VisionService()
        self.reasoning_engine = reasoning_engine or MultimodalReasoningEngine()
        self.response_generator = response_generator or MultimodalResponseGenerator()
        self.evidence_validator = evidence_validator or MultimodalEvidenceValidator()
        self.confidence_assessor = confidence_assessor or MultimodalConfidenceAssessor()
        self.ambiguity_detector = ambiguity_detector or MultimodalAmbiguityDetector()
        self.missing_info_detector = missing_info_detector or MultimodalMissingInformationDetector()
        self.fallback_handler = fallback_handler or MultimodalFallbackHandler()
        self.followup_resolver = followup_resolver or MultimodalFollowUpResolver()

        # Orchestrator pipeline
        self.orchestrator = create_multimodal_pipeline(
            vision_service=self.vision_service,
            reasoning_engine=self.reasoning_engine,
            response_generator=self.response_generator,
        )

    def get_or_create_session(self, session_id: str) -> MultimodalConversationSession:
        """Get or initialize a bounded conversation session."""
        return self.session_manager.get_or_create_session(session_id)

    def clear_session(self, session_id: str) -> bool:
        """Reset conversation session history for the given session ID."""
        return self.session_manager.delete_session(session_id)

    def process_interaction(
        self,
        session_id: str,
        query: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        file_name: Optional[str] = None,
    ) -> MultimodalUIResult:
        """Process a single multimodal user interaction through the full pipeline.

        Args:
            session_id: Active session identifier.
            query: User text input (or None for image-only).
            image_bytes: Raw uploaded image bytes (or None for text-only).
            file_name: Optional name of the uploaded image file.

        Returns:
            MultimodalUIResult: Structured result with response, evidence, and safety diagnostics.
        """
        clean_query = query.strip() if query and query.strip() else None

        if clean_query is None and image_bytes is None:
            raise ValueError("Interaction requires at least a query string or an image upload.")

        session = self.get_or_create_session(session_id)

        # 1. Ingest image if provided
        artifact: Optional[ImageArtifact] = None
        images: List[ImageArtifact] = []
        if image_bytes is not None:
            artifact = ingest_image(image_bytes, file_name=file_name)
            images = [artifact]

        # 2. Resolve follow-up query if session has prior turns
        followup_res: Optional[FollowUpResolution] = None
        effective_query = clean_query
        if session.get_turn_count() > 0 and clean_query:
            try:
                followup_res = self.followup_resolver.resolve(
                    query=clean_query,
                    session=session,
                    active_artifact=artifact,
                )
                effective_query = followup_res.resolved_query
            except Exception:
                effective_query = clean_query

        # 3. Construct canonical MultimodalRequest
        request = MultimodalRequest(
            query=effective_query,
            images=images,
            session_id=session_id,
        )

        # 4. Perform visual feature extraction if image present
        visual_output: Optional[Dict[str, Any]] = None
        if artifact is not None:
            try:
                visual_output = self.vision_service.analyze(artifact)
            except Exception:
                visual_output = None

        # 5. Build unified multimodal context
        context = build_multimodal_context(
            request=request,
            artifact=artifact,
            visual_output=visual_output,
        )

        # 6. Evaluate Ambiguity, Missing Information, Evidence, and Confidence
        ambiguity_res = self.ambiguity_detector.assess_ambiguity(
            query=effective_query or "",
            context=context,
        )

        # Assess missing information
        missing_info_res = assess_multimodal_missing_information(
            request=request,
            context=context,
        )

        # Validate evidence and assess confidence from context items
        evidence_items = context.evidence_items if (context and context.evidence_items) else None
        evidence_val = self.evidence_validator.validate_evidence(evidence=evidence_items) if evidence_items else None
        conf_res = self.confidence_assessor.assess(evidence=evidence_items) if evidence_items else None

        # 7. Evaluate Safe Fallback
        fallback_decision = self.fallback_handler.evaluate(
            query=effective_query,
            context=context,
            request=request,
            evidence_items=evidence_items,
            evidence_validation=evidence_val,
            confidence_assessment=conf_res,
            ambiguity_assessment=ambiguity_res,
            missing_info_assessment=missing_info_res,
        )

        # 8. Generate response according to fallback decision
        if fallback_decision.action != FallbackAction.PROCEED:
            response = fallback_decision.to_multimodal_response(
                query=effective_query or "",
                session_id=session_id,
                modality=request.modality,
                evidence_items=evidence_items,
            )
        else:
            # Full reasoning execution
            try:
                reasoning_result = self.reasoning_engine.reason(context)
                response = self.response_generator.generate_response(reasoning_result, context=context)
                # Re-validate with response-level visual evidence
                if response.visual_evidence:
                    evidence_val = self.evidence_validator.validate_evidence(evidence=response.visual_evidence)
                    conf_res = self.confidence_assessor.assess(evidence=response.visual_evidence)
            except Exception as exc:
                fallback_decision = FallbackDecision(
                    action=FallbackAction.SAFE_LIMITED_RESPONSE,
                    should_proceed=False,
                    reason=f"Reasoning engine encountered an unexpected error: {str(exc)}",
                    safe_message=f"I encountered a processing limitation while analyzing your request: {str(exc)}",
                )
                response = fallback_decision.to_multimodal_response(
                    query=effective_query or "",
                    session_id=session_id,
                    modality=request.modality,
                    evidence_items=evidence_items,
                )

        # 9. Record completed interaction turn into conversation session
        session.add_turn(
            query=clean_query or "",
            response=response,
            modality=request.modality,
            context=context,
        )

        turn_idx = session.total_turns_count

        return MultimodalUIResult(
            response=response,
            fallback_decision=fallback_decision,
            evidence_validation=evidence_val,
            confidence_assessment=conf_res,
            ambiguity_assessment=ambiguity_res,
            missing_info_assessment=missing_info_res,
            followup_resolution=followup_res,
            turn_index=turn_idx,
            modality=request.modality,
        )
