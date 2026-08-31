"""Multimodal Orchestrator and Pipeline Architecture (Phase 5 — Day 24).

Establishes the high-level orchestration interface, routing logic, and integration
hooks for the Multimodal AI Assistant. Coordinates text-only, image-only, and
text+image interaction flows across Days 25–29 downstream components.
"""

from typing import Any, Callable, Dict, List, Optional, Union

from .models import (
    AmbiguityDetails,
    ImageArtifact,
    ModalityType,
    MultimodalRequest,
    MultimodalResponse,
    VisualEvidenceItem,
    validate_multimodal_request,
)


# -----------------------------------------------------------------------------
# Component Hook Types (Extension Points for Days 25–28)
# -----------------------------------------------------------------------------

ImageUnderstandingHook = Callable[[ImageArtifact], Dict[str, Any]]
ReasoningEngineHook = Callable[[MultimodalRequest], MultimodalResponse]
AmbiguityDetectorHook = Callable[[MultimodalRequest], AmbiguityDetails]
GroundingValidatorHook = Callable[[MultimodalResponse], MultimodalResponse]


# -----------------------------------------------------------------------------
# Multimodal Orchestrator Service
# -----------------------------------------------------------------------------

class MultimodalOrchestrator:
    """Central orchestrator for multimodal request lifecycle and modality routing.

    Serves as the architectural boundary for Phase 5, coordinating input validation,
    modality detection, cross-modal context assembly, and downstream component
    dispatch (image understanding, reasoning, grounding, session management).
    """

    def __init__(
        self,
        image_understanding_hook: Optional[ImageUnderstandingHook] = None,
        reasoning_engine_hook: Optional[ReasoningEngineHook] = None,
        ambiguity_detector_hook: Optional[AmbiguityDetectorHook] = None,
        grounding_validator_hook: Optional[GroundingValidatorHook] = None,
    ) -> None:
        """Initialize the orchestrator with optional downstream component hooks.

        Args:
            image_understanding_hook: Hook for Day 25 visual understanding & feature extraction.
            reasoning_engine_hook: Hook for Day 26 joint cross-modal reasoning.
            ambiguity_detector_hook: Hook for Day 28 visual ambiguity detection.
            grounding_validator_hook: Hook for Day 28 evidence validation and grounding.
        """
        self.image_understanding_hook = image_understanding_hook
        self.reasoning_engine_hook = reasoning_engine_hook
        self.ambiguity_detector_hook = ambiguity_detector_hook
        self.grounding_validator_hook = grounding_validator_hook

    def process(self, request: Union[MultimodalRequest, Dict[str, Any]]) -> MultimodalResponse:
        """Process an incoming multimodal request through the orchestration pipeline.

        Pipeline Stages:
            1. Input Coercion & Validation
            2. Modality Detection (TEXT_ONLY | IMAGE_ONLY | TEXT_AND_IMAGE)
            3. Ambiguity & Sufficiency Assessment (Hook)
            4. Modality-Specific Handler Dispatch
            5. Evidence Validation & Grounding (Hook)
            6. Response Packaging

        Args:
            request: Validated MultimodalRequest or raw dictionary payload.

        Returns:
            MultimodalResponse: Structured, grounded response contract.

        Raises:
            TypeError: If request payload is of an unsupported type.
            ValueError: If request contains invalid values or missing inputs.
        """
        # Stage 1: Coerce & Validate Input
        coerced_request = self._coerce_request(request)
        validate_multimodal_request(coerced_request)

        # Stage 2: Modality Detection
        modality = coerced_request.modality

        # Stage 3: Ambiguity Check Hook (Day 28 integration point)
        ambiguity = AmbiguityDetails()
        if self.ambiguity_detector_hook is not None:
            try:
                ambiguity = self.ambiguity_detector_hook(coerced_request)
            except Exception as e:
                ambiguity = AmbiguityDetails(
                    is_ambiguous=False,
                    missing_aspects=[f"Ambiguity check warning: {str(e)}"],
                )

        if ambiguity.is_ambiguous and ambiguity.clarification_question:
            return MultimodalResponse(
                query=coerced_request.query or "",
                answer=ambiguity.clarification_question,
                modality=modality,
                ambiguity=ambiguity,
                grounded=True,
                session_id=coerced_request.session_id,
                formatted_markdown=f"**Clarification Required:** {ambiguity.clarification_question}",
            )

        # Stage 4: Downstream Handler Dispatch or Hook Invocation
        if self.reasoning_engine_hook is not None:
            response = self.reasoning_engine_hook(coerced_request)
        else:
            response = self._route_by_modality(coerced_request, modality)

        # Stage 5: Grounding Validation Hook (Day 28 integration point)
        if self.grounding_validator_hook is not None:
            try:
                response = self.grounding_validator_hook(response)
            except Exception as e:
                response.warning_message = f"Grounding validation note: {str(e)}"

        return response

    def process_query(
        self,
        query: Optional[str] = None,
        images: Optional[List[ImageArtifact]] = None,
        session_id: str = "default_multimodal_session",
    ) -> MultimodalResponse:
        """Convenience method for invoking orchestrator with separate query and image args."""
        req = MultimodalRequest(
            query=query,
            images=images if images is not None else [],
            session_id=session_id,
        )
        return self.process(req)

    # -------------------------------------------------------------------------
    # Internal Routing and Default Handlers
    # -------------------------------------------------------------------------

    def _coerce_request(self, payload: Union[MultimodalRequest, Dict[str, Any]]) -> MultimodalRequest:
        """Coerce raw dictionary or validate existing MultimodalRequest."""
        if isinstance(payload, MultimodalRequest):
            return payload
        elif isinstance(payload, dict):
            return MultimodalRequest(
                query=payload.get("query"),
                images=payload.get("images", []),
                session_id=payload.get("session_id", "default_multimodal_session"),
                temperature=payload.get("temperature", 0.1),
                max_output_tokens=payload.get("max_output_tokens"),
                metadata_filters=payload.get("metadata_filters", {}),
            )
        raise TypeError(f"Expected MultimodalRequest or dict, got {type(payload).__name__}")

    def _route_by_modality(self, request: MultimodalRequest, modality: ModalityType) -> MultimodalResponse:
        """Route request based on detected modality."""
        if modality == ModalityType.TEXT_ONLY:
            return self._handle_text_only(request)
        elif modality == ModalityType.IMAGE_ONLY:
            return self._handle_image_only(request)
        elif modality == ModalityType.TEXT_AND_IMAGE:
            return self._handle_text_and_image(request)
        raise ValueError(f"Unknown modality type: {modality}")

    def _handle_text_only(self, request: MultimodalRequest) -> MultimodalResponse:
        """Handle pure text request routing."""
        query_str = request.query or ""
        answer = f"[Text-Only Pipeline] Processed query: '{query_str}'"
        return MultimodalResponse(
            query=query_str,
            answer=answer,
            modality=ModalityType.TEXT_ONLY,
            grounded=True,
            session_id=request.session_id,
            formatted_markdown=f"### Response\n{answer}",
        )

    def _handle_image_only(self, request: MultimodalRequest) -> MultimodalResponse:
        """Handle image-only upload routing (automatic visual breakdown)."""
        image_count = len(request.images)
        metadata = {}
        if image_count > 0:
            first_img = request.images[0]
            metadata = {
                "format": first_img.format,
                "width": first_img.width,
                "height": first_img.height,
                "mime_type": first_img.mime_type,
            }

        answer = (
            f"[Image-Only Pipeline] Received {image_count} image(s). "
            f"Primary image: {metadata.get('format', 'N/A')} ({metadata.get('width', 0)}x{metadata.get('height', 0)}px)."
        )

        evidence = [
            VisualEvidenceItem(
                description=f"Image artifact format {metadata.get('format')}, size {metadata.get('width')}x{metadata.get('height')}",
                region_label="Global Image",
                confidence=1.0,
            )
        ]

        return MultimodalResponse(
            query="",
            answer=answer,
            modality=ModalityType.IMAGE_ONLY,
            visual_evidence=evidence,
            image_metadata=metadata,
            grounded=True,
            session_id=request.session_id,
            formatted_markdown=f"### Visual Inspection\n{answer}",
        )

    def _handle_text_and_image(self, request: MultimodalRequest) -> MultimodalResponse:
        """Handle joint cross-modal request routing."""
        query_str = request.query or ""
        image_count = len(request.images)
        metadata = {}
        if image_count > 0:
            first_img = request.images[0]
            metadata = {
                "format": first_img.format,
                "width": first_img.width,
                "height": first_img.height,
                "mime_type": first_img.mime_type,
            }

        answer = (
            f"[Cross-Modal Pipeline] Analyzed query '{query_str}' against {image_count} visual artifact(s)."
        )

        evidence = [
            VisualEvidenceItem(
                description=f"Visual context for query '{query_str}' from {metadata.get('format', 'image')}",
                region_label="Primary Visual Context",
                confidence=1.0,
            )
        ]

        return MultimodalResponse(
            query=query_str,
            answer=answer,
            modality=ModalityType.TEXT_AND_IMAGE,
            visual_evidence=evidence,
            image_metadata=metadata,
            grounded=True,
            session_id=request.session_id,
            formatted_markdown=f"### Multimodal Analysis\n{answer}",
        )
