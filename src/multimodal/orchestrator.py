"""Multimodal Orchestrator and Pipeline Architecture (Phase 5 — Day 24 & Day 25).

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
        vision_service: Optional[Any] = None,
        reasoning_engine: Optional[Any] = None,
        response_generator: Optional[Any] = None,
        image_understanding_hook: Optional[ImageUnderstandingHook] = None,
        reasoning_engine_hook: Optional[ReasoningEngineHook] = None,
        ambiguity_detector_hook: Optional[AmbiguityDetectorHook] = None,
        grounding_validator_hook: Optional[GroundingValidatorHook] = None,
        enable_reasoning_pipeline: bool = False,
    ) -> None:
        """Initialize the orchestrator with optional downstream component hooks.

        Args:
            vision_service: Optional VisionService instance (Day 25 vision layer).
            reasoning_engine: Optional MultimodalReasoningEngine instance (Day 26 reasoning layer).
            response_generator: Optional MultimodalResponseGenerator instance (Day 26 response layer).
            image_understanding_hook: Hook for Day 25 visual understanding & feature extraction.
            reasoning_engine_hook: Hook for Day 26 joint cross-modal reasoning.
            ambiguity_detector_hook: Hook for Day 28 visual ambiguity detection.
            grounding_validator_hook: Hook for Day 28 evidence validation and grounding.
            enable_reasoning_pipeline: If True, wires up the complete Day 26 reasoning pipeline.
        """
        self.vision_service = vision_service
        self.reasoning_engine = reasoning_engine
        self.response_generator = response_generator
        self.enable_reasoning_pipeline = enable_reasoning_pipeline or (reasoning_engine is not None)

        self.image_understanding_hook = image_understanding_hook
        if self.vision_service is not None and self.image_understanding_hook is None:
            if hasattr(self.vision_service, "as_hook"):
                self.image_understanding_hook = self.vision_service.as_hook()

        self.reasoning_engine_hook = reasoning_engine_hook
        if self.reasoning_engine_hook is None and self.enable_reasoning_pipeline:
            self._setup_default_reasoning_pipeline()

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

    def process_image_file(
        self,
        file_path: Any,
        query: Optional[str] = None,
        session_id: str = "default_multimodal_session",
    ) -> MultimodalResponse:
        """Ingest an image file from disk and process it end-to-end through the pipeline."""
        from .ingestion import ingest_image
        artifact = ingest_image(file_path)
        return self.process_query(query=query, images=[artifact], session_id=session_id)

    def process_image_bytes(
        self,
        data: bytes,
        query: Optional[str] = None,
        file_name: Optional[str] = None,
        session_id: str = "default_multimodal_session",
    ) -> MultimodalResponse:
        """Ingest raw image bytes and process them end-to-end through the pipeline."""
        from .ingestion import ingest_image
        artifact = ingest_image(data, file_name=file_name)
        return self.process_query(query=query, images=[artifact], session_id=session_id)

    def _setup_default_reasoning_pipeline(self) -> None:
        """Configure default Day 25 vision, Day 26 context, reasoning, and response generation."""
        from .context import build_multimodal_context
        from .reasoning import MultimodalReasoningEngine
        from .response_generator import MultimodalResponseGenerator
        from .vision import VisionService

        if self.vision_service is None:
            self.vision_service = VisionService()
        if self.reasoning_engine is None:
            self.reasoning_engine = MultimodalReasoningEngine()
        if self.response_generator is None:
            self.response_generator = MultimodalResponseGenerator()

        def _pipeline_hook(request: MultimodalRequest) -> MultimodalResponse:
            first_img = request.images[0] if request.images else None
            visual_output = None
            if first_img is not None and self.vision_service is not None:
                visual_output = self.vision_service.analyze(first_img)

            context = build_multimodal_context(
                request=request,
                artifact=first_img,
                visual_output=visual_output,
            )

            result = self.reasoning_engine.reason(context)
            if self.response_generator is not None:
                return self.response_generator.generate_response(result, context=context)
            return result.to_multimodal_response()

        self.reasoning_engine_hook = _pipeline_hook

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
        metadata: Dict[str, Any] = {}
        first_img: Optional[ImageArtifact] = request.images[0] if image_count > 0 else None

        if first_img:
            metadata = {
                "format": first_img.format,
                "width": first_img.width,
                "height": first_img.height,
                "mime_type": first_img.mime_type,
            }

        evidence: List[VisualEvidenceItem] = []

        # If vision service is connected, use it for structured visual understanding
        if self.vision_service is not None and first_img is not None:
            visual_output = self.vision_service.analyze(first_img)
            evidence = visual_output.to_visual_evidence_items()
            metadata.update(visual_output.visual_attributes)
            answer = f"[Visual Understanding] {visual_output.scene_description}"
        elif self.image_understanding_hook is not None and first_img is not None:
            hook_res = self.image_understanding_hook(first_img)
            desc = hook_res.get("scene_description", f"Observed {metadata.get('format')} image")
            evidence = [
                VisualEvidenceItem(
                    description=desc,
                    region_label="Global Scene",
                    confidence=float(hook_res.get("confidence", 1.0)),
                )
            ]
            answer = f"[Visual Understanding] {desc}"
        else:
            answer = (
                f"[Image-Only Pipeline] Received {image_count} image(s). "
                f"Primary image: {metadata.get('format', 'N/A')} ({metadata.get('width', 0)}x{metadata.get('height', 0)}px)."
            )
            if first_img:
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
        """Handle joint request visual feature extraction (cross-modal reasoning belongs to Day 26)."""
        query_str = request.query or ""
        image_count = len(request.images)
        metadata: Dict[str, Any] = {}
        first_img: Optional[ImageArtifact] = request.images[0] if image_count > 0 else None

        if first_img:
            metadata = {
                "format": first_img.format,
                "width": first_img.width,
                "height": first_img.height,
                "mime_type": first_img.mime_type,
            }

        evidence: List[VisualEvidenceItem] = []

        # If vision service is connected, extract structured visual information only
        if self.vision_service is not None and first_img is not None:
            visual_output = self.vision_service.analyze(first_img)
            evidence = visual_output.to_visual_evidence_items()
            metadata.update(visual_output.visual_attributes)
            answer = (
                f"[Visual Understanding] Extracted visual features for query '{query_str}': "
                f"{visual_output.scene_description}"
            )
        elif self.image_understanding_hook is not None and first_img is not None:
            hook_res = self.image_understanding_hook(first_img)
            desc = hook_res.get("scene_description", f"Observed {metadata.get('format')} image")
            evidence = [
                VisualEvidenceItem(
                    description=desc,
                    region_label="Visual Context",
                    confidence=float(hook_res.get("confidence", 1.0)),
                )
            ]
            answer = f"[Visual Understanding] Extracted visual features for query '{query_str}': {desc}"
        else:
            answer = (
                f"[Cross-Modal Pipeline] Analyzed query '{query_str}' against {image_count} visual artifact(s)."
            )
            if first_img:
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


# -----------------------------------------------------------------------------
# Pipeline Factory Function
# -----------------------------------------------------------------------------

def create_multimodal_pipeline(
    vision_service: Optional[Any] = None,
    reasoning_engine: Optional[Any] = None,
    response_generator: Optional[Any] = None,
    ambiguity_detector_hook: Optional[AmbiguityDetectorHook] = None,
    grounding_validator_hook: Optional[GroundingValidatorHook] = None,
) -> MultimodalOrchestrator:
    """Create a fully integrated Day 26 multimodal reasoning pipeline orchestrator.

    Integrates:
      User Input / Image -> Ingestion -> Preprocessing -> Vision Understanding ->
      Unified Multimodal Context -> Multimodal Reasoning -> Multimodal Response

    Args:
        vision_service: Optional VisionService (defaults to VisionService()).
        reasoning_engine: Optional ReasoningEngine (defaults to MultimodalReasoningEngine()).
        response_generator: Optional ResponseGenerator (defaults to MultimodalResponseGenerator()).
        ambiguity_detector_hook: Optional hook for Day 28 ambiguity detection.
        grounding_validator_hook: Optional hook for Day 28 grounding validation.

    Returns:
        MultimodalOrchestrator: Pre-configured, fully integrated multimodal reasoning pipeline.
    """
    from .reasoning import MultimodalReasoningEngine
    from .response_generator import MultimodalResponseGenerator
    from .vision import VisionService

    return MultimodalOrchestrator(
        vision_service=vision_service or VisionService(),
        reasoning_engine=reasoning_engine or MultimodalReasoningEngine(),
        response_generator=response_generator or MultimodalResponseGenerator(),
        ambiguity_detector_hook=ambiguity_detector_hook,
        grounding_validator_hook=grounding_validator_hook,
        enable_reasoning_pipeline=True,
    )
