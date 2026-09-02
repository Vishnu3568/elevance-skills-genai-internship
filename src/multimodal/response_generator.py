"""Evidence-Aware Multimodal Response Generation (Phase 5 — Day 26 Step 3).

Transforms structured ReasoningResult instances and MultimodalContext into
canonical, evidence-grounded MultimodalResponse contracts while distinguishing
raw evidence from interpretation, preserving provenance, and handling insufficient evidence safely.
"""

from typing import Any, Dict, List, Optional

from .context import MultimodalContext, UnifiedEvidenceItem
from .models import (
    ModalityType,
    MultimodalResponse,
    VisualEvidenceItem,
    validate_multimodal_response,
)
from .reasoning import ReasoningResult


# -----------------------------------------------------------------------------
# Response Generator Service
# -----------------------------------------------------------------------------

class MultimodalResponseGenerator:
    """Generates canonical MultimodalResponse contracts from ReasoningResult instances."""

    def __init__(self, default_confidence_threshold: float = 0.5) -> None:
        """Initialize response generator.

        Args:
            default_confidence_threshold: Minimum confidence score below which
                                          insufficient-evidence warning is emitted.
        """
        self.confidence_threshold = default_confidence_threshold

    def generate_response(
        self,
        reasoning_result: ReasoningResult,
        context: Optional[MultimodalContext] = None,
    ) -> MultimodalResponse:
        """Transform a ReasoningResult into a canonical MultimodalResponse.

        Args:
            reasoning_result: Structured deduction from the reasoning engine.
            context: Optional original MultimodalContext for additional provenance.

        Returns:
            MultimodalResponse: Structured, validated multimodal response.

        Raises:
            TypeError: If reasoning_result is not a ReasoningResult.
            ValueError: If session_id or required fields are malformed.
        """
        if not isinstance(reasoning_result, ReasoningResult):
            raise TypeError(
                f"Expected ReasoningResult instance, got {type(reasoning_result).__name__}"
            )

        modality = reasoning_result.modality
        session_id = reasoning_result.session_id
        query_str = reasoning_result.interpreted_query or ""
        confidence = reasoning_result.confidence

        # 1. Separate and map evidence
        visual_evidence: List[VisualEvidenceItem] = []
        text_evidence: List[str] = list(reasoning_result.textual_evidence)
        combined_evidence: List[UnifiedEvidenceItem] = list(reasoning_result.combined_evidence)

        # Modality-aware evidence filtering
        if modality == ModalityType.TEXT_ONLY:
            visual_evidence = []
        elif modality == ModalityType.IMAGE_ONLY:
            visual_evidence = list(reasoning_result.visual_evidence)
            text_evidence = []
        elif modality == ModalityType.TEXT_AND_IMAGE:
            visual_evidence = list(reasoning_result.visual_evidence)

        # 2. Extract and preserve evidence provenance in image_metadata
        metadata: Dict[str, Any] = dict(reasoning_result.metadata.get("visual_attributes", {}))
        provenance_records: List[Dict[str, Any]] = []

        for ev in combined_evidence:
            if ev.provenance:
                provenance_records.append(ev.provenance.to_dict())

        metadata["provenance_records"] = provenance_records
        metadata["confidence"] = confidence
        metadata["engine_name"] = reasoning_result.engine_name

        # 3. Check for insufficient evidence
        is_insufficient = False
        warning_msg: Optional[str] = None

        if confidence < self.confidence_threshold:
            is_insufficient = True
            warning_msg = (
                f"Low confidence ({confidence:.2f}): insufficient evidence available "
                f"in provided context to formulate an unconstrained conclusion."
            )
        elif modality == ModalityType.TEXT_AND_IMAGE and (len(visual_evidence) == 0 or len(text_evidence) == 0):
            is_insufficient = True
            warning_msg = (
                "Joint text and image context is incomplete: missing either visual or textual evidence."
            )
        elif modality == ModalityType.IMAGE_ONLY and len(visual_evidence) == 0:
            is_insufficient = True
            warning_msg = "Image artifact contains no observable visual evidence."
        elif modality == ModalityType.TEXT_ONLY and len(text_evidence) == 0:
            is_insufficient = True
            warning_msg = "Textual query contains no substantive context."

        # 4. Formulate answer text and formatted markdown
        answer = reasoning_result.reasoning_summary
        if is_insufficient and warning_msg:
            answer += f"\n\n[Caution: Evidence Limited] {warning_msg}"

        formatted_md = self._format_markdown_response(
            modality=modality,
            answer=answer,
            reasoning_steps=reasoning_result.reasoning_steps,
            text_evidence=text_evidence,
            visual_evidence=visual_evidence,
            confidence=confidence,
            warning_msg=warning_msg,
        )

        response = MultimodalResponse(
            query=query_str,
            answer=answer,
            modality=modality,
            visual_evidence=visual_evidence,
            image_metadata=metadata if metadata else None,
            grounded=not is_insufficient,
            warning_message=warning_msg,
            formatted_markdown=formatted_md,
            session_id=session_id,
        )

        validate_multimodal_response(response)
        return response

    def _format_markdown_response(
        self,
        modality: ModalityType,
        answer: str,
        reasoning_steps: List[str],
        text_evidence: List[str],
        visual_evidence: List[VisualEvidenceItem],
        confidence: float,
        warning_msg: Optional[str] = None,
    ) -> str:
        """Construct structured markdown clearly delineating interpretation from raw evidence."""
        header_title = {
            ModalityType.TEXT_ONLY: "Textual Analysis",
            ModalityType.IMAGE_ONLY: "Visual Inspection",
            ModalityType.TEXT_AND_IMAGE: "Multimodal Analysis",
        }.get(modality, "Response")

        md_parts: List[str] = [f"### {header_title}"]

        if warning_msg:
            md_parts.append(f"> ⚠️ **Evidence Warning:** {warning_msg}")

        md_parts.append(f"#### Synthesis\n{answer}")

        # Distinguish reasoning trace from raw evidence
        if reasoning_steps:
            steps_lines = "\n".join(f"- {step}" for step in reasoning_steps)
            md_parts.append(f"#### Reasoning Trace\n{steps_lines}")

        # Evidentiary Basis Section
        ev_sections: List[str] = []
        if text_evidence:
            text_lines = "\n".join(f"- *\"{txt}\"*" for txt in text_evidence)
            ev_sections.append(f"**Textual Evidence:**\n{text_lines}")

        if visual_evidence:
            vis_lines = "\n".join(
                f"- **[{item.region_label or 'Global'}]:** {item.description} (conf: {item.confidence:.2f})"
                for item in visual_evidence
            )
            ev_sections.append(f"**Visual Evidence:**\n{vis_lines}")

        if ev_sections:
            md_parts.append("#### Evidentiary Basis\n" + "\n\n".join(ev_sections))

        md_parts.append(f"**Confidence Assessment:** {confidence:.2f}")
        return "\n\n".join(md_parts)


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------

def generate_multimodal_response(
    reasoning_result: ReasoningResult,
    context: Optional[MultimodalContext] = None,
    confidence_threshold: float = 0.5,
) -> MultimodalResponse:
    """Generate a canonical MultimodalResponse using standard generator parameters."""
    generator = MultimodalResponseGenerator(default_confidence_threshold=confidence_threshold)
    return generator.generate_response(reasoning_result=reasoning_result, context=context)
