"""Vision Model Integration and Visual Understanding Layer (Phase 5 — Day 25 Step 3).

Establishes the vision model abstraction, provider interface, structured visual output
contract, deterministic mock provider, and Gemini vision integration point for
extracting visual information from PreprocessedImage instances.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Callable, Dict, List, Optional, Union

from .models import (
    ImageArtifact,
    VisualEvidenceItem,
)
from .preprocessing import (
    PreprocessedImage,
    preprocess_image,
)


# -----------------------------------------------------------------------------
# Structured Visual Output Contract
# -----------------------------------------------------------------------------

@dataclass
class StructuredVisualOutput:
    """Structured container for visual information extracted by a vision model."""

    scene_description: str
    detected_objects: List[str] = field(default_factory=list)
    visible_text: List[str] = field(default_factory=list)
    visual_attributes: Dict[str, Any] = field(default_factory=dict)
    spatial_observations: List[str] = field(default_factory=list)
    confidence: float = 1.0
    provider_name: str = "deterministic_mock"
    raw_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_visual_evidence_items(self) -> List[VisualEvidenceItem]:
        """Convert structured visual findings into canonical VisualEvidenceItem instances."""
        evidence: List[VisualEvidenceItem] = []

        # 1. Primary scene description
        if self.scene_description.strip():
            evidence.append(
                VisualEvidenceItem(
                    description=self.scene_description.strip(),
                    region_label="Global Scene",
                    confidence=self.confidence,
                )
            )

        # 2. Detected objects
        for obj in self.detected_objects:
            if obj.strip():
                evidence.append(
                    VisualEvidenceItem(
                        description=f"Detected entity/object: {obj.strip()}",
                        region_label="Visual Feature",
                        confidence=self.confidence,
                    )
                )

        # 3. Visible text / OCR / Chart annotations
        for txt in self.visible_text:
            if txt.strip():
                evidence.append(
                    VisualEvidenceItem(
                        description=f"Visible text: '{txt.strip()}'",
                        region_label="Text / Label",
                        confidence=self.confidence,
                    )
                )

        # 4. Spatial observations
        for obs in self.spatial_observations:
            if obs.strip():
                evidence.append(
                    VisualEvidenceItem(
                        description=obs.strip(),
                        region_label="Spatial Region",
                        confidence=self.confidence,
                    )
                )

        return evidence

    def to_dict(self) -> Dict[str, Any]:
        """Serialize structured visual output to dictionary."""
        return asdict(self)


# -----------------------------------------------------------------------------
# Vision Model Provider Interface
# -----------------------------------------------------------------------------

class VisionModelProvider(ABC):
    """Abstract interface for vision-capable models and backends."""

    @abstractmethod
    def analyze(self, image: PreprocessedImage) -> StructuredVisualOutput:
        """Extract structured visual information from a preprocessed image.

        Args:
            image: Normalized PreprocessedImage.

        Returns:
            StructuredVisualOutput: Structured visual observations and features.
        """
        pass


# -----------------------------------------------------------------------------
# Deterministic Mock Provider (Development / Offline / CI Testing)
# -----------------------------------------------------------------------------

class DeterministicMockVisionProvider(VisionModelProvider):
    """Deterministic development and unit-test provider.

    Extracts verified visual attributes, aspect ratios, orientations, and frame
    boundaries without downloading external neural weights or calling cloud APIs.
    """

    def __init__(self, provider_name: str = "deterministic_mock") -> None:
        self.provider_name = provider_name

    def analyze(self, image: PreprocessedImage) -> StructuredVisualOutput:
        """Deterministically inspect preprocessed image properties."""
        if not isinstance(image, PreprocessedImage):
            raise TypeError(f"Expected PreprocessedImage, got {type(image).__name__}")

        w, h = image.width, image.height
        aspect = image.aspect_ratio

        # Determine spatial orientation
        if w > h:
            orientation = "landscape"
        elif h > w:
            orientation = "portrait"
        else:
            orientation = "square"

        scene_desc = (
            f"Image ({image.format}, {w}x{h}px, {orientation}) "
            f"with {image.channels}-channel {image.mode} color space."
        )

        detected_objects = [
            f"visual_canvas_{w}x{h}",
            f"color_profile_{image.mode.lower()}",
            f"boundary_box_{orientation}",
        ]

        spatial_observations = [
            f"Canvas bounds span x=[0, {w}], y=[0, {h}]",
            f"Aspect ratio is {aspect:.4f} ({orientation} layout)",
        ]

        visual_attributes = {
            "width": w,
            "height": h,
            "format": image.format,
            "mode": image.mode,
            "channels": image.channels,
            "aspect_ratio": aspect,
            "orientation": orientation,
            "is_resized": image.is_resized,
            "file_name": image.metadata.get("file_name"),
        }

        # If original artifact has text-like markers or specific metadata, propagate
        visible_text: List[str] = []
        if image.artifact.file_name:
            visible_text.append(f"filename:{image.artifact.file_name}")

        return StructuredVisualOutput(
            scene_description=scene_desc,
            detected_objects=detected_objects,
            visible_text=visible_text,
            visual_attributes=visual_attributes,
            spatial_observations=spatial_observations,
            confidence=1.0,
            provider_name=self.provider_name,
            metadata=dict(image.metadata),
        )


# -----------------------------------------------------------------------------
# Gemini Vision Provider (Production Multimodal Backend Integration Point)
# -----------------------------------------------------------------------------

class GeminiVisionProvider(VisionModelProvider):
    """Production provider using Google Gemini 2.5 Flash for visual information extraction.

    Formats the preprocessed image with a structured visual understanding prompt
    requesting scene description, objects, visible text, and spatial layout in JSON.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash") -> None:
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("GOOGLE_API_KEY", "")
        self.model_name = model_name

    def analyze(self, image: PreprocessedImage) -> StructuredVisualOutput:
        """Call Gemini vision API to extract structured visual information."""
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not configured for GeminiVisionProvider. "
                "Use DeterministicMockVisionProvider for offline/testing environments."
            )

        if not isinstance(image, PreprocessedImage):
            raise TypeError(f"Expected PreprocessedImage, got {type(image).__name__}")

        try:
            # pyrefly: ignore [missing-import]
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
            from langchain_core.messages import HumanMessage  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "langchain-google-genai is required for GeminiVisionProvider."
            ) from exc

        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.1,
            max_retries=2,
        )

        b64_str = image.artifact.base64_str
        if not b64_str:
            import base64
            b64_str = base64.b64encode(image.to_bytes()).decode("ascii")

        mime = image.mime_type or "image/jpeg"
        image_url_str = f"data:{mime};base64,{b64_str}"

        prompt_text = (
            "Analyze the provided image and extract its visual information. "
            "Return a clean JSON object with the following schema:\n"
            "{\n"
            '  "scene_description": "Detailed factual description of visual scene",\n'
            '  "detected_objects": ["list", "of", "detected", "entities"],\n'
            '  "visible_text": ["any", "visible", "text", "labels", "or", "values"],\n'
            '  "visual_attributes": {"key": "value"},\n'
            '  "spatial_observations": ["spatial", "layout", "notes"],\n'
            '  "confidence": 0.95\n'
            "}\n"
            "Do NOT include markdown backticks around the JSON."
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": image_url_str},
            ]
        )

        try:
            response = llm.invoke([message])
            raw_text = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            raise RuntimeError(f"Gemini vision API call failed: {exc}") from exc

        return self._parse_gemini_response(raw_text, image)

    def _parse_gemini_response(self, raw_text: str, image: PreprocessedImage) -> StructuredVisualOutput:
        """Parse Gemini response into StructuredVisualOutput."""
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except Exception:
            # Safe recovery if model returned non-JSON text
            return StructuredVisualOutput(
                scene_description=cleaned,
                detected_objects=[],
                visible_text=[],
                visual_attributes={"width": image.width, "height": image.height},
                spatial_observations=[],
                confidence=0.8,
                provider_name="gemini-2.5-flash",
                raw_response=raw_text,
                metadata=dict(image.metadata),
            )

        return StructuredVisualOutput(
            scene_description=str(parsed.get("scene_description", "")),
            detected_objects=list(parsed.get("detected_objects", [])),
            visible_text=list(parsed.get("visible_text", [])),
            visual_attributes=dict(parsed.get("visual_attributes", {})),
            spatial_observations=list(parsed.get("spatial_observations", [])),
            confidence=float(parsed.get("confidence", 1.0)),
            provider_name="gemini-2.5-flash",
            raw_response=raw_text,
            metadata=dict(image.metadata),
        )


# -----------------------------------------------------------------------------
# High-Level Vision Service
# -----------------------------------------------------------------------------

class VisionService:
    """High-level service coordinating image preprocessing and vision model analysis."""

    def __init__(self, provider: Optional[VisionModelProvider] = None) -> None:
        """Initialize VisionService with a vision provider (defaults to DeterministicMockVisionProvider)."""
        self.provider = provider or DeterministicMockVisionProvider()

    def analyze(self, image: Union[PreprocessedImage, ImageArtifact]) -> StructuredVisualOutput:
        """Analyze an image artifact or preprocessed image.

        Args:
            image: PreprocessedImage or raw ImageArtifact (automatically preprocessed).

        Returns:
            StructuredVisualOutput: Structured visual information.

        Raises:
            TypeError: If input is not PreprocessedImage or ImageArtifact.
            ValueError: If image data is corrupted or cannot be processed.
        """
        if isinstance(image, ImageArtifact):
            preprocessed = preprocess_image(image)
        elif isinstance(image, PreprocessedImage):
            preprocessed = image
        else:
            raise TypeError(
                f"Expected PreprocessedImage or ImageArtifact, got {type(image).__name__}"
            )

        return self.provider.analyze(preprocessed)

    def as_hook(self) -> Callable[[ImageArtifact], Dict[str, Any]]:
        """Return an adapter callable compatible with MultimodalOrchestrator's image_understanding_hook."""
        def hook(artifact: ImageArtifact) -> Dict[str, Any]:
            output = self.analyze(artifact)
            return output.to_dict()

        return hook
