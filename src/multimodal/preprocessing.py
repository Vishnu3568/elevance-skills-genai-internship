"""Image Preprocessing and Normalization Layer (Phase 5 — Day 25 Step 2).

Safely decodes, normalizes, and validates incoming ImageArtifact instances
for downstream vision processing without mutating the original artifact.
"""

from dataclasses import dataclass, field
import io
import math
from typing import Any, Dict, Optional, Tuple

from PIL import Image, UnidentifiedImageError

from .models import (
    ImageArtifact,
    MAX_IMAGE_DIMENSION,
    MIN_IMAGE_DIMENSION,
    validate_image_artifact,
)


# -----------------------------------------------------------------------------
# Preprocessed Image Container
# -----------------------------------------------------------------------------

@dataclass
class PreprocessedImage:
    """Represents a normalized, decoded image prepared for downstream vision models."""

    artifact: ImageArtifact
    pil_image: Image.Image
    width: int
    height: int
    mode: str
    channels: int
    format: str
    mime_type: str
    aspect_ratio: float
    is_resized: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_bytes(self, format_override: Optional[str] = None, quality: int = 95) -> bytes:
        """Serialize the normalized PIL image back into bytes.

        Args:
            format_override: Optional format ('JPEG', 'PNG', 'WEBP'). Defaults to artifact format.
            quality: Compression quality for lossy formats (JPEG, WEBP).

        Returns:
            bytes: Encoded image bytes.
        """
        fmt = (format_override or self.format).upper()
        if fmt == "JPG":
            fmt = "JPEG"

        buf = io.BytesIO()
        save_kwargs: Dict[str, Any] = {"format": fmt}
        if fmt in ("JPEG", "WEBP"):
            save_kwargs["quality"] = quality

        self.pil_image.save(buf, **save_kwargs)
        return buf.getvalue()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize preprocessed image metadata to dictionary (excluding raw PIL object)."""
        return {
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "channels": self.channels,
            "format": self.format,
            "mime_type": self.mime_type,
            "aspect_ratio": self.aspect_ratio,
            "is_resized": self.is_resized,
            "metadata": dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# Image Preprocessor
# -----------------------------------------------------------------------------

class ImagePreprocessor:
    """Deterministic image preprocessing and normalization service."""

    def __init__(
        self,
        target_mode: str = "RGB",
        max_target_dimension: int = 2048,
        background_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Initialize image preprocessor with normalization parameters.

        Args:
            target_mode: Target color mode (default "RGB").
            max_target_dimension: Maximum allowed dimension (downsamples if exceeded).
            background_color: Background color for alpha compositing (default white).
        """
        if target_mode != "RGB":
            raise ValueError(f"Currently only 'RGB' target_mode is supported, got '{target_mode}'")

        if max_target_dimension < MIN_IMAGE_DIMENSION or max_target_dimension > MAX_IMAGE_DIMENSION:
            raise ValueError(
                f"max_target_dimension ({max_target_dimension}) must be within "
                f"[{MIN_IMAGE_DIMENSION}, {MAX_IMAGE_DIMENSION}]."
            )

        self.target_mode = target_mode
        self.max_target_dimension = max_target_dimension
        self.background_color = background_color

    def preprocess(self, artifact: ImageArtifact) -> PreprocessedImage:
        """Safely preprocess an ImageArtifact for downstream vision processing.

        Args:
            artifact: Canonical ImageArtifact to normalize.

        Returns:
            PreprocessedImage: Decoded, color-normalized, and dimension-bounded container.

        Raises:
            TypeError: If artifact is not an ImageArtifact.
            ValueError: If image bytes are corrupted, unreadable, or invalid.
        """
        if not isinstance(artifact, ImageArtifact):
            raise TypeError(f"Expected ImageArtifact instance, got {type(artifact).__name__}")

        # Validate input artifact contract invariants
        validate_image_artifact(artifact)

        # Safely decode bytes into a new PIL Image instance
        try:
            raw_pil = Image.open(io.BytesIO(artifact.data))
            # Load pixel data into memory so we can manipulate it safely
            raw_pil.load()
        except (UnidentifiedImageError, SyntaxError) as exc:
            raise ValueError(f"Failed to decode image artifact bytes: {exc}") from exc
        except (OSError, IOError) as exc:
            raise ValueError(f"Corrupted or truncated image artifact bytes: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Unexpected error decoding image artifact: {exc}") from exc

        orig_w, orig_h = raw_pil.size
        orig_mode = raw_pil.mode

        # 1. Color mode normalization (standardize transparency and color profiles to RGB)
        normalized_pil = self._normalize_color_mode(raw_pil)

        # 2. Dimension bounding (downscale if larger than max_target_dimension while preserving aspect ratio)
        resample_filter = getattr(Image, "Resampling", Image).LANCZOS
        is_resized = False
        curr_w, curr_h = normalized_pil.size

        if curr_w > self.max_target_dimension or curr_h > self.max_target_dimension:
            scale = min(self.max_target_dimension / curr_w, self.max_target_dimension / curr_h)
            new_w = max(MIN_IMAGE_DIMENSION, int(math.floor(curr_w * scale)))
            new_h = max(MIN_IMAGE_DIMENSION, int(math.floor(curr_h * scale)))
            normalized_pil = normalized_pil.resize((new_w, new_h), resample=resample_filter)
            curr_w, curr_h = normalized_pil.size
            is_resized = True

        aspect_ratio = round(curr_w / curr_h, 4) if curr_h > 0 else 1.0

        metadata: Dict[str, Any] = {
            "original_width": orig_w,
            "original_height": orig_h,
            "original_mode": orig_mode,
            "original_size_bytes": len(artifact.data),
            "file_name": artifact.file_name,
            "mime_type": artifact.mime_type,
            "format": artifact.format,
            "is_resized": is_resized,
        }

        return PreprocessedImage(
            artifact=artifact,
            pil_image=normalized_pil,
            width=curr_w,
            height=curr_h,
            mode=self.target_mode,
            channels=3,
            format=artifact.format,
            mime_type=artifact.mime_type,
            aspect_ratio=aspect_ratio,
            is_resized=is_resized,
            metadata=metadata,
        )

    def _normalize_color_mode(self, img: Image.Image) -> Image.Image:
        """Normalize PIL image color mode to standard RGB with proper background compositing."""
        if img.mode == "RGB":
            # Return copy to avoid altering original PIL reference
            return img.copy()

        # Handle alpha channels (RGBA, LA) or palette with transparency (P)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba_img = img.convert("RGBA")
            bg = Image.new("RGB", rgba_img.size, self.background_color)
            # Alpha compositing using the alpha band as mask
            bg.paste(rgba_img, mask=rgba_img.split()[3])
            return bg

        # Handle grayscale (L, 1) or other color profiles (CMYK, YCbCr)
        return img.convert("RGB")


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------

def preprocess_image(
    artifact: ImageArtifact,
    max_target_dimension: int = 2048,
    target_mode: str = "RGB",
) -> PreprocessedImage:
    """Preprocess and normalize an ImageArtifact using standard parameters.

    Args:
        artifact: Canonical ImageArtifact.
        max_target_dimension: Maximum allowed width or height (default 2048).
        target_mode: Target color mode (default 'RGB').

    Returns:
        PreprocessedImage: Validated and normalized preprocessed image.
    """
    preprocessor = ImagePreprocessor(
        target_mode=target_mode,
        max_target_dimension=max_target_dimension,
    )
    return preprocessor.preprocess(artifact)
