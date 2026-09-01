"""Image Ingestion Layer for Multimodal AI Assistant (Phase 5 — Day 25 Step 1).

Provides safe, validated ingestion of images from local file paths, raw bytes,
or in-memory binary streams into the canonical ImageArtifact contract without
modifying the underlying source or invoking vision models.
"""

import base64
import io
import os
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Union

from PIL import Image, UnidentifiedImageError

from .models import (
    ImageArtifact,
    MAX_IMAGE_SIZE_BYTES,
    SUPPORTED_MIME_TYPES,
)


# Canonical format to MIME mapping
FORMAT_TO_MIME: Dict[str, str] = {
    "JPEG": "image/jpeg",
    "JPG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


def ingest_image(
    source: Union[str, Path, bytes, BinaryIO],
    file_name: Optional[str] = None,
    compute_base64: bool = True,
) -> ImageArtifact:
    """Safely ingest an image from a path, raw bytes, or binary stream into an ImageArtifact.

    Args:
        source: File path (str/Path), raw bytes, or readable binary stream.
        file_name: Optional explicit file name or identifier.
        compute_base64: Whether to generate a base64 string representation (default True).

    Returns:
        ImageArtifact: Validated canonical image artifact contract.

    Raises:
        FileNotFoundError: If a file path is provided but does not exist.
        TypeError: If the source is of an unsupported type.
        ValueError: If the image is empty, corrupted, oversized, out of bounds,
                    or in an unsupported format.
    """
    resolved_bytes, resolved_file_name = _resolve_source_bytes(source, file_name)

    if len(resolved_bytes) == 0:
        raise ValueError("Cannot ingest empty image data (0 bytes).")

    if len(resolved_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise ValueError(
            f"Image size ({len(resolved_bytes)} bytes) exceeds maximum permitted limit of "
            f"{MAX_IMAGE_SIZE_BYTES} bytes (10 MB)."
        )

    # Safely inspect image using PIL in memory
    try:
        with Image.open(io.BytesIO(resolved_bytes)) as img:
            raw_format = img.format
            width, height = img.size
            # Verify file stream integrity (catches truncated data / broken chunks)
            img.verify()
    except (UnidentifiedImageError, SyntaxError) as exc:
        raise ValueError(f"Unidentified or corrupted image format: {exc}") from exc
    except (OSError, IOError) as exc:
        raise ValueError(f"Corrupted or unreadable image stream: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to inspect image: {exc}") from exc

    if not raw_format:
        raise ValueError("Could not determine image format from header.")

    norm_format = raw_format.strip().upper()
    if norm_format not in FORMAT_TO_MIME:
        raise ValueError(
            f"Unsupported image format '{raw_format}'. Supported formats: "
            f"{sorted(list(FORMAT_TO_MIME.keys()))}."
        )

    mime_type = FORMAT_TO_MIME[norm_format]

    # Generate base64 string if requested
    base64_str: Optional[str] = None
    if compute_base64:
        base64_str = base64.b64encode(resolved_bytes).decode("ascii")

    # Construct canonical ImageArtifact (triggers contract validation for dimensions, bounds, etc.)
    return ImageArtifact(
        data=resolved_bytes,
        mime_type=mime_type,
        width=width,
        height=height,
        format=norm_format,
        base64_str=base64_str,
        file_name=resolved_file_name,
    )


def _resolve_source_bytes(
    source: Union[str, Path, bytes, BinaryIO],
    file_name: Optional[str],
) -> tuple[bytes, Optional[str]]:
    """Extract raw byte content and file name from supported source types without altering source."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found at '{path}'")
        if not path.is_file():
            raise ValueError(f"Path '{path}' is a directory, not a regular image file.")

        with open(path, "rb") as f:
            data = f.read()
        name = file_name or path.name
        return data, name

    elif isinstance(source, bytes):
        return source, file_name

    elif hasattr(source, "read"):
        # Handle BinaryIO / BytesIO / UploadedFile
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass

        if hasattr(source, "getvalue"):
            data = source.getvalue()
        else:
            data = source.read()

        name = file_name or getattr(source, "name", None)
        return data, name

    raise TypeError(
        f"Unsupported source type '{type(source).__name__}'. "
        f"Expected str, Path, bytes, or binary stream."
    )


class ImageIngestionService:
    """Service wrapper for standard image ingestion operations."""

    def __init__(self, compute_base64: bool = True) -> None:
        self.compute_base64 = compute_base64

    def ingest(
        self,
        source: Union[str, Path, bytes, BinaryIO],
        file_name: Optional[str] = None,
    ) -> ImageArtifact:
        """Ingest image using instance settings."""
        return ingest_image(source=source, file_name=file_name, compute_base64=self.compute_base64)

    def ingest_from_file(self, file_path: Union[str, Path]) -> ImageArtifact:
        """Ingest directly from local file path."""
        return self.ingest(source=file_path)

    def ingest_from_bytes(self, data: bytes, file_name: Optional[str] = None) -> ImageArtifact:
        """Ingest directly from raw bytes."""
        return self.ingest(source=data, file_name=file_name)
