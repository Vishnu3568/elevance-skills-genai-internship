"""Unit tests for Phase 5 Multimodal AI Assistant image preprocessing & validation layer (Day 25 Step 2).
"""

import io
import os
import sys
from typing import Any
import unittest
from PIL import Image

# Ensure src is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from multimodal.models import ImageArtifact
from multimodal.preprocessing import ImagePreprocessor, PreprocessedImage, preprocess_image


class TestImagePreprocessing(unittest.TestCase):
    """Tests for image decoding, color normalization, dimension bounding, and metadata preservation."""

    @classmethod
    def setUpClass(cls):
        # Generate valid test artifacts
        cls.jpeg_bytes = cls._generate_bytes("JPEG", (160, 120), "RGB", color=(200, 50, 50))
        cls.png_bytes = cls._generate_bytes("PNG", (200, 150), "RGB", color=(50, 200, 50))
        cls.webp_bytes = cls._generate_bytes("WEBP", (300, 200), "RGB", color=(50, 50, 200))
        cls.rgba_bytes = cls._generate_bytes("PNG", (100, 100), "RGBA", color=(255, 0, 0, 128))
        cls.gray_bytes = cls._generate_bytes("PNG", (80, 80), "L", color=128)

        cls.jpeg_artifact = ImageArtifact(
            data=cls.jpeg_bytes,
            mime_type="image/jpeg",
            width=160,
            height=120,
            format="JPEG",
            file_name="sample.jpg",
        )
        cls.png_artifact = ImageArtifact(
            data=cls.png_bytes,
            mime_type="image/png",
            width=200,
            height=150,
            format="PNG",
            file_name="sample.png",
        )
        cls.webp_artifact = ImageArtifact(
            data=cls.webp_bytes,
            mime_type="image/webp",
            width=300,
            height=200,
            format="WEBP",
            file_name="sample.webp",
        )
        cls.rgba_artifact = ImageArtifact(
            data=cls.rgba_bytes,
            mime_type="image/png",
            width=100,
            height=100,
            format="PNG",
            file_name="alpha.png",
        )
        cls.gray_artifact = ImageArtifact(
            data=cls.gray_bytes,
            mime_type="image/png",
            width=80,
            height=80,
            format="PNG",
            file_name="gray.png",
        )

    @staticmethod
    def _generate_bytes(fmt: str, size: tuple[int, int], mode: str, color: Any) -> bytes:
        img = Image.new(mode, size, color=color)
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return buf.getvalue()

    def test_valid_png_preprocessing(self):
        preprocessed = preprocess_image(self.png_artifact)
        self.assertIsInstance(preprocessed, PreprocessedImage)
        self.assertEqual(preprocessed.width, 200)
        self.assertEqual(preprocessed.height, 150)
        self.assertEqual(preprocessed.mode, "RGB")
        self.assertEqual(preprocessed.channels, 3)
        self.assertEqual(preprocessed.format, "PNG")
        self.assertEqual(preprocessed.mime_type, "image/png")
        self.assertEqual(preprocessed.aspect_ratio, 1.3333)
        self.assertFalse(preprocessed.is_resized)
        self.assertEqual(preprocessed.metadata["file_name"], "sample.png")

    def test_valid_jpeg_preprocessing(self):
        preprocessed = preprocess_image(self.jpeg_artifact)
        self.assertIsInstance(preprocessed, PreprocessedImage)
        self.assertEqual(preprocessed.width, 160)
        self.assertEqual(preprocessed.height, 120)
        self.assertEqual(preprocessed.mode, "RGB")
        self.assertEqual(preprocessed.format, "JPEG")
        self.assertEqual(preprocessed.mime_type, "image/jpeg")
        self.assertFalse(preprocessed.is_resized)

    def test_valid_webp_preprocessing(self):
        preprocessed = preprocess_image(self.webp_artifact)
        self.assertIsInstance(preprocessed, PreprocessedImage)
        self.assertEqual(preprocessed.width, 300)
        self.assertEqual(preprocessed.height, 200)
        self.assertEqual(preprocessed.mode, "RGB")
        self.assertEqual(preprocessed.format, "WEBP")
        self.assertEqual(preprocessed.mime_type, "image/webp")

    def test_rgba_alpha_mode_normalization(self):
        # RGBA image should be composited onto white background and output as RGB
        preprocessed = preprocess_image(self.rgba_artifact)
        self.assertEqual(preprocessed.mode, "RGB")
        self.assertEqual(preprocessed.channels, 3)
        self.assertEqual(preprocessed.pil_image.mode, "RGB")
        self.assertEqual(preprocessed.metadata["original_mode"], "RGBA")

    def test_grayscale_mode_normalization(self):
        # Grayscale image (mode L) should be normalized to 3-channel RGB
        preprocessed = preprocess_image(self.gray_artifact)
        self.assertEqual(preprocessed.mode, "RGB")
        self.assertEqual(preprocessed.channels, 3)
        self.assertEqual(preprocessed.pil_image.mode, "RGB")
        self.assertEqual(preprocessed.metadata["original_mode"], "L")

    def test_dimension_downscaling_exceeding_max_bounds(self):
        # Create a large artifact (3000 x 1500)
        large_bytes = self._generate_bytes("JPEG", (3000, 1500), "RGB", color=(100, 100, 100))
        large_artifact = ImageArtifact(
            data=large_bytes,
            mime_type="image/jpeg",
            width=3000,
            height=1500,
            format="JPEG",
        )

        preprocessor = ImagePreprocessor(max_target_dimension=1000)
        preprocessed = preprocessor.preprocess(large_artifact)

        self.assertTrue(preprocessed.is_resized)
        self.assertEqual(preprocessed.width, 1000)
        self.assertEqual(preprocessed.height, 500)
        self.assertEqual(preprocessed.aspect_ratio, 2.0)
        self.assertEqual(preprocessed.metadata["original_width"], 3000)
        self.assertEqual(preprocessed.metadata["original_height"], 1500)

    def test_non_mutating_behavior(self):
        # Ensure original ImageArtifact is never mutated
        orig_data = self.png_artifact.data
        orig_w = self.png_artifact.width
        orig_h = self.png_artifact.height
        orig_fmt = self.png_artifact.format

        preprocessor = ImagePreprocessor(max_target_dimension=50)
        preprocessed = preprocessor.preprocess(self.png_artifact)

        self.assertTrue(preprocessed.is_resized)
        self.assertNotEqual(preprocessed.width, self.png_artifact.width)

        # Invariants on original artifact
        self.assertEqual(self.png_artifact.data, orig_data)
        self.assertEqual(self.png_artifact.width, orig_w)
        self.assertEqual(self.png_artifact.height, orig_h)
        self.assertEqual(self.png_artifact.format, orig_fmt)

    def test_deterministic_output(self):
        res1 = preprocess_image(self.jpeg_artifact)
        res2 = preprocess_image(self.jpeg_artifact)

        self.assertEqual(res1.width, res2.width)
        self.assertEqual(res1.height, res2.height)
        self.assertEqual(res1.aspect_ratio, res2.aspect_ratio)
        self.assertEqual(res1.to_bytes(), res2.to_bytes())

    def test_preprocessed_image_serialization(self):
        preprocessed = preprocess_image(self.png_artifact)
        d = preprocessed.to_dict()

        self.assertEqual(d["width"], 200)
        self.assertEqual(d["height"], 150)
        self.assertEqual(d["mode"], "RGB")
        self.assertEqual(d["format"], "PNG")
        self.assertFalse(d["is_resized"])
        self.assertIn("metadata", d)

        # Test to_bytes serialization
        encoded_bytes = preprocessed.to_bytes()
        self.assertIsInstance(encoded_bytes, bytes)
        self.assertTrue(len(encoded_bytes) > 0)

        # Test format override in to_bytes
        jpeg_bytes = preprocessed.to_bytes(format_override="JPEG")
        self.assertTrue(len(jpeg_bytes) > 0)

    def test_corrupted_data_rejection(self):
        # Create an artifact instance with corrupted internal data bytes
        bad_artifact = object.__new__(ImageArtifact)
        bad_artifact.data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"broken_garbage"
        bad_artifact.mime_type = "image/png"
        bad_artifact.width = 100
        bad_artifact.height = 100
        bad_artifact.format = "PNG"
        bad_artifact.base64_str = None
        bad_artifact.file_name = None

        with self.assertRaises(ValueError) as ctx:
            preprocess_image(bad_artifact)
        self.assertTrue(
            "corrupted" in str(ctx.exception).lower()
            or "failed" in str(ctx.exception).lower()
            or "unidentified" in str(ctx.exception).lower()
        )

    def test_invalid_artifact_type_rejection(self):
        with self.assertRaises(TypeError):
            preprocess_image("not-an-artifact")  # type: ignore

    def test_invalid_preprocessor_config_rejection(self):
        with self.assertRaises(ValueError):
            ImagePreprocessor(target_mode="CMYK")

        with self.assertRaises(ValueError):
            ImagePreprocessor(max_target_dimension=5)

        with self.assertRaises(ValueError):
            ImagePreprocessor(max_target_dimension=5000)


if __name__ == "__main__":
    unittest.main()
