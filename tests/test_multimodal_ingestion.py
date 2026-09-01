"""Unit tests for Phase 5 Multimodal AI Assistant image ingestion layer (Day 25 Step 1).
"""

import hashlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

# Ensure src is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from multimodal.ingestion import ImageIngestionService, ingest_image
from multimodal.models import ImageArtifact, MAX_IMAGE_SIZE_BYTES


class TestImageIngestion(unittest.TestCase):
    """Tests for safe image ingestion, validation, and ImageArtifact construction."""

    @classmethod
    def setUpClass(cls):
        # Create valid test images in-memory
        cls.jpeg_bytes = cls._generate_image_bytes("JPEG", (120, 80), color=(255, 0, 0))
        cls.png_bytes = cls._generate_image_bytes("PNG", (200, 150), color=(0, 255, 0))
        cls.webp_bytes = cls._generate_image_bytes("WEBP", (320, 240), color=(0, 0, 255))
        cls.gif_bytes = cls._generate_image_bytes("GIF", (100, 100), color=(255, 255, 0))
        cls.bmp_bytes = cls._generate_image_bytes("BMP", (100, 100), color=(255, 0, 255))

    @staticmethod
    def _generate_image_bytes(fmt: str, size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
        img = Image.new("RGB", size, color=color)
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return buf.getvalue()

    def test_valid_jpeg_ingestion_from_bytes(self):
        artifact = ingest_image(self.jpeg_bytes, file_name="test.jpg")
        self.assertIsInstance(artifact, ImageArtifact)
        self.assertEqual(artifact.format, "JPEG")
        self.assertEqual(artifact.mime_type, "image/jpeg")
        self.assertEqual(artifact.width, 120)
        self.assertEqual(artifact.height, 80)
        self.assertEqual(artifact.file_name, "test.jpg")
        self.assertIsNotNone(artifact.base64_str)
        self.assertEqual(artifact.data, self.jpeg_bytes)

    def test_valid_png_ingestion_from_bytes(self):
        artifact = ingest_image(self.png_bytes, file_name="test.png")
        self.assertIsInstance(artifact, ImageArtifact)
        self.assertEqual(artifact.format, "PNG")
        self.assertEqual(artifact.mime_type, "image/png")
        self.assertEqual(artifact.width, 200)
        self.assertEqual(artifact.height, 150)
        self.assertEqual(artifact.file_name, "test.png")
        self.assertIsNotNone(artifact.base64_str)

    def test_valid_webp_ingestion_from_bytes(self):
        artifact = ingest_image(self.webp_bytes, file_name="test.webp")
        self.assertIsInstance(artifact, ImageArtifact)
        self.assertEqual(artifact.format, "WEBP")
        self.assertEqual(artifact.mime_type, "image/webp")
        self.assertEqual(artifact.width, 320)
        self.assertEqual(artifact.height, 240)
        self.assertEqual(artifact.file_name, "test.webp")

    def test_valid_ingestion_from_file_path(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(self.png_bytes)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as f:
                initial_hash = hashlib.sha256(f.read()).hexdigest()

            artifact = ingest_image(tmp_path)
            self.assertIsInstance(artifact, ImageArtifact)
            self.assertEqual(artifact.width, 200)
            self.assertEqual(artifact.height, 150)
            self.assertEqual(artifact.format, "PNG")
            self.assertEqual(artifact.file_name, os.path.basename(tmp_path))

            # Verify source file remained 100% untouched
            with open(tmp_path, "rb") as f:
                post_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(initial_hash, post_hash, "Source file was modified during ingestion!")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_ingest_from_binary_stream(self):
        stream = io.BytesIO(self.jpeg_bytes)
        artifact = ingest_image(stream, file_name="stream_test.jpg")
        self.assertIsInstance(artifact, ImageArtifact)
        self.assertEqual(artifact.format, "JPEG")
        self.assertEqual(artifact.width, 120)
        self.assertEqual(artifact.height, 80)

    def test_empty_image_rejection(self):
        with self.assertRaises(ValueError) as ctx:
            ingest_image(b"")
        self.assertIn("empty image", str(ctx.exception).lower())

    def test_unsupported_format_rejection(self):
        # GIF is unsupported
        with self.assertRaises(ValueError) as ctx:
            ingest_image(self.gif_bytes)
        self.assertIn("unsupported image format", str(ctx.exception).lower())

        # BMP is unsupported
        with self.assertRaises(ValueError) as ctx:
            ingest_image(self.bmp_bytes)
        self.assertIn("unsupported image format", str(ctx.exception).lower())

    def test_non_image_bytes_rejection(self):
        with self.assertRaises(ValueError):
            ingest_image(b"This is just a regular text file content.")

    def test_corrupted_image_rejection(self):
        # Truncated JPEG (broken stream)
        truncated_jpeg = self.jpeg_bytes[:len(self.jpeg_bytes) // 3]
        with self.assertRaises(ValueError) as ctx:
            ingest_image(truncated_jpeg)
        self.assertTrue(
            "corrupted" in str(ctx.exception).lower() or "unidentified" in str(ctx.exception).lower()
        )

        # Truncated PNG
        truncated_png = self.png_bytes[:30]
        with self.assertRaises(ValueError):
            ingest_image(truncated_png)

    def test_oversized_image_rejection(self):
        oversized = b"0" * (MAX_IMAGE_SIZE_BYTES + 1)
        with self.assertRaises(ValueError) as ctx:
            ingest_image(oversized)
        self.assertIn("exceeds maximum permitted limit", str(ctx.exception))

    def test_dimensions_out_of_bounds_rejection(self):
        # Too small (< 10px)
        tiny_bytes = self._generate_image_bytes("PNG", (8, 8), color=(10, 10, 10))
        with self.assertRaises(ValueError) as ctx:
            ingest_image(tiny_bytes)
        self.assertIn("out of bounds", str(ctx.exception))

    def test_nonexistent_file_path_rejection(self):
        with self.assertRaises(FileNotFoundError):
            ingest_image("non_existent_image_path_12345.png")

    def test_directory_path_rejection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                ingest_image(tmpdir)

    def test_invalid_source_type_rejection(self):
        with self.assertRaises(TypeError):
            ingest_image(12345)  # type: ignore

    def test_image_ingestion_service(self):
        service = ImageIngestionService(compute_base64=True)
        art1 = service.ingest_from_bytes(self.png_bytes, file_name="service.png")
        self.assertIsInstance(art1, ImageArtifact)
        self.assertEqual(art1.format, "PNG")
        self.assertIsNotNone(art1.base64_str)

        service_no_b64 = ImageIngestionService(compute_base64=False)
        art2 = service_no_b64.ingest_from_bytes(self.png_bytes)
        self.assertIsNone(art2.base64_str)


if __name__ == "__main__":
    unittest.main()
