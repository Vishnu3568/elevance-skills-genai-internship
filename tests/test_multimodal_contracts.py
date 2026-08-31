"""Unit tests for Phase 5 Multimodal AI Assistant data contracts and models.
"""

import sys
import os
import unittest

# Ensure src is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from multimodal.models import (
    AmbiguityDetails,
    ImageArtifact,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_SIZE_BYTES,
    MIN_IMAGE_DIMENSION,
    ModalityType,
    MultimodalRequest,
    MultimodalResponse,
    SUPPORTED_MIME_TYPES,
    VisualEvidenceItem,
    validate_ambiguity_details,
    validate_image_artifact,
    validate_multimodal_request,
    validate_multimodal_response,
    validate_visual_evidence_item,
)


class TestModalityType(unittest.TestCase):
    """Tests for ModalityType enumeration."""

    def test_modality_type_values(self):
        self.assertEqual(ModalityType.TEXT_ONLY.value, "text_only")
        self.assertEqual(ModalityType.IMAGE_ONLY.value, "image_only")
        self.assertEqual(ModalityType.TEXT_AND_IMAGE.value, "text_and_image")


class TestImageArtifact(unittest.TestCase):
    """Tests for ImageArtifact model and validation."""

    def setUp(self):
        self.valid_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10"

    def test_valid_image_artifact_construction(self):
        artifact = ImageArtifact(
            data=self.valid_bytes,
            mime_type="image/png",
            width=100,
            height=100,
            format="PNG",
            base64_str="iVBORw0KGgoAAAANSUhEUgAA...",
            file_name="sample.png",
        )
        self.assertEqual(artifact.data, self.valid_bytes)
        self.assertEqual(artifact.mime_type, "image/png")
        self.assertEqual(artifact.width, 100)
        self.assertEqual(artifact.height, 100)
        self.assertEqual(artifact.format, "PNG")
        self.assertEqual(artifact.file_name, "sample.png")

        # Test to_dict serialization
        d = artifact.to_dict()
        self.assertNotIn("data", d)
        self.assertEqual(d["data_size_bytes"], len(self.valid_bytes))
        self.assertEqual(d["mime_type"], "image/png")

    def test_supported_mime_types(self):
        for mime in ["image/jpeg", "image/png", "image/webp", "IMAGE/PNG", " image/jpeg "]:
            artifact = ImageArtifact(
                data=self.valid_bytes,
                mime_type=mime,
                width=100,
                height=100,
                format="TEST",
            )
            self.assertIsNotNone(artifact)

    def test_invalid_mime_type_rejection(self):
        for invalid_mime in ["image/gif", "image/svg+xml", "application/pdf", "text/plain", ""]:
            with self.assertRaises(ValueError):
                ImageArtifact(
                    data=self.valid_bytes,
                    mime_type=invalid_mime,
                    width=100,
                    height=100,
                    format="PNG",
                )

    def test_empty_bytes_rejection(self):
        with self.assertRaises(ValueError):
            ImageArtifact(
                data=b"",
                mime_type="image/png",
                width=100,
                height=100,
                format="PNG",
            )

    def test_oversized_image_rejection(self):
        oversized = b"0" * (MAX_IMAGE_SIZE_BYTES + 1)
        with self.assertRaises(ValueError):
            ImageArtifact(
                data=oversized,
                mime_type="image/png",
                width=100,
                height=100,
                format="PNG",
            )

    def test_dimension_bounds(self):
        # Valid boundary dimensions
        min_art = ImageArtifact(data=self.valid_bytes, mime_type="image/png", width=10, height=10, format="PNG")
        self.assertEqual(min_art.width, 10)
        max_art = ImageArtifact(data=self.valid_bytes, mime_type="image/png", width=4096, height=4096, format="PNG")
        self.assertEqual(max_art.width, 4096)

        # Width out of bounds
        with self.assertRaises(ValueError):
            ImageArtifact(data=self.valid_bytes, mime_type="image/png", width=9, height=100, format="PNG")
        with self.assertRaises(ValueError):
            ImageArtifact(data=self.valid_bytes, mime_type="image/png", width=4097, height=100, format="PNG")

        # Height out of bounds
        with self.assertRaises(ValueError):
            ImageArtifact(data=self.valid_bytes, mime_type="image/png", width=100, height=9, format="PNG")
        with self.assertRaises(ValueError):
            ImageArtifact(data=self.valid_bytes, mime_type="image/png", width=100, height=4097, format="PNG")

    def test_type_validation(self):
        with self.assertRaises(TypeError):
            ImageArtifact(data="not-bytes", mime_type="image/png", width=100, height=100, format="PNG")  # type: ignore
        with self.assertRaises(TypeError):
            ImageArtifact(data=self.valid_bytes, mime_type=123, width=100, height=100, format="PNG")  # type: ignore
        with self.assertRaises(TypeError):
            ImageArtifact(data=self.valid_bytes, mime_type="image/png", width="100", height=100, format="PNG")  # type: ignore
        with self.assertRaises(TypeError):
            ImageArtifact(data=self.valid_bytes, mime_type="image/png", width=True, height=100, format="PNG")  # type: ignore


class TestMultimodalRequest(unittest.TestCase):
    """Tests for MultimodalRequest model and validation."""

    def setUp(self):
        self.valid_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10"
        self.artifact = ImageArtifact(
            data=self.valid_bytes,
            mime_type="image/png",
            width=200,
            height=200,
            format="PNG",
        )

    def test_text_only_request(self):
        req = MultimodalRequest(query="What is deep learning?")
        self.assertEqual(req.query, "What is deep learning?")
        self.assertEqual(len(req.images), 0)
        self.assertEqual(req.modality, ModalityType.TEXT_ONLY)
        self.assertEqual(req.session_id, "default_multimodal_session")
        self.assertEqual(req.temperature, 0.1)

    def test_image_only_request(self):
        req = MultimodalRequest(images=[self.artifact])
        self.assertIsNone(req.query)
        self.assertEqual(len(req.images), 1)
        self.assertEqual(req.modality, ModalityType.IMAGE_ONLY)

    def test_text_and_image_request(self):
        req = MultimodalRequest(query="Explain this chart", images=[self.artifact])
        self.assertEqual(req.query, "Explain this chart")
        self.assertEqual(len(req.images), 1)
        self.assertEqual(req.modality, ModalityType.TEXT_AND_IMAGE)

    def test_empty_request_rejection(self):
        with self.assertRaises(ValueError):
            MultimodalRequest(query="", images=[])
        with self.assertRaises(ValueError):
            MultimodalRequest(query=None, images=[])

    def test_invalid_query_type(self):
        with self.assertRaises(TypeError):
            MultimodalRequest(query=12345)  # type: ignore

    def test_invalid_images_type(self):
        with self.assertRaises(TypeError):
            MultimodalRequest(query="test", images="not-a-list")  # type: ignore
        with self.assertRaises(TypeError):
            MultimodalRequest(query="test", images=["not-an-artifact"])  # type: ignore

    def test_safe_defaults_isolation(self):
        req1 = MultimodalRequest(query="Question 1")
        req2 = MultimodalRequest(query="Question 2")
        req1.images.append(self.artifact)
        self.assertEqual(len(req1.images), 1)
        self.assertEqual(len(req2.images), 0)  # Verify list isolation

        req1.metadata_filters["category"] = "ai"
        self.assertNotIn("category", req2.metadata_filters)  # Verify dict isolation

    def test_temperature_bounds(self):
        req = MultimodalRequest(query="test", temperature=0.0)
        self.assertEqual(req.temperature, 0.0)
        req_max = MultimodalRequest(query="test", temperature=2.0)
        self.assertEqual(req_max.temperature, 2.0)

        with self.assertRaises(ValueError):
            MultimodalRequest(query="test", temperature=-0.1)
        with self.assertRaises(ValueError):
            MultimodalRequest(query="test", temperature=2.1)

    def test_max_output_tokens_validation(self):
        req = MultimodalRequest(query="test", max_output_tokens=1024)
        self.assertEqual(req.max_output_tokens, 1024)

        with self.assertRaises(ValueError):
            MultimodalRequest(query="test", max_output_tokens=0)
        with self.assertRaises(TypeError):
            MultimodalRequest(query="test", max_output_tokens="1024")  # type: ignore

    def test_to_dict_serialization(self):
        req = MultimodalRequest(query="Hello", images=[self.artifact], session_id="s1")
        d = req.to_dict()
        self.assertEqual(d["query"], "Hello")
        self.assertEqual(d["image_count"], 1)
        self.assertEqual(d["modality"], "text_and_image")
        self.assertEqual(d["session_id"], "s1")


class TestVisualEvidenceAndAmbiguity(unittest.TestCase):
    """Tests for VisualEvidenceItem and AmbiguityDetails models."""

    def test_visual_evidence_defaults_and_validation(self):
        item = VisualEvidenceItem(description="Bar chart showing 85% accuracy")
        self.assertEqual(item.description, "Bar chart showing 85% accuracy")
        self.assertIsNone(item.region_label)
        self.assertEqual(item.confidence, 1.0)

        with self.assertRaises(ValueError):
            VisualEvidenceItem(description="")
        with self.assertRaises(ValueError):
            VisualEvidenceItem(description="test", confidence=1.5)
        with self.assertRaises(ValueError):
            VisualEvidenceItem(description="test", confidence=-0.1)
        with self.assertRaises(TypeError):
            VisualEvidenceItem(description=123)  # type: ignore

    def test_ambiguity_details_defaults_and_validation(self):
        amb = AmbiguityDetails()
        self.assertFalse(amb.is_ambiguous)
        self.assertIsNone(amb.clarification_question)
        self.assertEqual(amb.missing_aspects, [])

        amb_active = AmbiguityDetails(
            is_ambiguous=True,
            clarification_question="Which curve on the chart are you referring to?",
            missing_aspects=["target_curve", "x_axis_unit"],
        )
        self.assertTrue(amb_active.is_ambiguous)
        self.assertEqual(len(amb_active.missing_aspects), 2)

        with self.assertRaises(ValueError):
            AmbiguityDetails(is_ambiguous=True, clarification_question="")
        with self.assertRaises(TypeError):
            AmbiguityDetails(is_ambiguous="yes")  # type: ignore


class TestMultimodalResponse(unittest.TestCase):
    """Tests for MultimodalResponse model and serialization."""

    def test_response_construction_and_defaults(self):
        evidence = VisualEvidenceItem(description="Loss curves plateauing at epoch 20", region_label="Main plot")
        resp = MultimodalResponse(
            query="Summarize performance",
            answer="The model converges at epoch 20.",
            modality=ModalityType.TEXT_AND_IMAGE,
            visual_evidence=[evidence],
            grounded=True,
            warning_message=None,
            image_metadata={"format": "PNG", "width": 800, "height": 600},
            formatted_markdown="### Visual Summary\nThe model converges at epoch 20.",
            session_id="session_123",
        )
        self.assertEqual(resp.query, "Summarize performance")
        self.assertEqual(resp.answer, "The model converges at epoch 20.")
        self.assertEqual(resp.modality, ModalityType.TEXT_AND_IMAGE)
        self.assertEqual(len(resp.visual_evidence), 1)
        self.assertTrue(resp.grounded)
        self.assertEqual(resp.session_id, "session_123")

        # Test dictionary serialization
        d = resp.to_dict()
        self.assertEqual(d["query"], "Summarize performance")
        self.assertEqual(d["modality"], "text_and_image")
        self.assertEqual(len(d["visual_evidence"]), 1)
        self.assertEqual(d["visual_evidence"][0]["region_label"], "Main plot")
        self.assertEqual(d["image_metadata"]["width"], 800)

    def test_response_validation_rules(self):
        with self.assertRaises(TypeError):
            MultimodalResponse(
                query=123,  # type: ignore
                answer="Ans",
                modality=ModalityType.TEXT_ONLY,
            )
        with self.assertRaises(TypeError):
            MultimodalResponse(
                query="Q",
                answer="A",
                modality="not-a-modality-enum",  # type: ignore
            )
        with self.assertRaises(ValueError):
            MultimodalResponse(
                query="Q",
                answer="A",
                modality=ModalityType.TEXT_ONLY,
                session_id="",
            )


if __name__ == "__main__":
    unittest.main()
