"""Integration and validation tests for the complete Multimodal Reasoning Pipeline (Phase 5 — Day 26 Step 4).

Validates end-to-end flow:
  User Query / Image
          ↓
  Image Ingestion
          ↓
  Image Preprocessing
          ↓
  Vision Understanding
          ↓
  MultimodalContext
          ↓
  MultimodalReasoningEngine
          ↓
  ReasoningResult
          ↓
  MultimodalResponseGenerator
          ↓
  MultimodalResponse
"""

import io
import os
import sys
import tempfile
from typing import Any
import unittest
from PIL import Image

# Ensure src is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from multimodal.context import (
    EvidenceProvenance,
    MultimodalContext,
    UnifiedEvidenceItem,
    build_multimodal_context,
)
from multimodal.ingestion import ingest_image
from multimodal.models import (
    ImageArtifact,
    ModalityType,
    MultimodalRequest,
    MultimodalResponse,
    VisualEvidenceItem,
)
from multimodal.orchestrator import (
    MultimodalOrchestrator,
    create_multimodal_pipeline,
)
from multimodal.preprocessing import preprocess_image
from multimodal.reasoning import (
    MultimodalReasoningEngine,
    ReasoningResult,
)
from multimodal.response_generator import MultimodalResponseGenerator
from multimodal.vision import VisionService


class TestMultimodalReasoningPipeline(unittest.TestCase):
    """End-to-end integration and validation tests for the Day 26 multimodal reasoning pipeline."""

    @classmethod
    def setUpClass(cls):
        # Create a sample test image in a temporary file
        cls.temp_dir = tempfile.mkdtemp()
        cls.sample_img_path = os.path.join(cls.temp_dir, "sample_chart.png")

        img = Image.new("RGB", (200, 100), color="teal")
        img.save(cls.sample_img_path, format="PNG")

        with open(cls.sample_img_path, "rb") as f:
            cls.png_bytes = f.read()

        cls.artifact = ingest_image(cls.png_bytes, file_name="sample_chart.png")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.sample_img_path):
            try:
                os.remove(cls.sample_img_path)
            except OSError:
                pass
        if os.path.exists(cls.temp_dir):
            try:
                os.rmdir(cls.temp_dir)
            except OSError:
                pass

    def setUp(self):
        # Create a fresh fully integrated pipeline instance for each test
        self.pipeline = create_multimodal_pipeline()

    def test_text_only_end_to_end(self):
        """Verify full end-to-end execution of a text-only query through the pipeline."""
        req = MultimodalRequest(
            query="Summarize attention mechanisms in transformers",
            session_id="pipeline_text_sess",
        )
        resp = self.pipeline.process(req)

        self.assertIsInstance(resp, MultimodalResponse)
        self.assertEqual(resp.modality, ModalityType.TEXT_ONLY)
        self.assertEqual(resp.session_id, "pipeline_text_sess")
        self.assertEqual(resp.query, "Summarize attention mechanisms in transformers")
        self.assertIn("attention mechanisms", resp.answer)
        self.assertEqual(len(resp.visual_evidence), 0)
        self.assertTrue(resp.grounded)
        self.assertIn("### Textual Analysis", resp.formatted_markdown)

    def test_image_only_end_to_end(self):
        """Verify full end-to-end execution of an image-only query through the pipeline."""
        req = MultimodalRequest(
            images=[self.artifact],
            session_id="pipeline_img_sess",
        )
        resp = self.pipeline.process(req)

        self.assertIsInstance(resp, MultimodalResponse)
        self.assertEqual(resp.modality, ModalityType.IMAGE_ONLY)
        self.assertEqual(resp.session_id, "pipeline_img_sess")
        self.assertEqual(resp.query, "")
        self.assertTrue(len(resp.visual_evidence) > 0)
        self.assertTrue(resp.grounded)
        self.assertIn("### Visual Inspection", resp.formatted_markdown)
        self.assertIn("Visual Evidence", resp.formatted_markdown)

    def test_text_and_image_end_to_end(self):
        """Verify full end-to-end execution of a joint text + image query through the pipeline."""
        query_text = "What is the dominant visual layout of this chart?"
        req = MultimodalRequest(
            query=query_text,
            images=[self.artifact],
            session_id="pipeline_joint_sess",
        )
        resp = self.pipeline.process(req)

        self.assertIsInstance(resp, MultimodalResponse)
        self.assertEqual(resp.modality, ModalityType.TEXT_AND_IMAGE)
        self.assertEqual(resp.session_id, "pipeline_joint_sess")
        self.assertEqual(resp.query, query_text)
        self.assertIn(query_text, resp.answer)
        self.assertTrue(len(resp.visual_evidence) > 0)
        self.assertTrue(resp.grounded)
        self.assertIn("### Multimodal Analysis", resp.formatted_markdown)
        self.assertIn("Textual Evidence", resp.formatted_markdown)
        self.assertIn("Visual Evidence", resp.formatted_markdown)

    def test_session_id_propagation(self):
        """Verify session_id propagates faithfully through all pipeline stages."""
        unique_session = "custom_session_id_alpha_99"
        resp = self.pipeline.process_query(
            query="Analyze latency",
            images=[self.artifact],
            session_id=unique_session,
        )
        self.assertEqual(resp.session_id, unique_session)

    def test_modality_propagation(self):
        """Verify modality detection and propagation across all 3 modalities."""
        text_resp = self.pipeline.process_query(query="Pure query", session_id="s_mod_1")
        self.assertEqual(text_resp.modality, ModalityType.TEXT_ONLY)

        img_resp = self.pipeline.process_query(images=[self.artifact], session_id="s_mod_2")
        self.assertEqual(img_resp.modality, ModalityType.IMAGE_ONLY)

        joint_resp = self.pipeline.process_query(query="Joint query", images=[self.artifact], session_id="s_mod_3")
        self.assertEqual(joint_resp.modality, ModalityType.TEXT_AND_IMAGE)

    def test_evidence_propagation(self):
        """Verify that visual and textual evidence propagate to the response contract."""
        resp = self.pipeline.process_query(
            query="Inspect chart dimensions",
            images=[self.artifact],
            session_id="s_ev_prop",
        )
        self.assertTrue(len(resp.visual_evidence) > 0)
        for ev in resp.visual_evidence:
            self.assertIsInstance(ev, VisualEvidenceItem)
            self.assertTrue(len(ev.description) > 0)

    def test_provenance_propagation(self):
        """Verify that provenance records are preserved in response metadata."""
        resp = self.pipeline.process_query(
            query="Verify origin tracking",
            images=[self.artifact],
            session_id="s_prov_prop",
        )
        self.assertIsNotNone(resp.image_metadata)
        prov_records = resp.image_metadata.get("provenance_records", [])
        self.assertTrue(len(prov_records) > 0)
        self.assertIn("source_id", prov_records[0])
        self.assertIn("provider", prov_records[0])

    def test_confidence_propagation(self):
        """Verify confidence scores propagate faithfully into metadata and formatted markdown."""
        resp = self.pipeline.process_query(
            query="Confidence check query",
            session_id="s_conf_prop",
        )
        self.assertIsNotNone(resp.image_metadata)
        self.assertIn("confidence", resp.image_metadata)
        self.assertEqual(resp.image_metadata["confidence"], 1.0)
        self.assertIn("**Confidence Assessment:** 1.00", resp.formatted_markdown)

    def test_invalid_input_handling(self):
        """Verify pipeline rejects invalid, empty, or corrupted payloads safely."""
        # 1. Invalid payload type
        with self.assertRaises(TypeError):
            self.pipeline.process(12345)  # type: ignore

        # 2. Empty request (no query and no images)
        with self.assertRaises(ValueError):
            self.pipeline.process_query()

        # 3. Invalid file path to process_image_file
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.pipeline.process_image_file("nonexistent_path_to_image_xyz.png")

        # 4. Corrupted image bytes to process_image_bytes
        with self.assertRaises(ValueError):
            self.pipeline.process_image_bytes(b"not_an_image_stream")

    def test_serialization_contains_no_raw_bytes(self):
        """Verify serialized MultimodalResponse contains no raw byte blobs."""
        resp = self.pipeline.process_query(
            query="Serialization safety audit",
            images=[self.artifact],
            session_id="s_bytes_audit",
        )
        d = resp.to_dict()

        self.assertIsInstance(d, dict)
        self.assertNotIn("data", d)
        self.assertNotIn("png_bytes", str(d))
        self.assertNotIn("b'\\x89PNG", str(d))
        self.assertEqual(d["session_id"], "s_bytes_audit")
        self.assertEqual(d["modality"], "text_and_image")

    def test_deterministic_pipeline_output(self):
        """Verify identical pipeline executions produce identical responses and outputs."""
        req = MultimodalRequest(
            query="Deterministic consistency query",
            images=[self.artifact],
            session_id="s_determ",
        )
        resp1 = self.pipeline.process(req)
        resp2 = self.pipeline.process(req)

        self.assertEqual(resp1.answer, resp2.answer)
        self.assertEqual(resp1.formatted_markdown, resp2.formatted_markdown)
        self.assertEqual(resp1.grounded, resp2.grounded)
        self.assertEqual(resp1.modality, resp2.modality)
        self.assertEqual(resp1.to_dict(), resp2.to_dict())

    def test_orchestrator_compatibility(self):
        """Verify all standard orchestrator entry points are compatible with the integrated pipeline."""
        # 1. process_query with query only
        r1 = self.pipeline.process_query(query="Explain concepts", session_id="o1")
        self.assertEqual(r1.modality, ModalityType.TEXT_ONLY)

        # 2. process_image_file with image on disk
        r2 = self.pipeline.process_image_file(self.sample_img_path, query="Describe file image", session_id="o2")
        self.assertEqual(r2.modality, ModalityType.TEXT_AND_IMAGE)
        self.assertTrue(len(r2.visual_evidence) > 0)

        # 3. process_image_bytes with raw in-memory bytes
        r3 = self.pipeline.process_image_bytes(self.png_bytes, session_id="o3")
        self.assertEqual(r3.modality, ModalityType.IMAGE_ONLY)
        self.assertTrue(len(r3.visual_evidence) > 0)


if __name__ == "__main__":
    unittest.main()
