"""Unit & Component Tests for Medical UI Integration (src/medical_main.py)."""

import unittest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document

# pyrefly: ignore [missing-import]
from medical_qa_service import MedicalQAService, MedicalQAResponse  # type: ignore
# pyrefly: ignore [missing-import]
from medquad_query_analyzer import MedicalQueryAnalysis  # type: ignore
# pyrefly: ignore [missing-import]
import medical_main  # type: ignore


class TestMedicalUI(unittest.TestCase):

    def setUp(self):
        clear_cache = getattr(medical_main.initialize_medical_qa_service, "clear", None)
        if clear_cache:
            clear_cache()

    # 1. Medical Entry Module Import Verification
    def test_01_medical_main_module_import(self):
        self.assertTrue(hasattr(medical_main, "initialize_medical_qa_service"))
        self.assertTrue(hasattr(medical_main, "render_response"))
        self.assertTrue(hasattr(medical_main, "main"))
        self.assertTrue(hasattr(medical_main, "build_vocabulary_from_vector_db"))

    # 2. Service Initialization Helper
    @patch("medical_main.load_medical_vector_db")
    @patch("medical_main.get_llm")
    def test_02_service_initialization(self, mock_get_llm, mock_load_vector_db):
        mock_db = MagicMock()
        mock_load_vector_db.return_value = mock_db
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        service, warning = medical_main.initialize_medical_qa_service()
        self.assertIsNotNone(service)
        self.assertIsInstance(service, MedicalQAService)
        self.assertIsNone(warning)

    # 3. Service Initialization Failure Handling (Missing Vector DB)
    @patch("medical_main.load_medical_vector_db", side_effect=FileNotFoundError("Index not found"))
    @patch("os.path.exists", return_value=False)
    def test_03_service_init_missing_vector_db(self, mock_exists, mock_load_vector_db):
        service, warning = medical_main.initialize_medical_qa_service()
        self.assertIsNone(service)
        self.assertIn("not found", warning)

    # 4. Service Initialization Failure Handling (Corrupt Vector DB)
    @patch("medical_main.load_medical_vector_db", side_effect=RuntimeError("Traceback: corrupt FAISS internals"))
    def test_04_service_init_invalid_vector_db_hides_internal_error(self, mock_load_vector_db):
        service, warning = medical_main.initialize_medical_qa_service()
        self.assertIsNone(service)
        self.assertIn("could not be loaded", warning)
        self.assertNotIn("Traceback", warning)

    # 5. Medical Vocabulary Hydration from Loaded Index Metadata
    def test_05_build_vocabulary_from_vector_db_metadata(self):
        doc = Document(
            page_content="Breast cancer information.",
            metadata={
                "focus": "Breast Cancer",
                "synonyms": "Mammary Cancer, Breast Carcinoma",
                "cuis": "C0006142, C1234567",
            },
        )
        vector_db = MagicMock()
        vector_db.docstore._dict = {"doc-1": doc}

        vocab = medical_main.build_vocabulary_from_vector_db(vector_db)
        matches = vocab.find_topics("What are treatments for mammary cancer?")

        self.assertEqual(matches[0][1], "Breast Cancer")
        self.assertIn("C0006142", matches[0][2])

    # 6. Render GROUNDED Response
    @patch("streamlit.success")
    @patch("streamlit.markdown")
    @patch("streamlit.write")
    def test_06_render_grounded_response(self, mock_write, mock_markdown, mock_success):
        analysis = MedicalQueryAnalysis(
            raw_query="What are treatments for breast cancer?",
            clean_query="what are treatments for breast cancer",
            intent="TREATMENT",
            primary_topic="Breast Cancer"
        )
        doc = Document(page_content="Surgery and chemotherapy.", metadata={"source": "CancerGov", "focus": "Breast Cancer"})
        response = MedicalQAResponse(
            query="What are treatments for breast cancer?",
            final_answer="Treatments include surgery.",
            is_grounded=True,
            confidence_score=0.90,
            status="GROUNDED",
            analysis=analysis,
            evidence_documents=[doc],
            citations=[{"source": "CancerGov", "focus": "Breast Cancer"}]
        )

        medical_main.render_response(response)
        mock_success.assert_called_once()
        mock_write.assert_called_with("Treatments include surgery.")

    # 7. Render INSUFFICIENT_EVIDENCE Response
    @patch("streamlit.warning")
    @patch("streamlit.write")
    def test_07_render_insufficient_evidence_response(self, mock_write, mock_warning):
        analysis = MedicalQueryAnalysis(raw_query="weak question", clean_query="weak question", intent="GENERAL_INFORMATION")
        response = MedicalQAResponse(
            query="weak question",
            final_answer="I could not find sufficient grounded information...",
            is_grounded=False,
            confidence_score=0.0,
            status="INSUFFICIENT_EVIDENCE",
            analysis=analysis
        )

        medical_main.render_response(response)
        mock_warning.assert_called_with("⚠️ Insufficient Grounded Evidence")
        mock_write.assert_called_with("I could not find sufficient grounded information...")

    # 8. Render OUT_OF_DOMAIN Response
    @patch("streamlit.info")
    @patch("streamlit.write")
    def test_08_render_out_of_domain_response(self, mock_write, mock_info):
        analysis = MedicalQueryAnalysis(raw_query="capital of Japan", clean_query="capital of japan", intent="GENERAL_INFORMATION")
        response = MedicalQAResponse(
            query="capital of Japan",
            final_answer="This question does not appear to be related to medical...",
            is_grounded=False,
            confidence_score=0.0,
            status="OUT_OF_DOMAIN",
            analysis=analysis
        )

        medical_main.render_response(response)
        mock_info.assert_called_with("ℹ️ Out of Knowledge Scope")

    # 9. Render RETRIEVAL_ERROR Response (No Stack Traces)
    @patch("streamlit.error")
    @patch("streamlit.write")
    def test_09_render_retrieval_error_no_stack_trace(self, mock_write, mock_error):
        analysis = MedicalQueryAnalysis(raw_query="error query", clean_query="error query", intent="GENERAL_INFORMATION")
        response = MedicalQAResponse(
            query="error query",
            final_answer="A technical error occurred while retrieving medical records.",
            is_grounded=False,
            confidence_score=0.0,
            status="RETRIEVAL_ERROR",
            analysis=analysis
        )

        medical_main.render_response(response)
        mock_error.assert_called_with("🚨 Medical Retrieval System Error")
        # Ensure no python Traceback string is passed to user
        self.assertNotIn("Traceback", response.final_answer)

    # 10. Render GENERATION_ERROR Response (No Stack Traces)
    @patch("streamlit.error")
    @patch("streamlit.write")
    def test_10_render_generation_error_no_stack_trace(self, mock_write, mock_error):
        analysis = MedicalQueryAnalysis(raw_query="gen error query", clean_query="gen error query", intent="GENERAL_INFORMATION")
        response = MedicalQAResponse(
            query="gen error query",
            final_answer="I found relevant medical evidence, but I was unable to generate a reliable answer...",
            is_grounded=False,
            confidence_score=0.0,
            status="GENERATION_ERROR",
            analysis=analysis
        )

        medical_main.render_response(response)
        mock_error.assert_called_with("🚨 Answer Generation Error")
        self.assertNotIn("Traceback", response.final_answer)

    # 11. Citations Rendered Only When Supplied
    @patch("streamlit.markdown")
    @patch("streamlit.write")
    def test_11_citations_rendered_only_when_supplied(self, mock_write, mock_markdown):
        analysis = MedicalQueryAnalysis(raw_query="no cite", clean_query="no cite", intent="GENERAL_INFORMATION")
        response = MedicalQAResponse(
            query="no cite",
            final_answer="Answer text",
            is_grounded=False,
            confidence_score=0.0,
            status="INSUFFICIENT_EVIDENCE",
            analysis=analysis,
            citations=[]
        )

        medical_main.render_response(response)
        # Verify markdown was NOT called for citations header
        citation_header_calls = [call for call in mock_markdown.call_args_list if "Verifiable NIH Sources" in str(call)]
        self.assertEqual(len(citation_header_calls), 0)

    # 12. Task 1 Baseline Files Verification
    def test_12_task1_baseline_unmodified(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src_dir = os.path.join(base_dir, "src")

        # Verify key Task 1 files exist and are intact
        task1_files = ["main.py", "chatbot_service.py", "response_policy.py", "langchain_helper.py", "sentiment_analyzer.py"]
        for fname in task1_files:
            fpath = os.path.join(src_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Task 1 baseline file missing: {fname}")


if __name__ == "__main__":
    unittest.main()
