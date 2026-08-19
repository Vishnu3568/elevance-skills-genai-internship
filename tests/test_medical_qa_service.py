"""Unit Tests for MedicalQAService Module with Safety Gating & Confidence Calibration."""

import unittest
import sys
import os
from typing import List, Tuple, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document

# pyrefly: ignore [missing-import]
from medquad_query_analyzer import MedicalQueryAnalyzer, MedicalVocabulary  # type: ignore
# pyrefly: ignore [missing-import]
from medical_qa_service import (  # type: ignore
    MedicalQAService,
    MedicalQAResponse,
    OUT_OF_DOMAIN_MESSAGE,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    GENERATION_ERROR_MESSAGE,
    RETRIEVAL_ERROR_MESSAGE,
    get_confidence_tier,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD
)


class MockVectorDB:
    """Mock vector database for deterministic testing without external models."""

    def __init__(self, candidates: List[Tuple[Document, float]], should_raise: bool = False):
        self.candidates = candidates
        self.should_raise = should_raise

    def similarity_search_with_score(self, query: str, k: int = 8) -> List[Tuple[Document, float]]:
        if self.should_raise:
            raise RuntimeError("Simulated FAISS vector search corruption")
        return self.candidates[:k]

    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        if self.should_raise:
            raise RuntimeError("Simulated FAISS vector search corruption")
        return [doc for doc, _ in self.candidates[:k]]


class MockLLM:
    """Mock LLM capturing prompts and returning deterministic answers."""

    def __init__(self, response_text: str = "Mock grounded answer based on NIH evidence.", should_raise: bool = False):
        self.response_text = response_text
        self.should_raise = should_raise
        self.prompts_received: List[str] = []
        self.call_count: int = 0

    def invoke(self, prompt: str) -> Any:
        self.call_count += 1
        self.prompts_received.append(prompt)
        if self.should_raise:
            raise RuntimeError("Simulated upstream LLM network error")

        class MockResponse:
            def __init__(self, content):
                self.content = content
        return MockResponse(self.response_text)


class TestMedicalQAService(unittest.TestCase):

    def setUp(self):
        # Set up test vocabulary
        self.vocab = MedicalVocabulary()
        self.vocab.add_topic(
            focus="Breast Cancer",
            synonyms=["Mammary Cancer", "Breast Carcinoma"],
            cuis=["C0006142"]
        )
        self.vocab.add_topic(
            focus="Diabetes Mellitus",
            synonyms=["Diabetes"],
            cuis=["C0011849"]
        )
        self.vocab.add_topic(
            focus="Influenza",
            synonyms=["Flu"],
            cuis=["C0015780"]
        )
        self.analyzer = MedicalQueryAnalyzer(vocabulary=self.vocab)

        # Standard sample documents
        self.doc_breast_treatment = Document(
            page_content="Breast cancer treatments include surgery, chemotherapy, and radiation.",
            metadata={
                "source": "CancerGov",
                "source_url": "https://www.cancer.gov/types/breast",
                "focus": "Breast Cancer",
                "question": "What are the treatments for Breast Cancer?",
                "question_type": "treatment"
            }
        )
        self.doc_flu_symptoms = Document(
            page_content="Symptoms of flu include fever, cough, and chills.",
            metadata={
                "source": "CDC",
                "source_url": "https://www.cdc.gov/flu",
                "focus": "Influenza",
                "question": "What are the symptoms of Flu?",
                "question_type": "symptoms"
            }
        )

    # 1. Invalid Input Validation
    def test_01_invalid_input_validation(self):
        mock_db = MockVectorDB([])
        service = MedicalQAService(vector_db=mock_db, analyzer=self.analyzer)

        with self.assertRaises(TypeError):
            service.process_query(12345)  # type: ignore

        with self.assertRaises(ValueError):
            service.process_query("")

        with self.assertRaises(ValueError):
            service.process_query("   \t \n  ")

    # 2. High-Confidence Grounded Query
    def test_02_high_confidence_grounded_query(self):
        # Distance 0.10 -> semantic 0.909 + topic 0.20 + intent 0.20 = 1.309 (HIGH)
        mock_db = MockVectorDB([(self.doc_breast_treatment, 0.10)])
        mock_llm = MockLLM(response_text="High confidence breast cancer treatment answer.")
        service = MedicalQAService(vector_db=mock_db, analyzer=self.analyzer, llm=mock_llm)

        response = service.process_query("What are the treatments for breast cancer?")
        self.assertTrue(response.is_grounded)
        self.assertEqual(response.status, "GROUNDED")
        self.assertGreaterEqual(response.confidence_score, HIGH_CONFIDENCE_THRESHOLD)
        self.assertEqual(get_confidence_tier(response.confidence_score), "HIGH")
        self.assertEqual(response.final_answer, "High confidence breast cancer treatment answer.")
        self.assertEqual(mock_llm.call_count, 1)

    # 3. Medium-Confidence Grounded Query
    def test_03_medium_confidence_grounded_query(self):
        # Distance 0.50 -> semantic 0.667 + topic 0.0 + intent 0.0 + entity 0.05 = 0.717 (MEDIUM)
        doc_generic = Document(
            page_content="Medical treatment options.",
            metadata={"source": "NIH", "focus": "General", "question_type": "information"}
        )
        mock_db = MockVectorDB([(doc_generic, 0.50)])
        mock_llm = MockLLM(response_text="Medium confidence treatment overview.")
        service = MedicalQAService(vector_db=mock_db, analyzer=self.analyzer, llm=mock_llm)

        response = service.process_query("What is the surgery treatment option?")
        self.assertTrue(response.is_grounded)
        self.assertEqual(response.status, "GROUNDED")
        self.assertEqual(get_confidence_tier(response.confidence_score), "MEDIUM")
        self.assertEqual(mock_llm.call_count, 1)

    # 4. Low-but-Acceptable Grounded Query
    def test_04_low_confidence_grounded_query(self):
        # Distance 1.00 -> semantic 0.50 + topic 0.0 + intent 0.0 + entity 0.05 = 0.55 (LOW: [0.50, 0.65))
        doc_low = Document(
            page_content="Discussion on pain management.",
            metadata={"source": "NIH", "focus": "General", "question_type": "treatment"}
        )
        mock_db = MockVectorDB([(doc_low, 1.00)])
        mock_llm = MockLLM(response_text="Low confidence response.")
        service = MedicalQAService(vector_db=mock_db, analyzer=self.analyzer, llm=mock_llm, relevance_threshold=0.50)

        response = service.process_query("Information about pain")
        self.assertTrue(response.is_grounded)
        self.assertEqual(response.status, "GROUNDED")
        self.assertEqual(get_confidence_tier(response.confidence_score), "LOW")
        self.assertEqual(mock_llm.call_count, 1)

    # 5. Below-Threshold Medical Query -> INSUFFICIENT_EVIDENCE
    def test_05_below_threshold_insufficient_evidence(self):
        # Distance 3.00 -> semantic 0.25 (below threshold 0.50)
        doc_weak = Document(
            page_content="Weakly related snippet.",
            metadata={"source": "NIH", "focus": "Diabetes Mellitus", "question_type": "information"}
        )
        mock_db = MockVectorDB([(doc_weak, 3.00)])
        mock_llm = MockLLM()
        service = MedicalQAService(vector_db=mock_db, analyzer=self.analyzer, llm=mock_llm, relevance_threshold=0.80)

        response = service.process_query("What causes diabetes?")
        self.assertFalse(response.is_grounded)
        self.assertEqual(response.status, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(response.confidence_score, 0.0)
        self.assertEqual(response.final_answer, INSUFFICIENT_EVIDENCE_MESSAGE)
        self.assertEqual(mock_llm.call_count, 0)

    # 6. Out-of-Domain Query -> OUT_OF_DOMAIN
    def test_06_out_of_domain_query(self):
        doc_unrelated = Document(
            page_content="Unrelated text snippet.",
            metadata={"source": "NIH", "focus": "General", "question_type": "information"}
        )
        # Distance 1.50 -> score ~ 0.40
        mock_db = MockVectorDB([(doc_unrelated, 1.50)])
        mock_llm = MockLLM()
        service = MedicalQAService(vector_db=mock_db, analyzer=self.analyzer, llm=mock_llm)

        response = service.process_query("What is the capital of Japan?")
        self.assertFalse(response.is_grounded)
        self.assertEqual(response.status, "OUT_OF_DOMAIN")
        self.assertEqual(response.confidence_score, 0.0)
        self.assertEqual(response.final_answer, OUT_OF_DOMAIN_MESSAGE)
        self.assertEqual(mock_llm.call_count, 0)

    # 7. No Topic + No Medical Entities -> OUT_OF_DOMAIN
    def test_07_no_medical_entities_query(self):
        doc_unrelated = Document(
            page_content="Some general data.",
            metadata={"source": "NIH", "focus": "General", "question_type": "information"}
        )
        mock_db = MockVectorDB([(doc_unrelated, 1.20)])
        mock_llm = MockLLM()
        service = MedicalQAService(vector_db=mock_db, analyzer=self.analyzer, llm=mock_llm)

        response = service.process_query("Tell me about quantum physics.")
        self.assertFalse(response.is_grounded)
        self.assertEqual(response.status, "OUT_OF_DOMAIN")
        self.assertEqual(response.final_answer, OUT_OF_DOMAIN_MESSAGE)
        self.assertEqual(mock_llm.call_count, 0)

    # 8. Empty Retrieval Candidate List
    def test_08_empty_retrieval_candidate_list(self):
        mock_db = MockVectorDB([])
        mock_llm = MockLLM()
        service = MedicalQAService(vector_db=mock_db, analyzer=self.analyzer, llm=mock_llm)

        # Medical query with 0 candidates returned
        response = service.process_query("What are the treatments for breast cancer?")
        self.assertFalse(response.is_grounded)
        self.assertEqual(response.status, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(response.confidence_score, 0.0)
        self.assertEqual(response.final_answer, INSUFFICIENT_EVIDENCE_MESSAGE)
        self.assertEqual(mock_llm.call_count, 0)

    # 9. Retrieval Exception -> RETRIEVAL_ERROR
    def test_09_retrieval_error_handling(self):
        mock_db = MockVectorDB([], should_raise=True)
        mock_llm = MockLLM()
        service = MedicalQAService(vector_db=mock_db, analyzer=self.analyzer, llm=mock_llm)

        response = service.process_query("What are the treatments for breast cancer?")
        self.assertFalse(response.is_grounded)
        self.assertEqual(response.status, "RETRIEVAL_ERROR")
        self.assertEqual(response.confidence_score, 0.0)
        self.assertEqual(response.final_answer, RETRIEVAL_ERROR_MESSAGE)
        self.assertEqual(mock_llm.call_count, 0)

    # 10. LLM Failure -> GENERATION_ERROR
    def test_10_generation_error_handling(self):
        mock_db = MockVectorDB([(self.doc_breast_treatment, 0.20)])
        failing_llm = MockLLM(should_raise=True)
        service = MedicalQAService(vector_db=mock_db, analyzer=self.analyzer, llm=failing_llm)

        response = service.process_query("What are the treatments for breast cancer?")
        self.assertFalse(response.is_grounded)
        self.assertEqual(response.status, "GENERATION_ERROR")
        self.assertEqual(response.confidence_score, 0.0)
        self.assertEqual(response.final_answer, GENERATION_ERROR_MESSAGE)
        self.assertEqual(len(response.evidence_documents), 1)
        self.assertEqual(len(response.citations), 1)

    # 11. Confidence Tier Helper Function
    def test_11_confidence_tier_helper(self):
        self.assertEqual(get_confidence_tier(0.95), "HIGH")
        self.assertEqual(get_confidence_tier(0.80), "HIGH")
        self.assertEqual(get_confidence_tier(0.75), "MEDIUM")
        self.assertEqual(get_confidence_tier(0.65), "MEDIUM")
        self.assertEqual(get_confidence_tier(0.55), "LOW")
        self.assertEqual(get_confidence_tier(0.50), "LOW")
        self.assertEqual(get_confidence_tier(0.40), "INSUFFICIENT")

    # 12. Citation Extraction & Deduplication
    def test_12_citation_extraction_and_deduplication(self):
        doc_dup1 = Document(
            page_content="Text A",
            metadata={"source": "CancerGov", "source_url": "https://cancer.gov", "focus": "Breast Cancer"}
        )
        doc_dup2 = Document(
            page_content="Text B",
            metadata={"source": "CancerGov", "source_url": "https://cancer.gov", "focus": "Breast Cancer"}
        )
        doc_unique = Document(
            page_content="Text C",
            metadata={"source": "CDC", "source_url": "https://cdc.gov", "focus": "Influenza"}
        )

        mock_db = MockVectorDB([(doc_dup1, 0.2), (doc_dup2, 0.3), (doc_unique, 0.4)])
        service = MedicalQAService(vector_db=mock_db, analyzer=self.analyzer)

        response = service.process_query("What is breast cancer?")
        self.assertEqual(len(response.evidence_documents), 3)
        self.assertEqual(len(response.citations), 2)
        sources = {c["source"] for c in response.citations}
        self.assertIn("CancerGov", sources)
        self.assertIn("CDC", sources)

    # 13. Service `retrieve_evidence` wrapper
    def test_13_retrieve_evidence_wrapper(self):
        mock_db = MockVectorDB([(self.doc_breast_treatment, 0.3)])
        service = MedicalQAService(vector_db=mock_db, analyzer=self.analyzer)

        candidates = service.retrieve_evidence("treatments for breast cancer", top_k=5, final_k=2)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].document.metadata["focus"], "Breast Cancer")
        self.assertGreater(candidates[0].final_score, 0.5)

    # 14. Unknown Disease + Unrelated Evidence -> Blocked (INSUFFICIENT_EVIDENCE, LLM Bypassed)
    def test_14_unknown_disease_unrelated_evidence_blocked(self):
        doc_flu = Document(
            page_content="Flu symptoms include fever and chills.",
            metadata={"source": "CDC", "focus": "Influenza", "question_type": "symptoms"}
        )
        mock_db = MockVectorDB([(doc_flu, 0.53)])
        mock_llm = MockLLM(response_text="Fake response")
        service = MedicalQAService(vector_db=mock_db, analyzer=self.analyzer, llm=mock_llm)

        # Asking for diabetes symptoms when only Influenza is retrieved
        response = service.process_query("What are the symptoms of diabetes?")
        self.assertFalse(response.is_grounded)
        self.assertEqual(response.status, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(response.confidence_score, 0.0)
        self.assertEqual(mock_llm.call_count, 0)
        self.assertIn("could not find sufficient grounded information", response.final_answer)

    # 15. Unknown Treatment/Entity + Unrelated Evidence -> Blocked (INSUFFICIENT_EVIDENCE, LLM Bypassed)
    def test_15_unknown_treatment_unrelated_evidence_blocked(self):
        doc_breast = Document(
            page_content="Breast cancer surgery details.",
            metadata={"source": "CancerGov", "focus": "Breast Cancer", "question_type": "treatment"}
        )
        mock_db = MockVectorDB([(doc_breast, 0.60)])
        mock_llm = MockLLM(response_text="Fake response")
        service = MedicalQAService(vector_db=mock_db, analyzer=self.analyzer, llm=mock_llm)

        # Asking for chemotherapy side effects when only breast surgery doc is retrieved
        response = service.process_query("What are the side effects of chemotherapy?")
        self.assertFalse(response.is_grounded)
        self.assertEqual(response.status, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(response.confidence_score, 0.0)
        self.assertEqual(mock_llm.call_count, 0)

    # 16. Known Topic + Matching Evidence -> GROUNDED (LLM Invoked 1x)
    def test_16_known_topic_matching_evidence_grounded(self):
        mock_db = MockVectorDB([(self.doc_breast_treatment, 0.10)])
        mock_llm = MockLLM(response_text="Verified breast cancer treatments.")
        service = MedicalQAService(vector_db=mock_db, analyzer=self.analyzer, llm=mock_llm)

        response = service.process_query("What are the treatments for breast cancer?")
        self.assertTrue(response.is_grounded)
        self.assertEqual(response.status, "GROUNDED")
        self.assertGreater(response.confidence_score, 0.80)
        self.assertEqual(mock_llm.call_count, 1)
        self.assertEqual(response.final_answer, "Verified breast cancer treatments.")

    # 17. Known Topic + Unrelated Evidence -> Blocked (INSUFFICIENT_EVIDENCE, LLM Bypassed)
    def test_17_known_topic_unrelated_evidence_blocked(self):
        doc_stroke = Document(
            page_content="Stroke rehabilitation options.",
            metadata={"source": "NINDS", "focus": "Stroke", "question_type": "treatment"}
        )
        mock_db = MockVectorDB([(doc_stroke, 0.40)])
        mock_llm = MockLLM(response_text="Fake response")
        service = MedicalQAService(vector_db=mock_db, analyzer=self.analyzer, llm=mock_llm)

        # Asking for breast cancer treatments when only Stroke doc is returned
        response = service.process_query("What are the treatments for breast cancer?")
        self.assertFalse(response.is_grounded)
        self.assertEqual(response.status, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(response.confidence_score, 0.0)
        self.assertEqual(mock_llm.call_count, 0)

    # 18. Out-of-Domain Query -> OUT_OF_DOMAIN (LLM Bypassed)
    def test_18_out_of_domain_query_bypasses_llm(self):
        doc_flu = Document(
            page_content="Flu symptoms.",
            metadata={"source": "CDC", "focus": "Influenza", "question_type": "symptoms"}
        )
        mock_db = MockVectorDB([(doc_flu, 0.90)])
        mock_llm = MockLLM(response_text="Fake response")
        service = MedicalQAService(vector_db=mock_db, analyzer=self.analyzer, llm=mock_llm)

        response = service.process_query("What is the capital of Japan?")
        self.assertFalse(response.is_grounded)
        self.assertEqual(response.status, "OUT_OF_DOMAIN")
        self.assertEqual(response.confidence_score, 0.0)
        self.assertEqual(mock_llm.call_count, 0)
        self.assertIn("does not appear to be related to medical", response.final_answer)

    # 19. Confidence Score Bounded in [0.0, 1.0]
    def test_19_confidence_score_bounded_in_unit_interval(self):
        # Even with high cumulative boosts (e.g. 1.45), score should not exceed 1.0 (100%)
        mock_db = MockVectorDB([(self.doc_breast_treatment, 0.05)])
        mock_llm = MockLLM(response_text="Answer")
        service = MedicalQAService(vector_db=mock_db, analyzer=self.analyzer, llm=mock_llm)

        response = service.process_query("What are the treatments for breast cancer?")
        self.assertTrue(response.is_grounded)
        self.assertLessEqual(response.confidence_score, 1.0)
        self.assertGreaterEqual(response.confidence_score, 0.0)
        self.assertEqual(get_confidence_tier(response.confidence_score), "HIGH")


if __name__ == "__main__":
    unittest.main()
