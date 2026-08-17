"""Integration Tests for MedQuAD Medical Retrieval and Metadata-Aware Reranking Pipeline."""

import unittest
import sys
import os
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# pyrefly: ignore [missing-import]
from medquad_parser import parse_medquad_directory  # type: ignore
# pyrefly: ignore [missing-import]
from medquad_indexer import create_medical_vector_db, load_medical_vector_db  # type: ignore
# pyrefly: ignore [missing-import]
from medquad_query_analyzer import MedicalQueryAnalyzer, MedicalVocabulary  # type: ignore
# pyrefly: ignore [missing-import]
from medquad_retriever import (  # type: ignore
    retrieve_medical_evidence,
    retrieve_medical_evidence_with_scores
)


class TestMedQuADRetrievalIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.fixtures_dir = os.path.join(cls.base_dir, "tests", "fixtures", "medquad")
        cls.temp_index_path = os.path.join(cls.base_dir, "scratch", "faiss_index_integration_test")

        if os.path.exists(cls.temp_index_path):
            shutil.rmtree(cls.temp_index_path)

        # 1. Parse fixture records and build MedicalVocabulary
        cls.records = parse_medquad_directory(cls.fixtures_dir)
        cls.vocabulary = MedicalVocabulary.from_records(cls.records)
        cls.analyzer = MedicalQueryAnalyzer(vocabulary=cls.vocabulary)

        # 2. Build temporary FAISS vector database
        cls.vector_db = create_medical_vector_db(
            xml_dir=cls.fixtures_dir,
            index_path=cls.temp_index_path
        )

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary test index
        if os.path.exists(cls.temp_index_path):
            shutil.rmtree(cls.temp_index_path)

    # 1. Realistic Query A: Treatment Query
    def test_01_treatment_query_integration(self):
        query = "What are the treatments for breast cancer?"
        results = retrieve_medical_evidence(
            query=query,
            vector_db=self.vector_db,
            analyzer=self.analyzer,
            top_k=5,
            final_k=2
        )
        self.assertGreater(len(results), 0)
        top_doc = results[0]
        self.assertEqual(top_doc.metadata.get("focus"), "Breast Cancer")
        self.assertEqual(top_doc.metadata.get("question_type"), "treatment")
        self.assertIn("surgery", top_doc.page_content.lower())

    # 2. Realistic Query B: General Information Query
    def test_02_information_query_integration(self):
        query = "What is breast cancer?"
        results = retrieve_medical_evidence(
            query=query,
            vector_db=self.vector_db,
            analyzer=self.analyzer,
            top_k=5,
            final_k=2
        )
        self.assertGreater(len(results), 0)
        top_doc = results[0]
        self.assertEqual(top_doc.metadata.get("focus"), "Breast Cancer")
        self.assertEqual(top_doc.metadata.get("question_type"), "information")

    # 3. Realistic Query C: Symptoms Query
    def test_03_symptoms_query_integration(self):
        query = "What are the symptoms of flu?"
        results = retrieve_medical_evidence(
            query=query,
            vector_db=self.vector_db,
            analyzer=self.analyzer,
            top_k=5,
            final_k=2
        )
        self.assertGreater(len(results), 0)
        top_doc = results[0]
        self.assertEqual(top_doc.metadata.get("focus"), "Influenza")
        self.assertEqual(top_doc.metadata.get("question_type"), "symptoms")

    # 4. Realistic Query D: Stroke Treatment Query
    def test_04_stroke_query_integration(self):
        query = "How is stroke treated?"
        results = retrieve_medical_evidence(
            query=query,
            vector_db=self.vector_db,
            analyzer=self.analyzer,
            top_k=5,
            final_k=2
        )
        self.assertGreater(len(results), 0)
        top_doc = results[0]
        self.assertEqual(top_doc.metadata.get("focus"), "Stroke")
        self.assertEqual(top_doc.metadata.get("question_type"), "treatment")

    # 5. Realistic Query E: Synonym Query Resolution
    def test_05_synonym_query_integration(self):
        query = "What are the treatments for mammary cancer?"
        results = retrieve_medical_evidence(
            query=query,
            vector_db=self.vector_db,
            analyzer=self.analyzer,
            top_k=5,
            final_k=2
        )
        self.assertGreater(len(results), 0)
        top_doc = results[0]
        # Resolves synonym "mammary cancer" to canonical focus "Breast Cancer"
        self.assertEqual(top_doc.metadata.get("focus"), "Breast Cancer")
        self.assertEqual(top_doc.metadata.get("question_type"), "treatment")

    # 6. End-to-End Metadata Reranking (Treatment ranks above Information)
    def test_06_metadata_reranking_end_to_end(self):
        query = "What are the treatments for breast cancer?"
        scored_candidates = retrieve_medical_evidence_with_scores(
            query=query,
            vector_db=self.vector_db,
            analyzer=self.analyzer,
            top_k=5,
            final_k=3
        )
        self.assertGreaterEqual(len(scored_candidates), 2)
        # Verify rank 1 is treatment and rank 2 is information
        self.assertEqual(scored_candidates[0].document.metadata.get("question_type"), "treatment")
        self.assertEqual(scored_candidates[1].document.metadata.get("question_type"), "information")
        self.assertGreater(scored_candidates[0].final_score, scored_candidates[1].final_score)

    # 7. Irrelevant / Unknown Query
    def test_07_irrelevant_query_handling(self):
        query = "What is the capital of Japan?"
        results = retrieve_medical_evidence(
            query=query,
            vector_db=self.vector_db,
            analyzer=self.analyzer,
            top_k=5,
            final_k=2
        )
        # Should not crash, returns standard documents without fake confidence claims
        self.assertIsInstance(results, list)

    # 8. Retrieval Limits (top_k and final_k)
    def test_08_retrieval_limits(self):
        query = "cancer"
        results = retrieve_medical_evidence(
            query=query,
            vector_db=self.vector_db,
            analyzer=self.analyzer,
            top_k=5,
            final_k=2
        )
        self.assertLessEqual(len(results), 2)

    # 9. Score Transparency Verification
    def test_09_score_transparency(self):
        query = "What are flu symptoms?"
        candidates = retrieve_medical_evidence_with_scores(
            query=query,
            vector_db=self.vector_db,
            analyzer=self.analyzer,
            top_k=5,
            final_k=3
        )
        for c in candidates:
            expected_final = c.semantic_score + c.topic_boost + c.intent_boost + c.entity_boost
            self.assertAlmostEqual(c.final_score, expected_final, places=4)
            self.assertGreater(c.semantic_score, 0.0)


if __name__ == "__main__":
    unittest.main()
