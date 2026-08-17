"""Unit Tests for MedQuAD Retriever and Metadata-Aware Reranker Module."""

import unittest
import sys
import os
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document

# pyrefly: ignore [missing-import]
from medquad_query_analyzer import MedicalQueryAnalyzer, MedicalVocabulary  # type: ignore
# pyrefly: ignore [missing-import]
from medquad_retriever import (  # type: ignore
    retrieve_medical_evidence,
    retrieve_medical_evidence_with_scores,
    TOPIC_BOOST_WEIGHT,
    INTENT_BOOST_WEIGHT,
    ENTITY_BOOST_WEIGHT
)


class MockVectorDB:
    """Mock vector database for testing retrieval and reranking deterministically."""

    def __init__(self, candidates: List[Tuple[Document, float]]):
        self.candidates = candidates

    def similarity_search_with_score(self, query: str, k: int = 8) -> List[Tuple[Document, float]]:
        return self.candidates[:k]

    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        return [doc for doc, _ in self.candidates[:k]]


class TestMedQuADRetriever(unittest.TestCase):

    def setUp(self):
        # Set up vocabulary
        self.vocab = MedicalVocabulary()
        self.vocab.add_topic(
            focus="Breast Cancer",
            synonyms=["Mammary Cancer", "Breast Carcinoma"],
            cuis=["C0006142"]
        )
        self.vocab.add_topic(
            focus="Diabetes Mellitus",
            synonyms=["Diabetes", "Type 2 Diabetes"],
            cuis=["C0011849"]
        )
        self.analyzer = MedicalQueryAnalyzer(vocabulary=self.vocab)

    # 1. Empty Query Validation
    def test_01_empty_query_validation(self):
        mock_db = MockVectorDB([])
        with self.assertRaises(TypeError):
            retrieve_medical_evidence(123, mock_db)  # type: ignore

        with self.assertRaises(ValueError):
            retrieve_medical_evidence("", mock_db)

        with self.assertRaises(ValueError):
            retrieve_medical_evidence("   \t  ", mock_db)

    # 2. Topic Match receives boost, mismatch does not
    def test_02_topic_boost(self):
        doc_matched = Document(
            page_content="Breast cancer treatments include surgery.",
            metadata={"focus": "Breast Cancer", "question_type": "information"}
        )
        doc_unmatched = Document(
            page_content="Diabetes symptoms include thirst.",
            metadata={"focus": "Diabetes Mellitus", "question_type": "information"}
        )
        # Give equal raw distances
        mock_db = MockVectorDB([(doc_matched, 0.5), (doc_unmatched, 0.5)])

        scored = retrieve_medical_evidence_with_scores(
            query="What is breast cancer?",
            vector_db=mock_db,
            analyzer=self.analyzer,
            final_k=2
        )
        self.assertEqual(len(scored), 2)
        # doc_matched should have topic boost (+0.20), doc_unmatched should have 0.0
        self.assertAlmostEqual(scored[0].topic_boost, TOPIC_BOOST_WEIGHT)
        self.assertAlmostEqual(scored[1].topic_boost, 0.0)
        self.assertGreater(scored[0].final_score, scored[1].final_score)

    # 3. Intent / Question Type Match Boost
    def test_03_intent_boost(self):
        doc_treatment = Document(
            page_content="Chemotherapy and radiation for cancer.",
            metadata={"focus": "Breast Cancer", "question_type": "treatment"}
        )
        doc_symptoms = Document(
            page_content="Lumps and pain in breast.",
            metadata={"focus": "Breast Cancer", "question_type": "symptoms"}
        )
        mock_db = MockVectorDB([(doc_treatment, 0.5), (doc_symptoms, 0.5)])

        scored = retrieve_medical_evidence_with_scores(
            query="What are the treatments for breast cancer?",
            vector_db=mock_db,
            analyzer=self.analyzer,
            final_k=2
        )
        # Treatment query should boost doc_treatment (+0.20)
        self.assertAlmostEqual(scored[0].intent_boost, INTENT_BOOST_WEIGHT)
        self.assertAlmostEqual(scored[1].intent_boost, 0.0)
        self.assertEqual(scored[0].document.metadata["question_type"], "treatment")

    # 4. Symptoms Intent Boost
    def test_04_symptoms_intent_boost(self):
        doc_symptoms = Document(
            page_content="Common symptoms of diabetes.",
            metadata={"focus": "Diabetes Mellitus", "question_type": "symptoms"}
        )
        doc_causes = Document(
            page_content="Causes of diabetes.",
            metadata={"focus": "Diabetes Mellitus", "question_type": "causes"}
        )
        mock_db = MockVectorDB([(doc_symptoms, 0.5), (doc_causes, 0.5)])

        scored = retrieve_medical_evidence_with_scores(
            query="What are the symptoms of diabetes?",
            vector_db=mock_db,
            analyzer=self.analyzer,
            final_k=2
        )
        self.assertAlmostEqual(scored[0].intent_boost, INTENT_BOOST_WEIGHT)
        self.assertEqual(scored[0].document.metadata["question_type"], "symptoms")

    # 5. Entity Overlap Boost
    def test_05_entity_overlap_boost(self):
        doc_with_entities = Document(
            page_content="Patients with diabetes often suffer from fatigue, fever, and blurred vision.",
            metadata={"focus": "Diabetes Mellitus", "question_type": "symptoms"}
        )
        doc_without_entities = Document(
            page_content="Overview of metabolic conditions.",
            metadata={"focus": "Diabetes Mellitus", "question_type": "symptoms"}
        )
        mock_db = MockVectorDB([(doc_with_entities, 0.5), (doc_without_entities, 0.5)])

        scored = retrieve_medical_evidence_with_scores(
            query="Does diabetes cause fever and blurred vision?",
            vector_db=mock_db,
            analyzer=self.analyzer,
            final_k=2
        )
        self.assertGreater(scored[0].entity_boost, scored[1].entity_boost)

    # 6. Crucial Reranking Test: Metadata boosts change candidate ranking!
    def test_06_metadata_boosts_change_ranking(self):
        # Candidate A: Matches topic + treatment intent, but had LOWER initial semantic similarity (distance 0.50)
        doc_a = Document(
            page_content="Breast cancer treatments include surgery and chemotherapy.",
            metadata={"focus": "Breast Cancer", "question_type": "treatment"}
        )
        # Candidate B: Matches topic but NOT treatment intent, had HIGHEST initial similarity (distance 0.25)
        doc_b = Document(
            page_content="Breast cancer is an abnormal proliferation of breast cells.",
            metadata={"focus": "Breast Cancer", "question_type": "information"}
        )
        # Candidate C: Unmatched topic (Diabetes), matched treatment intent, medium similarity (distance 0.30)
        doc_c = Document(
            page_content="Diabetes treatment involves insulin therapy.",
            metadata={"focus": "Diabetes Mellitus", "question_type": "treatment"}
        )

        mock_db = MockVectorDB([(doc_b, 0.25), (doc_c, 0.30), (doc_a, 0.50)])

        # Before reranking in raw vector DB: Candidate B is #1 (0.25), C is #2 (0.30), A is #3 (0.50)
        results = retrieve_medical_evidence_with_scores(
            query="What are the treatments for breast cancer?",
            vector_db=mock_db,
            analyzer=self.analyzer,
            final_k=3
        )

        # After metadata-aware reranking:
        # Candidate A gets: semantic (1/(1+0.5) = 0.667) + topic (0.20) + intent (0.20) + entity (0.10) = 1.167
        # Candidate B gets: semantic (1/(1+0.25) = 0.800) + topic (0.20) + intent (0.00) + entity (0.05) = 1.050
        # Candidate C gets: semantic (1/(1+0.3) = 0.769) + topic (0.00) + intent (0.20) + entity (0.05) = 1.019
        # Therefore: Candidate A MUST win #1 rank!
        self.assertEqual(results[0].document.metadata["focus"], "Breast Cancer")
        self.assertEqual(results[0].document.metadata["question_type"], "treatment")
        self.assertEqual(results[0].document, doc_a)
        self.assertEqual(results[1].document, doc_b)
        self.assertEqual(results[2].document, doc_c)

    # 7. final_k limits returned documents
    def test_07_final_k_limit(self):
        docs = [
            (Document(page_content=f"Doc {i}", metadata={"focus": "Breast Cancer"}), 0.5)
            for i in range(10)
        ]
        mock_db = MockVectorDB(docs)

        results = retrieve_medical_evidence(
            query="Breast cancer",
            vector_db=mock_db,
            analyzer=self.analyzer,
            top_k=8,
            final_k=3
        )
        self.assertEqual(len(results), 3)

    # 8. Missing Metadata Safety
    def test_08_missing_metadata_safety(self):
        doc_no_meta = Document(page_content="Medical text without metadata.", metadata={})
        doc_none_meta = Document(page_content="Medical text with None values.", metadata={"focus": None, "question_type": None})
        mock_db = MockVectorDB([(doc_no_meta, 0.4), (doc_none_meta, 0.4)])

        results = retrieve_medical_evidence(
            query="What is breast cancer?",
            vector_db=mock_db,
            analyzer=self.analyzer,
            final_k=2
        )
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
