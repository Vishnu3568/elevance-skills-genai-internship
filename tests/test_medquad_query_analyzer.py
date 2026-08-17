"""Unit Tests for MedQuAD Medical Query Analyzer Module."""

import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# pyrefly: ignore [missing-import]
from medquad_query_analyzer import (  # type: ignore
    MedicalQueryAnalyzer,
    MedicalVocabulary,
    MedicalEntity,
    MedicalQueryAnalysis
)


class TestMedicalQueryAnalyzer(unittest.TestCase):

    def setUp(self):
        # Create an in-memory vocabulary fixture
        self.vocab = MedicalVocabulary()
        self.vocab.add_topic(
            focus="Breast Cancer",
            synonyms=["Mammary Cancer", "Breast Carcinoma"],
            cuis=["C0006142", "C0006143"]
        )
        self.vocab.add_topic(
            focus="Diabetes Mellitus",
            synonyms=["Diabetes", "Type 2 Diabetes", "Type 2 Diabetes Mellitus"],
            cuis=["C0011849"]
        )
        self.vocab.add_topic(
            focus="Stroke",
            synonyms=["Cerebrovascular Accident"],
            cuis=["C0038454"]
        )
        self.vocab.add_topic(
            focus="Influenza",
            synonyms=["Flu"],
            cuis=["C0015780"]
        )
        self.analyzer = MedicalQueryAnalyzer(vocabulary=self.vocab)

    # 1. Empty Query Validation
    def test_01_empty_query_validation(self):
        with self.assertRaises(TypeError):
            self.analyzer.analyze(12345)  # type: ignore

        with self.assertRaises(ValueError):
            self.analyzer.analyze("")

        with self.assertRaises(ValueError):
            self.analyzer.analyze("   \t\n  ")

    # 2. Breast Cancer Topic Detection
    def test_02_breast_cancer_topic_detection(self):
        res = self.analyzer.analyze("What is Breast Cancer?")
        self.assertEqual(res.primary_topic, "Breast Cancer")
        self.assertIn("C0006142", res.matched_cuis)
        entity_texts = [e.text for e in res.entities if e.category == "DISEASE"]
        self.assertIn("breast cancer", entity_texts)

    # 3. Synonym Topic Detection
    def test_03_synonym_topic_detection(self):
        res = self.analyzer.analyze("Tell me about Mammary Cancer.")
        self.assertEqual(res.primary_topic, "Breast Cancer")
        self.assertIn("C0006142", res.matched_cuis)

    # 4. Multi-word Topic Detection
    def test_04_multi_word_topic_detection(self):
        res = self.analyzer.analyze("What is Type 2 Diabetes Mellitus?")
        self.assertEqual(res.primary_topic, "Diabetes Mellitus")
        self.assertIn("C0011849", res.matched_cuis)

    # 5. Diabetes Topic Detection
    def test_05_diabetes_topic_detection(self):
        res = self.analyzer.analyze("What causes Diabetes?")
        self.assertEqual(res.primary_topic, "Diabetes Mellitus")
        self.assertEqual(res.intent, "CAUSES_GENETICS")

    # 6. Symptom Detection
    def test_06_symptom_detection(self):
        res = self.analyzer.analyze("I have fever and shortness of breath.")
        symptoms = [e.text for e in res.entities if e.category == "SYMPTOM"]
        self.assertIn("fever", symptoms)
        self.assertIn("shortness of breath", symptoms)

    # 7. Treatment Detection
    def test_07_treatment_detection(self):
        res = self.analyzer.analyze("Will I need chemotherapy and surgery?")
        treatments = [e.text for e in res.entities if e.category == "TREATMENT"]
        self.assertIn("chemotherapy", treatments)
        self.assertIn("surgery", treatments)

    # 8. Treatment Intent
    def test_08_treatment_intent(self):
        res = self.analyzer.analyze("What are the treatments for breast cancer?")
        self.assertEqual(res.intent, "TREATMENT")
        self.assertEqual(res.raw_qtype_match, "treatment")

    # 9. Symptoms Intent
    def test_09_symptoms_intent(self):
        res = self.analyzer.analyze("What are the early signs and symptoms of diabetes?")
        self.assertEqual(res.intent, "SYMPTOMS")
        self.assertEqual(res.raw_qtype_match, "symptoms")

    # 10. Causes Intent
    def test_10_causes_intent(self):
        res = self.analyzer.analyze("What causes a stroke?")
        self.assertEqual(res.intent, "CAUSES_GENETICS")
        self.assertEqual(res.raw_qtype_match, "causes")

    # 11. Diagnosis Intent
    def test_11_diagnosis_intent(self):
        res = self.analyzer.analyze("How is breast cancer diagnosed with screening tests?")
        self.assertEqual(res.intent, "DIAGNOSIS")
        self.assertEqual(res.raw_qtype_match, "exams and tests")

    # 12. Safety / Side-Effect Intent
    def test_12_safety_side_effect_intent(self):
        res = self.analyzer.analyze("What are the side effects and drug interactions?")
        self.assertEqual(res.intent, "SAFETY_ADVERSE")
        self.assertEqual(res.raw_qtype_match, "side effects")

    # 13. General Information Intent
    def test_13_general_information_intent(self):
        res = self.analyzer.analyze("What is influenza?")
        self.assertEqual(res.intent, "GENERAL_INFORMATION")
        self.assertEqual(res.raw_qtype_match, "information")

    # 14. CUI Propagation
    def test_14_cui_propagation(self):
        res = self.analyzer.analyze("Can stroke cause blurred vision?")
        self.assertEqual(res.primary_topic, "Stroke")
        self.assertEqual(res.matched_cuis, ["C0038454"])
        disease_entity = next((e for e in res.entities if e.category == "DISEASE"), None)
        self.assertIsNotNone(disease_entity)
        self.assertEqual(disease_entity.cui, "C0038454")

    # 15. Longest Phrase Wins
    def test_15_longest_phrase_wins(self):
        res = self.analyzer.analyze("Information on Type 2 Diabetes Mellitus.")
        self.assertEqual(res.primary_topic, "Diabetes Mellitus")
        disease_entities = [e.text for e in res.entities if e.category == "DISEASE"]
        self.assertIn("type 2 diabetes mellitus", disease_entities)

    # 16. Word-Boundary Behavior
    def test_16_word_boundary_behavior(self):
        # "pain" should NOT match in "painting"
        res = self.analyzer.analyze("I enjoy painting on weekends.")
        symptoms = [e.text for e in res.entities if e.category == "SYMPTOM"]
        self.assertNotIn("pain", symptoms)

    # 17. Unknown Medical Topic
    def test_17_unknown_medical_topic(self):
        res = self.analyzer.analyze("What are the treatments for unknown xyz condition?")
        self.assertIsNone(res.primary_topic)
        self.assertEqual(res.matched_cuis, [])
        self.assertEqual(res.intent, "TREATMENT")

    # 18. Multiple Entities in One Query
    def test_18_multiple_entities_in_one_query(self):
        res = self.analyzer.analyze("Does breast cancer cause chest pain, and is surgery effective?")
        self.assertEqual(res.primary_topic, "Breast Cancer")
        categories = {e.category for e in res.entities}
        self.assertIn("DISEASE", categories)
        self.assertIn("SYMPTOM", categories)
        self.assertIn("TREATMENT", categories)


if __name__ == "__main__":
    unittest.main()
