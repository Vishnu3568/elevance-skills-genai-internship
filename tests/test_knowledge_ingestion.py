import unittest
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src.knowledge_base.ingestion import (
    DUPLICATE,
    INVALID,
    NEW,
    UPDATED,
    classify_updates,
    load_knowledge_csv,
    normalize_text,
)


class TestKnowledgeIngestion(unittest.TestCase):

    def test_new_record(self):
        existing = pd.DataFrame(
            [
                {
                    "prompt": "What is Python?",
                    "response": "Python is a programming language.",
                }
            ]
        )

        incoming = pd.DataFrame(
            [
                {
                    "prompt": "What is Java?",
                    "response": "Java is a programming language.",
                }
            ]
        )

        result = classify_updates(existing, incoming)

        self.assertEqual(result[0]["status"], NEW)

    def test_duplicate_record(self):
        existing = pd.DataFrame(
            [
                {
                    "prompt": "What is Python?",
                    "response": "Python is a programming language.",
                }
            ]
        )

        incoming = pd.DataFrame(
            [
                {
                    "prompt": "  WHAT IS PYTHON?  ",
                    "response": "Python is a programming language.",
                }
            ]
        )

        result = classify_updates(existing, incoming)

        self.assertEqual(result[0]["status"], DUPLICATE)

    def test_updated_record(self):
        existing = pd.DataFrame(
            [
                {
                    "prompt": "What is Python?",
                    "response": "Python is a programming language.",
                }
            ]
        )

        incoming = pd.DataFrame(
            [
                {
                    "prompt": "What is Python?",
                    "response": "Python is a high-level programming language.",
                }
            ]
        )

        result = classify_updates(existing, incoming)

        self.assertEqual(result[0]["status"], UPDATED)

    def test_invalid_record(self):
        existing = pd.DataFrame(
            [
                {
                    "prompt": "What is Python?",
                    "response": "Python is a programming language.",
                }
            ]
        )

        incoming = pd.DataFrame(
            [
                {
                    "prompt": "",
                    "response": "Some answer.",
                }
            ]
        )

        result = classify_updates(existing, incoming)

        self.assertEqual(result[0]["status"], INVALID)


# Standalone pytest functions for pytest compatibility
def test_new_record():
    TestKnowledgeIngestion().test_new_record()

def test_duplicate_record():
    TestKnowledgeIngestion().test_duplicate_record()

def test_updated_record():
    TestKnowledgeIngestion().test_updated_record()

def test_invalid_record():
    TestKnowledgeIngestion().test_invalid_record()


if __name__ == "__main__":
    unittest.main()
