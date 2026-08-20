import unittest
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src.knowledge_base.store import apply_updates


class TestKnowledgeStore(unittest.TestCase):

    def test_apply_new_record(self):
        knowledge_base = pd.DataFrame(
            [
                {
                    "prompt": "What is Python?",
                    "response": "Python is a programming language.",
                }
            ]
        )

        updates = [
            {
                "prompt": "What is Java?",
                "response": "Java is a programming language.",
                "status": "NEW",
            }
        ]

        result = apply_updates(knowledge_base, updates)

        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[1]["prompt"], "What is Java?")

    def test_apply_updated_record(self):
        knowledge_base = pd.DataFrame(
            [
                {
                    "prompt": "What is Python?",
                    "response": "Python is a programming language.",
                }
            ]
        )

        updates = [
            {
                "prompt": "What is Python?",
                "response": "Python is a high-level programming language.",
                "status": "UPDATED",
            }
        ]

        result = apply_updates(knowledge_base, updates)

        self.assertEqual(len(result), 1)
        self.assertEqual(
            result.iloc[0]["response"],
            "Python is a high-level programming language."
        )

    def test_ignore_duplicate(self):
        knowledge_base = pd.DataFrame(
            [
                {
                    "prompt": "What is Python?",
                    "response": "Python is a programming language.",
                }
            ]
        )

        updates = [
            {
                "prompt": "What is Python?",
                "response": "Python is a programming language.",
                "status": "DUPLICATE",
            }
        ]

        result = apply_updates(knowledge_base, updates)

        self.assertEqual(len(result), 1)

    def test_ignore_invalid(self):
        knowledge_base = pd.DataFrame(
            [
                {
                    "prompt": "What is Python?",
                    "response": "Python is a programming language.",
                }
            ]
        )

        updates = [
            {
                "prompt": "",
                "response": "Some answer.",
                "status": "INVALID",
            }
        ]

        result = apply_updates(knowledge_base, updates)

        self.assertEqual(len(result), 1)


# Standalone functions for pytest compatibility
def test_apply_new_record():
    TestKnowledgeStore().test_apply_new_record()

def test_apply_updated_record():
    TestKnowledgeStore().test_apply_updated_record()

def test_ignore_duplicate():
    TestKnowledgeStore().test_ignore_duplicate()

def test_ignore_invalid():
    TestKnowledgeStore().test_ignore_invalid()


if __name__ == "__main__":
    unittest.main()
