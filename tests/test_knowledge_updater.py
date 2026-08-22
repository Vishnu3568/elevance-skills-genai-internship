import unittest
import sys
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src.knowledge_base.updater import update_knowledge_base


class TestKnowledgeUpdater(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_updated_records_abort_incremental_update(self):
        base = Path(self.temp_dir)

        knowledge_base_path = base / "knowledge_base.csv"
        update_source_path = base / "updates.csv"

        pd.DataFrame(
            [
                {
                    "prompt": "What is Python?",
                    "response": "Python is a programming language.",
                }
            ]
        ).to_csv(
            knowledge_base_path,
            index=False,
        )

        pd.DataFrame(
            [
                {
                    "prompt": "What is Python?",
                    "response": "Python is a high-level language.",
                }
            ]
        ).to_csv(
            update_source_path,
            index=False,
        )

        history_path = str(base / "history.jsonl")

        with self.assertRaises(ValueError) as context:
            update_knowledge_base(
                knowledge_base_path=str(knowledge_base_path),
                update_source_path=str(update_source_path),
                vector_store_path="unused",
                history_path=history_path,
            )
        self.assertIn("UPDATED records require a vector-store rebuild", str(context.exception))
        # Ensure no audit record was written on abort
        self.assertFalse(os.path.exists(history_path))

    def test_faiss_failure_preserves_knowledge_base_csv(self):
        base = Path(self.temp_dir)

        knowledge_base_path = base / "knowledge_base.csv"
        update_source_path = base / "updates.csv"
        non_existent_vector_path = str(base / "non_existent_faiss_dir")
        history_path = str(base / "history.jsonl")

        initial_data = [
            {
                "prompt": "What is Python?",
                "response": "Python is a programming language.",
            }
        ]
        pd.DataFrame(initial_data).to_csv(
            knowledge_base_path,
            index=False,
            encoding="utf-8",
        )

        pd.DataFrame(
            [
                {
                    "prompt": "What is Java?",
                    "response": "Java is a programming language.",
                }
            ]
        ).to_csv(
            update_source_path,
            index=False,
            encoding="latin1",
        )

        # Vector store load will fail because path does not exist
        with self.assertRaises(Exception):
            update_knowledge_base(
                knowledge_base_path=str(knowledge_base_path),
                update_source_path=str(update_source_path),
                vector_store_path=non_existent_vector_path,
                history_path=history_path,
            )

        # Verify that the knowledge base file on disk was NOT mutated
        current_kb = pd.read_csv(knowledge_base_path, encoding="utf-8")
        self.assertEqual(len(current_kb), 1)
        self.assertEqual(current_kb.iloc[0]["prompt"], "What is Python?")
        # Ensure no SUCCESS audit record was written on failure
        self.assertFalse(os.path.exists(history_path))


# Standalone functions for pytest compatibility
def test_updated_records_abort_incremental_update():
    test_case = TestKnowledgeUpdater()
    test_case.setUp()
    try:
        test_case.test_updated_records_abort_incremental_update()
    finally:
        test_case.tearDown()

def test_faiss_failure_preserves_knowledge_base_csv():
    test_case = TestKnowledgeUpdater()
    test_case.setUp()
    try:
        test_case.test_faiss_failure_preserves_knowledge_base_csv()
    finally:
        test_case.tearDown()


if __name__ == "__main__":
    unittest.main()
