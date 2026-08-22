import unittest
import sys
import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src.knowledge_base.audit import (
    record_update,
    load_update_history,
    get_last_successful_update,
)


class TestKnowledgeAudit(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.history_path = os.path.join(self.temp_dir, "test_history.jsonl")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_record_update_creates_valid_entry(self):
        summary = {
            "existing_records": 76,
            "incoming_records": 3,
            "final_records": 79,
            "new": 3,
            "updated": 0,
            "duplicate": 0,
            "invalid": 0,
        }
        entry = record_update(
            history_path=self.history_path,
            update_summary=summary,
            source="dataset/test_source.csv",
            status="SUCCESS",
        )

        # Check required fields
        required_fields = [
            "timestamp", "source", "existing_records", "incoming_records",
            "final_records", "new", "updated", "duplicate", "invalid", "status"
        ]
        for field in required_fields:
            self.assertIn(field, entry)

        self.assertEqual(entry["source"], "dataset/test_source.csv")
        self.assertEqual(entry["existing_records"], 76)
        self.assertEqual(entry["final_records"], 79)
        self.assertEqual(entry["new"], 3)
        self.assertEqual(entry["status"], "SUCCESS")

        # Verify ISO timestamp
        dt = datetime.fromisoformat(entry["timestamp"])
        self.assertIsNotNone(dt)

    def test_load_update_history_returns_all_entries(self):
        summary_1 = {"existing_records": 76, "incoming_records": 3, "final_records": 79, "new": 3}
        summary_2 = {"existing_records": 79, "incoming_records": 2, "final_records": 81, "new": 2}

        record_update(self.history_path, summary_1, source="source1.csv")
        record_update(self.history_path, summary_2, source="source2.csv")

        history = load_update_history(self.history_path)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["source"], "source1.csv")
        self.assertEqual(history[1]["source"], "source2.csv")
        self.assertEqual(history[1]["final_records"], 81)

    def test_get_last_successful_update(self):
        summary_1 = {"existing_records": 76, "incoming_records": 3, "final_records": 79, "new": 3}
        summary_2 = {"existing_records": 79, "incoming_records": 2, "final_records": 81, "new": 2}

        record_update(self.history_path, summary_1, source="source1.csv", status="SUCCESS")
        record_update(self.history_path, summary_2, source="source2.csv", status="FAILED")

        last_success = get_last_successful_update(self.history_path)
        self.assertIsNotNone(last_success)
        self.assertEqual(last_success["source"], "source1.csv")
        self.assertEqual(last_success["status"], "SUCCESS")

    def test_load_non_existent_history_returns_empty_list(self):
        non_existent = os.path.join(self.temp_dir, "non_existent.jsonl")
        history = load_update_history(non_existent)
        self.assertEqual(history, [])
        self.assertIsNone(get_last_successful_update(non_existent))

    def test_load_malformed_lines_handled_safely(self):
        with open(self.history_path, "w", encoding="utf-8") as f:
            f.write('{"timestamp": "2026-08-22T00:00:00Z", "status": "SUCCESS", "source": "valid.csv"}\n')
            f.write('corrupt non-json line\n')
            f.write('\n')
            f.write('{"timestamp": "2026-08-22T01:00:00Z", "status": "SUCCESS", "source": "valid2.csv"}\n')

        history = load_update_history(self.history_path)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["source"], "valid.csv")
        self.assertEqual(history[1]["source"], "valid2.csv")


# Standalone functions for pytest compatibility
def test_record_update_creates_valid_entry():
    t = TestKnowledgeAudit()
    t.setUp()
    try:
        t.test_record_update_creates_valid_entry()
    finally:
        t.tearDown()


if __name__ == "__main__":
    unittest.main()
