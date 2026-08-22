import unittest
import sys
import os
import json
import time
import shutil
import tempfile
from pathlib import Path

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src.knowledge_base.scheduler import KnowledgeBaseScheduler
from src.knowledge_base.vector_store import create_knowledge_documents
from src.knowledge_base.audit import load_update_history
from src.langchain_helper import get_instructor_embeddings
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS


class TestKnowledgeSchedulerIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base = Path(self.temp_dir)

        self.kb_path = self.base / "knowledge_base.csv"
        self.faiss_dir = self.base / "faiss_test"
        self.config_path = self.base / "sources.json"
        self.source_csv = self.base / "incoming_updates.csv"
        self.history_path = self.base / "history.jsonl"

        # 1. Initialize 1 baseline record
        pd.DataFrame([
            {"prompt": "What is Python?", "response": "Python is a language."}
        ]).to_csv(self.kb_path, index=False, encoding="utf-8")

        # 2. Build initial FAISS
        initial_doc = create_knowledge_documents([
            {"prompt": "What is Python?", "response": "Python is a language.", "row": 0}
        ])
        embeddings = get_instructor_embeddings()
        init_db = FAISS.from_documents(initial_doc, embeddings)
        init_db.save_local(str(self.faiss_dir))

        # 3. Create 1 incoming source CSV with 1 NEW record
        pd.DataFrame([
            {"prompt": "What is PyTorch?", "response": "PyTorch is a deep learning framework."}
        ]).to_csv(self.source_csv, index=False, encoding="latin1")

        # 4. Create sources.json config pointing to the source CSV
        sources_data = [
            {
                "name": "deep_learning_updates",
                "path": str(self.source_csv),
                "format": "csv",
                "enabled": True,
            }
        ]
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(sources_data, f)

        # 5. Initialize Scheduler with isolated paths and fast interval
        self.scheduler = KnowledgeBaseScheduler(
            config_path=str(self.config_path),
            knowledge_base_path=str(self.kb_path),
            vector_store_path=str(self.faiss_dir),
            history_path=str(self.history_path),
            interval_seconds=0.2,
        )

    def tearDown(self):
        self.scheduler.stop(timeout=10.0)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_scheduler_lifecycle_updates_knowledge_and_records_audit(self):
        # Start the background periodic scheduler
        self.assertTrue(self.scheduler.start())
        self.assertTrue(self.scheduler.is_running())

        # Wait for first cycle to complete
        time.sleep(1.0)

        # Stop scheduler cleanly (allow enough time for embedding model load)
        self.assertTrue(self.scheduler.stop(timeout=10.0))
        self.assertFalse(self.scheduler.is_running())

        status = self.scheduler.get_status()
        self.assertEqual(status["last_status"], "SUCCESS")
        self.assertGreaterEqual(status["total_runs"], 1)

        # Verify KB expanded from 1 -> 2
        final_kb = pd.read_csv(self.kb_path, encoding="utf-8")
        self.assertEqual(len(final_kb), 2)
        self.assertEqual(final_kb.iloc[1]["prompt"], "What is PyTorch?")

        # Verify FAISS expanded from 1 -> 2
        embeddings = get_instructor_embeddings()
        final_db = FAISS.load_local(str(self.faiss_dir), embeddings, allow_dangerous_deserialization=True)
        self.assertEqual(final_db.index.ntotal, 2)
        self.assertEqual(len(final_db.docstore._dict), 2)

        # Verify audit log was recorded
        history = load_update_history(str(self.history_path))
        self.assertGreaterEqual(len(history), 1)
        self.assertEqual(history[0]["status"], "SUCCESS")
        self.assertEqual(history[0]["new"], 1)
        self.assertEqual(history[0]["final_records"], 2)


# Standalone function for pytest compatibility
def test_scheduler_lifecycle_updates_knowledge_and_records_audit():
    t = TestKnowledgeSchedulerIntegration()
    t.setUp()
    try:
        t.test_scheduler_lifecycle_updates_knowledge_and_records_audit()
    finally:
        t.tearDown()


if __name__ == "__main__":
    unittest.main()
