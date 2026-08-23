import unittest
import sys
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    # pyrefly: ignore [missing-import]
    from src.knowledge_base.rebuild import rebuild_knowledge_base  # type: ignore
    # pyrefly: ignore [missing-import]
    from src.knowledge_base.vector_store import create_knowledge_documents  # type: ignore
    # pyrefly: ignore [missing-import]
    from src.knowledge_base.audit import load_update_history  # type: ignore
    # pyrefly: ignore [missing-import]
    import src.langchain_helper as langchain_helper  # type: ignore
except ImportError:
    # pyrefly: ignore [missing-import]
    from knowledge_base.rebuild import rebuild_knowledge_base  # type: ignore
    # pyrefly: ignore [missing-import]
    from knowledge_base.vector_store import create_knowledge_documents  # type: ignore
    # pyrefly: ignore [missing-import]
    from knowledge_base.audit import load_update_history  # type: ignore
    # pyrefly: ignore [missing-import]
    import langchain_helper  # type: ignore

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS


class TestKnowledgeRebuild(unittest.TestCase):
    """Unit and integration tests for safe full-rebuild orchestration."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base = Path(self.temp_dir)

        self.kb_path = self.base / "knowledge_base.csv"
        self.faiss_dir = self.base / "faiss_test"
        self.updates_path = self.base / "updates.csv"
        self.history_path = self.base / "history.jsonl"

        # Baseline knowledge base: 2 records
        pd.DataFrame([
            {
                "prompt": "What is your return policy?",
                "response": "30 days with receipt",
            },
            {
                "prompt": "Where are you located?",
                "response": "Main Street",
            },
        ]).to_csv(self.kb_path, index=False, encoding="utf-8")

        # Build initial FAISS store
        docs = create_knowledge_documents([
            {"prompt": "What is your return policy?", "response": "30 days with receipt", "row": 0},
            {"prompt": "Where are you located?", "response": "Main Street", "row": 1},
        ])
        embeddings = langchain_helper.get_instructor_embeddings()
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(str(self.faiss_dir))

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pure_rebuild_from_existing_kb(self):
        """Test rebuilding FAISS index directly from active knowledge base CSV."""
        # Delete existing FAISS index to verify fresh construction
        shutil.rmtree(str(self.faiss_dir), ignore_errors=True)

        res = rebuild_knowledge_base(
            knowledge_base_path=str(self.kb_path),
            update_source_path=None,
            vector_store_path=str(self.faiss_dir),
            history_path=str(self.history_path),
        )

        self.assertEqual(res["status"], "REBUILD_SUCCESS")
        self.assertEqual(res["existing_records"], 2)
        self.assertEqual(res["incoming_records"], 0)
        self.assertEqual(res["final_records"], 2)

        # Verify FAISS exists and contains 2 records
        embeddings = langchain_helper.get_instructor_embeddings()
        rebuilt_db = FAISS.load_local(str(self.faiss_dir), embeddings, allow_dangerous_deserialization=True)
        self.assertEqual(rebuilt_db.index.ntotal, 2)
        self.assertEqual(len(rebuilt_db.docstore._dict), 2)

        # Verify audit log
        history = load_update_history(str(self.history_path))
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["status"], "REBUILD_SUCCESS")
        self.assertEqual(history[0]["final_records"], 2)

    def test_merge_rebuild_with_updated_and_new_records(self):
        """Test rebuild with incoming CSV containing UPDATED, NEW, DUPLICATE, and INVALID records."""
        pd.DataFrame([
            {
                "prompt": "What is your return policy?",
                "response": "60 days no questions asked",  # UPDATED
            },
            {
                "prompt": "Do you offer international shipping?",
                "response": "Yes worldwide",  # NEW
            },
            {
                "prompt": "Where are you located?",
                "response": "Main Street",  # DUPLICATE
            },
            {
                "prompt": "",
                "response": "Missing prompt",  # INVALID
            },
        ]).to_csv(self.updates_path, index=False, encoding="latin1")

        res = rebuild_knowledge_base(
            knowledge_base_path=str(self.kb_path),
            update_source_path=str(self.updates_path),
            vector_store_path=str(self.faiss_dir),
            history_path=str(self.history_path),
        )

        self.assertEqual(res["status"], "REBUILD_SUCCESS")
        self.assertEqual(res["existing_records"], 2)
        self.assertEqual(res["incoming_records"], 4)
        self.assertEqual(res["updated_records"], 1)
        self.assertEqual(res["new_records"], 1)
        self.assertEqual(res["duplicate_records"], 1)
        self.assertEqual(res["invalid_records"], 1)
        self.assertEqual(res["final_records"], 3)

        # Verify CSV updated
        df = pd.read_csv(self.kb_path, encoding="utf-8")
        self.assertEqual(len(df), 3)
        return_policy_row = df[df["prompt"] == "What is your return policy?"].iloc[0]
        self.assertEqual(return_policy_row["response"], "60 days no questions asked")

        # Verify FAISS updated
        embeddings = langchain_helper.get_instructor_embeddings()
        rebuilt_db = FAISS.load_local(str(self.faiss_dir), embeddings, allow_dangerous_deserialization=True)
        self.assertEqual(rebuilt_db.index.ntotal, 3)
        self.assertEqual(len(rebuilt_db.docstore._dict), 3)

        # Verify audit log
        history = load_update_history(str(self.history_path))
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["status"], "REBUILD_SUCCESS")
        self.assertEqual(history[0]["updated"], 1)
        self.assertEqual(history[0]["new"], 1)

    def test_rebuild_failure_preserves_original_state_and_logs_failure(self):
        """Test that any staging failure cleanly rolls back and preserves the original CSV and FAISS."""
        # Read original CSV and FAISS ntotal
        orig_df = pd.read_csv(self.kb_path, encoding="utf-8")
        orig_count = len(orig_df)

        embeddings = langchain_helper.get_instructor_embeddings()
        orig_db = FAISS.load_local(str(self.faiss_dir), embeddings, allow_dangerous_deserialization=True)
        orig_ntotal = orig_db.index.ntotal

        # Mock FAISS.from_documents to raise an error during staging
        with patch.object(FAISS, "from_documents", side_effect=RuntimeError("Simulated embedding failure")):
            with self.assertRaises(RuntimeError):
                rebuild_knowledge_base(
                    knowledge_base_path=str(self.kb_path),
                    update_source_path=None,
                    vector_store_path=str(self.faiss_dir),
                    history_path=str(self.history_path),
                )

        # Assert CSV was not modified
        current_df = pd.read_csv(self.kb_path, encoding="utf-8")
        self.assertEqual(len(current_df), orig_count)
        self.assertEqual(current_df.iloc[0]["response"], "30 days with receipt")

        # Assert FAISS was not modified
        current_db = FAISS.load_local(str(self.faiss_dir), embeddings, allow_dangerous_deserialization=True)
        self.assertEqual(current_db.index.ntotal, orig_ntotal)

        # Assert audit log contains REBUILD_FAILED
        history = load_update_history(str(self.history_path))
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["status"], "REBUILD_FAILED")
        self.assertIn("Simulated embedding failure", history[0]["error"])

    def test_missing_files_raise_file_not_found_and_log_failure(self):
        """Test that missing knowledge base or update source raises FileNotFoundError and logs failure."""
        missing_source = self.base / "missing_source.csv"

        with self.assertRaises(FileNotFoundError):
            rebuild_knowledge_base(
                knowledge_base_path=str(self.kb_path),
                update_source_path=str(missing_source),
                vector_store_path=str(self.faiss_dir),
                history_path=str(self.history_path),
            )

        history = load_update_history(str(self.history_path))
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["status"], "REBUILD_FAILED")
        self.assertIn("not found", history[0]["error"])


if __name__ == "__main__":
    unittest.main()
