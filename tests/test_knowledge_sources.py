import unittest
import sys
import os
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src.knowledge_base.sources import (
    load_source_config,
    validate_source_entry,
    get_enabled_sources,
    process_configured_sources,
)
from src.knowledge_base.vector_store import create_knowledge_documents
from src.langchain_helper import get_instructor_embeddings
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS


class TestKnowledgeSources(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "sources.json")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_valid_source_config(self):
        sources = [
            {
                "name": "source_a",
                "path": "data/source_a.csv",
                "format": "csv",
                "enabled": True,
                "description": "Source A description",
            },
            {
                "name": "source_b",
                "path": "data/source_b.csv",
                "format": "csv",
                "enabled": False,
            }
        ]
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(sources, f)

        loaded = load_source_config(self.config_path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["name"], "source_a")
        self.assertTrue(loaded[0]["enabled"])

    def test_load_missing_config_raises_file_not_found(self):
        non_existent = os.path.join(self.temp_dir, "non_existent.json")
        with self.assertRaises(FileNotFoundError):
            load_source_config(non_existent)

    def test_load_malformed_json_raises_value_error(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write("invalid json { [")
        with self.assertRaises(ValueError) as ctx:
            load_source_config(self.config_path)
        self.assertIn("Malformed JSON", str(ctx.exception))

    def test_validate_source_entry_missing_fields(self):
        with self.assertRaises(ValueError) as ctx:
            validate_source_entry({"name": "test", "path": "test.csv"})
        self.assertIn("missing required fields", str(ctx.exception))

    def test_validate_source_entry_invalid_format(self):
        entry = {
            "name": "test",
            "path": "test.pdf",
            "format": "pdf",
            "enabled": True,
        }
        with self.assertRaises(ValueError) as ctx:
            validate_source_entry(entry)
        self.assertIn("Unsupported format", str(ctx.exception))

    def test_get_enabled_sources_filters_disabled(self):
        sources = [
            {"name": "s1", "path": "s1.csv", "format": "csv", "enabled": True},
            {"name": "s2", "path": "s2.csv", "format": "csv", "enabled": False},
            {"name": "s3", "path": "s3.csv", "format": "csv", "enabled": True},
        ]
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(sources, f)

        enabled = get_enabled_sources(self.config_path)
        self.assertEqual(len(enabled), 2)
        self.assertEqual(enabled[0]["name"], "s1")
        self.assertEqual(enabled[1]["name"], "s3")

    def test_process_configured_sources_multiple_sources(self):
        base = Path(self.temp_dir)
        kb_path = base / "knowledge_base.csv"
        s1_path = base / "source1.csv"
        s2_path = base / "source2.csv"
        faiss_dir = base / "faiss_test"
        history_path = base / "history.jsonl"

        # 1. Base knowledge base with 1 record
        pd.DataFrame([{"prompt": "Initial Question?", "response": "Initial Answer."}]).to_csv(
            kb_path, index=False, encoding="utf-8"
        )

        # 2. Initial in-memory FAISS saved to faiss_dir
        initial_doc = create_knowledge_documents([{"prompt": "Initial Question?", "response": "Initial Answer.", "row": 0}])
        embeddings = get_instructor_embeddings()
        init_db = FAISS.from_documents(initial_doc, embeddings)
        init_db.save_local(str(faiss_dir))

        # 3. Source 1 with 1 NEW record
        pd.DataFrame([{"prompt": "Question One?", "response": "Answer One."}]).to_csv(
            s1_path, index=False, encoding="latin1"
        )

        # 4. Source 2 with 1 NEW record and 1 DUPLICATE
        pd.DataFrame([
            {"prompt": "Question Two?", "response": "Answer Two."},
            {"prompt": "Question One?", "response": "Answer One."},
        ]).to_csv(
            s2_path, index=False, encoding="latin1"
        )

        # 5. Config with both sources
        sources_config = [
            {"name": "source_one", "path": str(s1_path), "format": "csv", "enabled": True},
            {"name": "source_two", "path": str(s2_path), "format": "csv", "enabled": True},
        ]
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(sources_config, f)

        # 6. Process configured sources
        outcomes = process_configured_sources(
            config_path=self.config_path,
            knowledge_base_path=str(kb_path),
            vector_store_path=str(faiss_dir),
            history_path=str(history_path),
        )

        self.assertEqual(len(outcomes), 2)
        self.assertEqual(outcomes[0]["status"], "SUCCESS")
        self.assertEqual(outcomes[0]["result"]["new"], 1)
        self.assertEqual(outcomes[1]["status"], "SUCCESS")
        self.assertEqual(outcomes[1]["result"]["new"], 1)
        self.assertEqual(outcomes[1]["result"]["duplicate"], 1)

        # Final KB should have 3 records (1 initial + 1 from s1 + 1 from s2)
        final_kb = pd.read_csv(kb_path, encoding="utf-8")
        self.assertEqual(len(final_kb), 3)

        # Final FAISS should have 3 vectors
        final_db = FAISS.load_local(str(faiss_dir), embeddings, allow_dangerous_deserialization=True)
        self.assertEqual(final_db.index.ntotal, 3)

    def test_process_configured_sources_missing_source_file_handled(self):
        base = Path(self.temp_dir)
        kb_path = base / "knowledge_base.csv"
        faiss_dir = base / "faiss_test"

        pd.DataFrame([{"prompt": "Q?", "response": "A."}]).to_csv(kb_path, index=False, encoding="utf-8")

        sources_config = [
            {"name": "missing_source", "path": "does_not_exist.csv", "format": "csv", "enabled": True}
        ]
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(sources_config, f)

        outcomes = process_configured_sources(
            config_path=self.config_path,
            knowledge_base_path=str(kb_path),
            vector_store_path=str(faiss_dir),
        )

        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["status"], "ERROR")
        self.assertIn("Source file not found", outcomes[0]["error"])


# Standalone function for pytest compatibility
def test_load_valid_source_config():
    t = TestKnowledgeSources()
    t.setUp()
    try:
        t.test_load_valid_source_config()
    finally:
        t.tearDown()


if __name__ == "__main__":
    unittest.main()
