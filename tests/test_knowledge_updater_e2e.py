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
from src.knowledge_base.store import load_knowledge_base
from src.langchain_helper import get_instructor_embeddings
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS


class TestKnowledgeUpdaterE2E(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Temporary paths
        self.temp_kb_path = os.path.join(self.temp_dir, "knowledge_base.csv")
        self.temp_faiss_dir = os.path.join(self.temp_dir, "faiss_index")
        self.temp_history_path = os.path.join(self.temp_dir, "history.jsonl")

        # Source paths
        baseline_kb_path = os.path.join(self.base_dir, "dataset", "dataset.csv")
        self.updates_path = os.path.join(self.base_dir, "dataset", "knowledge_updates.csv")

        # Copy 76-row baseline to temporary isolated workspace
        shutil.copyfile(baseline_kb_path, self.temp_kb_path)

        # Build initial 76-row isolated FAISS index
        from src.knowledge_base.vector_store import create_knowledge_documents
        df_base = pd.read_csv(baseline_kb_path, encoding="latin1")
        initial_docs = create_knowledge_documents(
            [
                {"prompt": row["prompt"], "response": row["response"], "row": idx}
                for idx, row in df_base.iterrows()
            ]
        )
        embeddings = get_instructor_embeddings()
        init_db = FAISS.from_documents(initial_docs, embeddings)
        init_db.save_local(self.temp_faiss_dir)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_isolated_end_to_end_update_and_retrieval(self):
        # 1. Execute end-to-end update orchestrator on isolated copies
        summary = update_knowledge_base(
            knowledge_base_path=self.temp_kb_path,
            update_source_path=self.updates_path,
            vector_store_path=self.temp_faiss_dir,
            history_path=self.temp_history_path,
        )

        # 2. Verify summary dictionary
        self.assertEqual(summary["existing_records"], 76)
        self.assertEqual(summary["incoming_records"], 3)
        self.assertEqual(summary["final_records"], 79)
        self.assertEqual(summary["new"], 3)
        self.assertEqual(summary["updated"], 0)
        self.assertEqual(summary["duplicate"], 0)
        self.assertEqual(summary["invalid"], 0)

        # 3. Verify managed CSV file contains 79 rows
        updated_kb = load_knowledge_base(self.temp_kb_path)
        self.assertEqual(len(updated_kb), 79)

        # 4. Verify FAISS index contains 79 vectors and 79 docstore items
        embeddings = get_instructor_embeddings()
        updated_faiss = FAISS.load_local(
            self.temp_faiss_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        self.assertEqual(updated_faiss.index.ntotal, 79)
        self.assertEqual(len(updated_faiss.docstore._dict), 79)

        # 5. Verify retrieval of newly added knowledge
        query = "Does the bootcamp provide practical projects?"
        retriever = updated_faiss.as_retriever(score_threshold=0.7)
        if hasattr(retriever, "invoke"):
            retrieved_docs = retriever.invoke(query)
        elif hasattr(retriever, "get_relevant_documents"):
            retrieved_docs = retriever.get_relevant_documents(query)
        else:
            retrieved_docs = updated_faiss.similarity_search(query, k=3)

        self.assertGreater(len(retrieved_docs), 0)
        top_content = retrieved_docs[0].page_content
        self.assertIn("Does the bootcamp provide practical projects?", top_content)
        self.assertIn("real-world business scenarios", top_content)

        # 6. Verify persistent audit record was written on success
        from src.knowledge_base.audit import load_update_history, get_last_successful_update
        history = load_update_history(self.temp_history_path)
        self.assertEqual(len(history), 1)
        last_update = get_last_successful_update(self.temp_history_path)
        self.assertIsNotNone(last_update)
        self.assertEqual(last_update["status"], "SUCCESS")
        self.assertEqual(last_update["new"], 3)
        self.assertEqual(last_update["final_records"], 79)


# Standalone function for pytest compatibility
def test_isolated_end_to_end_update_and_retrieval():
    test_case = TestKnowledgeUpdaterE2E()
    test_case.setUp()
    try:
        test_case.test_isolated_end_to_end_update_and_retrieval()
    finally:
        test_case.tearDown()


if __name__ == "__main__":
    unittest.main()
