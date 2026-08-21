import unittest
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src.knowledge_base.ingestion import classify_updates
from src.knowledge_base.vector_store import (
    add_documents_to_vector_store,
    create_knowledge_documents,
)
from src.langchain_helper import get_instructor_embeddings
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS


class TestFaissDynamicUpdate(unittest.TestCase):

    def test_faiss_add_new_knowledge(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        kb_path = os.path.join(base_dir, "dataset", "knowledge_base.csv")
        updates_path = os.path.join(base_dir, "dataset", "knowledge_updates.csv")

        existing = pd.read_csv(kb_path, encoding="utf-8")
        incoming = pd.read_csv(updates_path, encoding="latin1")

        classified = classify_updates(existing, incoming)

        new_records = [
            {
                "prompt": item["prompt"],
                "response": item["response"],
                "row": len(existing) + index,
            }
            for index, item in enumerate(classified)
            if item["status"] == "NEW"
        ]

        existing_documents = create_knowledge_documents(
            [
                {
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "row": index,
                }
                for index, row in existing.iterrows()
            ]
        )

        new_documents = create_knowledge_documents(new_records)

        embeddings = get_instructor_embeddings()

        # In-memory FAISS creation without touching disk persisted faiss_index
        vector_store = FAISS.from_documents(
            existing_documents,
            embeddings,
        )

        self.assertEqual(vector_store.index.ntotal, 76)

        add_documents_to_vector_store(
            vector_store,
            new_documents,
        )

        self.assertEqual(vector_store.index.ntotal, 79)
        self.assertEqual(len(vector_store.docstore._dict), 79)


# Standalone function for pytest compatibility
def test_faiss_add_new_knowledge():
    TestFaissDynamicUpdate().test_faiss_add_new_knowledge()


if __name__ == "__main__":
    unittest.main()
