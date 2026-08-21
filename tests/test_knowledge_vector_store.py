import unittest
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src.knowledge_base.vector_store import (
    add_documents_to_vector_store,
    create_knowledge_documents,
)


class TestKnowledgeVectorStore(unittest.TestCase):

    def test_create_knowledge_document(self):
        records = [
            {
                "prompt": "What is Java?",
                "response": "Java is a programming language.",
                "row": 76,
            }
        ]

        documents = create_knowledge_documents(records)

        self.assertEqual(len(documents), 1)
        self.assertEqual(
            documents[0].page_content,
            "prompt: What is Java?\nresponse: Java is a programming language.",
        )
        self.assertEqual(documents[0].metadata["source"], "What is Java?")
        self.assertEqual(documents[0].metadata["row"], 76)

    def test_empty_records_create_no_documents(self):
        documents = create_knowledge_documents([])
        self.assertEqual(documents, [])


# Standalone functions for pytest compatibility
def test_create_knowledge_document():
    TestKnowledgeVectorStore().test_create_knowledge_document()

def test_empty_records_create_no_documents():
    TestKnowledgeVectorStore().test_empty_records_create_no_documents()


if __name__ == "__main__":
    unittest.main()
