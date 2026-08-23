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
    from src.chatbot_service import ChatbotService, ChatbotResponse  # type: ignore
    # pyrefly: ignore [missing-import]
    from src.knowledge_base.updater import update_knowledge_base  # type: ignore
    # pyrefly: ignore [missing-import]
    from src.knowledge_base.vector_store import create_knowledge_documents  # type: ignore
    # pyrefly: ignore [missing-import]
    from src.knowledge_base.audit import load_update_history  # type: ignore
    # pyrefly: ignore [missing-import]
    import src.langchain_helper as src_lh  # type: ignore
except ImportError:
    # pyrefly: ignore [missing-import]
    from chatbot_service import ChatbotService, ChatbotResponse  # type: ignore
    # pyrefly: ignore [missing-import]
    from knowledge_base.updater import update_knowledge_base  # type: ignore
    # pyrefly: ignore [missing-import]
    from knowledge_base.vector_store import create_knowledge_documents  # type: ignore
    # pyrefly: ignore [missing-import]
    from knowledge_base.audit import load_update_history  # type: ignore
    # pyrefly: ignore [missing-import]
    import langchain_helper as src_lh  # type: ignore

try:
    # pyrefly: ignore [missing-import]
    import langchain_helper as raw_lh  # type: ignore
except ImportError:
    raw_lh = src_lh

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatResult, ChatGeneration
    from langchain_core.messages import AIMessage
except ImportError:
    from langchain.chat_models.base import BaseChatModel
    from langchain.schema import ChatResult, ChatGeneration, AIMessage


class FakeQAChatModel(BaseChatModel):
    """Deterministic ChatModel for integration testing that accurately evaluates
    whether the retrieved context answers the question.
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        full_text = " ".join(m.content if hasattr(m, "content") else str(m) for m in messages)
        parts = full_text.split("QUESTION:")
        context_part = parts[0] if len(parts) > 1 else full_text
        question_part = parts[1].lower() if len(parts) > 1 else full_text.lower()

        if "shipping" in question_part:
            if "50 countries worldwide" in context_part:
                content = "Yes, we ship to over 50 countries worldwide."
            else:
                content = "I don't know."
        elif "return" in question_part:
            if "30 days with receipt" in context_part:
                content = "30 days with receipt."
            else:
                content = "I don't know."
        else:
            content = "I don't know."

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "fake-qa-chat-model"


class TestChatbotDynamicRetrievalIntegration(unittest.TestCase):
    """End-to-end integration test proving that a live ChatbotService instance
    retrieves newly added knowledge dynamically without restarting.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base = Path(self.temp_dir)

        self.kb_path = self.base / "knowledge_base.csv"
        self.faiss_dir = self.base / "faiss_test"
        self.updates_path = self.base / "updates.csv"
        self.history_path = self.base / "history.jsonl"

        # 1. Create baseline knowledge base with exactly 1 FAQ
        pd.DataFrame([
            {
                "prompt": "What is your return policy?",
                "response": "30 days with receipt",
            }
        ]).to_csv(self.kb_path, index=False, encoding="utf-8")

        # 2. Build isolated FAISS index for baseline
        initial_doc = create_knowledge_documents([
            {
                "prompt": "What is your return policy?",
                "response": "30 days with receipt",
                "row": 0,
            }
        ])
        embeddings = src_lh.get_instructor_embeddings()
        init_db = FAISS.from_documents(initial_doc, embeddings)
        init_db.save_local(str(self.faiss_dir))

        # 3. Create isolated incoming updates CSV with 1 NEW FAQ
        pd.DataFrame([
            {
                "prompt": "Do you offer international shipping?",
                "response": "Yes, we ship to over 50 countries worldwide",
            }
        ]).to_csv(self.updates_path, index=False, encoding="latin1")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_live_chatbot_service_dynamic_retrieval_without_restart(self):
        fake_llm = FakeQAChatModel()

        # Patch vectordb_file_path and get_llm across all module namespaces
        with patch.object(src_lh, "vectordb_file_path", str(self.faiss_dir)), \
             patch.object(raw_lh, "vectordb_file_path", str(self.faiss_dir)), \
             patch.object(src_lh, "get_llm", return_value=fake_llm), \
             patch.object(raw_lh, "get_llm", return_value=fake_llm):

            # 1. Instantiate ONE ChatbotService instance
            service = ChatbotService()

            query = "Do you offer international shipping?"

            # 2. Query BEFORE the dynamic update
            res_before = service.process_query(query)

            self.assertIsInstance(res_before, ChatbotResponse)
            # Before update, the FAQ does not exist in FAISS
            self.assertTrue(res_before.is_ood)
            self.assertFalse(any("50 countries" in doc.page_content for doc in res_before.source_documents))
            self.assertFalse(any("international shipping" in doc.page_content.lower() for doc in res_before.source_documents))
            self.assertIn("I don't know", res_before.raw_answer)

            # 3. Execute dynamic update using temporary sandbox paths ONLY
            update_result = update_knowledge_base(
                knowledge_base_path=str(self.kb_path),
                update_source_path=str(self.updates_path),
                vector_store_path=str(self.faiss_dir),
                history_path=str(self.history_path),
            )

            # 4. Verify update metrics & persistence
            self.assertEqual(update_result["existing_records"], 1)
            self.assertEqual(update_result["incoming_records"], 1)
            self.assertEqual(update_result["final_records"], 2)
            self.assertEqual(update_result["new"], 1)
            self.assertEqual(update_result["duplicate"], 0)
            self.assertEqual(update_result["updated"], 0)

            # Verify FAISS contains 2 documents/vectors
            embeddings = src_lh.get_instructor_embeddings()
            updated_db = FAISS.load_local(str(self.faiss_dir), embeddings, allow_dangerous_deserialization=True)
            self.assertEqual(updated_db.index.ntotal, 2)
            self.assertEqual(len(updated_db.docstore._dict), 2)

            # Verify audit log contains SUCCESS
            history = load_update_history(str(self.history_path))
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["status"], "SUCCESS")
            self.assertEqual(history[0]["new"], 1)

            # 5. Query AFTER the update using the EXACT SAME ChatbotService instance (NO RESTART)
            res_after = service.process_query(query)

            self.assertIsInstance(res_after, ChatbotResponse)
            self.assertFalse(res_after.is_ood)
            self.assertGreater(len(res_after.source_documents), 0)

            # Verify retrieved document content
            retrieved_content = " ".join(doc.page_content for doc in res_after.source_documents)
            self.assertIn("Yes, we ship to over 50 countries worldwide", retrieved_content)
            self.assertIn("Yes, we ship to over 50 countries worldwide", res_after.raw_answer)

            # Verify response policy executed normally
            self.assertIsNotNone(res_after.final_answer)
            self.assertIn("Yes, we ship to over 50 countries worldwide", res_after.final_answer)


# Standalone function for pytest compatibility
def test_live_chatbot_service_dynamic_retrieval():
    t = TestChatbotDynamicRetrievalIntegration()
    t.setUp()
    try:
        t.test_live_chatbot_service_dynamic_retrieval_without_restart()
    finally:
        t.tearDown()


if __name__ == "__main__":
    unittest.main()
