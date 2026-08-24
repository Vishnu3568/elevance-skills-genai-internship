import unittest
import sys
import os
import shutil
import tempfile
import re
from pathlib import Path
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb import (  # type: ignore
        ScientificPaper,
        build_scientific_vector_store,
        ScientificRetriever,
        ScientificGenerator,
        ChatMessage,
        ConversationalResponse,
        ScientificConversationSession,
        ScientificConversationalService,
        condense_followup_query,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb import (  # type: ignore
        ScientificPaper,
        build_scientific_vector_store,
        ScientificRetriever,
        ScientificGenerator,
        ChatMessage,
        ConversationalResponse,
        ScientificConversationSession,
        ScientificConversationalService,
        condense_followup_query,
    )

try:
    from langchain_core.embeddings import Embeddings
    from langchain_core.messages import AIMessage
except ImportError:
    from langchain.embeddings.base import Embeddings
    from langchain.schema import AIMessage


class DeterministicFakeEmbeddings(Embeddings):
    """Fast, deterministic local embeddings for unit testing."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def _embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        words = re.findall(r"\w+", text.lower())
        for word in words:
            idx = abs(hash(word)) % self.dim
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


class FakeConversationalLLM:
    """Deterministic fake LLM responding to condensation and answer generation prompts."""

    def __init__(self):
        self.invocations: List[str] = []

    def invoke(self, prompt: str):
        prompt_str = str(prompt)
        self.invocations.append(prompt_str)

        # Condensation prompt response
        if "STANDALONE QUESTION:" in prompt_str:
            if "its limitations" in prompt_str.lower():
                return AIMessage(content="What are the limitations of Retrieval-Augmented Generation (2005.11401)?")
            return AIMessage(content="Explain self-attention in Transformer models (1706.03762).")

        # QA prompt response
        if "Retrieval-Augmented Generation" in prompt_str or "2005.11401" in prompt_str:
            return AIMessage(content="RAG (arXiv: 2005.11401) combines parametric memory with non-parametric dense retrieval.")
        if "Attention Is All You Need" in prompt_str or "1706.03762" in prompt_str:
            return AIMessage(content="The Transformer architecture (arXiv: 1706.03762) uses multi-head self-attention.")

        return AIMessage(content="General scientific explanation based on retrieved evidence.")


class TestScientificConversation(unittest.TestCase):
    """Unit and multi-turn tests for conversational follow-up and session context management."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base = Path(self.temp_dir)
        self.embeddings = DeterministicFakeEmbeddings()

        self.paper1 = ScientificPaper(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            abstract="The dominant sequence transduction models rely on self-attention...",
            categories=["cs.CL", "cs.LG"],
            primary_category="cs.CL",
            published_date="2017-06-12",
            doi="10.48550/arXiv.1706.03762",
            concepts=["Transformer", "Self-Attention"],
        )

        self.paper2 = ScientificPaper(
            arxiv_id="2005.11401",
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            authors=["Patrick Lewis", "Ethan Perez"],
            abstract="RAG explores neural language models with dense retrieval parametric memory...",
            categories=["cs.CL", "cs.AI"],
            primary_category="cs.CL",
            published_date="2020-05-22",
            doi="10.48550/arXiv.2005.11401",
            concepts=["RAG", "Dense Retrieval"],
        )

        store_path = self.base / "faiss_test"
        self.vector_store = build_scientific_vector_store(
            papers=[self.paper1, self.paper2],
            store_path=str(store_path),
            embeddings=self.embeddings,
        )
        self.retriever = ScientificRetriever(self.vector_store, default_k=2)
        self.fake_llm = FakeConversationalLLM()
        self.generator = ScientificGenerator(llm=self.fake_llm)
        self.service = ScientificConversationalService(
            retriever=self.retriever,
            generator=self.generator,
        )

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_history_passthrough(self):
        query = "What is LoRA: Low-Rank Adaptation?"
        condensed = condense_followup_query(query, chat_history=[], llm=self.fake_llm)
        self.assertEqual(condensed, query)

    def test_followup_coreference_resolution_with_llm(self):
        history = [
            ChatMessage(role="user", content="Explain Retrieval-Augmented Generation."),
            ChatMessage(
                role="assistant",
                content="RAG (2005.11401) combines parametric memory with dense retrieval.",
            ),
        ]
        condensed = condense_followup_query("What are its limitations?", chat_history=history, llm=self.fake_llm)
        self.assertEqual(condensed, "What are the limitations of Retrieval-Augmented Generation (2005.11401)?")

    def test_followup_coreference_resolution_heuristic_without_llm(self):
        history = [
            ChatMessage(role="user", content="Tell me about Transformer models."),
            ChatMessage(role="assistant", content="The Transformer replaces RNNs with attention."),
        ]
        condensed = condense_followup_query("What are its limitations?", chat_history=history, llm=None)
        self.assertIn("Transformer models", condensed)

    def test_history_bounding_and_pruning(self):
        session = ScientificConversationSession(session_id="test_session", max_history_turns=2)

        # Add 3 turn pairs (6 messages) -> should retain last 4 messages (2 turn pairs)
        for i in range(1, 4):
            session.add_user_message(f"User query {i}")
            session.add_assistant_message(f"Assistant answer {i}")

        messages = session.get_messages()
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0].content, "User query 2")
        self.assertEqual(messages[-1].content, "Assistant answer 3")

    def test_session_clear_resets_history(self):
        session = ScientificConversationSession()
        session.add_user_message("Query 1")
        session.add_assistant_message("Answer 1")
        self.assertEqual(len(session.get_messages()), 2)

        session.clear()
        self.assertEqual(len(session.get_messages()), 0)
        self.assertEqual(session.get_history_text(), "No previous conversation.")

    def test_multi_turn_grounded_conversation_service(self):
        session = ScientificConversationSession()

        # Turn 1: Initial query
        resp1 = self.service.chat(
            query="What is Retrieval-Augmented Generation?",
            session=session,
        )
        self.assertIsInstance(resp1, ConversationalResponse)
        self.assertEqual(resp1.condensed_query, "What is Retrieval-Augmented Generation?")
        self.assertIn("2005.11401", resp1.answer.answer)
        self.assertTrue(resp1.validation.is_grounded)
        self.assertEqual(len(session.get_messages()), 2)

        # Turn 2: Follow-up question with coreference
        resp2 = self.service.chat(
            query="What are its limitations?",
            session=session,
        )
        self.assertIsInstance(resp2, ConversationalResponse)
        self.assertEqual(resp2.query, "What are its limitations?")
        self.assertIn("Retrieval-Augmented Generation", resp2.condensed_query)
        self.assertTrue(resp2.validation.is_grounded)
        self.assertEqual(len(session.get_messages()), 4)

    def test_source_preservation_in_session_messages(self):
        session = ScientificConversationSession()
        self.service.chat(query="Explain Attention Is All You Need", session=session)

        assistant_msg = session.get_messages()[1]
        self.assertEqual(assistant_msg.role, "assistant")
        self.assertGreaterEqual(len(assistant_msg.sources), 1)
        self.assertEqual(assistant_msg.sources[0].arxiv_id, "1706.03762")

    def test_input_validation_empty_user_or_assistant_message(self):
        session = ScientificConversationSession()
        with self.assertRaises(ValueError):
            session.add_user_message("   ")

        with self.assertRaises(ValueError):
            session.add_assistant_message("")

        with self.assertRaises(ValueError):
            condense_followup_query("  ", [])

    def test_conversation_isolation_from_production_stores(self):
        """Verify conversational service operations never modify customer or medical stores."""
        proj_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        customer_faiss = proj_root / "faiss_index"
        medical_faiss = proj_root / "faiss_index_medical"

        cust_mtime = customer_faiss.stat().st_mtime if customer_faiss.exists() else None
        med_mtime = medical_faiss.stat().st_mtime if medical_faiss.exists() else None

        self.service.chat("Turn 1 test query")
        self.service.chat("Turn 2 follow up query")

        if customer_faiss.exists():
            self.assertEqual(customer_faiss.stat().st_mtime, cust_mtime)
        if medical_faiss.exists():
            self.assertEqual(medical_faiss.stat().st_mtime, med_mtime)


if __name__ == "__main__":
    unittest.main()
