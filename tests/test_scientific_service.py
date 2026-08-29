import json
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
        StructuredPaperAnalysis,
        PaperUnderstandingService,
        build_scientific_vector_store,
        ScientificRetriever,
        ScientificGenerator,
        ScientificExpertService,
        ScientificExpertResponse,
        ScientificConversationSession,
        detect_scientific_intent,
        INTENT_GENERAL,
        INTENT_PAPER_LOOKUP,
        INTENT_SUMMARY,
        INTENT_CONCEPT_EXPLANATION,
        INTENT_COMPARISON,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb import (  # type: ignore
        ScientificPaper,
        StructuredPaperAnalysis,
        PaperUnderstandingService,
        build_scientific_vector_store,
        ScientificRetriever,
        ScientificGenerator,
        ScientificExpertService,
        ScientificExpertResponse,
        ScientificConversationSession,
        detect_scientific_intent,
        INTENT_GENERAL,
        INTENT_PAPER_LOOKUP,
        INTENT_SUMMARY,
        INTENT_CONCEPT_EXPLANATION,
        INTENT_COMPARISON,
    )

try:
    from langchain_core.embeddings import Embeddings
    from langchain_core.messages import AIMessage
except ImportError:
    from langchain.embeddings.base import Embeddings
    from langchain.schema import AIMessage


class DeterministicFakeEmbeddings(Embeddings):
    """Deterministic fast local embeddings for testing."""

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


class FakeExpertLLM:
    """Mock LLM handling general QA, summary, concept explanation, and comparison prompts."""

    def __init__(self):
        self.invocations: List[str] = []

    def invoke(self, prompt: str):
        prompt_str = str(prompt)
        self.invocations.append(prompt_str)

        # Condensation prompt
        if "STANDALONE QUESTION:" in prompt_str:
            if "its limitations" in prompt_str.lower():
                return AIMessage(content="What are the limitations of Retrieval-Augmented Generation (2005.11401)?")
            if "compare" in prompt_str.lower() and ("it" in prompt_str.lower() or "rnn" in prompt_str.lower()):
                return AIMessage(content="Compare Transformers (1706.03762) with RNNs.")
            if "masking" in prompt_str.lower() or "bert" in prompt_str.lower():
                return AIMessage(content="Explain the masked language model mechanism in BERT.")
            if "summarize" in prompt_str.lower() and ("this" in prompt_str.lower() or "paper" in prompt_str.lower()):
                return AIMessage(content="Summarize the paper Attention Is All You Need (1706.03762).")
            return AIMessage(content="Explain self-attention in Transformer models (1706.03762).")

        # Summary prompt
        if "GROUNDED PAPER SUMMARY:" in prompt_str:
            if "2005.11401" in prompt_str or "Retrieval-Augmented" in prompt_str:
                return AIMessage(
                    content=(
                        "### 📄 Paper Information\n"
                        "- Title: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2005.11401)\n"
                        "- Authors: Patrick Lewis et al.\n\n"
                        "### 📝 Executive Summary\n"
                        "Combines parametric memory with non-parametric retrieval for factual QA.\n\n"
                        "### 💡 Key Contributions\n"
                        "1. RAG-Token and RAG-Sequence architectures\n2. End-to-end differentiable retrieval-generation\n\n"
                        "### 🔬 Method & Architecture\n"
                        "Dense passage retriever combined with BART seq2seq generator.\n\n"
                        "### 📊 Important Findings & Results\n"
                        "State of the art on Open-Domain QA.\n\n"
                        "### ⚠️ Limitations & Open Challenges\n"
                        "Inference latency due to dense retrieval."
                    )
                )
            return AIMessage(
                content=(
                    "### 📄 Paper Information\n"
                    "- Title: Attention Is All You Need (1706.03762)\n"
                    "- Authors: Ashish Vaswani et al.\n\n"
                    "### 📝 Executive Summary\n"
                    "Introduces the Transformer architecture replacing recurrence with attention.\n\n"
                    "### 💡 Key Contributions\n"
                    "1. Multi-head self-attention\n2. Positional encodings\n3. Scalable training\n\n"
                    "### 🔬 Method & Architecture\n"
                    "Encoder-decoder stack with scaled dot-product attention.\n\n"
                    "### 📊 Important Findings & Results\n"
                    "State of the art on WMT translation benchmarks.\n\n"
                    "### ⚠️ Limitations & Open Challenges\n"
                    "Quadratic memory complexity with respect to sequence length."
                )
            )

        # Comparison prompt
        if "GROUNDED TECHNICAL COMPARISON:" in prompt_str:
            if "transformer" in prompt_str.lower() or "rnn" in prompt_str.lower():
                return AIMessage(
                    content=(
                        "### ⚖️ Scientific Comparison\n"
                        "Comparison between Transformers (1706.03762) and RNNs.\n\n"
                        "### 📊 Comparative Analysis\n"
                        "| Dimension | Transformers | RNNs |\n"
                        "|---|---|---|\n"
                        "| Parallelization | O(1) sequential ops | O(n) sequential ops |\n"
                        "| Long-range dependencies | Direct self-attention | Vanishing gradients |\n\n"
                        "### 🎯 Key Takeaways & Recommendations\n"
                        "Use Transformers for scalable sequence modeling."
                    )
                )
            return AIMessage(
                content=(
                    "### ⚖️ Scientific Comparison\n"
                    "Comparison between LoRA (2106.09685) and Full Fine-Tuning.\n\n"
                    "### 📊 Comparative Analysis\n"
                    "| Dimension | LoRA | Full Fine-Tuning |\n"
                    "|---|---|---|\n"
                    "| Trainable parameters | <1% | 100% |\n"
                    "| Memory | Low | High |\n\n"
                    "### 🎯 Key Takeaways & Recommendations\n"
                    "Use LoRA for parameter-efficient adaptation."
                )
            )

        # Concept explanation prompt
        if "GROUNDED CONCEPT EXPLANATION:" in prompt_str:
            concept_section = prompt_str.split("CONCEPT TO EXPLAIN:")[-1].lower() if "CONCEPT TO EXPLAIN:" in prompt_str else prompt_str.lower()
            if "bert" in concept_section or "mask" in concept_section:
                return AIMessage(
                    content=(
                        "### 🧠 Concept: Masked Language Modeling\n\n"
                        "#### 🔬 Technical Explanation\n"
                        "Randomly masks tokens and trains bidirectional representations as in 1706.03762.\n\n"
                        "#### 💡 Intuitive Explanation\n"
                        "Fill-in-the-blank cloze task to learn context from both directions simultaneously.\n\n"
                        "#### 🚀 Why It Matters in Modern AI\n"
                        "Enables deep contextual representations for pre-trained foundation models."
                    )
                )
            if "retrieval" in concept_section or "rag" in concept_section or "2005.11401" in concept_section:
                return AIMessage(
                    content=(
                        "### 🧠 Concept: Retrieval-Augmented Generation\n\n"
                        "#### 🔬 Technical Explanation\n"
                        "Combines parametric memory with dense retrieval as in 2005.11401.\n\n"
                        "#### 💡 Intuitive Explanation\n"
                        "Allows language models to look up external documents before answering.\n\n"
                        "#### 🚀 Why It Matters in Modern AI\n"
                        "Reduces hallucinations and enables verifiable citations."
                    )
                )
            return AIMessage(
                content=(
                    "### 🧠 Concept: Multi-Head Attention\n\n"
                    "#### 🔬 Technical Explanation\n"
                    "Computes multiple attention projections in parallel as in 1706.03762.\n\n"
                    "#### 💡 Intuitive Explanation\n"
                    "Allows the model to focus on different positions and representation subspaces simultaneously.\n\n"
                    "#### 🚀 Why It Matters in Modern AI\n"
                    "Foundation for LLMs like GPT, BERT, and LLaMA."
                )
            )

        # Structured Extraction prompt
        if "STRUCTURED JSON EXTRACTION:" in prompt_str:
            return AIMessage(
                content=json.dumps({
                    "research_problem": "Computational bottlenecks of recurrent sequence models.",
                    "motivation": "Sequential computation prevents parallelization during training.",
                    "methodology": "Solely rely on multi-head self-attention mechanisms without recurrence.",
                    "model_architecture": "Encoder-decoder Transformer with 8 parallel attention heads.",
                    "key_contributions": [
                        "First sequence transduction model based entirely on self-attention.",
                        "Significant reduction in training time and superior translation quality.",
                    ],
                    "datasets_benchmarks": ["WMT 2014 English-to-German", "WMT 2014 English-to-French"],
                    "key_findings": [
                        "Self-attention connects all pairs of input tokens with constant path length.",
                    ],
                    "quantitative_results": [
                        "28.4 BLEU on WMT 2014 English-to-German benchmark.",
                        "41.8 BLEU on WMT 2014 English-to-French benchmark.",
                    ],
                    "limitations": [
                        "Quadratic memory complexity with sequence length.",
                    ],
                    "open_questions": [
                        "Extending attention architectures to long-context inputs and other modalities.",
                    ],
                    "technical_concepts": [
                        "Transformer",
                        "Multi-Head Self-Attention",
                        "Positional Encoding",
                    ],
                })
            )

        # Default QA / General Prompt
        if "2005.11401" in prompt_str or "Retrieval-Augmented" in prompt_str or "rag" in prompt_str.lower():
            return AIMessage(content="RAG (arXiv: 2005.11401) integrates parametric memory with dense retrieval.")
        if "1706.03762" in prompt_str or "Attention" in prompt_str:
            return AIMessage(content="The Transformer (arXiv: 1706.03762) uses multi-head attention.")

        return AIMessage(content="Grounded scientific explanation from retrieved papers.")


class TestScientificService(unittest.TestCase):
    """Unit and end-to-end integration tests for the ScientificExpertService."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base = Path(self.temp_dir)
        self.embeddings = DeterministicFakeEmbeddings()

        self.paper1 = ScientificPaper(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            abstract="The dominant sequence transduction models are based on complex neural networks with attention...",
            categories=["cs.CL", "cs.LG"],
            primary_category="cs.CL",
            published_date="2017-06-12",
            doi="10.48550/arXiv.1706.03762",
            journal_ref="NeurIPS 2017",
            concepts=["Transformer", "Self-Attention", "Multi-Head Attention"],
        )

        self.paper2 = ScientificPaper(
            arxiv_id="2005.11401",
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            authors=["Patrick Lewis", "Ethan Perez"],
            abstract="Large pre-trained language models store factual knowledge in parametric memory with retrieval...",
            categories=["cs.CL", "cs.AI"],
            primary_category="cs.CL",
            published_date="2020-05-22",
            doi="10.48550/arXiv.2005.11401",
            concepts=["RAG", "Dense Retrieval"],
        )

        self.paper3 = ScientificPaper(
            arxiv_id="2106.09685",
            title="LoRA: Low-Rank Adaptation of Large Language Models",
            authors=["Edward J. Hu", "Yelong Shen"],
            abstract="An important paradigm consists of fine-tuning large models using low rank adaptation matrices...",
            categories=["cs.LG", "cs.AI"],
            primary_category="cs.LG",
            published_date="2021-06-17",
            doi="10.48550/arXiv.2106.09685",
            concepts=["LoRA", "PEFT", "Fine-Tuning"],
        )

        store_path = self.base / "faiss_test"
        self.vector_store = build_scientific_vector_store(
            papers=[self.paper1, self.paper2, self.paper3],
            store_path=str(store_path),
            embeddings=self.embeddings,
        )
        self.retriever = ScientificRetriever(self.vector_store, default_k=2)
        self.fake_llm = FakeExpertLLM()
        self.generator = ScientificGenerator(llm=self.fake_llm)
        self.service = ScientificExpertService(
            retriever=self.retriever,
            generator=self.generator,
        )

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_intent_detection(self):
        self.assertEqual(detect_scientific_intent("Compare LoRA and full fine-tuning"), INTENT_COMPARISON)
        self.assertEqual(detect_scientific_intent("Difference between dense and sparse retrieval"), INTENT_COMPARISON)
        self.assertEqual(detect_scientific_intent("Tell me about paper 1706.03762"), INTENT_PAPER_LOOKUP)
        self.assertEqual(detect_scientific_intent("1706.03762"), INTENT_PAPER_LOOKUP)
        self.assertEqual(detect_scientific_intent("Summarize Attention Is All You Need"), INTENT_SUMMARY)
        self.assertEqual(detect_scientific_intent("Explain multi-head attention like I'm a beginner"), INTENT_CONCEPT_EXPLANATION)
        self.assertEqual(detect_scientific_intent("Explain concept of self-attention"), INTENT_CONCEPT_EXPLANATION)
        self.assertEqual(detect_scientific_intent("What is Retrieval-Augmented Generation?"), INTENT_GENERAL)

    def test_paper_lookup_exact_and_versioned_arxiv_id(self):
        # Exact ID
        res1 = self.service.find_paper(arxiv_id="1706.03762")
        self.assertIsNotNone(res1)
        self.assertEqual(res1.arxiv_id, "1706.03762")
        self.assertEqual(res1.title, "Attention Is All You Need")

        # Versioned ID
        res2 = self.service.find_paper(arxiv_id="1706.03762v2")
        self.assertIsNotNone(res2)
        self.assertEqual(res2.arxiv_id, "1706.03762")

        # Non-existent ID
        res_none = self.service.find_paper(arxiv_id="9999.99999")
        self.assertIsNone(res_none)

    def test_paper_lookup_title_query(self):
        res = self.service.find_paper(title_query="Attention Is All You Need")
        self.assertIsNotNone(res)
        self.assertEqual(res.arxiv_id, "1706.03762")

    def test_ask_general_question_end_to_end(self):
        resp = self.service.ask("What is Retrieval-Augmented Generation?")
        self.assertIsInstance(resp, ScientificExpertResponse)
        self.assertIn("2005.11401", resp.answer)
        self.assertTrue(resp.grounded)
        self.assertGreaterEqual(len(resp.sources), 1)
        self.assertIn("### 🧠 Grounded Scientific Explanation", resp.formatted_markdown)

    def test_summarize_paper(self):
        resp = self.service.summarize_paper("1706.03762")
        self.assertIsInstance(resp, ScientificExpertResponse)
        self.assertEqual(resp.intent, INTENT_SUMMARY)
        self.assertIn("### 📄 Paper Information", resp.answer)
        self.assertIn("### 💡 Key Contributions", resp.answer)
        self.assertTrue(resp.grounded)
        self.assertEqual(resp.sources[0].arxiv_id, "1706.03762")

    def test_explain_concept(self):
        resp = self.service.explain_concept("Multi-Head Attention")
        self.assertIsInstance(resp, ScientificExpertResponse)
        self.assertEqual(resp.intent, INTENT_CONCEPT_EXPLANATION)
        self.assertIn("#### 🔬 Technical Explanation", resp.answer)
        self.assertIn("#### 💡 Intuitive Explanation", resp.answer)
        self.assertTrue(resp.grounded)

    def test_compare_topics(self):
        resp = self.service.compare("LoRA", "Full Fine-Tuning")
        self.assertIsInstance(resp, ScientificExpertResponse)
        self.assertEqual(resp.intent, INTENT_COMPARISON)
        self.assertIn("### ⚖️ Scientific Comparison", resp.answer)
        self.assertIn("### 📊 Comparative Analysis", resp.answer)
        self.assertTrue(resp.grounded)
        self.assertGreaterEqual(len(resp.sources), 1)

    def test_conversational_follow_up_through_service(self):
        self.service.clear_session()

        # Turn 1
        resp1 = self.service.ask("Explain Retrieval-Augmented Generation.")
        self.assertIn("2005.11401", resp1.answer)

        # Turn 2: Follow-up question with pronoun
        resp2 = self.service.ask("What are its limitations?")
        self.assertIn("Retrieval-Augmented Generation", resp2.condensed_query)
        self.assertTrue(resp2.grounded)
        self.assertEqual(len(self.service.get_session_messages()), 4)

    def test_failure_cases_and_input_validation(self):
        # Empty query
        with self.assertRaises(ValueError):
            self.service.ask("   ")

        with self.assertRaises(TypeError):
            self.service.ask(12345)  # type: ignore

        # Empty concept
        with self.assertRaises(ValueError):
            self.service.explain_concept("")

        with self.assertRaises(TypeError):
            self.service.explain_concept(12345)  # type: ignore

        # Empty compare
        with self.assertRaises(ValueError):
            self.service.compare("LoRA", "   ")

        # Unknown paper summary returns safe fallback
        unknown_summary = self.service.summarize_paper("9999.88888")
        self.assertFalse(unknown_summary.grounded)
        self.assertIn("Could not find scientific paper", unknown_summary.answer)

    def test_analyze_paper_by_arxiv_id_success(self):
        """Verify successful structured paper analysis for a known paper by arXiv ID."""
        analysis = self.service.analyze_paper("1706.03762")
        self.assertIsNotNone(analysis)
        self.assertIsInstance(analysis, StructuredPaperAnalysis)
        self.assertEqual(analysis.arxiv_id, "1706.03762")
        self.assertEqual(analysis.title, "Attention Is All You Need")
        self.assertIn("Encoder-decoder Transformer", analysis.model_architecture)
        self.assertEqual(len(analysis.key_contributions), 2)
        self.assertEqual(len(analysis.datasets_benchmarks), 2)
        self.assertEqual(len(analysis.quantitative_results), 2)
        self.assertEqual(len(analysis.limitations), 1)
        self.assertEqual(len(analysis.technical_concepts), 3)

    def test_analyze_paper_by_title_success(self):
        """Verify successful structured paper analysis by title query."""
        analysis = self.service.analyze_paper("Attention Is All You Need")
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.arxiv_id, "1706.03762")
        self.assertEqual(analysis.title, "Attention Is All You Need")

    def test_analyze_paper_non_existent_returns_none(self):
        """Verify that non-existent paper query gracefully returns None."""
        analysis = self.service.analyze_paper("9999.88888")
        self.assertIsNone(analysis)

    def test_analyze_paper_input_validation(self):
        """Verify empty and invalid type inputs for analyze_paper raise proper errors."""
        with self.assertRaises(ValueError):
            self.service.analyze_paper("   ")

        with self.assertRaises(ValueError):
            self.service.analyze_paper("")

        with self.assertRaises(TypeError):
            self.service.analyze_paper(12345)  # type: ignore

    def test_custom_understanding_service_injection(self):
        """Verify that an explicitly injected PaperUnderstandingService is used by ScientificExpertService."""
        custom_llm = FakeExpertLLM()
        custom_understanding = PaperUnderstandingService(llm=custom_llm)
        custom_service = ScientificExpertService(
            retriever=self.retriever,
            generator=self.generator,
            understanding_service=custom_understanding,
        )
        self.assertIs(custom_service.understanding_service, custom_understanding)

        analysis = custom_service.analyze_paper("1706.03762")
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.arxiv_id, "1706.03762")

    def test_analyze_paper_records_in_session_history(self):
        """Verify that analyze_paper appends user and assistant messages to active session."""
        initial_history_count = len(self.service.get_session_messages())
        analysis = self.service.analyze_paper("1706.03762")
        self.assertIsNotNone(analysis)

        updated_history = self.service.get_session_messages()
        self.assertEqual(len(updated_history), initial_history_count + 2)
        self.assertEqual(updated_history[-2].role, "user")
        self.assertIn("Analyze paper: 1706.03762", updated_history[-2].content)
        self.assertEqual(updated_history[-1].role, "assistant")
        self.assertIn("Extracted structured analysis for 'Attention Is All You Need'", updated_history[-1].content)

    def test_summarize_paper_by_title(self):
        """Verify that summarize_paper resolves by title and generates a structured summary."""
        resp = self.service.summarize_paper("Attention Is All You Need")
        self.assertIsInstance(resp, ScientificExpertResponse)
        self.assertEqual(resp.intent, INTENT_SUMMARY)
        self.assertIn("### 📄 Paper Information", resp.answer)
        self.assertIn("### 💡 Key Contributions", resp.answer)
        self.assertTrue(resp.grounded)
        self.assertEqual(resp.sources[0].title, "Attention Is All You Need")

    def test_summarize_paper_by_versioned_arxiv_id(self):
        """Verify that summarize_paper resolves versioned arXiv IDs like 2005.11401v2."""
        resp = self.service.summarize_paper("2005.11401v2")
        self.assertIsInstance(resp, ScientificExpertResponse)
        self.assertEqual(resp.intent, INTENT_SUMMARY)
        self.assertTrue(resp.grounded)
        self.assertEqual(resp.sources[0].arxiv_id, "2005.11401")

    def test_summarize_paper_input_validation_types(self):
        """Verify that summarize_paper enforces TypeError for non-strings and ValueError for empty strings."""
        with self.assertRaises(TypeError):
            self.service.summarize_paper(None)  # type: ignore

        with self.assertRaises(TypeError):
            self.service.summarize_paper(12345)  # type: ignore

        with self.assertRaises(ValueError):
            self.service.summarize_paper("")

        with self.assertRaises(ValueError):
            self.service.summarize_paper("   ")

    def test_summarize_paper_refusal_detection(self):
        """Verify that an LLM refusal / insufficient-evidence phrase marks the summary as ungrounded."""
        refusal_llm = FakeExpertLLM()
        refusal_llm.invoke = lambda prompt: AIMessage(  # type: ignore
            content="The provided scientific papers do not contain sufficient evidence to answer this question."
        )
        refusal_service = ScientificExpertService(
            retriever=self.retriever,
            generator=ScientificGenerator(llm=refusal_llm),
        )
        resp = refusal_service.summarize_paper("1706.03762")
        self.assertIsInstance(resp, ScientificExpertResponse)
        self.assertFalse(resp.grounded)
        self.assertIsNotNone(resp.warning_message)

    def test_summarize_paper_unsupported_citation_warning(self):
        """Verify that hallucinated citations in a summary produce an ungrounded warning."""
        hallucinating_llm = FakeExpertLLM()
        hallucinating_llm.invoke = lambda prompt: AIMessage(  # type: ignore
            content="### 📄 Paper Information\n- Paper: Attention (1706.03762)\n\nCites unsupported paper 9999.88888 as evidence."
        )
        hallucinating_service = ScientificExpertService(
            retriever=self.retriever,
            generator=ScientificGenerator(llm=hallucinating_llm),
        )
        resp = hallucinating_service.summarize_paper("1706.03762")
        self.assertIsInstance(resp, ScientificExpertResponse)
        self.assertFalse(resp.grounded)
        self.assertIn("9999.88888", str(resp.warning_message))

    def test_explain_concept_input_validation_types(self):
        """Verify that explain_concept enforces TypeError for non-strings and ValueError for empty strings."""
        with self.assertRaises(TypeError):
            self.service.explain_concept(None)  # type: ignore

        with self.assertRaises(TypeError):
            self.service.explain_concept(12345)  # type: ignore

        with self.assertRaises(ValueError):
            self.service.explain_concept("")

        with self.assertRaises(ValueError):
            self.service.explain_concept("   ")

    def test_explain_concept_no_retrieval_evidence_safe_fallback(self):
        """Verify that when no scientific papers are retrieved, explain_concept returns safely without recursion."""
        empty_retriever = ScientificRetriever(vector_store=self.retriever.vector_store)
        empty_retriever.retrieve = lambda query, k=3: []  # type: ignore

        safe_service = ScientificExpertService(
            retriever=empty_retriever,
            generator=self.generator,
        )
        resp = safe_service.explain_concept("Quantum Superposition in NLP")
        self.assertIsInstance(resp, ScientificExpertResponse)
        self.assertEqual(resp.intent, INTENT_CONCEPT_EXPLANATION)
        self.assertFalse(resp.grounded)
        self.assertEqual(len(resp.sources), 0)
        self.assertIsNotNone(resp.warning_message)
        self.assertIn("No supporting scientific literature evidence", str(resp.warning_message))

    def test_explain_concept_refusal_detection(self):
        """Verify that an LLM refusal phrase in concept explanation marks the response as ungrounded."""
        refusal_llm = FakeExpertLLM()
        refusal_llm.invoke = lambda prompt: AIMessage(  # type: ignore
            content="The provided scientific papers do not contain sufficient evidence to explain this concept."
        )
        refusal_service = ScientificExpertService(
            retriever=self.retriever,
            generator=ScientificGenerator(llm=refusal_llm),
        )
        resp = refusal_service.explain_concept("Multi-Head Attention")
        self.assertIsInstance(resp, ScientificExpertResponse)
        self.assertFalse(resp.grounded)
        self.assertIsNotNone(resp.warning_message)

    def test_explain_concept_unsupported_citation_warning(self):
        """Verify that hallucinated citations in concept explanation produce an ungrounded warning."""
        hallucinating_llm = FakeExpertLLM()
        hallucinating_llm.invoke = lambda prompt: AIMessage(  # type: ignore
            content="### 🧠 Concept: Multi-Head Attention\n\n#### 🔬 Technical Explanation\nIntroduced in 9999.88888 as a scalable mechanism."
        )
        hallucinating_service = ScientificExpertService(
            retriever=self.retriever,
            generator=ScientificGenerator(llm=hallucinating_llm),
        )
        resp = hallucinating_service.explain_concept("Multi-Head Attention")
        self.assertIsInstance(resp, ScientificExpertResponse)
        self.assertFalse(resp.grounded)
        self.assertIn("9999.88888", str(resp.warning_message))

    def test_followup_comparison_resolves_context(self):
        """Verify that follow-up comparison resolves pronouns to previous conversation context."""
        self.service.clear_session()
        resp1 = self.service.ask("Explain Transformers.")
        self.assertIn("1706.03762", resp1.answer)

        resp2 = self.service.ask("Compare it with RNNs.")
        self.assertEqual(resp2.query, "Compare it with RNNs.")
        self.assertIn("Transformers", resp2.condensed_query)
        self.assertEqual(resp2.intent, INTENT_COMPARISON)
        self.assertTrue(resp2.grounded)
        self.assertIn("Scientific Comparison", resp2.answer)
        self.assertEqual(self.service.get_session_messages()[-2].content, "Compare it with RNNs.")

    def test_followup_concept_explanation_resolves_context(self):
        """Verify that follow-up concept explanation resolves pronouns to previous conversation entity."""
        self.service.clear_session()
        resp1 = self.service.ask("What is BERT?")
        self.assertIsNotNone(resp1.answer)

        resp2 = self.service.ask("Explain its masking mechanism")
        self.assertEqual(resp2.query, "Explain its masking mechanism")
        self.assertIn("BERT", resp2.condensed_query)
        self.assertEqual(resp2.intent, INTENT_CONCEPT_EXPLANATION)
        self.assertTrue(resp2.grounded)
        self.assertIn("Concept: Masked Language Modeling", resp2.answer)
        self.assertEqual(self.service.get_session_messages()[-2].content, "Explain its masking mechanism")

    def test_followup_summary_resolves_context(self):
        """Verify that follow-up paper summarization resolves paper reference to previous turn."""
        self.service.clear_session()
        resp1 = self.service.ask("Tell me about paper 1706.03762")
        self.assertIsNotNone(resp1.answer)

        resp2 = self.service.ask("Summarize this paper")
        self.assertEqual(resp2.query, "Summarize this paper")
        self.assertIn("1706.03762", resp2.condensed_query)
        self.assertEqual(resp2.intent, INTENT_SUMMARY)
        self.assertTrue(resp2.grounded)
        self.assertIn("Paper Information", resp2.answer)
        self.assertEqual(self.service.get_session_messages()[-2].content, "Summarize this paper")

    def test_multi_session_context_isolation(self):
        """Verify that concurrent separate sessions do not bleed conversation context across sessions."""
        session_a = ScientificConversationSession("session_a")
        session_b = ScientificConversationSession("session_b")

        # Turn 1 in each session
        self.service.ask("Explain Transformers.", session=session_a)
        self.service.ask("What is BERT?", session=session_b)

        # Turn 2 follow-ups
        resp_a = self.service.ask("Compare it with RNNs.", session=session_a)
        resp_b = self.service.ask("Explain its masking mechanism", session=session_b)

        # Session A must resolve to Transformers and never mention BERT
        self.assertIn("Transformers", resp_a.condensed_query)
        self.assertNotIn("BERT", resp_a.condensed_query)

        # Session B must resolve to BERT and never mention Transformers
        self.assertIn("BERT", resp_b.condensed_query)
        self.assertNotIn("Transformers", resp_b.condensed_query)

        # Session message counts are isolated
        self.assertEqual(len(session_a.get_messages()), 4)
        self.assertEqual(len(session_b.get_messages()), 4)

    def test_followup_with_empty_history_is_safe(self):
        """Verify that asking a follow-up-like query with no history does not crash or loop."""
        empty_session = ScientificConversationSession("empty_session")
        resp = self.service.ask("Compare it with RNNs", session=empty_session)
        self.assertIsInstance(resp, ScientificExpertResponse)
        self.assertEqual(resp.query, "Compare it with RNNs")
        self.assertEqual(len(empty_session.get_messages()), 2)

    def test_followup_preserves_grounding_and_sources(self):
        """Verify that multi-turn follow-up queries retain grounded sources and citation verification."""
        self.service.clear_session()
        self.service.ask("Explain Retrieval-Augmented Generation.")
        resp2 = self.service.ask("What are its limitations?")

        self.assertTrue(resp2.grounded)
        self.assertGreaterEqual(len(resp2.sources), 1)
        self.assertGreaterEqual(len(resp2.citations), 1)
        self.assertEqual(resp2.citations[0].arxiv_id, "2005.11401")
        self.assertIn("2005.11401", resp2.answer)

    def test_service_isolation_from_production_stores(self):
        """Verify that ScientificExpertService never alters customer or medical stores."""
        proj_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        customer_faiss = proj_root / "faiss_index"
        medical_faiss = proj_root / "faiss_index_medical"

        cust_mtime = customer_faiss.stat().st_mtime if customer_faiss.exists() else None
        med_mtime = medical_faiss.stat().st_mtime if medical_faiss.exists() else None

        self.service.ask("Transformer test")
        self.service.summarize_paper("1706.03762")
        self.service.compare("LoRA", "Fine-tuning")
        self.service.analyze_paper("1706.03762")

        if customer_faiss.exists():
            self.assertEqual(customer_faiss.stat().st_mtime, cust_mtime)
        if medical_faiss.exists():
            self.assertEqual(medical_faiss.stat().st_mtime, med_mtime)


if __name__ == "__main__":
    unittest.main()

