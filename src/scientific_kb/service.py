"""Scientific Domain Expert Service.

Integrates Ingestion, Vector Store, Multi-Criteria Retrieval, Grounded LLM
Generation, Citation Validation, and Conversational Follow-Up into a single
high-level RAG service for scientific inquiry.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.conversation import (  # type: ignore
        ChatMessage,
        ConversationalResponse,
        ScientificConversationSession,
        ScientificConversationalService,
        condense_followup_query,
    )
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.generation import (  # type: ignore
        INSUFFICIENT_EVIDENCE_PHRASE,
        ScientificAnswer,
        ScientificGenerator,
        build_scientific_prompt,
    )
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.grounding import (  # type: ignore
        Citation,
        GroundingValidationResult,
        extract_cited_arxiv_ids,
        format_grounded_answer,
        validate_answer_grounding,
    )
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.retriever import (  # type: ignore
        ScientificRetrievalResult,
        ScientificRetriever,
        format_retrieval_context,
    )
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.models import (  # type: ignore
        StructuredPaperAnalysis,
    )
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.understanding import (  # type: ignore
        PaperUnderstandingService,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb.conversation import (  # type: ignore
        ChatMessage,
        ConversationalResponse,
        ScientificConversationSession,
        ScientificConversationalService,
        condense_followup_query,
    )
    # pyrefly: ignore [missing-import]
    from scientific_kb.generation import (  # type: ignore
        INSUFFICIENT_EVIDENCE_PHRASE,
        ScientificAnswer,
        ScientificGenerator,
        build_scientific_prompt,
    )
    # pyrefly: ignore [missing-import]
    from scientific_kb.grounding import (  # type: ignore
        Citation,
        GroundingValidationResult,
        extract_cited_arxiv_ids,
        format_grounded_answer,
        validate_answer_grounding,
    )
    # pyrefly: ignore [missing-import]
    from scientific_kb.retriever import (  # type: ignore
        ScientificRetrievalResult,
        ScientificRetriever,
        format_retrieval_context,
    )
    # pyrefly: ignore [missing-import]
    from scientific_kb.models import (  # type: ignore
        StructuredPaperAnalysis,
    )
    # pyrefly: ignore [missing-import]
    from scientific_kb.understanding import (  # type: ignore
        PaperUnderstandingService,
    )

# Intent Constants
INTENT_GENERAL = "general_question"
INTENT_PAPER_LOOKUP = "paper_lookup"
INTENT_SUMMARY = "paper_summary"
INTENT_CONCEPT_EXPLANATION = "concept_explanation"
INTENT_COMPARISON = "comparison"


@dataclass
class ScientificExpertResponse:
    """Canonical end-to-end response from the Scientific Expert Service."""

    query: str
    condensed_query: str
    answer: str
    intent: str
    sources: List[ScientificRetrievalResult] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    grounded: bool = True
    warning_message: Optional[str] = None
    formatted_markdown: str = ""


SUMMARY_PROMPT_TEMPLATE = """You are an expert scientific AI research assistant specializing in Computer Science, AI, ML, and NLP.

Provide a comprehensive, evidence-grounded summary of the target paper based strictly on the provided context.

Structure your response with the following exact sections:
### 📄 Paper Information
- Title:
- Authors:
- Publication Info / arXiv ID:

### 📝 Executive Summary
[Concise summary of the paper's core hypothesis and motivation]

### 💡 Key Contributions
1.
2.
3.

### 🔬 Method & Architecture
[Detailed explanation of the proposed approach, equations, or mechanism]

### 📊 Important Findings & Results
[Key empirical findings and conclusions from the paper]

### ⚠️ Limitations & Open Challenges
[Any mentioned constraints or areas of future exploration]

---
SCIENTIFIC EVIDENCE:
{context}
---

PAPER TO SUMMARIZE:
{query}

GROUNDED PAPER SUMMARY:"""

COMPARISON_PROMPT_TEMPLATE = """You are an expert scientific AI research assistant specializing in AI, Machine Learning, and NLP.

Provide a structured, evidence-grounded technical comparison between the two specified topics based strictly on the provided context.

Structure your response with:
### ⚖️ Scientific Comparison
Provide an overview comparing the core paradigms.

### 📊 Comparative Analysis
| Dimension | {topic_a} | {topic_b} |
|---|---|---|
| Core Mechanism | ... | ... |
| Computational Efficiency | ... | ... |
| Memory / Resource Usage | ... | ... |
| Primary Use Cases | ... | ... |
| Key Advantages | ... | ... |
| Main Limitations | ... | ... |

### 🎯 Key Takeaways & Recommendations
[Synthesis of trade-offs and when to use each approach]

---
SCIENTIFIC EVIDENCE:
{context}
---

TOPICS TO COMPARE:
Topic A: {topic_a}
Topic B: {topic_b}

GROUNDED TECHNICAL COMPARISON:"""

CONCEPT_EXPLANATION_PROMPT_TEMPLATE = """You are an expert scientific AI research assistant specializing in AI, Machine Learning, and NLP.

Explain the requested scientific concept thoroughly and accurately based on the provided paper evidence.

Structure your response with:
### 🧠 Concept: {concept}

#### 🔬 Technical Explanation
[Rigorous mathematical and architectural explanation]

#### 💡 Intuitive Explanation
[Intuitive analogy or clear high-level mental model]

#### 🚀 Why It Matters in Modern AI
[Practical significance, real-world impact, and performance improvements]

---
SCIENTIFIC EVIDENCE:
{context}
---

CONCEPT TO EXPLAIN:
{concept}

GROUNDED CONCEPT EXPLANATION:"""


def detect_scientific_intent(query: str) -> str:
    """Deterministically classify user query intent for specialized scientific workflows.

    Args:
        query (str): The raw user query.

    Returns:
        str: Detected intent identifier.
    """
    if not isinstance(query, str):
        return INTENT_GENERAL

    q = query.strip().lower()
    if not q:
        return INTENT_GENERAL

    # 1. Comparison Intent
    if (
        re.search(r"\b(compare|versus| vs | vs\. |difference between|trade-?offs between)\b", q)
        or (" and " in q and q.startswith("compare"))
    ):
        return INTENT_COMPARISON

    # 2. Paper Lookup Intent (explicit lookup phrases or standalone arXiv IDs)
    if re.search(r"\b(explain paper|tell me about paper|what is paper|lookup paper|show paper|find paper)\b", q):
        return INTENT_PAPER_LOOKUP

    # 3. Summary Intent
    if re.search(r"\b(summarize|summary of|give me a summary|brief summary|paper summary)\b", q):
        return INTENT_SUMMARY

    # 4. Concept Explanation Intent (explicit concept requests)
    if re.search(r"\b(explain concept|concept of|like i'm a beginner|for beginners)\b", q) or q.startswith("explain "):
        return INTENT_CONCEPT_EXPLANATION

    # 5. Standalone arXiv ID lookup (e.g. "1706.03762")
    if bool(re.match(r"^\s*\d{4}\.\d{4,5}(?:v\d+)?\s*$", q)):
        return INTENT_PAPER_LOOKUP

    return INTENT_GENERAL


class ScientificExpertService:
    """End-to-end domain expert service for scientific paper search, QA, and explanations."""

    def __init__(
        self,
        retriever: ScientificRetriever,
        generator: ScientificGenerator,
        session: Optional[ScientificConversationSession] = None,
        understanding_service: Optional[PaperUnderstandingService] = None,
    ):
        """Initialize the ScientificExpertService.

        Args:
            retriever (ScientificRetriever): Retriever over scientific FAISS vector store.
            generator (ScientificGenerator): Grounded answer generator.
            session (Optional[ScientificConversationSession]): Multi-turn conversation session.
            understanding_service (Optional[PaperUnderstandingService]): Structured paper analysis service.
        """
        if retriever is None:
            raise ValueError("retriever cannot be None.")
        if generator is None:
            raise ValueError("generator cannot be None.")

        self.retriever = retriever
        self.generator = generator
        self.session = session or ScientificConversationSession()
        self.conversational_service = ScientificConversationalService(
            retriever=self.retriever,
            generator=self.generator,
            default_session=self.session,
        )
        self.understanding_service = understanding_service or PaperUnderstandingService(
            llm=getattr(self.generator, "llm", None)
        )

    def find_paper(
        self,
        arxiv_id: Optional[str] = None,
        title_query: Optional[str] = None,
    ) -> Optional[ScientificRetrievalResult]:
        """Find a specific paper in the scientific vector store by arXiv ID or title.

        Args:
            arxiv_id (Optional[str]): Standard or versioned arXiv identifier.
            title_query (Optional[str]): Paper title or title substring.

        Returns:
            Optional[ScientificRetrievalResult]: Matched paper result, or None if not found.
        """
        # 1. Lookup by arXiv ID
        if arxiv_id:
            cleaned_id = arxiv_id.strip().lower()
            base_id = re.sub(r"v\d+$", "", cleaned_id)

            if hasattr(self.retriever.vector_store, "docstore"):
                docstore = self.retriever.vector_store.docstore._dict
                for doc in docstore.values():
                    meta = doc.metadata or {}
                    doc_id = str(meta.get("arxiv_id", "")).strip().lower()
                    doc_base_id = re.sub(r"v\d+$", "", doc_id)
                    if doc_id == cleaned_id or doc_base_id == base_id:
                        return ScientificRetrievalResult(
                            document=doc,
                            arxiv_id=str(meta.get("arxiv_id", "")),
                            title=str(meta.get("title", "")),
                            authors=str(meta.get("authors", "")),
                            primary_category=str(meta.get("primary_category", "")),
                            categories=list(meta.get("categories", [])),
                            published_date=str(meta.get("published_date", "")),
                            score=0.0,
                            concepts=list(meta.get("concepts", [])),
                            doi=str(meta.get("doi", "")),
                            journal_ref=str(meta.get("journal_ref", "")),
                            url=str(meta.get("url", "")),
                        )

            # Check retrieve results only if exact ID match
            candidates = self.retriever.retrieve(query=arxiv_id, k=3)
            for cand in candidates:
                if cand.arxiv_id.lower() == cleaned_id or re.sub(r"v\d+$", "", cand.arxiv_id.lower()) == base_id:
                    return cand
            return None

        # 2. Lookup by title query
        if title_query and title_query.strip():
            cleaned_title = title_query.strip().lower()

            if hasattr(self.retriever.vector_store, "docstore"):
                docstore = self.retriever.vector_store.docstore._dict
                for doc in docstore.values():
                    meta = doc.metadata or {}
                    doc_title = str(meta.get("title", "")).strip().lower()
                    if cleaned_title in doc_title or doc_title in cleaned_title:
                        return ScientificRetrievalResult(
                            document=doc,
                            arxiv_id=str(meta.get("arxiv_id", "")),
                            title=str(meta.get("title", "")),
                            authors=str(meta.get("authors", "")),
                            primary_category=str(meta.get("primary_category", "")),
                            categories=list(meta.get("categories", [])),
                            published_date=str(meta.get("published_date", "")),
                            score=0.0,
                            concepts=list(meta.get("concepts", [])),
                            doi=str(meta.get("doi", "")),
                            journal_ref=str(meta.get("journal_ref", "")),
                            url=str(meta.get("url", "")),
                        )

            # Fallback to similarity search on title
            candidates = self.retriever.retrieve(query=title_query, k=2)
            if candidates:
                return candidates[0]

        return None

    def ask(
        self,
        query: str,
        session: Optional[ScientificConversationSession] = None,
        category_filter: Optional[Union[str, List[str]]] = None,
        min_year: Optional[int] = None,
        author_filter: Optional[str] = None,
        concept_filter: Optional[Union[str, List[str]]] = None,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> ScientificExpertResponse:
        """Route user query through intent classification and execute grounded scientific RAG.

        Args:
            query (str): The user query.
            session (Optional[ScientificConversationSession]): Session to use.
            category_filter: Optional category filter.
            min_year: Optional minimum publication year.
            author_filter: Optional author filter.
            concept_filter: Optional concept filter.
            k: Top-k papers.
            score_threshold: Score threshold.

        Returns:
            ScientificExpertResponse: Complete grounded scientific response.
        """
        if not isinstance(query, str):
            raise TypeError(f"Expected query to be a string, got {type(query).__name__}")

        stripped_query = query.strip()
        if not stripped_query:
            raise ValueError("Query cannot be empty or whitespace-only.")

        active_session = session or self.session

        # 1. Condense follow-up query based on active session history before intent routing
        condensed_query = condense_followup_query(
            query=stripped_query,
            chat_history=active_session.get_messages(),
            llm=getattr(self.generator, "llm", None),
        )

        # 2. Intent detection on the resolved/condensed query
        intent = detect_scientific_intent(condensed_query)

        if intent == INTENT_COMPARISON:
            resp = self._handle_comparison_query(condensed_query, active_session)
            resp.query = stripped_query
            resp.condensed_query = condensed_query
            if active_session.messages and active_session.messages[-2].role == "user":
                active_session.messages[-2].content = stripped_query
            return resp

        if intent == INTENT_PAPER_LOOKUP:
            resp = self._handle_paper_lookup_query(condensed_query, active_session)
            resp.query = stripped_query
            resp.condensed_query = condensed_query
            if active_session.messages and active_session.messages[-2].role == "user":
                active_session.messages[-2].content = stripped_query
            return resp

        if intent == INTENT_SUMMARY:
            resp = self._handle_summary_query(condensed_query, active_session)
            resp.query = stripped_query
            resp.condensed_query = condensed_query
            if active_session.messages and active_session.messages[-2].role == "user":
                active_session.messages[-2].content = stripped_query
            return resp

        if intent == INTENT_CONCEPT_EXPLANATION:
            resp = self._handle_concept_explanation_query(condensed_query, active_session)
            resp.query = stripped_query
            resp.condensed_query = condensed_query
            if active_session.messages and active_session.messages[-2].role == "user":
                active_session.messages[-2].content = stripped_query
            return resp

        # Default General Scientific QA via Conversational Service
        conv_resp = self.conversational_service.chat(
            query=stripped_query,
            session=active_session,
            category_filter=category_filter,
            min_year=min_year,
            author_filter=author_filter,
            concept_filter=concept_filter,
            k=k,
            score_threshold=score_threshold,
            condensed_query=condensed_query,
        )

        return ScientificExpertResponse(
            query=conv_resp.query,
            condensed_query=conv_resp.condensed_query,
            answer=conv_resp.answer.answer,
            intent=INTENT_GENERAL,
            sources=conv_resp.sources,
            citations=conv_resp.validation.valid_citations,
            grounded=conv_resp.validation.is_grounded,
            warning_message=conv_resp.validation.warning_message,
            formatted_markdown=conv_resp.formatted_response,
        )

    def summarize_paper(
        self,
        paper_query_or_id: str,
        session: Optional[ScientificConversationSession] = None,
    ) -> ScientificExpertResponse:
        """Generate a structured, evidence-grounded summary of a specific scientific paper."""
        if not isinstance(paper_query_or_id, str):
            raise TypeError("Paper identifier or title must be a string.")
        if not paper_query_or_id.strip():
            raise ValueError("Paper identifier or title cannot be empty.")

        active_session = session or self.session
        stripped_target = paper_query_or_id.strip()

        # 1. Check if an arXiv ID was specified
        target_ids = extract_cited_arxiv_ids(stripped_target)
        if target_ids:
            found = self.find_paper(arxiv_id=target_ids[0])
            retrieved_sources = [found] if found else []
        else:
            found = self.find_paper(title_query=stripped_target)
            retrieved_sources = [found] if found else self.retriever.retrieve(stripped_target, k=1)

        if not retrieved_sources:
            ans = ScientificAnswer(
                query=stripped_target,
                answer=f"Could not find scientific paper evidence for '{stripped_target}'.",
                sources=[],
                grounded=False,
            )
            val = GroundingValidationResult(
                is_grounded=False,
                valid_citations=[],
                unsupported_citations=[],
                warning_message=f"Paper not found: {stripped_target}",
            )
            return ScientificExpertResponse(
                query=stripped_target,
                condensed_query=stripped_target,
                answer=ans.answer,
                intent=INTENT_SUMMARY,
                sources=[],
                citations=[],
                grounded=False,
                warning_message=val.warning_message,
                formatted_markdown=format_grounded_answer(ans, val),
            )

        context_str = format_retrieval_context(retrieved_sources)
        prompt = SUMMARY_PROMPT_TEMPLATE.format(context=context_str, query=stripped_target)
        raw_output = self.generator._invoke_llm(prompt)
        clean_answer = raw_output.strip() if isinstance(raw_output, str) else str(raw_output).strip()

        norm_ans = clean_answer.lower()
        is_refusal_or_empty = (
            not clean_answer
            or "not contain sufficient evidence" in norm_ans
            or "insufficient evidence" in norm_ans
            or norm_ans.startswith("i don't know")
            or norm_ans.startswith("i do not know")
        )

        ans = ScientificAnswer(
            query=stripped_target,
            answer=clean_answer if clean_answer else INSUFFICIENT_EVIDENCE_PHRASE,
            sources=retrieved_sources,
            grounded=not is_refusal_or_empty,
            raw_response=raw_output,
        )
        val = validate_answer_grounding(ans, retrieved_sources)
        if is_refusal_or_empty:
            val = GroundingValidationResult(
                is_grounded=False,
                valid_citations=val.valid_citations,
                unsupported_citations=val.unsupported_citations,
                warning_message=val.warning_message or "Insufficient evidence or empty summary generated.",
            )
        formatted_md = format_grounded_answer(ans, val)

        active_session.add_user_message(f"Summarize paper: {stripped_target}")
        active_session.add_assistant_message(ans.answer, sources=retrieved_sources)

        return ScientificExpertResponse(
            query=stripped_target,
            condensed_query=stripped_target,
            answer=ans.answer,
            intent=INTENT_SUMMARY,
            sources=retrieved_sources,
            citations=val.valid_citations,
            grounded=val.is_grounded,
            warning_message=val.warning_message,
            formatted_markdown=formatted_md,
        )

    def analyze_paper(
        self,
        paper_query_or_id: str,
        session: Optional[ScientificConversationSession] = None,
    ) -> Optional[StructuredPaperAnalysis]:
        """Extract structured research understanding (methodology, contributions, results, limitations) from a paper.

        Args:
            paper_query_or_id (str): Paper arXiv ID, title, or natural language query.
            session (Optional[ScientificConversationSession]): Optional conversation session for history tracking.

        Returns:
            Optional[StructuredPaperAnalysis]: Validated structured analysis if paper is found, None otherwise.

        Raises:
            TypeError: If paper_query_or_id is not a string.
            ValueError: If paper_query_or_id is empty or whitespace-only.
        """
        if not isinstance(paper_query_or_id, str):
            raise TypeError("Paper identifier or title must be a string.")
        if not paper_query_or_id.strip():
            raise ValueError("Paper identifier or title cannot be empty.")

        active_session = session or self.session
        stripped_target = paper_query_or_id.strip()

        # 1. Resolve paper by arXiv ID
        target_ids = extract_cited_arxiv_ids(stripped_target)
        if target_ids:
            found = self.find_paper(arxiv_id=target_ids[0])
        else:
            # 2. Resolve paper by title or semantic retrieval
            found = self.find_paper(title_query=stripped_target)
            if not found:
                candidates = self.retriever.retrieve(stripped_target, k=1)
                found = candidates[0] if candidates else None

        if not found:
            return None

        # 3. Delegate to PaperUnderstandingService
        analysis = self.understanding_service.analyze_paper_structure(found)

        # 4. Record interaction in active session history
        active_session.add_user_message(f"Analyze paper: {stripped_target}")
        active_session.add_assistant_message(
            f"Extracted structured analysis for '{analysis.title}' (arXiv:{analysis.arxiv_id}).",
            sources=[found],
        )

        return analysis

    def explain_concept(
        self,
        concept: str,
        session: Optional[ScientificConversationSession] = None,
    ) -> ScientificExpertResponse:
        """Generate a grounded, two-tiered (technical & intuitive) explanation of an AI/ML concept."""
        if not isinstance(concept, str):
            raise TypeError("Concept name must be a string.")
        if not concept.strip():
            raise ValueError("Concept name cannot be empty.")

        active_session = session or self.session
        stripped_concept = concept.strip()

        retrieved_sources = self.retriever.retrieve(stripped_concept, k=3)
        if not retrieved_sources:
            ans = ScientificAnswer(
                query=stripped_concept,
                answer=INSUFFICIENT_EVIDENCE_PHRASE,
                sources=[],
                grounded=False,
                raw_response=INSUFFICIENT_EVIDENCE_PHRASE,
            )
            val = GroundingValidationResult(
                is_grounded=False,
                valid_citations=[],
                unsupported_citations=[],
                warning_message=f"No supporting scientific literature evidence was retrieved for concept '{stripped_concept}'.",
            )
            formatted_md = format_grounded_answer(ans, val)

            active_session.add_user_message(f"Explain concept: {stripped_concept}")
            active_session.add_assistant_message(ans.answer, sources=[])

            return ScientificExpertResponse(
                query=stripped_concept,
                condensed_query=stripped_concept,
                answer=ans.answer,
                intent=INTENT_CONCEPT_EXPLANATION,
                sources=[],
                citations=[],
                grounded=False,
                warning_message=val.warning_message,
                formatted_markdown=formatted_md,
            )

        context_str = format_retrieval_context(retrieved_sources)
        prompt = CONCEPT_EXPLANATION_PROMPT_TEMPLATE.format(context=context_str, concept=stripped_concept)
        raw_output = self.generator._invoke_llm(prompt)
        clean_answer = raw_output.strip() if isinstance(raw_output, str) else str(raw_output).strip()

        norm_ans = clean_answer.lower()
        is_refusal_or_empty = (
            not clean_answer
            or "not contain sufficient evidence" in norm_ans
            or "insufficient evidence" in norm_ans
            or norm_ans.startswith("i don't know")
            or norm_ans.startswith("i do not know")
        )

        ans = ScientificAnswer(
            query=stripped_concept,
            answer=clean_answer if clean_answer else INSUFFICIENT_EVIDENCE_PHRASE,
            sources=retrieved_sources,
            grounded=not is_refusal_or_empty,
            raw_response=raw_output,
        )
        val = validate_answer_grounding(ans, retrieved_sources)
        if is_refusal_or_empty:
            val = GroundingValidationResult(
                is_grounded=False,
                valid_citations=val.valid_citations,
                unsupported_citations=val.unsupported_citations,
                warning_message=val.warning_message or "Insufficient evidence or empty explanation generated.",
            )
        formatted_md = format_grounded_answer(ans, val)

        active_session.add_user_message(f"Explain concept: {stripped_concept}")
        active_session.add_assistant_message(ans.answer, sources=retrieved_sources)

        return ScientificExpertResponse(
            query=stripped_concept,
            condensed_query=stripped_concept,
            answer=ans.answer,
            intent=INTENT_CONCEPT_EXPLANATION,
            sources=retrieved_sources,
            citations=val.valid_citations,
            grounded=val.is_grounded,
            warning_message=val.warning_message,
            formatted_markdown=formatted_md,
        )

    def compare(
        self,
        topic_a: str,
        topic_b: str,
        session: Optional[ScientificConversationSession] = None,
    ) -> ScientificExpertResponse:
        """Generate a structured, evidence-grounded comparison between two scientific paradigms."""
        if not isinstance(topic_a, str) or not topic_a.strip() or not isinstance(topic_b, str) or not topic_b.strip():
            raise ValueError("Both topic_a and topic_b must be non-empty strings.")

        active_session = session or self.session
        clean_a = topic_a.strip()
        clean_b = topic_b.strip()

        # Retrieve evidence for both topics independently to ensure balanced coverage
        results_a = self.retriever.retrieve(clean_a, k=2)
        results_b = self.retriever.retrieve(clean_b, k=2)

        # Merge and deduplicate retrieved sources
        seen_ids = set()
        combined_sources: List[ScientificRetrievalResult] = []
        for res in results_a + results_b:
            if res.arxiv_id not in seen_ids:
                seen_ids.add(res.arxiv_id)
                combined_sources.append(res)

        if not combined_sources:
            ans = ScientificAnswer(
                query=f"Compare {clean_a} vs {clean_b}",
                answer=f"Could not find sufficient scientific literature evidence to compare '{clean_a}' and '{clean_b}'.",
                sources=[],
                grounded=False,
                raw_response="No evidence retrieved for comparison.",
            )
            val = GroundingValidationResult(
                is_grounded=False,
                valid_citations=[],
                unsupported_citations=[],
                warning_message=f"No supporting scientific evidence was retrieved for '{clean_a}' or '{clean_b}'.",
            )
            formatted_md = format_grounded_answer(ans, val)

            active_session.add_user_message(f"Compare {clean_a} vs {clean_b}")
            active_session.add_assistant_message(ans.answer, sources=[])

            return ScientificExpertResponse(
                query=f"Compare {clean_a} vs {clean_b}",
                condensed_query=f"Compare {clean_a} and {clean_b}",
                answer=ans.answer,
                intent=INTENT_COMPARISON,
                sources=[],
                citations=[],
                grounded=False,
                warning_message=val.warning_message,
                formatted_markdown=formatted_md,
            )

        context_str = format_retrieval_context(combined_sources)
        prompt = COMPARISON_PROMPT_TEMPLATE.format(
            context=context_str,
            topic_a=clean_a,
            topic_b=clean_b,
        )
        raw_output = self.generator._invoke_llm(prompt)

        ans = ScientificAnswer(
            query=f"Compare {clean_a} vs {clean_b}",
            answer=raw_output.strip(),
            sources=combined_sources,
            grounded=True,
            raw_response=raw_output,
        )
        val = validate_answer_grounding(ans, combined_sources)
        formatted_md = format_grounded_answer(ans, val)

        active_session.add_user_message(f"Compare {clean_a} vs {clean_b}")
        active_session.add_assistant_message(ans.answer, sources=combined_sources)

        return ScientificExpertResponse(
            query=f"Compare {clean_a} vs {clean_b}",
            condensed_query=f"Compare {clean_a} and {clean_b}",
            answer=ans.answer,
            intent=INTENT_COMPARISON,
            sources=combined_sources,
            citations=val.valid_citations,
            grounded=val.is_grounded,
            warning_message=val.warning_message,
            formatted_markdown=formatted_md,
        )

    def _handle_comparison_query(self, query: str, session: ScientificConversationSession) -> ScientificExpertResponse:
        match = re.search(r"(?:compare|difference between|versus)\s+(.+?)\s+(?:and|with|to|vs\.?)\s+(.+)", query, re.IGNORECASE)
        if match:
            topic_a = match.group(1).strip()
            topic_b = match.group(2).strip()
            return self.compare(topic_a, topic_b, session=session)
        return self._handle_general_fallback(query, INTENT_COMPARISON, session)

    def _handle_paper_lookup_query(self, query: str, session: ScientificConversationSession) -> ScientificExpertResponse:
        cited_ids = extract_cited_arxiv_ids(query)
        if cited_ids:
            found = self.find_paper(arxiv_id=cited_ids[0])
            if found:
                return self.summarize_paper(cited_ids[0], session=session)
        return self._handle_general_fallback(query, INTENT_PAPER_LOOKUP, session)

    def _handle_summary_query(self, query: str, session: ScientificConversationSession) -> ScientificExpertResponse:
        cleaned = re.sub(r"\b(summarize|summary of|give me a summary of|paper summary)\b", "", query, flags=re.IGNORECASE).strip()
        if not cleaned:
            cleaned = query
        return self.summarize_paper(cleaned, session=session)

    def _handle_concept_explanation_query(self, query: str, session: ScientificConversationSession) -> ScientificExpertResponse:
        cleaned = re.sub(r"\b(explain |explain concept|what is |how does |like i'm a beginner|for beginners)\b", "", query, flags=re.IGNORECASE).strip()
        if not cleaned:
            cleaned = query
        return self.explain_concept(cleaned, session=session)

    def _handle_general_fallback(self, query: str, intent: str, session: ScientificConversationSession) -> ScientificExpertResponse:
        resp = self.conversational_service.chat(query=query, session=session, condensed_query=query)
        return ScientificExpertResponse(
            query=resp.query,
            condensed_query=resp.condensed_query,
            answer=resp.answer.answer,
            intent=intent,
            sources=resp.sources,
            citations=resp.validation.valid_citations,
            grounded=resp.validation.is_grounded,
            warning_message=resp.validation.warning_message,
            formatted_markdown=resp.formatted_response,
        )

    def clear_session(self) -> None:
        """Clear the active conversation session."""
        self.session.clear()

    def get_session_messages(self) -> List[ChatMessage]:
        """Return the current conversation history messages."""
        return self.session.get_messages()
