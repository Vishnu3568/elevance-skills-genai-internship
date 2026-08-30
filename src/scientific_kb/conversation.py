"""Conversational follow-up and session context management for the Scientific Chatbot.

Provides bounded chat session state, contextual query condensation for multi-turn
follow-up questions, and end-to-end multi-turn grounded orchestration.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.generation import ScientificAnswer, ScientificGenerator  # type: ignore
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.grounding import (  # type: ignore
        GroundingValidationResult,
        format_grounded_answer,
        validate_answer_grounding,
    )
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.retriever import (  # type: ignore
        ScientificRetrievalResult,
        ScientificRetriever,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb.generation import ScientificAnswer, ScientificGenerator  # type: ignore
    # pyrefly: ignore [missing-import]
    from scientific_kb.grounding import (  # type: ignore
        GroundingValidationResult,
        format_grounded_answer,
        validate_answer_grounding,
    )
    # pyrefly: ignore [missing-import]
    from scientific_kb.retriever import (  # type: ignore
        ScientificRetrievalResult,
        ScientificRetriever,
    )

CONDENSATION_PROMPT_TEMPLATE = """Given the following conversation history and a follow-up question, rephrase the follow-up question into a standalone scientific question that can be understood completely without the prior conversation.

RULES:
1. Replace all pronouns and ambiguous references (such as "it", "its", "that architecture", "the method", "the model") with their specific scientific concepts from the conversation history.
2. If the question is already completely standalone, return it exactly as is.
3. Do NOT answer the question. Only output the standalone rephrased question.

---
CONVERSATION HISTORY:
{history}
---

FOLLOW-UP QUESTION:
{query}

STANDALONE QUESTION:"""

PRONOUN_PATTERN = re.compile(
    r"\b("
    r"the first contribution|the second contribution|the third contribution|"
    r"the first one|the second one|the third one|"
    r"the key contributions|the contributions|"
    r"the findings|the results|the datasets?|the authors?|"
    r"the former|the latter|"
    r"this paper|that paper|the paper|"
    r"the method|the model|the architecture|the approach|the algorithm|"
    r"it|its|this|that|these|those|they|their"
    r")\b",
    re.IGNORECASE,
)

FOLLOWUP_STARTS = (
    "what about", "how about", "and ", "why ", "how ", "who ", "which ",
    "compare with", "limitations", "who are", "who is",
    "which dataset", "what dataset", "what were", "what are",
    "can you", "could you", "tell me more", "explain the", "elaborate",
)



@dataclass
class ChatMessage:
    """Represents a single message in a multi-turn scientific conversation."""

    role: str  # "user" or "assistant"
    content: str
    sources: List[ScientificRetrievalResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ConversationalResponse:
    """Structured response from a multi-turn conversational turn."""

    query: str
    condensed_query: str
    answer: ScientificAnswer
    validation: GroundingValidationResult
    formatted_response: str
    sources: List[ScientificRetrievalResult] = field(default_factory=list)


class ScientificConversationSession:
    """Manages a bounded session history for a scientific chat conversation."""

    def __init__(self, session_id: str = "default", max_history_turns: int = 5):
        """Initialize the conversation session.

        Args:
            session_id (str): Unique identifier for this session.
            max_history_turns (int): Maximum number of turn pairs (user+assistant) to retain.
        """
        self.session_id = session_id
        self.max_history_turns = max_history_turns
        self.messages: List[ChatMessage] = []

    def add_user_message(self, content: str) -> ChatMessage:
        """Add a user message to the session."""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("User message content cannot be empty.")
        msg = ChatMessage(role="user", content=content.strip())
        self.messages.append(msg)
        self._prune_history()
        return msg

    def add_assistant_message(
        self,
        content: str,
        sources: Optional[List[ScientificRetrievalResult]] = None,
    ) -> ChatMessage:
        """Add an assistant message with optional sources to the session."""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Assistant message content cannot be empty.")
        msg = ChatMessage(
            role="assistant",
            content=content.strip(),
            sources=list(sources) if sources else [],
        )
        self.messages.append(msg)
        self._prune_history()
        return msg

    def _prune_history(self) -> None:
        """Prune messages to retain at most max_history_turns * 2 individual messages."""
        max_messages = self.max_history_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_messages(self) -> List[ChatMessage]:
        """Return the current list of messages in this session."""
        return list(self.messages)

    def get_history_text(self) -> str:
        """Format the recent conversation history as dialogue text."""
        if not self.messages:
            return "No previous conversation."

        history_lines = []
        for msg in self.messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            history_lines.append(f"{prefix}: {msg.content}")

        return "\n".join(history_lines)

    def clear(self) -> None:
        """Clear all messages from the session."""
        self.messages.clear()


def condense_followup_query(
    query: str,
    chat_history: List[ChatMessage],
    llm: Optional[Any] = None,
) -> str:
    """Condense a contextual follow-up query into a standalone scientific question.

    If chat history is empty or the query has no reference to past context,
    the original query is returned cleanly.

    Args:
        query (str): The raw user query.
        chat_history (List[ChatMessage]): Prior conversation history.
        llm (Optional[Any]): Optional LLM backend to execute query condensation.

    Returns:
        str: Standalone query for dense retrieval.
    """
    if not isinstance(query, str):
        raise TypeError(f"Expected query to be a string, got {type(query).__name__}")

    stripped_query = query.strip()
    if not stripped_query:
        raise ValueError("Query cannot be empty or whitespace-only.")

    # 1. If no chat history exists, return query as-is
    if not chat_history:
        return stripped_query

    # 2. Check if query contains anaphoric pronouns or references
    has_pronouns = bool(PRONOUN_PATTERN.search(stripped_query))
    is_short_followup = len(stripped_query.split()) <= 10 and (
        stripped_query.lower().startswith(FOLLOWUP_STARTS)
    )

    if not has_pronouns and not is_short_followup and llm is None:
        return stripped_query

    # 3. If LLM is provided, use prompt condensation
    if llm is not None:
        history_text = "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
            for m in chat_history
        )
        prompt = CONDENSATION_PROMPT_TEMPLATE.format(
            history=history_text,
            query=stripped_query,
        )

        try:
            if hasattr(llm, "invoke"):
                res = llm.invoke(prompt)
                output = str(res.content if hasattr(res, "content") else res)
            elif callable(llm):
                output = str(llm(prompt))
            elif hasattr(llm, "predict"):
                output = str(llm.predict(prompt))
            else:
                output = stripped_query

            condensed = output.strip().replace("Standalone Question:", "").strip()
            if condensed:
                return condensed
        except Exception:
            pass

    # 4. Deterministic heuristic condensation fallback when LLM is absent
    last_topic = ""
    for m in reversed(chat_history):
        if m.role == "assistant" and m.sources:
            if m.sources[0].arxiv_id and m.sources[0].arxiv_id not in m.sources[0].title:
                last_topic = f"{m.sources[0].title} ({m.sources[0].arxiv_id})"
            else:
                last_topic = m.sources[0].title
            break
        elif m.role == "user":
            clean_user = re.sub(r"^(?:explain|what is|tell me about|how does|summarize|compare)\s+", "", m.content, flags=re.IGNORECASE).strip()
            last_topic = clean_user if clean_user else m.content
            break

    if last_topic and (has_pronouns or is_short_followup):
        # A. Ordinals / specific contributions (e.g. "Explain the second one")
        if re.search(r"\b(first|second|third|former|latter)\b", stripped_query, re.IGNORECASE) and (
            "contribution" in stripped_query.lower() or "one" in stripped_query.lower() or "former" in stripped_query.lower() or "latter" in stripped_query.lower()
        ):
            if "contribution" in stripped_query.lower():
                resolved = re.sub(r"\b(the\s+(?:first|second|third|former|latter)\s+contribution)\b", rf"\1 of {last_topic}", stripped_query, flags=re.IGNORECASE)
            else:
                resolved = re.sub(r"\b(the\s+(?:first|second|third|former|latter)(?:\s+one)?)\b", rf"the \1 contribution of {last_topic}", stripped_query, flags=re.IGNORECASE)
                resolved = re.sub(r"\bthe\s+the\b", "the", resolved, flags=re.IGNORECASE)
                resolved = re.sub(r"\s+one\s+contribution\b", " contribution", resolved, flags=re.IGNORECASE)
            if resolved != stripped_query:
                return resolved

        # B. Author queries (e.g. "Who are the authors?")
        if re.search(r"\bwho\s+(?:is|are)\s+(?:the\s+)?authors?\b", stripped_query, re.IGNORECASE):
            return re.sub(r"\b(who\s+(?:is|are)\s+(?:the\s+)?authors?)\b", rf"\1 of {last_topic}", stripped_query, flags=re.IGNORECASE)

        # C. Dataset / benchmark queries (e.g. "What dataset did they use?")
        if re.search(r"\b(?:what|which)\s+(?:datasets?|benchmarks?)\b", stripped_query, re.IGNORECASE):
            if re.search(r"\bthey\b", stripped_query, re.IGNORECASE):
                return re.sub(r"\bthey\b", last_topic, stripped_query, flags=re.IGNORECASE)
            if re.search(r"\bthe\s+datasets?\b", stripped_query, re.IGNORECASE):
                return re.sub(r"\b(the\s+datasets?)\b", rf"\1 used in {last_topic}", stripped_query, flags=re.IGNORECASE)
            return f"{stripped_query} in {last_topic}"

        # D. Results / findings queries (e.g. "What were the results?")
        if re.search(r"\b(?:what\s+(?:were|are)\s+)?(the\s+(?:results|findings|key findings))\b", stripped_query, re.IGNORECASE):
            return re.sub(r"\b(the\s+(?:results|findings|key findings))\b", rf"\1 of {last_topic}", stripped_query, flags=re.IGNORECASE)

        # E. Specialized pronoun combinations (e.g. "its limitations", "its advantages")
        if re.search(r"\bits\s+(?:limitations?|advantages?|disadvantages?|contributions?|architecture|method|approach)\b", stripped_query, re.IGNORECASE):
            resolved = re.sub(r"\bits\s+(\w+)\b", rf"the \1 of {last_topic}", stripped_query, flags=re.IGNORECASE)
            if resolved != stripped_query:
                return resolved

        # F. Generic pronoun substitution
        resolved = PRONOUN_PATTERN.sub(last_topic, stripped_query, count=1)
        if resolved != stripped_query:
            return resolved

        return f"{stripped_query} in {last_topic}"

    return stripped_query


class ScientificConversationalService:
    """Full-pipeline conversational orchestrator for the Scientific Domain Expert Chatbot."""

    def __init__(
        self,
        retriever: ScientificRetriever,
        generator: ScientificGenerator,
        default_session: Optional[ScientificConversationSession] = None,
    ):
        """Initialize the conversational service.

        Args:
            retriever (ScientificRetriever): Scientific paper retriever.
            generator (ScientificGenerator): Grounded answer generator.
            default_session (Optional[ScientificConversationSession]): Optional default session.
        """
        if retriever is None:
            raise ValueError("retriever cannot be None.")
        if generator is None:
            raise ValueError("generator cannot be None.")

        self.retriever = retriever
        self.generator = generator
        self.default_session = default_session or ScientificConversationSession()

    def chat(
        self,
        query: str,
        session: Optional[ScientificConversationSession] = None,
        category_filter: Optional[Union[str, List[str]]] = None,
        min_year: Optional[int] = None,
        author_filter: Optional[str] = None,
        concept_filter: Optional[Union[str, List[str]]] = None,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        condensed_query: Optional[str] = None,
    ) -> ConversationalResponse:
        """Process a multi-turn user query, condense context, retrieve papers, and generate grounded answers.

        Args:
            query (str): User question.
            session (Optional[ScientificConversationSession]): Target session (defaults to self.default_session).
            category_filter (Optional[Union[str, List[str]]]): Category filter.
            min_year (Optional[int]): Minimum publication year.
            author_filter (Optional[str]): Author filter.
            concept_filter (Optional[Union[str, List[str]]]): Concept filter.
            k (Optional[int]): Top-k papers to retrieve.
            score_threshold (Optional[float]): Score threshold.
            condensed_query (Optional[str]): Pre-condensed standalone query if already resolved.

        Returns:
            ConversationalResponse: Complete response with answer, sources, validation, and Markdown.
        """
        active_session = session or self.default_session

        # 1. Condense follow-up query if not already pre-condensed
        if condensed_query is not None and isinstance(condensed_query, str) and condensed_query.strip():
            resolved_query = condensed_query.strip()
        else:
            resolved_query = condense_followup_query(
                query=query,
                chat_history=active_session.get_messages(),
                llm=self.generator.llm,
            )

        # 2. Retrieve relevant scientific papers using condensed query
        retrieval_results = self.retriever.retrieve(
            query=resolved_query,
            k=k,
            score_threshold=score_threshold,
            category_filter=category_filter,
            min_year=min_year,
            author_filter=author_filter,
            concept_filter=concept_filter,
        )

        # 3. Generate grounded scientific answer
        answer = self.generator.generate_answer(
            query=resolved_query,
            retrieval_results=retrieval_results,
        )

        # 4. Validate answer grounding and citations
        validation = validate_answer_grounding(answer, retrieval_results)

        # 5. Format publication-grade Markdown response
        formatted_md = format_grounded_answer(answer, validation)

        # 6. Update session state
        active_session.add_user_message(query)
        active_session.add_assistant_message(answer.answer, sources=retrieval_results)

        return ConversationalResponse(
            query=query,
            condensed_query=resolved_query,
            answer=answer,
            validation=validation,
            formatted_response=formatted_md,
            sources=retrieval_results,
        )
