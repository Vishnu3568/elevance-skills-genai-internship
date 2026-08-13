"""Chatbot Service module for Customer Service Chatbot.

Encapsulates end-to-end pipeline orchestration: input validation, sentiment analysis,
RAG retrieval, OOD detection, and response policy application into a structured payload.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from sentiment_analyzer import analyze_sentiment
from langchain_helper import get_qa_chain
from response_policy import apply_response_policy


@dataclass
class ChatbotResponse:
    """Structured response object containing query results and execution metadata."""
    query: str
    final_answer: str
    raw_answer: str
    sentiment_label: str
    confidence_score: float
    is_ood: bool
    source_documents: list


class ChatbotService:
    """Service layer orchestrating sentiment analysis, RAG QA retrieval, and response policy."""

    def __init__(self):
        pass

    def process_query(self, query: str) -> ChatbotResponse:
        """Process a customer query through the end-to-end chatbot pipeline.

        Args:
            query (str): The customer query string.

        Returns:
            ChatbotResponse: Structured data object containing answers and metadata.

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is empty or whitespace-only.
        """
        if not isinstance(query, str):
            raise TypeError(f"Expected query to be a string, got {type(query).__name__}")

        stripped_query = query.strip()
        if not stripped_query:
            raise ValueError("Query cannot be empty or whitespace-only.")

        # 1. Sentiment Processing with Failure Isolation
        sentiment_label = "NEUTRAL"
        confidence_score = 0.0
        try:
            sent_res = analyze_sentiment(stripped_query)
            sentiment_label = sent_res.get("label", "neutral").upper()
            confidence_score = float(sent_res.get("score", 0.0))
        except Exception:
            sentiment_label = "NEUTRAL"
            confidence_score = 0.0

        # 2. RAG QA Processing
        chain = get_qa_chain()
        try:
            rag_res = chain.invoke({"query": stripped_query})
        except Exception:
            try:
                rag_res = chain({"query": stripped_query})
            except Exception:
                rag_res = chain(stripped_query)

        source_documents = []
        if isinstance(rag_res, dict):
            raw_answer = rag_res.get("result", str(rag_res))
            src_docs = rag_res.get("source_documents", [])
            if isinstance(src_docs, list):
                source_documents = src_docs
        else:
            raw_answer = str(rag_res)

        # 3. OOD Detection
        norm_ans = raw_answer.strip().lower().rstrip(".")
        is_ood = norm_ans in ("i don't know", "i do not know") or norm_ans.startswith("i don't know")

        # 4. Response Policy Application
        final_answer = apply_response_policy(raw_answer, sentiment_label, confidence_score)

        return ChatbotResponse(
            query=query,
            final_answer=final_answer,
            raw_answer=raw_answer,
            sentiment_label=sentiment_label,
            confidence_score=confidence_score,
            is_ood=is_ood,
            source_documents=source_documents
        )
