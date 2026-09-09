"""Cross-Lingual Retrieval Bridge for Multilingual Chatbot (Phase 6 Day 32).

Aligns non-English queries (Spanish, French, German, Hindi) into search-optimized
English representations based on Day 30 language detection and Day 31 canonical intent,
then retrieves relevant evidence from the existing English FAISS vector store.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.multilingual.models import (
    CrossLingualRetrievalResult,
    MultilingualIntent,
    MultilingualIntentResult,
    SupportedLanguage,
)


# -----------------------------------------------------------------------------
# Canonical English Retrieval Alignment Profiles
# -----------------------------------------------------------------------------

INTENT_RETRIEVAL_TEMPLATES: Dict[str, str] = {
    MultilingualIntent.REFUND_POLICY.value: "course refund policy money back guarantee cancellation return guidelines",
    MultilingualIntent.PREREQUISITES.value: "prerequisites beginner non-technical programming background laptop 4GB ram",
    MultilingualIntent.COURSE_DETAILS.value: "course duration lifetime access syllabus curriculum schedule datasets",
    MultilingualIntent.CAREER_ASSISTANCE.value: "job assistance virtual internship resume guarantee job placement recruiters",
    MultilingualIntent.SUPPORT_CONTACT.value: "contact instructors discord community questions support doubts",
    MultilingualIntent.PAYMENT_PRICING.value: "EMI payment options installments pricing cost fees discount",
    MultilingualIntent.TECHNICAL_SUPPORT.value: "technical error bug formula power bi mac virtual machine excel code not working",
    MultilingualIntent.GREETING.value: "hello welcome help questions",
    MultilingualIntent.GENERAL_INQUIRY.value: "bootcamp course training data science AI information",
}

# Cross-lingual technical term bridging
TECHNICAL_KEYWORD_MAP: Dict[str, str] = {
    "reembolso": "refund",
    "remboursement": "refund",
    "rückerstattung": "refund",
    "रिफंड": "refund",
    "वापसी": "refund",
    "cuotas": "EMI installments",
    "plazos": "installments",
    "échéances": "installments",
    "ratenzahlung": "EMI installments",
    "किस्त": "EMI installments",
    "ईएमआई": "EMI",
    "empleo": "job",
    "stage": "internship",
    "praktikum": "internship",
    "नौकरी": "job",
    "इंटर्नशिप": "internship",
    "duración": "duration",
    "durée": "duration",
    "dauer": "duration",
    "अवधि": "duration",
    "requisitos": "prerequisites",
    "prérequis": "prerequisites",
    "voraussetzungen": "prerequisites",
    "योग्यता": "prerequisites",
    "discord": "discord",
    "power bi": "power bi",
    "excel": "excel",
    "mac": "mac",
}


class CrossLingualRetriever:
    """Production-grade retrieval bridge connecting multilingual inputs to English vector stores."""

    def __init__(
        self,
        vector_store_retriever: Optional[Any] = None,
        custom_search_fn: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
    ) -> None:
        """Initialize the cross-lingual retriever.

        Args:
            vector_store_retriever: Optional pre-loaded LangChain retriever.
            custom_search_fn: Optional custom search function for test isolation.
        """
        self._retriever = vector_store_retriever
        self._custom_search_fn = custom_search_fn

    def _get_default_retriever(self) -> Optional[Any]:
        """Lazy-load the existing production FAISS retriever from langchain_helper."""
        if self._retriever is not None:
            return self._retriever

        try:
            from src.langchain_helper import get_qa_chain, vectordb_file_path, get_instructor_embeddings
            # pyrefly: ignore [missing-import]
            from langchain_community.vectorstores import FAISS  # type: ignore

            if os.path.exists(vectordb_file_path):
                embeddings = get_instructor_embeddings()
                try:
                    vectordb = FAISS.load_local(
                        vectordb_file_path,
                        embeddings,
                        allow_dangerous_deserialization=True,
                    )
                except TypeError:
                    vectordb = FAISS.load_local(vectordb_file_path, embeddings)
                self._retriever = vectordb.as_retriever(score_threshold=0.7)
                return self._retriever
        except Exception:
            pass
        return None

    def construct_aligned_query(
        self,
        text: str,
        language: str,
        intent_result: Optional[MultilingualIntentResult] = None,
    ) -> str:
        """Construct a search-oriented English retrieval representation for any query.

        Args:
            text (str): Raw user query.
            language (str): Detected language code (e.g., 'es', 'fr', 'de', 'hi', 'en').
            intent_result (Optional[MultilingualIntentResult]): Intent classification outcome.

        Returns:
            str: Aligned English search query.
        """
        if not text or not text.strip():
            return ""

        intent_val = intent_result.intent if intent_result else MultilingualIntent.GENERAL_INQUIRY.value

        # If already English, use the original text enriched with intent if helpful
        if language == SupportedLanguage.ENGLISH.value:
            return text.strip()

        # Build aligned query from intent template
        template = INTENT_RETRIEVAL_TEMPLATES.get(intent_val, "course training information")

        # Extract mapped domain keywords from the query
        norm_text = text.lower()
        extracted_terms: List[str] = []
        for foreign_term, english_term in TECHNICAL_KEYWORD_MAP.items():
            if foreign_term in norm_text and english_term not in extracted_terms:
                extracted_terms.append(english_term)

        # Include matched keywords from intent detector if present
        if intent_result and intent_result.matched_keywords:
            for kw in intent_result.matched_keywords:
                kw_lower = kw.lower()
                if kw_lower in TECHNICAL_KEYWORD_MAP:
                    mapped = TECHNICAL_KEYWORD_MAP[kw_lower]
                    if mapped not in extracted_terms:
                        extracted_terms.append(mapped)

        if extracted_terms:
            aligned = f"{template} {' '.join(extracted_terms)}"
        else:
            aligned = template

        return aligned.strip()

    def retrieve(
        self,
        query: str,
        language: str,
        intent_result: Optional[MultilingualIntentResult] = None,
        top_k: int = 3,
    ) -> CrossLingualRetrievalResult:
        """Execute cross-lingual retrieval against the English knowledge base.

        Args:
            query (str): Original user query.
            language (str): Detected language code.
            intent_result (Optional[MultilingualIntentResult]): Day 31 intent result.
            top_k (int): Maximum documents to retrieve.

        Returns:
            CrossLingualRetrievalResult: Structured retrieval result.
        """
        if not query or not isinstance(query, str) or not query.strip():
            return CrossLingualRetrievalResult(
                original_query="" if query is None else str(query),
                detected_language=language or SupportedLanguage.UNKNOWN.value,
                intent=intent_result.intent if intent_result else MultilingualIntent.UNKNOWN.value,
                aligned_query="",
                retrieved_documents=[],
                evidence_text="",
                is_evidence_found=False,
                confidence=0.0,
                metadata={"reason": "Empty or invalid query"},
            )

        intent_val = intent_result.intent if intent_result else MultilingualIntent.GENERAL_INQUIRY.value

        # Greetings do not require heavy vector retrieval
        if intent_val == MultilingualIntent.GREETING.value:
            return CrossLingualRetrievalResult(
                original_query=query,
                detected_language=language,
                intent=intent_val,
                aligned_query="greeting",
                retrieved_documents=[],
                evidence_text="User is greeting the assistant.",
                is_evidence_found=True,
                confidence=0.95,
                metadata={"type": "conversational_greeting"},
            )

        aligned_query = self.construct_aligned_query(query, language, intent_result)

        # 1. Custom search function hook (for test isolation)
        if self._custom_search_fn is not None:
            docs = self._custom_search_fn(aligned_query, top_k)
            evidence = "\n\n".join(d.get("page_content", "") for d in docs if d.get("page_content"))
            is_found = len(docs) > 0
            conf = 0.90 if is_found else 0.0
            return CrossLingualRetrievalResult(
                original_query=query,
                detected_language=language,
                intent=intent_val,
                aligned_query=aligned_query,
                retrieved_documents=docs,
                evidence_text=evidence,
                is_evidence_found=is_found,
                confidence=conf,
                metadata={"retriever_source": "custom_fn", "doc_count": len(docs)},
            )

        # 2. Production FAISS vector store retrieval
        retriever = self._get_default_retriever()
        retrieved_docs: List[Dict[str, Any]] = []

        if retriever is not None:
            try:
                raw_docs: List[Any] = []
                if hasattr(retriever, "invoke"):
                    res = retriever.invoke(aligned_query)
                    raw_docs = list(res) if isinstance(res, (list, tuple)) else [res]
                elif hasattr(retriever, "get_relevant_documents"):
                    res = retriever.get_relevant_documents(aligned_query)
                    raw_docs = list(res) if isinstance(res, (list, tuple)) else [res]
                elif callable(retriever):
                    res = retriever(aligned_query)
                    raw_docs = list(res) if isinstance(res, (list, tuple)) else [res]

                for doc in raw_docs[:top_k]:
                    content = getattr(doc, "page_content", str(doc))
                    doc_meta = getattr(doc, "metadata", {})
                    retrieved_docs.append({
                        "page_content": content,
                        "metadata": doc_meta,
                    })
            except Exception:
                # Retrieval failure isolation
                pass

        evidence_text = "\n\n".join(str(d.get("page_content", "")) for d in retrieved_docs if d.get("page_content"))

        is_evidence_found = len(retrieved_docs) > 0
        confidence = 0.85 if is_evidence_found else 0.0

        return CrossLingualRetrievalResult(
            original_query=query,
            detected_language=language,
            intent=intent_val,
            aligned_query=aligned_query,
            retrieved_documents=retrieved_docs,
            evidence_text=evidence_text,
            is_evidence_found=is_evidence_found,
            confidence=confidence,
            metadata={
                "retriever_source": "faiss_production" if retriever else "unloaded",
                "doc_count": len(retrieved_docs),
            },
        )
