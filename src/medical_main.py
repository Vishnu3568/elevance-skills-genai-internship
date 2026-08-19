"""Streamlit User Interface for MedQuAD Medical Q&A Assistant.

Provides a professional, safety-gated, session-aware chat interface for medical Q&A,
exposing evidence-grounded answers, retrieval confidence tiers, NIH citations,
and supporting evidence transparency.
"""

import os
import sys
from typing import Tuple, Optional
import streamlit as st

# Disable NLTK import security hook that blocks regex in CWD
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.path.abspath('.')]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_helper import get_llm
from medquad_indexer import load_medical_vector_db, DEFAULT_MEDICAL_INDEX_PATH
from medquad_query_analyzer import MedicalQueryAnalyzer, MedicalVocabulary
from medical_qa_service import (
    MedicalQAService,
    MedicalQAResponse,
    get_confidence_tier,
)

MISSING_MEDICAL_INDEX_MESSAGE = (
    "Medical vector database index not found at 'faiss_index_medical/'. "
    "Please build the index first using: python src/medquad_indexer.py --xml_dir <path_to_medquad_xmls>"
)
INVALID_MEDICAL_INDEX_MESSAGE = (
    "Medical vector database at 'faiss_index_medical/' could not be loaded. "
    "Please rebuild the medical FAISS index using: python src/medquad_indexer.py --xml_dir <path_to_medquad_xmls>"
)


def _split_metadata_list(value) -> list:
    """Convert comma-delimited vector-store metadata into a clean string list."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def build_vocabulary_from_vector_db(vector_db) -> MedicalVocabulary:
    """Build a MedicalVocabulary from loaded FAISS document metadata."""
    vocab = MedicalVocabulary()
    docstore = getattr(vector_db, "docstore", None)
    documents = getattr(docstore, "_dict", {})

    for doc in documents.values():
        metadata = getattr(doc, "metadata", {}) or {}
        focus = metadata.get("focus", "")
        if not focus:
            continue
        vocab.add_topic(
            focus=str(focus),
            synonyms=_split_metadata_list(metadata.get("synonyms", "")),
            cuis=_split_metadata_list(metadata.get("cuis", "")),
        )

    return vocab


@st.cache_resource
def initialize_medical_qa_service() -> Tuple[Optional[MedicalQAService], Optional[str]]:
    """Initialize the MedicalQAService with cached vector DB, vocabulary, and LLM.

    Returns:
        Tuple[Optional[MedicalQAService], Optional[str]]: Service instance and any error/warning message.
    """
    try:
        vector_db = load_medical_vector_db(DEFAULT_MEDICAL_INDEX_PATH)
    except FileNotFoundError:
        return None, MISSING_MEDICAL_INDEX_MESSAGE
    except Exception:
        return None, INVALID_MEDICAL_INDEX_MESSAGE

    vocab = build_vocabulary_from_vector_db(vector_db)
    analyzer = MedicalQueryAnalyzer(vocabulary=vocab)

    llm_instance = None
    warning_msg = None
    try:
        llm_instance = get_llm()
    except Exception as err:
        warning_msg = f"LLM Warning: {err} Operating in retrieval-gated fallback mode."

    service = MedicalQAService(
        vector_db=vector_db,
        analyzer=analyzer,
        llm=llm_instance,
        relevance_threshold=0.50
    )

    return service, warning_msg


def render_response(response: MedicalQAResponse):
    """Render a MedicalQAResponse payload in Streamlit with safety status and transparency."""

    # 1. Render Status-specific Badges & Warnings
    if response.status == "GROUNDED":
        tier = get_confidence_tier(response.confidence_score)
        if tier == "HIGH":
            st.success(f"🔒 Knowledge-base retrieval confidence: **{response.confidence_score:.1%}** (HIGH)")
        elif tier == "MEDIUM":
            st.info(f"ℹ️ Knowledge-base retrieval confidence: **{response.confidence_score:.1%}** (MEDIUM)")
        else:
            st.warning(f"⚠️ Knowledge-base retrieval confidence: **{response.confidence_score:.1%}** (LOW)")
    elif response.status == "INSUFFICIENT_EVIDENCE":
        st.warning("⚠️ Insufficient Grounded Evidence")
    elif response.status == "OUT_OF_DOMAIN":
        st.info("ℹ️ Out of Knowledge Scope")
    elif response.status == "RETRIEVAL_ERROR":
        st.error("🚨 Medical Retrieval System Error")
    elif response.status == "GENERATION_ERROR":
        st.error("🚨 Answer Generation Error")

    # 2. Render Main Answer
    st.markdown("### Answer")
    st.write(response.final_answer)

    # 3. Render Citations & Sources (if available)
    if response.citations:
        st.markdown("#### 📚 Verifiable NIH Sources")
        for cite in response.citations:
            source_name = cite.get("source", "NIH MedQuAD")
            focus_topic = cite.get("focus", "")
            url = cite.get("source_url", "")

            label_parts = [f"**{source_name}**"]
            if focus_topic:
                label_parts.append(f"({focus_topic})")

            label_str = " ".join(label_parts)
            if url:
                st.markdown(f"- {label_str}: [{url}]({url})")
            else:
                st.markdown(f"- {label_str}")

    # 4. Collapsible Supporting Evidence Transparency
    if response.evidence_documents:
        with st.expander("🔍 View supporting medical evidence (NIH ground-truth)"):
            for idx, doc in enumerate(response.evidence_documents, 1):
                meta = doc.metadata or {}
                st.markdown(f"**Evidence #{idx}** — *Focus:* `{meta.get('focus', 'N/A')}` | *Type:* `{meta.get('question_type', 'N/A')}` | *Source:* `{meta.get('source', 'N/A')}`")
                if meta.get("question"):
                    st.caption(f"**Question:** {meta.get('question')}")
                st.text_area(f"Content #{idx}", value=doc.page_content, height=100, key=f"evidence_text_{idx}_{hash(doc.page_content[:30])}")

    # 5. Collapsible Query Analysis Transparency (System Debug Info)
    if response.analysis:
        with st.expander("📊 System Debug Info (Query Analysis)"):
            st.caption("*System analysis details only — not a clinical diagnosis or medical assessment.*")
            st.json({
                "raw_query": response.analysis.raw_query,
                "clean_query": response.analysis.clean_query,
                "intent": response.analysis.intent,
                "raw_qtype_match": response.analysis.raw_qtype_match,
                "primary_topic": response.analysis.primary_topic,
                "matched_cuis": response.analysis.matched_cuis,
                "entities": [
                    {"text": e.text, "category": e.category, "cui": e.cui, "confidence": e.confidence}
                    for e in response.analysis.entities
                ]
            })


def main():
    """Main Streamlit application entry point for Medical Q&A Assistant."""
    st.set_page_config(
        page_title="Medical Q&A Assistant",
        page_icon="🏥",
        layout="wide"
    )

    st.title("🏥 Medical Q&A Assistant 🩺")
    st.markdown(
        "Welcome to the **MedQuAD Medical Q&A Assistant**. Ask any health question to receive "
        "evidence-grounded medical information compiled from verified National Institutes of Health (NIH) resources. "
        "All answers are strictly safety-gated and backed by verifiable citations."
    )
    st.divider()

    # Initialize Sidebar & Session State
    if "medical_messages" not in st.session_state:
        st.session_state.medical_messages = []

    with st.sidebar:
        st.header("⚙️ Controls")
        if st.button("🧹 Clear Conversation", use_container_width=True):
            st.session_state.medical_messages = []
            st.rerun()

        st.divider()
        st.markdown("### ℹ️ About MedQuAD")
        st.caption(
            "Answers are grounded in 47k+ medical Q&A pairs from 12 NIH resources "
            "(CancerGov, CDC, GARD, NINDS, NIDDK, etc.)."
        )

    # Initialize Service
    service, init_msg = initialize_medical_qa_service()
    if init_msg and ("not found" in init_msg.lower() or "could not be loaded" in init_msg.lower()):
        st.error(init_msg)
        return
    elif init_msg:
        st.warning(init_msg)

    if service is None:
        st.error("Failed to initialize Medical Q&A Service.")
        return

    # Render Chat History
    for msg in st.session_state.medical_messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                render_response(msg["response"])

    # Chat Input
    query_input = st.chat_input("Ask a medical question (e.g., 'What are the treatments for breast cancer?')...")
    if query_input:
        clean_input = query_input.strip()
        if clean_input:
            # Display user message
            with st.chat_message("user"):
                st.write(clean_input)

            # Process query through MedicalQAService
            with st.chat_message("assistant"):
                with st.spinner("Analyzing query & searching NIH medical knowledge base..."):
                    try:
                        response = service.process_query(clean_input)
                        render_response(response)

                        # Save to session history
                        st.session_state.medical_messages.append({"role": "user", "content": clean_input})
                        st.session_state.medical_messages.append({"role": "assistant", "response": response})
                    except Exception as err:
                        st.error(f"An unexpected error occurred while processing your request: {err}")


if __name__ == "__main__":
    main()
