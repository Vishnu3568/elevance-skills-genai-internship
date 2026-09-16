"""Unified Streamlit User Interface for ElevanceSkills Cross-Task AI Platform (Day 36).

Provides an integrated, multi-domain conversational interface orchestrating Customer Support,
Medical Clinical Q&A, Scientific Literature Research, Multimodal Visual Intelligence,
and Cross-Cutting Multilingual Understanding into a single experience.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Disable NLTK import security hook that blocks regex in CWD
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.path.abspath(".")]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

try:
    from src.cross_task import (
        CitationItem,
        CrossTaskService,
        DomainType,
        EvidenceItem,
        UnifiedRequest,
        UnifiedResponse,
    )
    from src.multimodal.ingestion import ingest_image
except ImportError:
    from cross_task import (  # type: ignore
        CitationItem,
        CrossTaskService,
        DomainType,
        EvidenceItem,
        UnifiedRequest,
        UnifiedResponse,
    )
    from multimodal.ingestion import ingest_image  # type: ignore


# -----------------------------------------------------------------------------
# Streamlit UI Configuration
# -----------------------------------------------------------------------------

def configure_page() -> None:
    """Configure Streamlit layout and page metadata."""
    st.set_page_config(
        page_title="ElevanceSkills Unified AI Platform",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def initialize_session_state() -> None:
    """Initialize conversation history and session identifier."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"unified_session_{int(time.time())}"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "cross_task_service" not in st.session_state:
        st.session_state.cross_task_service = CrossTaskService()


def render_sidebar() -> Tuple[Optional[str], Optional[Any]]:
    """Render sidebar controls for manual domain override and image attachments."""
    st.sidebar.title("🌐 Platform Controls")
    st.sidebar.markdown("---")

    st.sidebar.subheader("🎯 Domain Routing")
    domain_choice = st.sidebar.selectbox(
        "Domain Routing Mode",
        options=[
            "Auto-Detect (Smart Router)",
            "Customer Support (EdTech FAQ)",
            "Medical Q&A (MedQuAD Clinical)",
            "Scientific Research (arXiv AI/ML)",
            "Multimodal Intelligence",
        ],
        index=0,
        help="Leave on Auto-Detect for automatic intent classification and language identification.",
    )

    domain_override: Optional[str] = None
    if domain_choice == "Customer Support (EdTech FAQ)":
        domain_override = DomainType.CUSTOMER_SUPPORT.value
    elif domain_choice == "Medical Q&A (MedQuAD Clinical)":
        domain_override = DomainType.MEDICAL.value
    elif domain_choice == "Scientific Research (arXiv AI/ML)":
        domain_override = DomainType.SCIENTIFIC.value
    elif domain_choice == "Multimodal Intelligence":
        domain_override = DomainType.MULTIMODAL.value

    st.sidebar.markdown("---")
    st.sidebar.subheader("🖼️ Multimodal Image Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Attach Image (Optional)",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload a diagram, medical scan, or UI image for cross-modal reasoning.",
    )

    image_artifact = None
    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.read()
            image_artifact = ingest_image(
                source=image_bytes,
                file_name=uploaded_file.name,
            )
            import io
            st.sidebar.image(io.BytesIO(image_bytes), caption=f"Loaded: {uploaded_file.name}", use_column_width=True)
            st.sidebar.success(f"Image ingested: {image_artifact.width}x{image_artifact.height}px")
        except Exception as e:
            st.sidebar.error(f"Image ingestion error: {str(e)}")

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset Conversation Session"):
        st.session_state.messages = []
        st.session_state.session_id = f"unified_session_{int(time.time())}"
        st.rerun()

    st.sidebar.caption(f"Active Session: `{st.session_state.session_id}`")
    return domain_override, image_artifact


def render_response_metadata(resp: UnifiedResponse) -> None:
    """Render domain badges, language indicators, confidence meters, and citations."""
    col1, col2, col3, col4 = st.columns(4)

    # Domain Badge
    domain_icons = {
        "customer_support": "💼 Customer Support",
        "medical": "🩺 Medical Q&A",
        "scientific": "🔬 Scientific Research",
        "multimodal": "👁️ Multimodal AI",
        "general_chitchat": "💬 General Chitchat",
    }
    col1.metric("Routed Domain", domain_icons.get(resp.domain, resp.domain.upper()))

    # Language Badge
    lang_display = f"{resp.detected_language_name} ({resp.detected_language})"
    col2.metric("Detected Language", lang_display)

    # Confidence Tier
    confidence_pct = f"{resp.confidence_score * 100:.1f}% ({resp.confidence_tier})"
    col3.metric("Grounded Confidence", confidence_pct)

    # Execution Latency
    col4.metric("Latency", f"{resp.execution_time_ms:.1f} ms")

    # Sentiment Alert (if detected)
    if resp.sentiment:
        st.info(f"📊 Detected Sentiment Tone: **{resp.sentiment}**")

    # Warning / Fallback Notice
    if resp.warning_message:
        st.warning(f"⚠️ Notice: {resp.warning_message}")

    # Evidence Expander
    if resp.evidence:
        with st.expander(f"📑 Grounded Evidence Items ({len(resp.evidence)})", expanded=False):
            for i, ev in enumerate(resp.evidence, 1):
                st.markdown(f"**{i}. [{ev.source}]** {ev.description}")

    # Citations Expander
    if resp.citations:
        with st.expander(f"📚 Provenance & Citations ({len(resp.citations)})", expanded=False):
            for i, cit in enumerate(resp.citations, 1):
                url_str = f" ([Link]({cit.url}))" if cit.url else ""
                authors_str = f" — *{cit.authors}*" if cit.authors else ""
                st.markdown(f"{i}. **{cit.title}** (ID: `{cit.source_id}`){authors_str}{url_str}")


def main() -> None:
    """Main execution loop for Unified Streamlit application."""
    configure_page()
    initialize_session_state()

    st.title("🌐 ElevanceSkills Unified AI Assistant")
    st.markdown(
        "*End-to-End Multimodal, Multilingual, Medical, Scientific, and Customer Support Architecture.*"
    )

    domain_override, image_artifact = render_sidebar()

    # Display conversation messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "response_obj" in msg:
                render_response_metadata(msg["response_obj"])

    # User chat input
    user_query = st.chat_input("Ask a question across Customer Support, Medical, Science, or upload an image...")

    if user_query:
        # Display user query
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Dispatch through CrossTaskService
        with st.chat_message("assistant"):
            with st.spinner("Analyzing request across specialist domains..."):
                service: CrossTaskService = st.session_state.cross_task_service
                request = UnifiedRequest(
                    query=user_query,
                    image=image_artifact,
                    session_id=st.session_state.session_id,
                    domain_override=domain_override,
                )
                response: UnifiedResponse = service.process_request(request)

                st.markdown(response.final_text_response)
                render_response_metadata(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.final_text_response,
                    "response_obj": response,
                })


if __name__ == "__main__":
    main()
