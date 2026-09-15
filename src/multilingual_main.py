"""Streamlit User Interface for Multilingual Customer Service AI Assistant (Phase 6 — Day 35).

Provides a session-aware, cross-lingual conversational interface supporting:
1. Multi-turn conversational continuity & context retention
2. Dynamic language switching across 5 languages (English, Spanish, French, German, Hindi)
3. Intra-utterance code-switching & Romanized Hinglish understanding
4. Competing intent ambiguity alerts and localized clarification notices
5. Evidence grounding transparency and session management
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import uuid

import streamlit as st

# Disable NLTK import security hook that blocks regex in CWD
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.path.abspath(".")]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.multilingual import (
        LanguageIdentificationResult,
        MultilingualConversationSession,
        MultilingualIntent,
        MultilingualIntentResult,
        MultilingualResponse,
        MultilingualService,
        MultilingualTextRequest,
        SupportedLanguage,
    )
except ImportError:
    from multilingual import (
        LanguageIdentificationResult,
        MultilingualConversationSession,
        MultilingualIntent,
        MultilingualIntentResult,
        MultilingualResponse,
        MultilingualService,
        MultilingualTextRequest,
        SupportedLanguage,
    )


# -----------------------------------------------------------------------------
# Language Option Configuration
# -----------------------------------------------------------------------------

LANGUAGE_OPTIONS: Dict[str, Optional[str]] = {
    "Auto-Detect (Adaptive)": None,
    "English (en)": SupportedLanguage.ENGLISH.value,
    "Spanish (es)": SupportedLanguage.SPANISH.value,
    "French (fr)": SupportedLanguage.FRENCH.value,
    "German (de)": SupportedLanguage.GERMAN.value,
    "Hindi (hi)": SupportedLanguage.HINDI.value,
}


# -----------------------------------------------------------------------------
# Service Initializer (Streamlit Cached Resource)
# -----------------------------------------------------------------------------

@st.cache_resource
def initialize_multilingual_service() -> Tuple[Optional[MultilingualService], Optional[str]]:
    """Initialize and cache the MultilingualService instance.

    Returns:
        Tuple[Optional[MultilingualService], Optional[str]]: Service instance and optional error message.
    """
    try:
        service = MultilingualService()
        return service, None
    except Exception as exc:
        return None, f"Failed to initialize Multilingual Service: {str(exc)}"


# -----------------------------------------------------------------------------
# Response Rendering Component
# -----------------------------------------------------------------------------

def render_multilingual_response(processed: Dict[str, Any]) -> None:
    """Render structured response metadata, badges, and answer in Streamlit.

    Args:
        processed (Dict[str, Any]): Processed request dictionary from MultilingualService.process_request.
    """
    detection = processed.get("detection", {})
    intent = processed.get("intent", {})
    retrieval = processed.get("retrieval", {})
    response_meta = processed.get("response", {})
    is_grounded = processed.get("is_grounded", True)
    lang_name = processed.get("language_name", "Unknown")
    is_forced = processed.get("is_forced", False)
    is_mixed = detection.get("is_mixed_language", False)
    secondary_lang = detection.get("secondary_language")
    is_ambiguous = intent.get("is_ambiguous", False) or response_meta.get("is_ambiguous", False)
    clarification = response_meta.get("clarification_prompt")
    competing = intent.get("competing_intents", [])

    # 1. Badges and Metadata
    col1, col2, col3 = st.columns([1.2, 1.2, 1.0])

    with col1:
        lang_conf = detection.get("confidence", 0.0)
        mode_str = "Forced" if is_forced else "Detected"
        if is_mixed and secondary_lang:
            sec_name = SupportedLanguage.get_language_name(secondary_lang)
            st.info(f"🌐 **Language**: {lang_name} + {sec_name} *(Mixed/Code-Switching)*")
        else:
            st.info(f"🌐 **Language**: {lang_name} *({mode_str}: {lang_conf:.1%})*")

    with col2:
        intent_val = intent.get("intent", "UNKNOWN")
        intent_conf = intent.get("confidence", 0.0)
        st.info(f"🎯 **Intent**: `{intent_val}` *({intent_conf:.1%})*")

    with col3:
        if is_grounded:
            st.success("🟢 **Grounded Answer**")
        else:
            st.warning("🛡️ **Fallback Notice**")

    # 2. Ambiguity & Clarification Warnings
    if is_ambiguous:
        comp_str = ", ".join(f"`{c}`" for c in competing) if competing else "`Multiple Topics`"
        st.warning(f"⚠️ **Ambiguity Detected**: Competing intents identified ({comp_str}).")
        if clarification:
            st.info(f"💡 **Clarification Prompt**: *{clarification}*")

    # 3. Context Follow-Up Notification
    if processed.get("is_followup", False):
        st.caption(f"🔗 *Context-resolved follow-up referencing prior conversation topic.*")

    # 4. Final Answer
    st.markdown(processed.get("final_answer", ""))

    # 5. Supporting Knowledge Evidence Expander
    docs = retrieval.get("retrieved_documents", [])
    if docs:
        with st.expander(f"🔍 Supporting Knowledge Base Evidence ({len(docs)} documents)", expanded=False):
            st.caption(f"**Aligned Search Query**: `{retrieval.get('aligned_query', '')}`")
            for idx, doc in enumerate(docs, 1):
                content = doc.get("page_content", "")
                meta = doc.get("metadata", {})
                st.markdown(f"**Document {idx}** *(Source: {meta.get('source', 'Knowledge Base')})*")
                st.markdown(f"```text\n{content}\n```")


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------

def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="Multilingual AI Assistant",
        page_icon="🌐",
        layout="wide",
    )

    st.title("🌐 MULTILINGUAL CUSTOMER SERVICE ASSISTANT")
    st.caption("Cross-Lingual RAG, Context Retention, Code-Switching & Competing Intent Handling (Phase 6)")

    # 1. Initialize Service
    service, error = initialize_multilingual_service()
    if error or service is None:
        st.error(error or "Failed to load Multilingual Service.")
        return

    # 2. Manage Session State
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session-{uuid.uuid4().hex[:8]}"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    session_id = st.session_state.session_id

    # 3. Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Assistant Configuration")

        selected_label = st.selectbox(
            "Language Mode",
            options=list(LANGUAGE_OPTIONS.keys()),
            index=0,
            help="Select 'Auto-Detect' for adaptive matrix language detection, or force a specific response language.",
        )
        selected_key = str(selected_label or "Auto-Detect (Adaptive)")
        forced_language = LANGUAGE_OPTIONS.get(selected_key)

        st.divider()
        st.markdown(f"**Active Session ID**:\n`{session_id}`")

        active_session = service.get_session(session_id)
        turn_count = len(active_session.turns) if active_session else 0
        st.markdown(f"**Turn History**: `{turn_count}` turns recorded")

        if active_session and active_session.active_language:
            st.markdown(f"**Session Language**: `{SupportedLanguage.get_language_name(active_session.active_language)}`")

        if st.button("🔄 Reset / New Session", use_container_width=True):
            service.clear_session(session_id)
            st.session_state.session_id = f"session-{uuid.uuid4().hex[:8]}"
            st.session_state.messages = []
            st.rerun()

        st.divider()
        with st.expander("ℹ️ Supported Languages & Features"):
            st.markdown(
                "- **English (`en`)**: Native support\n"
                "- **Spanish (`es`)**: Native support\n"
                "- **French (`fr`)**: Native support\n"
                "- **German (`de`)**: Native support\n"
                "- **Hindi (`hi`)**: Devanagari & Hinglish\n"
                "- **Code-Switching**: Mixed-language queries\n"
                "- **Multi-Turn**: Pronoun & entity resolution\n"
                "- **Grounding**: Strict non-hallucination\n"
            )

    # 4. Render Conversation History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                if "processed" in msg:
                    render_multilingual_response(msg["processed"])
                else:
                    st.markdown(msg["content"])

    # 5. User Query Input
    user_input = st.chat_input("Ask a question in English, Spanish, French, German, Hindi, or Hinglish...")
    if user_input and user_input.strip():
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Process through MultilingualService
        with st.chat_message("assistant"):
            with st.spinner("Processing multilingual query..."):
                try:
                    request = MultilingualTextRequest(
                        text=user_input,
                        forced_language=forced_language,
                        session_id=session_id,
                    )
                    processed = service.process_request(request)
                    render_multilingual_response(processed)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": processed.get("final_answer", ""),
                        "processed": processed,
                    })
                except Exception as exc:
                    st.error(f"An error occurred while processing your request: {str(exc)}")


if __name__ == "__main__":
    main()
