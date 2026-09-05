"""Streamlit User Interface for Multimodal AI Assistant (Phase 5 — Day 29).

Provides an enterprise-grade, session-aware chat interface for multimodal reasoning,
supporting text-only, image-only, and joint text+image queries with evidence transparency,
confidence badges, clarification notices, missing-information handling, and safe fallback behavior.
"""

import os
import sys
from typing import Optional, Tuple
import uuid

import streamlit as st

# Disable NLTK import security hook that blocks regex in CWD
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.path.abspath(".")]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.multimodal import (
        FallbackAction,
        ModalityType,
        MultimodalAssistantService,
        MultimodalResponse,
        MultimodalUIResult,
    )
except ImportError:
    from multimodal import (
        FallbackAction,
        ModalityType,
        MultimodalAssistantService,
        MultimodalResponse,
        MultimodalUIResult,
    )


# -----------------------------------------------------------------------------
# Service Initializer (Streamlit-Cached Resource)
# -----------------------------------------------------------------------------

@st.cache_resource
def initialize_multimodal_service() -> Tuple[Optional[MultimodalAssistantService], Optional[str]]:
    """Initialize and cache the MultimodalAssistantService instance.

    Returns:
        Tuple[Optional[MultimodalAssistantService], Optional[str]]: Service and optional warning.
    """
    try:
        service = MultimodalAssistantService()
        return service, None
    except Exception as exc:
        return None, f"Failed to initialize Multimodal Assistant Service: {str(exc)}"


# -----------------------------------------------------------------------------
# Response Rendering Component
# -----------------------------------------------------------------------------

def render_multimodal_response(result: MultimodalUIResult) -> None:
    """Render a MultimodalUIResult in Streamlit with safety badges and transparency.

    Args:
        result: The structured MultimodalUIResult contract from the backend service.
    """
    response = result.response
    fallback_decision = result.fallback_decision
    action = fallback_decision.action

    # 1. Status and Safety Badge
    if action == FallbackAction.PROCEED:
        conf_val = ""
        if result.confidence_assessment and result.confidence_assessment.aggregate_confidence is not None:
            score = result.confidence_assessment.aggregate_confidence
            conf_val = f" — Confidence: **{score:.1%}**"
        st.success(f"🟢 **Grounded Analysis**{conf_val}")
    elif action == FallbackAction.ASK_CLARIFICATION:
        st.warning("🟡 **Clarification Required** (Ambiguity Detected)")
    elif action == FallbackAction.REQUEST_MISSING_INFORMATION:
        st.info("🟠 **Additional Information Required** (Incomplete Context)")
    elif action == FallbackAction.SAFE_LIMITED_RESPONSE:
        st.warning("🛡️ **Safe Limited Response** (Qualified Answering)")
    elif action == FallbackAction.DECLINE_UNSUPPORTED_CLAIM:
        st.error("⛔ **Unsupported Claim Declined** (Insufficient / Unreliable Evidence)")
    else:
        st.caption(f"ℹ️ **Status**: `{action.value}`")

    # 2. Ambiguity & Missing Information Notices
    if action == FallbackAction.ASK_CLARIFICATION and fallback_decision.clarification_needed:
        st.info(f"**Clarification Question:** {fallback_decision.clarification_needed}")
        if result.ambiguity_assessment and result.ambiguity_assessment.candidate_interpretations:
            st.markdown(
                "**Possible Interpretations:**\n"
                + "\n".join(f"- {c}" for c in result.ambiguity_assessment.candidate_interpretations)
            )

    if action == FallbackAction.REQUEST_MISSING_INFORMATION and fallback_decision.missing_requirements:
        st.warning(
            "**Missing Information Needed:**\n"
            + "\n".join(f"- {req}" for req in fallback_decision.missing_requirements)
        )

    # 3. Main Assistant Answer / Safe Text
    st.markdown("### 💬 Assistant Response")
    st.write(response.answer)

    # 4. Evidence Transparency Expander
    has_visual_evidence = bool(response.visual_evidence and len(response.visual_evidence) > 0)
    if has_visual_evidence:
        with st.expander("🔍 Verified Visual Evidence & Grounding", expanded=False):
            st.markdown(f"**Total Verified Evidence Items:** {len(response.visual_evidence)}")
            for idx, item in enumerate(response.visual_evidence, start=1):
                region_str = f" `[{item.region_label}]`" if item.region_label else ""
                conf_pct = f"({item.confidence:.0%})" if item.confidence is not None else ""
                st.markdown(f"{idx}. **{item.description}**{region_str} {conf_pct}")

    # 5. Diagnostic Details Expander
    with st.expander("🛠️ Decision & Modality Diagnostics", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Modality:** `{result.modality.value}`")
            st.write(f"**Grounded:** `{response.grounded}`")
            st.write(f"**Turn Index:** `{result.turn_index}`")
        with col2:
            st.write(f"**Action:** `{action.value}`")
            st.write(f"**Reason:** {fallback_decision.reason}")

        if result.followup_resolution and result.followup_resolution.is_followup:
            st.info(f"🔄 **Resolved Query:** `{result.followup_resolution.resolved_query}`")


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------

def main() -> None:
    """Main Streamlit application entry point for the Multimodal AI Assistant."""
    st.set_page_config(
        page_title="Multimodal AI Assistant | Sentinel NLP",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("👁️ Multimodal AI Assistant")
    st.caption("Phase 5 — Cross-Modal Reasoning, Evidence Transparency & Safe Grounding")

    # 1. Initialize Service
    service, init_error = initialize_multimodal_service()
    if init_error or service is None:
        st.error(f"Initialization Error: {init_error}")
        st.stop()

    # 2. Session State Initialization
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 3. Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Session & Controls")
        st.markdown(f"**Active Session ID:** `{st.session_state.session_id}`")
        st.markdown(f"**Total Turns:** `{len(st.session_state.chat_history)}`")

        if st.button("🗑️ Reset Conversation Session", use_container_width=True):
            service.clear_session(st.session_state.session_id)
            st.session_state.chat_history = []
            st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
            st.success("Conversation session reset successfully.")
            st.rerun()

        st.divider()
        st.markdown("### 🖼️ Supported Image Formats")
        st.markdown("- **PNG** (`.png`)\n- **JPEG** (`.jpg`, `.jpeg`)\n- **WebP** (`.webp`)")
        st.caption("Max file size: **10 MB**")

        st.divider()
        st.markdown("### 🛡️ Safety Guarantees")
        st.markdown(
            "- Zero Raw Image Byte Leakage\n"
            "- Structural Evidence Grounding\n"
            "- Multi-Turn Context Retention\n"
            "- Ambiguity & Missing Info Refusal\n"
            "- Isolated Session State"
        )

    # 4. Multimodal Input Form
    with st.form("multimodal_input_form", clear_on_submit=False):
        st.subheader("📥 Submit Multimodal Query")

        uploaded_file = st.file_uploader(
            "Upload Image (Optional for text-only, Required for visual queries):",
            type=["png", "jpg", "jpeg", "webp"],
            help="Select a PNG, JPEG, or WebP image artifact (Max 10 MB)",
        )

        # Image preview if uploaded
        if uploaded_file is not None:
            try:
                from PIL import Image
                preview_img = Image.open(uploaded_file)
                st.image(preview_img, caption=f"Preview: {uploaded_file.name}", width=320)
            except Exception as e:
                st.warning(f"Could not render image preview: {e}")

        user_query = st.text_area(
            "Your Question / Query:",
            placeholder="e.g., 'Analyze this diagram and explain its core components' or 'Explain transformers'",
            height=90,
        )

        submitted = st.form_submit_button("🚀 Submit Multimodal Query", use_container_width=True)

    # 5. Process Interaction on Submit
    if submitted:
        query_text = user_query.strip() if user_query else None
        image_bytes: Optional[bytes] = None
        file_name: Optional[str] = None

        if uploaded_file is not None:
            uploaded_file.seek(0)
            image_bytes = uploaded_file.read()
            file_name = uploaded_file.name

        if not query_text and image_bytes is None:
            st.warning("⚠️ Please provide at least a text query or an image upload.")
        else:
            with st.spinner("Processing cross-modal reasoning and validating evidence..."):
                try:
                    result = service.process_interaction(
                        session_id=st.session_state.session_id,
                        query=query_text,
                        image_bytes=image_bytes,
                        file_name=file_name,
                    )

                    # Store in chat history
                    st.session_state.chat_history.append({
                        "query": query_text or "(Image Upload Only)",
                        "has_image": image_bytes is not None,
                        "file_name": file_name,
                        "result": result,
                    })

                except Exception as exc:
                    st.error(f"❌ Error during multimodal processing: {str(exc)}")

    # 6. Render Chronological Conversation History
    st.divider()
    st.subheader("💬 Conversation History")

    if not st.session_state.chat_history:
        st.info("No turns in this session yet. Submit a query or upload an image above to start.")
    else:
        for turn_idx, turn in enumerate(st.session_state.chat_history, start=1):
            with st.container():
                st.markdown(f"#### 👤 User (Turn {turn_idx})")
                query_display = turn["query"]
                if turn.get("has_image") and turn.get("file_name"):
                    st.markdown(f"*{query_display}* *(Image: `{turn['file_name']}`)*")
                else:
                    st.markdown(f"*{query_display}*")

                st.markdown("#### 🤖 Assistant")
                render_multimodal_response(turn["result"])
                st.divider()


if __name__ == "__main__":
    main()
