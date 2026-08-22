import os
import sys
from typing import Optional

os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
import streamlit as st

# Add current directory and parent directory to sys.path while excluding '.' to avoid NLTK import issues
sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.path.abspath('.')]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_helper import create_vector_db
from chatbot_service import ChatbotService
from response_policy import HIGH_CONFIDENCE_THRESHOLD, apply_response_policy
from knowledge_base.scheduler import KnowledgeBaseScheduler
from knowledge_base.audit import get_last_successful_update, DEFAULT_HISTORY_PATH


_GLOBAL_SCHEDULER_INSTANCE: Optional[KnowledgeBaseScheduler] = None


@st.cache_resource
def get_knowledge_scheduler() -> KnowledgeBaseScheduler:
    """Streamlit-safe cached singleton for the KnowledgeBaseScheduler."""
    global _GLOBAL_SCHEDULER_INSTANCE
    if _GLOBAL_SCHEDULER_INSTANCE is None:
        _GLOBAL_SCHEDULER_INSTANCE = KnowledgeBaseScheduler(interval_seconds=300.0)
        _GLOBAL_SCHEDULER_INSTANCE.start()
    return _GLOBAL_SCHEDULER_INSTANCE


def render_sync_status(scheduler: KnowledgeBaseScheduler):
    """Render the Knowledge Base Synchronization & Status section."""
    with st.expander("🔄 Dynamic Knowledge Base & Sync Status", expanded=False):
        status = scheduler.get_status()
        is_running = status.get("is_running", False)
        st.markdown(
            f"**Scheduler Status**: {'🟢 Active (Daemon Running)' if is_running else '⚪ Idle / Stopped'}"
        )
        st.markdown(f"**Polling Interval**: `{status.get('interval_seconds', 300.0):.0f}s` (every 5 minutes)")

        last_sync = get_last_successful_update(status.get("history_path") or DEFAULT_HISTORY_PATH)
        if last_sync:
            st.markdown(f"**Last Successful Sync**: `{last_sync.get('timestamp', 'N/A')}`")
            st.markdown(
                f"**Total Records**: `{last_sync.get('final_records', 'N/A')}` "
                f"(*+{last_sync.get('new', 0)} new, {last_sync.get('duplicate', 0)} duplicate*)"
            )
        else:
            st.markdown("**Last Successful Sync**: `None yet recorded`")

        if status.get("last_error"):
            st.error(f"Last Background Error: {status['last_error']}")

        if st.button("🔄 Sync Knowledge Sources Now"):
            with st.spinner("Checking and updating knowledge sources..."):
                outcome = scheduler.run_once()
                if outcome.get("status") == "SUCCESS":
                    st.success("Knowledge sources synchronized successfully!")
                elif outcome.get("status") == "SKIPPED":
                    st.info("Sync skipped: another update cycle is currently active.")
                else:
                    st.error(f"Sync failed: {outcome.get('error')}")


def main():
    st.title(" CUSTOMER SERVICE CHATBOT 🤖")

    scheduler = get_knowledge_scheduler()
    render_sync_status(scheduler)

    btn = st.button("Create Knowledgebase")
    if btn:
        with st.spinner("Creating knowledge base..."):
            create_vector_db()
            st.success("Knowledge base created successfully!")

    question = st.text_input("Question: ")

    if question and question.strip():
        with st.spinner("Fetching answer..."):
            service = ChatbotService()
            try:
                response = service.process_query(question)

                # Display sentiment UI badge
                label = response.sentiment_label
                score = response.confidence_score
                if label == "POSITIVE":
                    st.info(f"Detected Sentiment: **{label}** (Confidence: {score:.2%})")
                elif label == "NEGATIVE":
                    st.warning(f"Detected Sentiment: **{label}** (Confidence: {score:.2%})")
                else:
                    st.caption(f"Detected Sentiment: **{label}** (Confidence: {score:.2%})")

                st.header("Answer")
                st.write(response.final_answer)
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")


if __name__ == "__main__":
    main()