import os
import sys

os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
import streamlit as st

# Add current directory and parent directory to sys.path while excluding '.' to avoid NLTK import issues
sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.path.abspath('.')]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_helper import create_vector_db
from chatbot_service import ChatbotService
from response_policy import HIGH_CONFIDENCE_THRESHOLD, apply_response_policy

st.title(" CUSTOMER SERVICE CHATBOT 🤖")
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