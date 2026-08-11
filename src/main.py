import os
import sys

os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
import streamlit as st

# Add current directory and parent directory to sys.path while excluding '.' to avoid NLTK import issues
sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.path.abspath('.')]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_helper import get_qa_chain, create_vector_db
from sentiment_analyzer import analyze_sentiment

st.title(" CUSTOMER SERVICE CHATBOT 🤖")
btn = st.button("Create Knowledgebase")
if btn:
    with st.spinner("Creating knowledge base..."):
        create_vector_db()
        st.success("Knowledge base created successfully!")

question = st.text_input("Question: ")

if question and question.strip():
    # Analyze user sentiment before generating response
    sentiment_label = "NEUTRAL"
    try:
        sentiment_res = analyze_sentiment(question)
        sentiment_label = sentiment_res.get("label", "neutral").upper()
        score = sentiment_res.get("score", 0.0)
        
        if sentiment_label == "POSITIVE":
            st.info(f"Detected Sentiment: **{sentiment_label}** (Confidence: {score:.2%})")
        elif sentiment_label == "NEGATIVE":
            st.warning(f"Detected Sentiment: **{sentiment_label}** (Confidence: {score:.2%})")
        else:
            st.caption(f"Detected Sentiment: **{sentiment_label}** (Confidence: {score:.2%})")
    except Exception as e:
        st.warning(f"Sentiment analysis unavailable: {str(e)}")

    with st.spinner("Fetching answer..."):
        chain = get_qa_chain()
        try:
            response = chain.invoke({"query": question})
        except Exception:
            try:
                response = chain({"query": question})
            except Exception:
                response = chain(question)

        st.header("Answer")
        if isinstance(response, dict):
            answer = response.get("result", str(response))
        else:
            answer = str(response)

        # Adapt tone based on sentiment if answer is grounded (not 'I don't know.')
        norm_ans = answer.strip().lower().rstrip(".")
        is_ood = norm_ans in ("i don't know", "i do not know") or norm_ans.startswith("i don't know")
        
        if not is_ood:
            if sentiment_label == "NEGATIVE":
                answer = f"I am sorry to hear about your experience and frustration. {answer}"
            elif sentiment_label == "POSITIVE":
                answer = f"Thank you for your positive feedback! {answer}"

        st.write(answer)