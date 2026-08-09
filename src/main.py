import os
import sys

os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
import streamlit as st

# Add current directory and parent directory to sys.path while excluding '.' to avoid NLTK import issues
sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.path.abspath('.')]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_helper import get_qa_chain, create_vector_db

st.title(" CUSTOMER SERVICE CHATBOT 🤖")
btn = st.button("Create Knowledgebase")
if btn:
    with st.spinner("Creating knowledge base..."):
        create_vector_db()
        st.success("Knowledge base created successfully!")

question = st.text_input("Question: ")

if question:
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

        st.write(answer)