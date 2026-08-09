import os
import sys

# Disable NLTK import security hook that blocks regex in CWD
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.path.abspath('.')]

from dotenv import load_dotenv
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import CSVLoader
    from langchain_community.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
except ImportError:
    from langchain.vectorstores import FAISS
    from langchain.document_loaders import CSVLoader
    from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

try:
    from langchain.chains import RetrievalQA
except ImportError:
    try:
        from langchain_classic.chains import RetrievalQA
    except ImportError:
        from langchain_community.chains import RetrievalQA

load_dotenv()

# Determine project paths robustly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "dataset.csv")
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")

vectordb_file_path = os.path.join(BASE_DIR, "faiss_index")

def get_instructor_embeddings():
    try:
        return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    except Exception as e:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Lazy initialization of LLM
def get_llm():
    api_key = os.environ.get("GOOGLE_API_KEY", "")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not configured. "
            "Add it to the .env file before starting the application."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.1,
        max_retries=3,
    )


def create_vector_db():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")

    loader = CSVLoader(file_path=DATASET_PATH, source_column="prompt")
    data = loader.load()

    embeddings = get_instructor_embeddings()
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    embeddings = get_instructor_embeddings()

    if not os.path.exists(vectordb_file_path):
        create_vector_db()

    try:
        vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
    except TypeError:
        vectordb = FAISS.load_local(vectordb_file_path, embeddings)

    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    llm = get_llm()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("hello?"))

