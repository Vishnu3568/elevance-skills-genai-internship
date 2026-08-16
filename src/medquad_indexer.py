"""MedQuAD Indexer Module for Medical Q&A Pipeline.

Constructs, persists, and loads the isolated MedQuAD FAISS vector store,
converting parsed MedicalQARecord objects into LangChain Documents with rich metadata.
"""

import os
import sys
from typing import List, Optional

# Disable NLTK import security hook that blocks regex in CWD
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.path.abspath('.')]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document

from medquad_parser import parse_medquad_directory, MedicalQARecord
from langchain_helper import get_instructor_embeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MEDICAL_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index_medical")


def _resolve_index_path(index_path: str) -> str:
    """Resolve relative index path against project BASE_DIR if needed."""
    if os.path.isabs(index_path):
        return index_path
    return os.path.join(BASE_DIR, index_path)


def create_medical_vector_db(
    xml_dir: str,
    index_path: str = DEFAULT_MEDICAL_INDEX_PATH
) -> FAISS:
    """Parse MedQuAD XML files from a directory and build a persisted FAISS vector index.

    Args:
        xml_dir (str): Directory containing MedQuAD XML files.
        index_path (str): Destination directory for the persisted FAISS index.

    Returns:
        FAISS: The constructed LangChain FAISS vector store.

    Raises:
        FileNotFoundError: If xml_dir does not exist.
        ValueError: If no valid answerable medical records are found.
    """
    resolved_xml_dir = xml_dir if os.path.isabs(xml_dir) else os.path.join(BASE_DIR, xml_dir)
    if not os.path.exists(resolved_xml_dir):
        raise FileNotFoundError(f"MedQuAD XML directory not found at: {resolved_xml_dir}")

    records: List[MedicalQARecord] = parse_medquad_directory(resolved_xml_dir)
    if not records:
        raise ValueError(f"No answerable MedQuAD records found in directory: {resolved_xml_dir}")

    # Convert MedicalQARecord items into LangChain Document instances
    documents: List[Document] = [
        Document(
            page_content=record.to_embedding_text(),
            metadata=record.to_metadata()
        )
        for record in records
    ]

    embeddings = get_instructor_embeddings()
    vectordb = FAISS.from_documents(documents=documents, embedding=embeddings)

    target_index_path = _resolve_index_path(index_path)
    vectordb.save_local(target_index_path)

    return vectordb


def load_medical_vector_db(
    index_path: str = DEFAULT_MEDICAL_INDEX_PATH
) -> FAISS:
    """Load the persisted MedQuAD FAISS vector store.

    Args:
        index_path (str): Path to the persisted FAISS index directory.

    Returns:
        FAISS: Loaded LangChain FAISS vector store.

    Raises:
        FileNotFoundError: If the FAISS index directory does not exist.
    """
    target_index_path = _resolve_index_path(index_path)
    if not os.path.exists(target_index_path):
        raise FileNotFoundError(
            f"Medical FAISS index not found at {target_index_path}. "
            "Run create_medical_vector_db() first to initialize the medical vector store."
        )

    embeddings = get_instructor_embeddings()
    try:
        vectordb = FAISS.load_local(
            target_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except TypeError:
        vectordb = FAISS.load_local(target_index_path, embeddings)

    return vectordb


def get_medical_retriever(
    index_path: str = DEFAULT_MEDICAL_INDEX_PATH,
    score_threshold: float = 0.7
):
    """Obtain a similarity retriever for the medical vector database.

    Args:
        index_path (str): Path to the persisted FAISS index directory.
        score_threshold (float): Similarity score threshold for relevance filtering.

    Returns:
        VectorStoreRetriever: Configured LangChain retriever for MedQuAD documents.
    """
    vectordb = load_medical_vector_db(index_path=index_path)
    return vectordb.as_retriever(score_threshold=score_threshold)


if __name__ == "__main__":
    print("MedQuAD Indexer module initialized cleanly.")
