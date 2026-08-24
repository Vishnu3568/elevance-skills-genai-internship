"""Vector store construction and management for the Scientific Domain Expert Chatbot.

Provides LangChain Document formatting, FAISS index construction, loading,
and incremental paper addition for arXiv scientific papers.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.docstore.document import Document
    from langchain.vectorstores import FAISS

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.models import ScientificPaper  # type: ignore
    # pyrefly: ignore [missing-import]
    from src.langchain_helper import get_instructor_embeddings  # type: ignore
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb.models import ScientificPaper  # type: ignore
    # pyrefly: ignore [missing-import]
    from langchain_helper import get_instructor_embeddings  # type: ignore

DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "faiss_index_scientific",
)


def create_paper_documents(papers: List[ScientificPaper]) -> List[Document]:
    """Convert a list of ScientificPaper objects into LangChain Documents.

    The page_content is structured specifically for dense semantic retrieval,
    focusing on Title, Categories, Key Concepts, and Abstract, while keeping
    citation-heavy metadata in the Document metadata dictionary.

    Args:
        papers (List[ScientificPaper]): List of canonical ScientificPaper instances.

    Returns:
        List[Document]: LangChain Documents ready for embedding and indexing.
    """
    documents: List[Document] = []

    for paper in papers:
        categories_str = ", ".join(paper.categories)
        concepts_str = ", ".join(paper.concepts) if paper.concepts else "None"

        page_content = (
            f"Title: {paper.title}\n"
            f"Categories: {paper.primary_category} ({categories_str})\n"
            f"Concepts: {concepts_str}\n"
            f"Abstract: {paper.abstract}"
        )

        metadata: Dict[str, Any] = {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": ", ".join(paper.authors),
            "primary_category": paper.primary_category,
            "categories": list(paper.categories),
            "published_date": paper.published_date,
            "updated_date": paper.updated_date or "",
            "doi": paper.doi or "",
            "journal_ref": paper.journal_ref or "",
            "concepts": list(paper.concepts),
            "url": f"https://arxiv.org/abs/{paper.arxiv_id}",
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def build_scientific_vector_store(
    papers: List[ScientificPaper],
    store_path: str = DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
    embeddings: Optional[Any] = None,
) -> FAISS:
    """Build and persist a fresh FAISS vector store from scientific papers.

    Args:
        papers (List[ScientificPaper]): List of papers to index.
        store_path (str): Target directory to save the FAISS index.
        embeddings (Optional[Any]): Embedding model to use (defaults to get_instructor_embeddings).

    Returns:
        FAISS: The constructed and persisted FAISS vector store.

    Raises:
        ValueError: If papers list is empty.
    """
    if not papers:
        raise ValueError("Cannot build scientific vector store with empty papers list.")

    documents = create_paper_documents(papers)

    if embeddings is None:
        embeddings = get_instructor_embeddings()

    vector_store = FAISS.from_documents(documents, embeddings)

    target_path = Path(store_path)
    target_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(target_path))

    return vector_store


def load_scientific_vector_store(
    store_path: str = DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
    embeddings: Optional[Any] = None,
) -> FAISS:
    """Load an existing scientific FAISS vector store from disk.

    Args:
        store_path (str): Path to the saved FAISS directory.
        embeddings (Optional[Any]): Embedding model to use (defaults to get_instructor_embeddings).

    Returns:
        FAISS: Loaded vector store instance.

    Raises:
        FileNotFoundError: If the index directory does not exist.
    """
    target_path = Path(store_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Scientific vector store not found at: {store_path}")

    if embeddings is None:
        embeddings = get_instructor_embeddings()

    try:
        vector_store = FAISS.load_local(
            str(target_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except TypeError:
        vector_store = FAISS.load_local(str(target_path), embeddings)

    return vector_store


def add_papers_to_vector_store(
    vector_store: FAISS,
    papers: List[ScientificPaper],
) -> FAISS:
    """Add new scientific papers to an existing in-memory FAISS vector store.

    Args:
        vector_store (FAISS): Existing FAISS vector store instance.
        papers (List[ScientificPaper]): List of new papers to add.

    Returns:
        FAISS: The updated vector store instance.
    """
    if not papers:
        return vector_store

    documents = create_paper_documents(papers)
    vector_store.add_documents(documents)
    return vector_store
