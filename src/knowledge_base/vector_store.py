from typing import List

try:
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.docstore.document import Document
    from langchain.vectorstores import FAISS


def create_knowledge_documents(records: List[dict]) -> List[Document]:
    """Convert accepted knowledge records into LangChain Documents."""

    documents = []

    for record in records:
        documents.append(
            Document(
                page_content=(
                    f"prompt: {record['prompt']}\n"
                    f"response: {record['response']}"
                ),
                metadata={
                    "source": record["prompt"],
                    "row": record.get("row"),
                },
            )
        )

    return documents


def add_documents_to_vector_store(
    vector_store: FAISS,
    documents: List[Document],
) -> List[str]:
    """Add knowledge documents to an existing FAISS vector store."""

    if not documents:
        return []

    return vector_store.add_documents(documents)
