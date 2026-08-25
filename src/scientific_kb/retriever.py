"""Scientific retrieval layer for semantic search and multi-criteria metadata filtering.

Provides dense semantic retrieval over arXiv papers with configurable distance
thresholding, category filtering, publication year filtering, author filtering,
and evidence context formatting.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

try:
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.docstore.document import Document
    from langchain.vectorstores import FAISS


@dataclass
class ScientificRetrievalResult:
    """Structured result representing a retrieved scientific paper and its score."""

    document: Document
    arxiv_id: str
    title: str
    authors: str
    primary_category: str
    categories: List[str]
    published_date: str
    score: float
    concepts: List[str] = field(default_factory=list)
    doi: str = ""
    journal_ref: str = ""
    url: str = ""


def format_retrieval_context(results: List[ScientificRetrievalResult]) -> str:
    """Format a list of ScientificRetrievalResult objects into clean evidence context.

    Args:
        results (List[ScientificRetrievalResult]): Retrieved paper results.

    Returns:
        str: Formatted context string suitable for grounding and LLM synthesis.
    """
    if not results:
        return "No relevant scientific papers found."

    context_blocks = []
    for idx, res in enumerate(results, 1):
        categories_str = ", ".join(res.categories) if res.categories else res.primary_category
        concepts_str = ", ".join(res.concepts) if res.concepts else "None"
        doi_str = f" | DOI: {res.doi}" if res.doi else ""
        journal_str = f" | Journal: {res.journal_ref}" if res.journal_ref else ""

        block = (
            f"[Paper {idx}]\n"
            f"Title: {res.title}\n"
            f"arXiv ID: {res.arxiv_id}\n"
            f"Authors: {res.authors}\n"
            f"Published: {res.published_date}{doi_str}{journal_str}\n"
            f"Categories: {res.primary_category} ({categories_str})\n"
            f"Concepts: {concepts_str}\n"
            f"URL: {res.url}\n"
            f"Content:\n{res.document.page_content}"
        )
        context_blocks.append(block)

    return "\n\n---\n\n".join(context_blocks)


class ScientificRetriever:
    """Retriever orchestrating semantic FAISS search with score thresholding and metadata filters."""

    def __init__(
        self,
        vector_store: FAISS,
        default_k: int = 4,
        score_threshold: Optional[float] = None,
    ):
        """Initialize the ScientificRetriever.

        Args:
            vector_store (FAISS): Underlying FAISS vector store.
            default_k (int): Default number of top documents to retrieve.
            score_threshold (Optional[float]): Default maximum L2 distance threshold (lower is closer).
        """
        if vector_store is None:
            raise ValueError("vector_store cannot be None.")

        self.vector_store = vector_store
        self.default_k = default_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        category_filter: Optional[Union[str, List[str]]] = None,
        min_year: Optional[int] = None,
        author_filter: Optional[str] = None,
        concept_filter: Optional[Union[str, List[str]]] = None,
    ) -> List[ScientificRetrievalResult]:
        """Perform semantic search with distance thresholding and metadata filtering.

        Args:
            query (str): The search query string.
            k (Optional[int]): Number of documents to return (defaults to self.default_k).
            score_threshold (Optional[float]): Maximum L2 distance threshold (defaults to self.score_threshold).
            category_filter (Optional[Union[str, List[str]]]): Category tag or list of tags to filter by.
            min_year (Optional[int]): Minimum publication year (e.g. 2020).
            author_filter (Optional[str]): Author name or substring (case-insensitive).
            concept_filter (Optional[Union[str, List[str]]]): Concept tag or list of concept tags.

        Returns:
            List[ScientificRetrievalResult]: Filtered and ranked scientific retrieval results.

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is empty or whitespace-only.
        """
        if not isinstance(query, str):
            raise TypeError(f"Expected query to be a string, got {type(query).__name__}")

        stripped_query = query.strip()
        if not stripped_query:
            raise ValueError("Query cannot be empty or whitespace-only.")

        target_k = k if k is not None else self.default_k
        if target_k <= 0:
            return []

        active_threshold = score_threshold if score_threshold is not None else self.score_threshold

        # Determine total available documents in the vector store dynamically
        total_vectors = 0
        if hasattr(self.vector_store, "index") and hasattr(self.vector_store.index, "ntotal"):
            total_vectors = int(self.vector_store.index.ntotal)
        elif hasattr(self.vector_store, "docstore") and hasattr(self.vector_store.docstore, "_dict"):
            total_vectors = len(self.vector_store.docstore._dict)

        has_filters = bool(
            category_filter
            or (min_year is not None)
            or author_filter
            or concept_filter
            or (active_threshold is not None)
        )

        if has_filters:
            # When metadata filters or score thresholds are active, search up to the full
            # corpus size to guarantee that any matching document in the store can be considered.
            fetch_k = total_vectors if total_vectors > 0 else max(target_k * 10, 100)
        else:
            fetch_k = min(target_k, total_vectors) if total_vectors > 0 else target_k

        fetch_k = max(fetch_k, 1)

        # 1. Execute similarity search with distance score (L2 distance: lower is more similar)
        raw_results = self.vector_store.similarity_search_with_score(stripped_query, k=fetch_k)


        # Normalize filter parameters
        target_categories: Optional[List[str]] = None
        if category_filter:
            if isinstance(category_filter, str):
                target_categories = [category_filter.strip().lower()]
            elif isinstance(category_filter, list):
                target_categories = [c.strip().lower() for c in category_filter if isinstance(c, str)]

        target_concepts: Optional[List[str]] = None
        if concept_filter:
            if isinstance(concept_filter, str):
                target_concepts = [concept_filter.strip().lower()]
            elif isinstance(concept_filter, list):
                target_concepts = [c.strip().lower() for c in concept_filter if isinstance(c, str)]

        author_needle = author_filter.strip().lower() if author_filter else None

        filtered_results: List[ScientificRetrievalResult] = []

        for doc, score in raw_results:
            # 2. Score threshold check (for L2 distance, lower score = more relevant)
            if active_threshold is not None and score > active_threshold:
                continue

            meta = doc.metadata or {}

            # 3. Category filter check
            if target_categories:
                doc_primary = str(meta.get("primary_category", "")).lower()
                doc_categories = [str(c).lower() for c in meta.get("categories", [])]
                if not (doc_primary in target_categories or any(c in target_categories for c in doc_categories)):
                    continue

            # 4. Publication year filter check
            if min_year is not None:
                pub_date = str(meta.get("published_date", "")).strip()
                # Parse year from leading 4 digits (e.g. "2020-05-22" -> 2020)
                try:
                    year = int(pub_date[:4])
                    if year < min_year:
                        continue
                except (ValueError, IndexError):
                    continue

            # 5. Author filter check
            if author_needle:
                doc_authors = str(meta.get("authors", "")).lower()
                if author_needle not in doc_authors:
                    continue

            # 6. Concept filter check
            if target_concepts:
                doc_concepts = [str(c).lower() for c in meta.get("concepts", [])]
                if not any(tc in doc_concepts for tc in target_concepts):
                    continue

            filtered_results.append(
                ScientificRetrievalResult(
                    document=doc,
                    arxiv_id=str(meta.get("arxiv_id", "")),
                    title=str(meta.get("title", "")),
                    authors=str(meta.get("authors", "")),
                    primary_category=str(meta.get("primary_category", "")),
                    categories=list(meta.get("categories", [])),
                    published_date=str(meta.get("published_date", "")),
                    score=float(score),
                    concepts=list(meta.get("concepts", [])),
                    doi=str(meta.get("doi", "")),
                    journal_ref=str(meta.get("journal_ref", "")),
                    url=str(meta.get("url", "")),
                )
            )

            if len(filtered_results) >= target_k:
                break

        return filtered_results

    def format_retrieval_context(self, results: List[ScientificRetrievalResult]) -> str:
        """Convenience method to format results using format_retrieval_context."""
        return format_retrieval_context(results)
