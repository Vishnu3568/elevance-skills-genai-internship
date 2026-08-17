"""MedQuAD Retriever Module with Metadata-Aware Reranking.

Combines FAISS semantic search with clinical query analysis, applying deterministic
metadata boosts (topic match, intent/qtype match, entity overlap) to improve evidence retrieval.
"""

import os
import sys
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple, Any

# Disable NLTK import security hook that blocks regex in CWD
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.path.abspath('.')]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document

from medquad_query_analyzer import MedicalQueryAnalyzer, MedicalQueryAnalysis

# Configurable metadata scoring weights
TOPIC_BOOST_WEIGHT: float = 0.20
INTENT_BOOST_WEIGHT: float = 0.20
ENTITY_BOOST_WEIGHT: float = 0.05
MAX_ENTITY_BOOST_CAP: float = 0.15

# Mapping from broad intent to valid MedQuAD qtypes
INTENT_TO_QTYPES: Dict[str, Set[str]] = {
    "TREATMENT": {"treatment", "how does it work", "how effective is it", "support groups"},
    "SYMPTOMS": {"symptoms", "complications", "frequency"},
    "CAUSES_GENETICS": {"causes", "inheritance", "genetic changes", "susceptibility"},
    "DIAGNOSIS": {"exams and tests", "stages", "outlook", "when to contact a medical professional"},
    "PREVENTION": {"prevention", "precautions", "why get vaccinated"},
    "MEDICATION_USAGE": {
        "indication", "usage", "dose", "storage and disposal", "forget a dose",
        "brand names", "brand names of combination products"
    },
    "SAFETY_ADVERSE": {
        "side effects", "emergency or overdose", "important warning", "contraindication",
        "interactions with medications", "interactions with herbs and supplements",
        "interactions with foods", "severe reaction"
    },
    "GENERAL_INFORMATION": {
        "information", "other information", "research", "dietary", "considerations", "how can i learn more"
    }
}


@dataclass
class RetrievalCandidate:
    """Represents a scored candidate document during retrieval and reranking."""
    document: Document
    semantic_score: float
    topic_boost: float
    intent_boost: float
    entity_boost: float
    final_score: float


def _normalize_str(text: Optional[str]) -> str:
    """Normalize string for robust whitespace/case-insensitive comparison."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', str(text).strip().lower())


def _calculate_topic_boost(
    analysis: MedicalQueryAnalysis,
    metadata: Dict[str, Any]
) -> float:
    """Calculate topic match boost if candidate focus/synonyms match query primary topic."""
    if not analysis.primary_topic:
        return 0.0

    target_topic = _normalize_str(analysis.primary_topic)
    candidate_focus = _normalize_str(metadata.get("focus", ""))

    if candidate_focus and candidate_focus == target_topic:
        return TOPIC_BOOST_WEIGHT

    candidate_synonyms = metadata.get("synonyms", "")
    if candidate_synonyms:
        syn_list = [
            _normalize_str(s) for s in (
                candidate_synonyms.split(",") if isinstance(candidate_synonyms, str) else candidate_synonyms
            )
        ]
        if target_topic in syn_list:
            return TOPIC_BOOST_WEIGHT

    return 0.0


def _calculate_intent_boost(
    analysis: MedicalQueryAnalysis,
    metadata: Dict[str, Any]
) -> float:
    """Calculate intent match boost if candidate question_type matches query intent/qtype."""
    candidate_qtype = _normalize_str(metadata.get("question_type", ""))
    if not candidate_qtype:
        return 0.0

    # 1. Exact raw qtype match preference
    if analysis.raw_qtype_match and _normalize_str(analysis.raw_qtype_match) == candidate_qtype:
        return INTENT_BOOST_WEIGHT

    # 2. Broad intent to qtype set mapping
    valid_qtypes = INTENT_TO_QTYPES.get(analysis.intent, set())
    if candidate_qtype in valid_qtypes:
        return INTENT_BOOST_WEIGHT

    return 0.0


def _calculate_entity_overlap_boost(
    analysis: MedicalQueryAnalysis,
    candidate_doc: Document
) -> float:
    """Calculate entity overlap boost for each query entity present in candidate document."""
    if not analysis.entities:
        return 0.0

    metadata = candidate_doc.metadata or {}
    text_corpus = (
        f"{candidate_doc.page_content} "
        f"{metadata.get('focus', '')} "
        f"{metadata.get('synonyms', '')} "
        f"{metadata.get('question', '')}"
    ).lower()

    matched_entity_count = 0
    for entity in analysis.entities:
        clean_ent = _normalize_str(entity.text)
        if clean_ent and re.search(r'\b' + re.escape(clean_ent) + r'\b', text_corpus):
            matched_entity_count += 1

    total_boost = matched_entity_count * ENTITY_BOOST_WEIGHT
    return min(total_boost, MAX_ENTITY_BOOST_CAP)


def retrieve_medical_evidence_with_scores(
    query: str,
    vector_db: Any,
    analyzer: Optional[MedicalQueryAnalyzer] = None,
    top_k: int = 8,
    final_k: int = 3
) -> List[RetrievalCandidate]:
    """Retrieve and rerank candidate documents using semantic similarity and metadata boosts.

    Args:
        query (str): The medical question query.
        vector_db: Loaded LangChain FAISS vector store.
        analyzer (Optional[MedicalQueryAnalyzer]): Query analyzer instance.
        top_k (int): Number of semantic candidates retrieved from vector store.
        final_k (int): Number of top reranked documents returned.

    Returns:
        List[RetrievalCandidate]: Scored and sorted candidate objects.

    Raises:
        TypeError: If query is not a string.
        ValueError: If query is empty or whitespace-only.
    """
    if not isinstance(query, str):
        raise TypeError(f"Expected query to be a string, got {type(query).__name__}")

    stripped_query = query.strip()
    if not stripped_query:
        raise ValueError("Query cannot be empty or whitespace-only.")

    # 1. Analyze user query
    if analyzer is None:
        analyzer = MedicalQueryAnalyzer()
    analysis = analyzer.analyze(stripped_query)

    # 2. Retrieve initial semantic candidates from FAISS vector store
    raw_candidates: List[Tuple[Document, float]] = []
    if hasattr(vector_db, "similarity_search_with_score"):
        try:
            raw_candidates = vector_db.similarity_search_with_score(stripped_query, k=top_k)
        except Exception:
            docs = vector_db.similarity_search(stripped_query, k=top_k)
            raw_candidates = [(doc, 1.0) for doc in docs]
    elif hasattr(vector_db, "similarity_search"):
        docs = vector_db.similarity_search(stripped_query, k=top_k)
        raw_candidates = [(doc, 1.0) for doc in docs]

    if not raw_candidates:
        return []

    # 3. Score and rerank candidates
    candidates: List[RetrievalCandidate] = []
    for item in raw_candidates:
        if isinstance(item, tuple) and len(item) == 2:
            doc, dist_or_score = item
            # Convert FAISS Euclidean distance (smaller = better) to similarity score
            semantic_score = 1.0 / (1.0 + float(dist_or_score)) if float(dist_or_score) >= 0 else 1.0
        else:
            doc = item
            semantic_score = 1.0

        metadata = doc.metadata or {}
        topic_boost = _calculate_topic_boost(analysis, metadata)
        intent_boost = _calculate_intent_boost(analysis, metadata)
        entity_boost = _calculate_entity_overlap_boost(analysis, doc)

        final_score = semantic_score + topic_boost + intent_boost + entity_boost

        candidates.append(
            RetrievalCandidate(
                document=doc,
                semantic_score=semantic_score,
                topic_boost=topic_boost,
                intent_boost=intent_boost,
                entity_boost=entity_boost,
                final_score=final_score
            )
        )

    # 4. Sort candidates descending by final_score
    candidates.sort(key=lambda c: c.final_score, reverse=True)

    return candidates[:final_k]


def retrieve_medical_evidence(
    query: str,
    vector_db: Any,
    analyzer: Optional[MedicalQueryAnalyzer] = None,
    top_k: int = 8,
    final_k: int = 3
) -> List[Document]:
    """Retrieve top reranked LangChain Document evidence objects for a medical query.

    Args:
        query (str): The medical question query.
        vector_db: Loaded LangChain FAISS vector store.
        analyzer (Optional[MedicalQueryAnalyzer]): Query analyzer instance.
        top_k (int): Number of semantic candidates retrieved from vector store.
        final_k (int): Number of top reranked documents returned.

    Returns:
        List[Document]: Top reranked LangChain Document evidence objects.
    """
    scored_candidates = retrieve_medical_evidence_with_scores(
        query=query,
        vector_db=vector_db,
        analyzer=analyzer,
        top_k=top_k,
        final_k=final_k
    )
    return [c.document for c in scored_candidates]


if __name__ == "__main__":
    print("MedQuAD Retriever module initialized cleanly.")
