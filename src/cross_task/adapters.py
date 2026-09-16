"""Response Adapters for Cross-Task Integration (Day 36).

Converts specialist responses from Tasks 1–6 into the standardized UnifiedResponse contract
without information loss or hallucination.
"""

from typing import Any, Dict, List, Optional

from .models import (
    CitationItem,
    ConfidenceTier,
    DomainType,
    EvidenceItem,
    UnifiedResponse,
)


def _determine_confidence_tier(score: float) -> str:
    """Classify confidence score into deterministic rating tier."""
    if score >= 0.80:
        return ConfidenceTier.HIGH.value
    elif score >= 0.65:
        return ConfidenceTier.MEDIUM.value
    elif score >= 0.50:
        return ConfidenceTier.LOW.value
    return ConfidenceTier.INSUFFICIENT.value


def adapt_chatbot_response(
    resp: Any,
    query: str,
    detected_language: str = "en",
    detected_language_name: str = "English",
    session_id: str = "default_session",
    execution_time_ms: float = 0.0,
) -> UnifiedResponse:
    """Adapt Task 1/2 Customer Service response to UnifiedResponse."""
    evidence_items: List[EvidenceItem] = []
    citations: List[CitationItem] = []

    # Map source documents to evidence and citations
    raw_docs = getattr(resp, "source_documents", []) or []
    for idx, doc in enumerate(raw_docs):
        page_content = getattr(doc, "page_content", str(doc))
        metadata = getattr(doc, "metadata", {}) or {}
        evidence_items.append(
            EvidenceItem(
                description=page_content[:300] + ("..." if len(page_content) > 300 else ""),
                source="customer_knowledge_base",
                confidence=float(getattr(resp, "confidence_score", 1.0)),
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        )
        source_id = str(metadata.get("source", f"kb_item_{idx+1}"))
        citations.append(
            CitationItem(
                title=f"Knowledge Base FAQ ({source_id})",
                source_id=source_id,
                extra=metadata if isinstance(metadata, dict) else {},
            )
        )

    confidence = float(getattr(resp, "confidence_score", 1.0))
    tier = _determine_confidence_tier(confidence)
    is_ood = bool(getattr(resp, "is_ood", False))
    final_answer = getattr(resp, "final_answer", str(resp))

    return UnifiedResponse(
        query=query,
        final_text_response=final_answer,
        domain=DomainType.CUSTOMER_SUPPORT.value,
        detected_language=detected_language,
        detected_language_name=detected_language_name,
        confidence_score=confidence,
        confidence_tier=tier,
        evidence=evidence_items,
        citations=citations,
        is_grounded=not is_ood,
        sentiment=getattr(resp, "sentiment_label", None),
        execution_time_ms=execution_time_ms,
        session_id=session_id,
        metadata={
            "raw_answer": getattr(resp, "raw_answer", ""),
            "is_ood": is_ood,
        },
    )


def adapt_medical_response(
    resp: Any,
    query: str,
    detected_language: str = "en",
    detected_language_name: str = "English",
    session_id: str = "default_session",
    execution_time_ms: float = 0.0,
) -> UnifiedResponse:
    """Adapt Task 3 Medical Q&A response to UnifiedResponse."""
    evidence_items: List[EvidenceItem] = []
    citations: List[CitationItem] = []

    # Map evidence documents
    evidence_docs = getattr(resp, "evidence_documents", []) or []
    for doc in evidence_docs:
        content = getattr(doc, "page_content", str(doc))
        meta = getattr(doc, "metadata", {}) or {}
        evidence_items.append(
            EvidenceItem(
                description=content[:300] + ("..." if len(content) > 300 else ""),
                source="medquad_clinical_corpus",
                confidence=float(getattr(resp, "confidence_score", 1.0)),
                metadata=meta if isinstance(meta, dict) else {},
            )
        )

    # Map citations
    raw_citations = getattr(resp, "citations", []) or []
    for cit in raw_citations:
        if isinstance(cit, dict):
            citations.append(
                CitationItem(
                    title=str(cit.get("title", cit.get("source", "MedQuAD NIH Record"))),
                    source_id=str(cit.get("source_id", cit.get("doc_id", "NIH_CLINICAL"))),
                    url=cit.get("url"),
                    extra=cit,
                )
            )

    confidence = float(getattr(resp, "confidence_score", 0.0))
    tier = _determine_confidence_tier(confidence)
    is_grounded = bool(getattr(resp, "is_grounded", True))
    status = getattr(resp, "status", "GROUNDED")

    return UnifiedResponse(
        query=query,
        final_text_response=getattr(resp, "final_answer", str(resp)),
        domain=DomainType.MEDICAL.value,
        detected_language=detected_language,
        detected_language_name=detected_language_name,
        confidence_score=confidence,
        confidence_tier=tier,
        evidence=evidence_items,
        citations=citations,
        is_grounded=is_grounded,
        execution_time_ms=execution_time_ms,
        session_id=session_id,
        metadata={
            "medical_status": status,
            "analysis": getattr(resp, "analysis", None),
        },
    )


def adapt_scientific_response(
    resp: Any,
    query: str,
    detected_language: str = "en",
    detected_language_name: str = "English",
    session_id: str = "default_session",
    execution_time_ms: float = 0.0,
) -> UnifiedResponse:
    """Adapt Task 4 Scientific Domain Expert response to UnifiedResponse."""
    evidence_items: List[EvidenceItem] = []
    citations: List[CitationItem] = []

    # Map sources
    sources = getattr(resp, "sources", []) or []
    for src in sources:
        title = getattr(src, "title", "arXiv Paper")
        arxiv_id = getattr(src, "arxiv_id", "N/A")
        doc = getattr(src, "document", None)
        abstract = getattr(doc, "page_content", "") if doc is not None else ""
        authors_val = getattr(src, "authors", "")
        pub_date = getattr(src, "published_date", getattr(src, "published", ""))
        cats = getattr(src, "categories", [])
        score = float(getattr(src, "score", 0.0))
        confidence = max(0.0, min(1.0, 1.0 - (score / 10.0)))

        evidence_items.append(
            EvidenceItem(
                description=f"[{arxiv_id}] {title}: {abstract[:250]}..." if abstract else f"[{arxiv_id}] {title}",
                source=f"arxiv:{arxiv_id}",
                confidence=confidence,
                metadata={
                    "authors": authors_val,
                    "categories": cats,
                    "published": pub_date,
                },
            )
        )

    # Map citations
    raw_citations = getattr(resp, "citations", []) or []
    for cit in raw_citations:
        arxiv_id = getattr(cit, "arxiv_id", "")
        citations.append(
            CitationItem(
                title=getattr(cit, "title", f"arXiv:{arxiv_id}"),
                source_id=arxiv_id,
                url=getattr(cit, "url", f"https://arxiv.org/abs/{arxiv_id}"),
                authors=", ".join(getattr(cit, "authors", [])) if hasattr(cit, "authors") else None,
                extra={"published": getattr(cit, "published", "")},
            )
        )

    is_grounded = bool(getattr(resp, "grounded", True))
    warning = getattr(resp, "warning_message", None)
    confidence = 0.95 if is_grounded else 0.40
    tier = _determine_confidence_tier(confidence)

    return UnifiedResponse(
        query=query,
        final_text_response=getattr(resp, "answer", str(resp)),
        domain=DomainType.SCIENTIFIC.value,
        detected_language=detected_language,
        detected_language_name=detected_language_name,
        confidence_score=confidence,
        confidence_tier=tier,
        evidence=evidence_items,
        citations=citations,
        is_grounded=is_grounded,
        warning_message=warning,
        execution_time_ms=execution_time_ms,
        session_id=session_id,
        metadata={
            "intent": getattr(resp, "intent", "general_question"),
            "condensed_query": getattr(resp, "condensed_query", query),
        },
    )


def adapt_multimodal_response(
    resp: Any,
    query: str,
    detected_language: str = "en",
    detected_language_name: str = "English",
    session_id: str = "default_session",
    execution_time_ms: float = 0.0,
) -> UnifiedResponse:
    """Adapt Task 5 Multimodal Assistant response to UnifiedResponse."""
    evidence_items: List[EvidenceItem] = []

    visual_evidence = getattr(resp, "visual_evidence", []) or []
    for v in visual_evidence:
        evidence_items.append(
            EvidenceItem(
                description=getattr(v, "description", str(v)),
                source="visual_input_analysis",
                confidence=float(getattr(v, "confidence", 1.0)),
                region_label=getattr(v, "region_label", None),
            )
        )

    is_grounded = bool(getattr(resp, "grounded", True))
    confidence = 1.0 if is_grounded else 0.50
    tier = _determine_confidence_tier(confidence)
    modality = getattr(resp, "modality", None)
    modality_val = getattr(modality, "value", str(modality)) if modality else "text_and_image"

    return UnifiedResponse(
        query=query,
        final_text_response=getattr(resp, "answer", str(resp)),
        domain=DomainType.MULTIMODAL.value,
        detected_language=detected_language,
        detected_language_name=detected_language_name,
        confidence_score=confidence,
        confidence_tier=tier,
        evidence=evidence_items,
        is_grounded=is_grounded,
        warning_message=getattr(resp, "warning_message", None),
        execution_time_ms=execution_time_ms,
        session_id=getattr(resp, "session_id", session_id),
        metadata={
            "modality": modality_val,
            "ambiguity": getattr(resp, "ambiguity", None),
            "image_metadata": getattr(resp, "image_metadata", None),
        },
    )


def adapt_multilingual_response(
    resp: Any,
    query: str,
    domain: str = DomainType.CUSTOMER_SUPPORT.value,
    session_id: str = "default_session",
    execution_time_ms: float = 0.0,
) -> UnifiedResponse:
    """Adapt Task 6 Multilingual response to UnifiedResponse."""
    evidence_items: List[EvidenceItem] = []
    citations: List[CitationItem] = []

    src_docs = getattr(resp, "source_documents", []) or []
    for idx, doc in enumerate(src_docs):
        if isinstance(doc, dict):
            prompt_text = doc.get("prompt", "")
            resp_text = doc.get("response", "")
            evidence_items.append(
                EvidenceItem(
                    description=f"FAQ Q: {prompt_text} | A: {resp_text}",
                    source="multilingual_aligned_kb",
                    confidence=float(getattr(resp, "confidence_score", 1.0)),
                    metadata=doc,
                )
            )
            citations.append(
                CitationItem(
                    title=f"Knowledge Base Record ({doc.get('source', f'kb_{idx+1}')})",
                    source_id=str(doc.get("source", f"kb_{idx+1}")),
                    extra=doc,
                )
            )

    confidence = float(getattr(resp, "confidence_score", 1.0))
    tier = _determine_confidence_tier(confidence)
    lang_code = getattr(resp, "language", "en")

    return UnifiedResponse(
        query=query,
        final_text_response=getattr(resp, "final_answer", str(resp)),
        domain=domain,
        detected_language=lang_code,
        detected_language_name=lang_code.upper(),
        confidence_score=confidence,
        confidence_tier=tier,
        evidence=evidence_items,
        citations=citations,
        is_grounded=bool(getattr(resp, "is_grounded", True)),
        warning_message=getattr(resp, "clarification_prompt", None) if getattr(resp, "is_ambiguous", False) else None,
        execution_time_ms=execution_time_ms,
        session_id=session_id,
        metadata={
            "intent": getattr(resp, "intent", ""),
            "aligned_query": getattr(resp, "aligned_query", query),
            "is_ambiguous": getattr(resp, "is_ambiguous", False),
        },
    )
