"""Scientific Knowledge Base package for Task 4: Scientific Domain Expert Chatbot.
"""

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.models import ScientificPaper, validate_paper  # type: ignore
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.parser import (  # type: ignore
        SUPPORTED_CATEGORIES,
        is_target_domain,
        parse_paper,
    )
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.ingestion import (  # type: ignore
        IngestionResult,
        ingest_jsonl,
        ingest_papers,
    )
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.vector_store import (  # type: ignore
        DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
        add_papers_to_vector_store,
        build_scientific_vector_store,
        create_paper_documents,
        load_scientific_vector_store,
    )
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.retriever import (  # type: ignore
        ScientificRetrievalResult,
        ScientificRetriever,
        format_retrieval_context,
    )
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.generation import (  # type: ignore
        INSUFFICIENT_EVIDENCE_PHRASE,
        SCIENTIFIC_QA_PROMPT_TEMPLATE,
        ScientificAnswer,
        ScientificGenerator,
        build_scientific_prompt,
    )
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.grounding import (  # type: ignore
        Citation,
        GroundingValidationResult,
        extract_cited_arxiv_ids,
        format_grounded_answer,
        validate_answer_grounding,
    )
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.conversation import (  # type: ignore
        ChatMessage,
        ConversationalResponse,
        ScientificConversationSession,
        ScientificConversationalService,
        condense_followup_query,
    )
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.service import (  # type: ignore
        COMPARISON_PROMPT_TEMPLATE,
        CONCEPT_EXPLANATION_PROMPT_TEMPLATE,
        INTENT_COMPARISON,
        INTENT_CONCEPT_EXPLANATION,
        INTENT_GENERAL,
        INTENT_PAPER_LOOKUP,
        INTENT_SUMMARY,
        SUMMARY_PROMPT_TEMPLATE,
        ScientificExpertResponse,
        ScientificExpertService,
        detect_scientific_intent,
    )
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.exploration import (  # type: ignore
        AuthorStatistic,
        CategoryStatistic,
        ConceptCooccurrence,
        ConceptExtractionResult,
        CorpusSummary,
        RelatedPaperMatch,
        ScientificExplorationEngine,
        ScientificExplorationGraph,
        ScientificGraphEdge,
        ScientificGraphNode,
        TimelineEntry,
        extract_concepts_from_text,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb.models import ScientificPaper, validate_paper  # type: ignore
    # pyrefly: ignore [missing-import]
    from scientific_kb.parser import (  # type: ignore
        SUPPORTED_CATEGORIES,
        is_target_domain,
        parse_paper,
    )
    # pyrefly: ignore [missing-import]
    from scientific_kb.ingestion import (  # type: ignore
        IngestionResult,
        ingest_jsonl,
        ingest_papers,
    )
    # pyrefly: ignore [missing-import]
    from scientific_kb.vector_store import (  # type: ignore
        DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
        add_papers_to_vector_store,
        build_scientific_vector_store,
        create_paper_documents,
        load_scientific_vector_store,
    )
    # pyrefly: ignore [missing-import]
    from scientific_kb.retriever import (  # type: ignore
        ScientificRetrievalResult,
        ScientificRetriever,
        format_retrieval_context,
    )
    # pyrefly: ignore [missing-import]
    from scientific_kb.generation import (  # type: ignore
        INSUFFICIENT_EVIDENCE_PHRASE,
        SCIENTIFIC_QA_PROMPT_TEMPLATE,
        ScientificAnswer,
        ScientificGenerator,
        build_scientific_prompt,
    )
    # pyrefly: ignore [missing-import]
    from scientific_kb.grounding import (  # type: ignore
        Citation,
        GroundingValidationResult,
        extract_cited_arxiv_ids,
        format_grounded_answer,
        validate_answer_grounding,
    )
    # pyrefly: ignore [missing-import]
    from scientific_kb.conversation import (  # type: ignore
        ChatMessage,
        ConversationalResponse,
        ScientificConversationSession,
        ScientificConversationalService,
        condense_followup_query,
    )
    # pyrefly: ignore [missing-import]
    from scientific_kb.service import (  # type: ignore
        COMPARISON_PROMPT_TEMPLATE,
        CONCEPT_EXPLANATION_PROMPT_TEMPLATE,
        INTENT_COMPARISON,
        INTENT_CONCEPT_EXPLANATION,
        INTENT_GENERAL,
        INTENT_PAPER_LOOKUP,
        INTENT_SUMMARY,
        SUMMARY_PROMPT_TEMPLATE,
        ScientificExpertResponse,
        ScientificExpertService,
        detect_scientific_intent,
    )
    # pyrefly: ignore [missing-import]
    from scientific_kb.exploration import (  # type: ignore
        AuthorStatistic,
        CategoryStatistic,
        ConceptCooccurrence,
        ConceptExtractionResult,
        CorpusSummary,
        RelatedPaperMatch,
        ScientificExplorationEngine,
        ScientificExplorationGraph,
        ScientificGraphEdge,
        ScientificGraphNode,
        TimelineEntry,
        extract_concepts_from_text,
    )

__all__ = [
    "ScientificPaper",
    "validate_paper",
    "parse_paper",
    "is_target_domain",
    "SUPPORTED_CATEGORIES",
    "IngestionResult",
    "ingest_papers",
    "ingest_jsonl",
    "create_paper_documents",
    "build_scientific_vector_store",
    "load_scientific_vector_store",
    "add_papers_to_vector_store",
    "DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH",
    "ScientificRetrievalResult",
    "ScientificRetriever",
    "format_retrieval_context",
    "ScientificAnswer",
    "ScientificGenerator",
    "build_scientific_prompt",
    "SCIENTIFIC_QA_PROMPT_TEMPLATE",
    "INSUFFICIENT_EVIDENCE_PHRASE",
    "Citation",
    "GroundingValidationResult",
    "extract_cited_arxiv_ids",
    "validate_answer_grounding",
    "format_grounded_answer",
    "ChatMessage",
    "ScientificConversationSession",
    "condense_followup_query",
    "ScientificConversationalService",
    "ConversationalResponse",
    "ScientificExpertService",
    "ScientificExpertResponse",
    "detect_scientific_intent",
    "INTENT_GENERAL",
    "INTENT_PAPER_LOOKUP",
    "INTENT_SUMMARY",
    "INTENT_CONCEPT_EXPLANATION",
    "INTENT_COMPARISON",
    "SUMMARY_PROMPT_TEMPLATE",
    "COMPARISON_PROMPT_TEMPLATE",
    "CONCEPT_EXPLANATION_PROMPT_TEMPLATE",
    "extract_concepts_from_text",
    "ConceptExtractionResult",
    "ConceptCooccurrence",
    "CategoryStatistic",
    "AuthorStatistic",
    "TimelineEntry",
    "RelatedPaperMatch",
    "ScientificGraphNode",
    "ScientificGraphEdge",
    "ScientificExplorationGraph",
    "CorpusSummary",
    "ScientificExplorationEngine",
]
