"""Evidence grounding and citation validation for the Scientific Domain Expert Chatbot.

Provides citation extraction, verification against retrieved sources, detection
of unsupported/hallucinated citations, and rich Markdown answer formatting.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Union

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.generation import (  # type: ignore
        INSUFFICIENT_EVIDENCE_PHRASE,
        ScientificAnswer,
    )
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.retriever import ScientificRetrievalResult  # type: ignore
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb.generation import (  # type: ignore
        INSUFFICIENT_EVIDENCE_PHRASE,
        ScientificAnswer,
    )
    # pyrefly: ignore [missing-import]
    from scientific_kb.retriever import ScientificRetrievalResult  # type: ignore

# Regex pattern for matching arXiv identifiers (standard e.g. 1706.03762 or versioned 2005.11401v2)
ARXIV_ID_PATTERN = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b", re.IGNORECASE)


@dataclass
class Citation:
    """Structured representation of a verified scientific paper citation."""

    arxiv_id: str
    title: str
    url: str = ""
    doi: str = ""
    primary_category: str = ""


@dataclass
class GroundingValidationResult:
    """Result of validating an answer against retrieved paper evidence."""

    is_grounded: bool
    valid_citations: List[Citation] = field(default_factory=list)
    unsupported_citations: List[str] = field(default_factory=list)
    warning_message: Optional[str] = None


def extract_cited_arxiv_ids(text: str) -> List[str]:
    """Extract distinct arXiv identifiers from a text block, preserving first-seen ordering.

    Supports both standard (1706.03762) and versioned (2005.11401v2) arXiv IDs.

    Args:
        text (str): The text block to scan for citations.

    Returns:
        List[str]: Unique list of extracted arXiv IDs.
    """
    if not isinstance(text, str):
        return []

    matches = ARXIV_ID_PATTERN.findall(text)
    seen: Set[str] = set()
    distinct_ids: List[str] = []

    for raw_id in matches:
        cleaned_id = raw_id.strip()
        if cleaned_id and cleaned_id not in seen:
            seen.add(cleaned_id)
            distinct_ids.append(cleaned_id)

    return distinct_ids


def _strip_version(arxiv_id: str) -> str:
    """Helper to remove version suffix (e.g. '2005.11401v2' -> '2005.11401')."""
    return re.sub(r"v\d+$", "", arxiv_id, flags=re.IGNORECASE)


def validate_answer_grounding(
    answer: Union[str, ScientificAnswer],
    sources: List[ScientificRetrievalResult],
) -> GroundingValidationResult:
    """Validate that citations in a generated answer are grounded in retrieved sources.

    Args:
        answer (Union[str, ScientificAnswer]): Generated answer text or ScientificAnswer object.
        sources (List[ScientificRetrievalResult]): List of retrieved paper results.

    Returns:
        GroundingValidationResult: Detailed grounding and citation verification result.

    Raises:
        TypeError: If answer or sources have invalid types.
    """
    if isinstance(answer, ScientificAnswer):
        answer_text = answer.answer
    elif isinstance(answer, str):
        answer_text = answer
    else:
        raise TypeError(f"Expected answer to be str or ScientificAnswer, got {type(answer).__name__}")

    if not isinstance(sources, list):
        raise TypeError(f"Expected sources to be a list, got {type(sources).__name__}")

    stripped_answer = answer_text.strip()

    # 1. Check for explicit insufficient evidence response
    norm_ans = stripped_answer.lower()
    if (
        "not contain sufficient evidence" in norm_ans
        or "insufficient evidence" in norm_ans
        or stripped_answer == INSUFFICIENT_EVIDENCE_PHRASE
    ):
        return GroundingValidationResult(
            is_grounded=False,
            valid_citations=[],
            unsupported_citations=[],
            warning_message="Insufficient evidence to answer the question.",
        )

    # 2. Build index of retrieved sources
    retrieved_by_id = {}
    retrieved_by_base_id = {}
    retrieved_by_title = {}

    for src in sources:
        sid = src.arxiv_id.strip()
        retrieved_by_id[sid] = src
        retrieved_by_base_id[_strip_version(sid)] = src
        if src.title:
            retrieved_by_title[src.title.strip().lower()] = src

    # 3. Extract cited IDs from answer text
    cited_ids = extract_cited_arxiv_ids(stripped_answer)

    valid_citations: List[Citation] = []
    unsupported_citations: List[str] = []
    seen_valid_ids: Set[str] = set()

    for cid in cited_ids:
        base_id = _strip_version(cid)
        matching_src = retrieved_by_id.get(cid) or retrieved_by_base_id.get(base_id)

        if matching_src:
            if matching_src.arxiv_id not in seen_valid_ids:
                seen_valid_ids.add(matching_src.arxiv_id)
                valid_citations.append(
                    Citation(
                        arxiv_id=matching_src.arxiv_id,
                        title=matching_src.title,
                        url=matching_src.url,
                        doi=matching_src.doi,
                        primary_category=matching_src.primary_category,
                    )
                )
        else:
            unsupported_citations.append(cid)

    # 4. If no explicit IDs cited, check if any retrieved paper titles appear in text
    if not cited_ids:
        for title_lower, src in retrieved_by_title.items():
            if len(title_lower) >= 5 and title_lower in norm_ans:
                if src.arxiv_id not in seen_valid_ids:
                    seen_valid_ids.add(src.arxiv_id)
                    valid_citations.append(
                        Citation(
                            arxiv_id=src.arxiv_id,
                            title=src.title,
                            url=src.url,
                            doi=src.doi,
                            primary_category=src.primary_category,
                        )
                    )

    # 5. Determine grounding status
    if unsupported_citations:
        is_grounded = False
        warning = f"Answer contains {len(unsupported_citations)} unsupported citation(s) not in retrieved evidence: {', '.join(unsupported_citations)}"
    else:
        is_grounded = True
        warning = None

    return GroundingValidationResult(
        is_grounded=is_grounded,
        valid_citations=valid_citations,
        unsupported_citations=unsupported_citations,
        warning_message=warning,
    )


def format_grounded_answer(
    answer: Union[str, ScientificAnswer],
    validation: Optional[GroundingValidationResult] = None,
    fallback_sources: Optional[List[ScientificRetrievalResult]] = None,
) -> str:
    """Format a validated scientific answer into publication-quality Markdown with citations.

    Args:
        answer (Union[str, ScientificAnswer]): Answer text or ScientificAnswer instance.
        validation (Optional[GroundingValidationResult]): Precomputed validation result.
        fallback_sources (Optional[List[ScientificRetrievalResult]]): Sources if answer is a plain str.

    Returns:
        str: Clean Markdown formatted answer with source attribution and warnings.
    """
    if isinstance(answer, ScientificAnswer):
        answer_text = answer.answer
        sources = answer.sources
    else:
        answer_text = str(answer)
        sources = fallback_sources or []

    if validation is None:
        validation = validate_answer_grounding(answer_text, sources)

    sections = [
        "### 🧠 Grounded Scientific Explanation",
        answer_text.strip(),
    ]

    # Warning for unsupported citations
    if validation.unsupported_citations:
        warning_block = (
            f"> ⚠️ **Citation Warning**: The answer references {len(validation.unsupported_citations)} paper(s) "
            f"not present in the retrieved evidence: `{', '.join(validation.unsupported_citations)}`."
        )
        sections.append(warning_block)

    # Sources section
    citations_to_render = validation.valid_citations
    if not citations_to_render and sources and validation.is_grounded:
        # If no explicit inline citation was matched but retrieval sources exist, list retrieved context
        citations_to_render = [
            Citation(
                arxiv_id=s.arxiv_id,
                title=s.title,
                url=s.url,
                doi=s.doi,
                primary_category=s.primary_category,
            )
            for s in sources
        ]

    if citations_to_render:
        source_lines = ["### 📚 Verified Sources & References"]
        for idx, cit in enumerate(citations_to_render, 1):
            url_link = f"[`arXiv:{cit.arxiv_id}`]({cit.url})" if cit.url else f"`arXiv:{cit.arxiv_id}`"
            cat_str = f" | `{cit.primary_category}`" if cit.primary_category else ""
            doi_link = f" | [DOI: {cit.doi}](https://doi.org/{cit.doi})" if cit.doi else ""
            source_lines.append(f"{idx}. **{cit.title}** — {url_link}{cat_str}{doi_link}")
        sections.append("\n".join(source_lines))
    else:
        sections.append("### 📚 Verified Sources\n*No verified citations available.*")

    return "\n\n".join(sections)
