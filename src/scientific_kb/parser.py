"""Parser and domain filtering for arXiv scientific papers.
"""

from typing import Any, Dict, List, Optional, Set

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.models import ScientificPaper, validate_paper  # type: ignore
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb.models import ScientificPaper, validate_paper  # type: ignore


SUPPORTED_CATEGORIES: Set[str] = {
    "cs.CL",
    "cs.AI",
    "cs.LG",
    "cs.CV",
    "stat.ML",
}


def is_target_domain(paper: ScientificPaper) -> bool:
    """Check whether a paper belongs to the target scientific AI/ML/NLP domain.

    Args:
        paper (ScientificPaper): The paper instance to check.

    Returns:
        bool: True if at least one category matches the target AI/ML/NLP domain.
    """
    if not isinstance(paper, ScientificPaper):
        raise TypeError(f"Expected ScientificPaper instance, got {type(paper).__name__}")

    return any(cat.strip() in SUPPORTED_CATEGORIES for cat in paper.categories)


def parse_paper(raw_data: Dict[str, Any]) -> ScientificPaper:
    """Parse a raw arXiv-style dictionary into a validated ScientificPaper instance.

    Supports canonical and alternate field names, list/string formats for authors
    and categories, and derives primary category deterministically if omitted.

    Args:
        raw_data (Dict[str, Any]): Raw dictionary containing paper metadata.

    Returns:
        ScientificPaper: Validated canonical scientific paper instance.

    Raises:
        TypeError: If raw_data is not a dict or fields have invalid types.
        ValueError: If required fields are missing, empty, or fail validation.
    """
    if not isinstance(raw_data, dict):
        raise TypeError(f"Expected dictionary for raw_data, got {type(raw_data).__name__}")

    # 1. Extract arxiv_id
    raw_id = raw_data.get("arxiv_id")
    if raw_id is None:
        raw_id = raw_data.get("id")
    if not isinstance(raw_id, str):
        raise TypeError(f"arxiv_id must be a string, got {type(raw_id).__name__}")
    arxiv_id = raw_id.strip()

    # 2. Extract title
    raw_title = raw_data.get("title")
    if not isinstance(raw_title, str):
        raise TypeError(f"title must be a string, got {type(raw_title).__name__}")
    title = raw_title.strip()

    # 3. Extract authors (support list of strings or comma-separated string)
    raw_authors = raw_data.get("authors")
    authors: List[str] = []
    if isinstance(raw_authors, list):
        for idx, item in enumerate(raw_authors):
            if not isinstance(item, str):
                raise TypeError(f"Author at index {idx} must be a string, got {type(item).__name__}")
            cleaned = item.strip()
            if cleaned:
                authors.append(cleaned)
    elif isinstance(raw_authors, str):
        for part in raw_authors.split(","):
            cleaned = part.strip()
            if cleaned:
                authors.append(cleaned)
    else:
        raise TypeError(f"authors must be a list or comma-separated string, got {type(raw_authors).__name__}")

    # 4. Extract abstract
    raw_abstract = raw_data.get("abstract")
    if not isinstance(raw_abstract, str):
        raise TypeError(f"abstract must be a string, got {type(raw_abstract).__name__}")
    abstract = raw_abstract.strip()

    # 5. Extract categories (support list of strings or space-delimited string)
    raw_categories = raw_data.get("categories")
    categories: List[str] = []
    if isinstance(raw_categories, list):
        for idx, item in enumerate(raw_categories):
            if not isinstance(item, str):
                raise TypeError(f"Category at index {idx} must be a string, got {type(item).__name__}")
            cleaned = item.strip()
            if cleaned:
                categories.append(cleaned)
    elif isinstance(raw_categories, str):
        for part in raw_categories.split():
            cleaned = part.strip()
            if cleaned:
                categories.append(cleaned)
    else:
        raise TypeError(f"categories must be a list or space-delimited string, got {type(raw_categories).__name__}")

    # 6. Extract primary_category (use explicit if provided, else derive first category)
    raw_primary = raw_data.get("primary_category")
    if raw_primary is not None:
        if not isinstance(raw_primary, str):
            raise TypeError(f"primary_category must be a string, got {type(raw_primary).__name__}")
        primary_category = raw_primary.strip()
    elif categories:
        primary_category = categories[0]
    else:
        primary_category = ""

    # 7. Extract published_date
    raw_published = raw_data.get("published_date")
    if raw_published is None:
        raw_published = raw_data.get("published")
    if raw_published is None and isinstance(raw_data.get("versions"), list) and raw_data["versions"]:
        first_ver = raw_data["versions"][0]
        if isinstance(first_ver, dict):
            raw_published = first_ver.get("created")
    if not isinstance(raw_published, str):
        raise TypeError(f"published_date must be a string, got {type(raw_published).__name__}")
    published_date = raw_published.strip()

    # 8. Extract optional string fields
    raw_updated = raw_data.get("updated_date")
    if raw_updated is None:
        raw_updated = raw_data.get("updated")
    updated_date = raw_updated.strip() if isinstance(raw_updated, str) and raw_updated.strip() else None

    raw_doi = raw_data.get("doi")
    doi = raw_doi.strip() if isinstance(raw_doi, str) and raw_doi.strip() else None

    raw_journal = raw_data.get("journal_ref")
    if raw_journal is None:
        raw_journal = raw_data.get("journal-ref")
    journal_ref = raw_journal.strip() if isinstance(raw_journal, str) and raw_journal.strip() else None

    raw_summary = raw_data.get("summary")
    summary = raw_summary.strip() if isinstance(raw_summary, str) and raw_summary.strip() else None

    # 9. Extract concepts
    raw_concepts = raw_data.get("concepts")
    concepts: List[str] = []
    if raw_concepts is not None:
        if not isinstance(raw_concepts, list):
            raise TypeError(f"concepts must be a list of strings, got {type(raw_concepts).__name__}")
        for idx, item in enumerate(raw_concepts):
            if not isinstance(item, str):
                raise TypeError(f"Concept at index {idx} must be a string, got {type(item).__name__}")
            cleaned = item.strip()
            if cleaned:
                concepts.append(cleaned)

    paper = ScientificPaper(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        abstract=abstract,
        categories=categories,
        primary_category=primary_category,
        published_date=published_date,
        updated_date=updated_date,
        doi=doi,
        journal_ref=journal_ref,
        concepts=concepts,
        summary=summary,
    )

    validate_paper(paper)
    return paper
