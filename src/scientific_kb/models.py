"""Data models and validation for scientific papers in the Scientific Domain Expert Chatbot.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScientificPaper:
    """Canonical data model representing an arXiv scientific paper."""

    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    primary_category: str
    published_date: str
    updated_date: Optional[str] = None
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    concepts: List[str] = field(default_factory=list)
    summary: Optional[str] = None


def validate_paper(paper: ScientificPaper) -> None:
    """Validate that a ScientificPaper instance satisfies all schema invariants.

    Args:
        paper (ScientificPaper): The paper instance to validate.

    Raises:
        TypeError: If fields do not have the expected types.
        ValueError: If required fields are empty, whitespace-only, or contain invalid contents.
    """
    if not isinstance(paper, ScientificPaper):
        raise TypeError(f"Expected ScientificPaper instance, got {type(paper).__name__}")

    # 1. Validate arxiv_id
    if not isinstance(paper.arxiv_id, str):
        raise TypeError(f"arxiv_id must be a string, got {type(paper.arxiv_id).__name__}")
    if not paper.arxiv_id.strip():
        raise ValueError("arxiv_id cannot be empty or whitespace-only.")

    # 2. Validate title
    if not isinstance(paper.title, str):
        raise TypeError(f"title must be a string, got {type(paper.title).__name__}")
    if not paper.title.strip():
        raise ValueError("title cannot be empty or whitespace-only.")

    # 3. Validate authors
    if not isinstance(paper.authors, list):
        raise TypeError(f"authors must be a list of strings, got {type(paper.authors).__name__}")
    if not paper.authors:
        raise ValueError("authors must contain at least one author.")
    for idx, author in enumerate(paper.authors):
        if not isinstance(author, str):
            raise TypeError(f"Author at index {idx} must be a string, got {type(author).__name__}")
        if not author.strip():
            raise ValueError(f"Author at index {idx} cannot be empty or whitespace-only.")

    # 4. Validate abstract
    if not isinstance(paper.abstract, str):
        raise TypeError(f"abstract must be a string, got {type(paper.abstract).__name__}")
    if not paper.abstract.strip():
        raise ValueError("abstract cannot be empty or whitespace-only.")

    # 5. Validate categories
    if not isinstance(paper.categories, list):
        raise TypeError(f"categories must be a list of strings, got {type(paper.categories).__name__}")
    if not paper.categories:
        raise ValueError("categories must contain at least one category.")
    for idx, cat in enumerate(paper.categories):
        if not isinstance(cat, str):
            raise TypeError(f"Category at index {idx} must be a string, got {type(cat).__name__}")
        if not cat.strip():
            raise ValueError(f"Category at index {idx} cannot be empty or whitespace-only.")

    # 6. Validate primary_category
    if not isinstance(paper.primary_category, str):
        raise TypeError(f"primary_category must be a string, got {type(paper.primary_category).__name__}")
    if not paper.primary_category.strip():
        raise ValueError("primary_category cannot be empty or whitespace-only.")

    # 7. Validate published_date
    if not isinstance(paper.published_date, str):
        raise TypeError(f"published_date must be a string, got {type(paper.published_date).__name__}")
    if not paper.published_date.strip():
        raise ValueError("published_date cannot be empty or whitespace-only.")

    # 8. Validate optional string fields
    for field_name in ("updated_date", "doi", "journal_ref", "summary"):
        val = getattr(paper, field_name)
        if val is not None and not isinstance(val, str):
            raise TypeError(f"{field_name} must be a string or None, got {type(val).__name__}")

    # 9. Validate concepts
    if paper.concepts is not None:
        if not isinstance(paper.concepts, list):
            raise TypeError(f"concepts must be a list of strings, got {type(paper.concepts).__name__}")
        for idx, concept in enumerate(paper.concepts):
            if not isinstance(concept, str):
                raise TypeError(f"Concept at index {idx} must be a string, got {type(concept).__name__}")
            if not concept.strip():
                raise ValueError(f"Concept at index {idx} cannot be empty or whitespace-only.")


@dataclass
class StructuredPaperAnalysis:
    """Structured data model representing the extracted research understanding of a scientific paper."""

    arxiv_id: str
    title: str
    research_problem: str = ""
    motivation: str = ""
    methodology: str = ""
    model_architecture: str = ""
    key_contributions: List[str] = field(default_factory=list)
    datasets_benchmarks: List[str] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    quantitative_results: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    technical_concepts: List[str] = field(default_factory=list)

    def to_dict(self) -> "Dict[str, Any]":
        """Convert the structured analysis to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: "Dict[str, Any]") -> "StructuredPaperAnalysis":
        """Construct and validate a StructuredPaperAnalysis instance from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dictionary for StructuredPaperAnalysis, got {type(data).__name__}")

        instance = cls(
            arxiv_id=data.get("arxiv_id", ""),
            title=data.get("title", ""),
            research_problem=data.get("research_problem", ""),
            motivation=data.get("motivation", ""),
            methodology=data.get("methodology", ""),
            model_architecture=data.get("model_architecture", ""),
            key_contributions=list(data.get("key_contributions") or []),
            datasets_benchmarks=list(data.get("datasets_benchmarks") or []),
            key_findings=list(data.get("key_findings") or []),
            quantitative_results=list(data.get("quantitative_results") or []),
            limitations=list(data.get("limitations") or []),
            open_questions=list(data.get("open_questions") or []),
            technical_concepts=list(data.get("technical_concepts") or []),
        )
        validate_structured_analysis(instance)
        return instance


def validate_structured_analysis(analysis: StructuredPaperAnalysis) -> None:
    """Validate that a StructuredPaperAnalysis instance satisfies all schema invariants.

    Args:
        analysis (StructuredPaperAnalysis): The analysis instance to validate.

    Raises:
        TypeError: If fields do not have the expected types.
        ValueError: If required fields are empty, whitespace-only, or contain invalid contents.
    """
    if not isinstance(analysis, StructuredPaperAnalysis):
        raise TypeError(f"Expected StructuredPaperAnalysis instance, got {type(analysis).__name__}")

    # 1. Validate arxiv_id
    if not isinstance(analysis.arxiv_id, str):
        raise TypeError(f"arxiv_id must be a string, got {type(analysis.arxiv_id).__name__}")
    if not analysis.arxiv_id.strip():
        raise ValueError("arxiv_id cannot be empty or whitespace-only.")

    # 2. Validate title
    if not isinstance(analysis.title, str):
        raise TypeError(f"title must be a string, got {type(analysis.title).__name__}")
    if not analysis.title.strip():
        raise ValueError("title cannot be empty or whitespace-only.")

    # 3. Validate string fields
    string_fields = (
        ("research_problem", analysis.research_problem),
        ("motivation", analysis.motivation),
        ("methodology", analysis.methodology),
        ("model_architecture", analysis.model_architecture),
    )
    for field_name, val in string_fields:
        if not isinstance(val, str):
            raise TypeError(f"{field_name} must be a string, got {type(val).__name__}")

    # 4. Validate collection fields
    collection_fields = (
        ("key_contributions", analysis.key_contributions),
        ("datasets_benchmarks", analysis.datasets_benchmarks),
        ("key_findings", analysis.key_findings),
        ("quantitative_results", analysis.quantitative_results),
        ("limitations", analysis.limitations),
        ("open_questions", analysis.open_questions),
        ("technical_concepts", analysis.technical_concepts),
    )
    for field_name, col in collection_fields:
        if not isinstance(col, list):
            raise TypeError(f"{field_name} must be a list of strings, got {type(col).__name__}")
        for idx, item in enumerate(col):
            if not isinstance(item, str):
                raise TypeError(f"Item in {field_name} at index {idx} must be a string, got {type(item).__name__}")
            if not item.strip():
                raise ValueError(f"Item in {field_name} at index {idx} cannot be empty or whitespace-only.")

