"""Research-paper understanding and structured information-extraction service.

Extracts structured research dimensions (problem, motivation, methodology,
architecture, contributions, benchmarks, findings, results, limitations,
open questions, and technical concepts) from scientific papers and abstracts.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.models import (  # type: ignore
        ScientificPaper,
        StructuredPaperAnalysis,
        validate_structured_analysis,
    )
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.retriever import ScientificRetrievalResult  # type: ignore
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb.models import (  # type: ignore
        ScientificPaper,
        StructuredPaperAnalysis,
        validate_structured_analysis,
    )
    # pyrefly: ignore [missing-import]
    from scientific_kb.retriever import ScientificRetrievalResult  # type: ignore

STRUCTURED_EXTRACTION_PROMPT_TEMPLATE = """You are an expert scientific AI research assistant specializing in deep research-paper understanding, methodology extraction, and literature analysis across Artificial Intelligence, Machine Learning, and Natural Language Processing.

Analyze the provided scientific paper metadata and abstract carefully to extract a structured research understanding.

RULES:
1. Base your extraction strictly on the provided paper metadata and abstract evidence.
2. Do NOT fabricate or invent datasets, numerical results, architectures, contributions, open questions, or limitations.
3. If a field is not supported by the provided text, return:
   - "" (empty string) for scalar string fields.
   - [] (empty list) for list fields.
4. Preserve quantitative information, percentages, benchmarks, and metrics when explicitly present.
5. Distinguish actual reported results from high-level claims or aspirations.
6. Extract technical concepts and keywords that are explicitly supported by the text.
7. Return ONLY a valid, parseable JSON object matching the exact schema below. Do not wrap in Markdown fences, do not add commentary or explanations outside the JSON.

REQUIRED JSON SCHEMA:
{{
  "research_problem": "Core research challenge or question addressed by the paper",
  "motivation": "Why this problem matters and the limitations of previous approaches",
  "methodology": "The high-level approach, paradigm, or theoretical mechanism proposed",
  "model_architecture": "Specific neural architecture, components, or mathematical formulation",
  "key_contributions": ["List of explicit contributions made by the authors"],
  "datasets_benchmarks": ["List of datasets, benchmarks, or corpora mentioned"],
  "key_findings": ["List of primary qualitative or conceptual findings and takeaways"],
  "quantitative_results": ["List of specific quantitative results or empirical metrics reported"],
  "limitations": ["List of explicitly acknowledged constraints, trade-offs, or failure modes"],
  "open_questions": ["List of explicitly mentioned future work, open challenges, or unexplored directions"],
  "technical_concepts": ["List of key technical terms, methods, and algorithmic concepts"]
}}

---
SOURCE PAPER METADATA:
Title: {title}
arXiv ID: {arxiv_id}
Categories: {categories}
Authors: {authors}
Published Date: {published_date}
---

PAPER ABSTRACT / SCIENTIFIC EVIDENCE:
{abstract}
---

STRUCTURED JSON EXTRACTION:"""


def build_paper_understanding_prompt(
    paper_or_title: Union[ScientificPaper, ScientificRetrievalResult, Dict[str, Any], str],
    abstract: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    categories: Optional[Union[str, List[str]]] = None,
    authors: Optional[Union[str, List[str]]] = None,
    published_date: Optional[str] = None,
) -> str:
    """Build the standardized structured research-paper extraction prompt.

    Args:
        paper_or_title (Union[ScientificPaper, ScientificRetrievalResult, Dict[str, Any], str]):
            Paper object, dictionary, or title string.
        abstract (Optional[str]): Paper abstract text if not passed via object.
        arxiv_id (Optional[str]): arXiv identifier if not passed via object.
        categories (Optional[Union[str, List[str]]]): Categories string or list.
        authors (Optional[Union[str, List[str]]]): Authors string or list.
        published_date (Optional[str]): Publication timestamp string.

    Returns:
        str: Fully rendered extraction prompt.

    Raises:
        ValueError: If title or abstract is empty or whitespace-only.
        TypeError: If input types are invalid.
    """
    p_title = ""
    p_id = ""
    p_abstract = ""
    p_cats = ""
    p_authors = ""
    p_date = ""

    if isinstance(paper_or_title, ScientificPaper):
        p_title = paper_or_title.title
        p_id = paper_or_title.arxiv_id
        p_abstract = paper_or_title.abstract
        p_cats = ", ".join(paper_or_title.categories)
        p_authors = ", ".join(paper_or_title.authors)
        p_date = paper_or_title.published_date
    elif isinstance(paper_or_title, ScientificRetrievalResult):
        p_title = paper_or_title.title
        p_id = paper_or_title.arxiv_id
        p_cats = ", ".join(paper_or_title.categories) if isinstance(paper_or_title.categories, list) else str(paper_or_title.categories)
        p_authors = paper_or_title.authors
        p_date = paper_or_title.published_date
        # Extract abstract from page_content or fallback
        if paper_or_title.document and hasattr(paper_or_title.document, "page_content"):
            content = paper_or_title.document.page_content
            if "Abstract:" in content:
                p_abstract = content.split("Abstract:", 1)[1].strip()
            else:
                p_abstract = content.strip()
    elif isinstance(paper_or_title, dict):
        p_title = str(paper_or_title.get("title", "") or "")
        p_id = str(paper_or_title.get("arxiv_id") or paper_or_title.get("id") or "")
        p_abstract = str(paper_or_title.get("abstract", "") or "")
        raw_cats = paper_or_title.get("categories", "")
        p_cats = ", ".join(raw_cats) if isinstance(raw_cats, list) else str(raw_cats)
        raw_authors = paper_or_title.get("authors", "")
        p_authors = ", ".join(raw_authors) if isinstance(raw_authors, list) else str(raw_authors)
        p_date = str(paper_or_title.get("published_date") or paper_or_title.get("published") or "")
    elif isinstance(paper_or_title, str):
        p_title = paper_or_title
    else:
        raise TypeError(
            f"Expected paper_or_title to be ScientificPaper, ScientificRetrievalResult, dict, or str, got {type(paper_or_title).__name__}"
        )

    # Override with explicit keyword arguments if supplied
    if abstract is not None:
        p_abstract = abstract
    if arxiv_id is not None:
        p_id = arxiv_id
    if categories is not None:
        p_cats = ", ".join(categories) if isinstance(categories, list) else str(categories)
    if authors is not None:
        p_authors = ", ".join(authors) if isinstance(authors, list) else str(authors)
    if published_date is not None:
        p_date = published_date

    # Validate essential fields
    if not isinstance(p_title, str) or not p_title.strip():
        raise ValueError("Title cannot be empty or whitespace-only.")
    if not isinstance(p_abstract, str) or not p_abstract.strip():
        raise ValueError("Abstract cannot be empty or whitespace-only.")

    return STRUCTURED_EXTRACTION_PROMPT_TEMPLATE.format(
        title=p_title.strip(),
        arxiv_id=p_id.strip() or "N/A",
        categories=p_cats.strip() or "General AI/ML",
        authors=p_authors.strip() or "Unknown",
        published_date=p_date.strip() or "Unknown",
        abstract=p_abstract.strip(),
    )


class PaperUnderstandingService:
    """Service for extracting structured research understanding from scientific papers and abstracts."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        model_name: str = "google/flan-t5-base",
        temperature: float = 0.1,
    ):
        """Initialize the PaperUnderstandingService.

        Args:
            llm (Optional[Any]): Configurable LLM backend (LangChain model, callable, or adapter).
            model_name (str): Model identifier.
            temperature (float): Generation temperature for deterministic structured output.
        """
        self.llm = llm
        self.model_name = model_name
        self.temperature = temperature

    def _invoke_llm(self, prompt: str) -> str:
        """Invoke the configured LLM handling various invocation protocols."""
        if self.llm is None:
            raise RuntimeError(
                "No LLM backend configured. Please supply a valid LLM instance to PaperUnderstandingService."
            )

        if hasattr(self.llm, "invoke"):
            res = self.llm.invoke(prompt)
            if hasattr(res, "content"):
                return str(res.content)
            return str(res)

        if callable(self.llm):
            return str(self.llm(prompt))

        if hasattr(self.llm, "predict"):
            return str(self.llm.predict(prompt))

        raise TypeError(f"Unsupported LLM object type: {type(self.llm).__name__}")

    def _parse_llm_json(self, raw_output: str) -> Dict[str, Any]:
        """Parse raw LLM output into a dictionary, stripping markdown fences if present.

        Args:
            raw_output (str): Raw string output from LLM.

        Returns:
            Dict[str, Any]: Parsed JSON dictionary.

        Raises:
            ValueError: If JSON is malformed or not a dictionary.
        """
        if not isinstance(raw_output, str):
            raise TypeError(f"Expected raw_output to be a string, got {type(raw_output).__name__}")

        cleaned = raw_output.strip()
        # Remove Markdown JSON code fences if wrapped
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as decode_err:
            raise ValueError(f"Failed to parse LLM extraction response as valid JSON: {decode_err}")

        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object (dict) from LLM, got {type(parsed).__name__}")

        return parsed

    def analyze_paper_structure(
        self,
        paper: Union[ScientificPaper, ScientificRetrievalResult, Dict[str, Any], str],
        abstract: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        title: Optional[str] = None,
        categories: Optional[Union[str, List[str]]] = None,
        authors: Optional[Union[str, List[str]]] = None,
        published_date: Optional[str] = None,
    ) -> StructuredPaperAnalysis:
        """Extract structured research understanding from a scientific paper.

        Args:
            paper (Union[ScientificPaper, ScientificRetrievalResult, Dict[str, Any], str]):
                Paper object, dictionary, or title.
            abstract (Optional[str]): Abstract text override.
            arxiv_id (Optional[str]): arXiv identifier override.
            title (Optional[str]): Title override.
            categories (Optional[Union[str, List[str]]]): Category override.
            authors (Optional[Union[str, List[str]]]): Author override.
            published_date (Optional[str]): Published date override.

        Returns:
            StructuredPaperAnalysis: Fully validated structured understanding object.

        Raises:
            ValueError: If abstract or title is empty/missing, or if JSON extraction fails validation.
            TypeError: If paper or LLM output is malformed.
        """
        # Determine source title and arXiv ID to preserve authoritative identity
        source_title = ""
        source_id = ""

        if isinstance(paper, ScientificPaper):
            source_title = paper.title
            source_id = paper.arxiv_id
        elif isinstance(paper, ScientificRetrievalResult):
            source_title = paper.title
            source_id = paper.arxiv_id
        elif isinstance(paper, dict):
            source_title = str(paper.get("title", "") or "")
            source_id = str(paper.get("arxiv_id") or paper.get("id") or "")
        elif isinstance(paper, str):
            source_title = paper
        else:
            raise TypeError(f"Unsupported paper type: {type(paper).__name__}")

        if title is not None:
            source_title = title
        if arxiv_id is not None:
            source_id = arxiv_id

        if not source_title.strip():
            raise ValueError("Title cannot be empty or whitespace-only.")
        if not source_id.strip():
            raise ValueError("arxiv_id cannot be empty or whitespace-only.")

        # Build extraction prompt (this validates that abstract is not empty)
        prompt = build_paper_understanding_prompt(
            paper_or_title=paper,
            abstract=abstract,
            arxiv_id=source_id,
            categories=categories,
            authors=authors,
            published_date=published_date,
        )

        # Invoke LLM
        raw_output = self._invoke_llm(prompt)

        # Parse JSON
        parsed_data = self._parse_llm_json(raw_output)

        # Construct and validate StructuredPaperAnalysis, strictly preserving authoritative source ID & title
        analysis = StructuredPaperAnalysis(
            arxiv_id=source_id.strip(),
            title=source_title.strip(),
            research_problem=str(parsed_data.get("research_problem", "") or "").strip(),
            motivation=str(parsed_data.get("motivation", "") or "").strip(),
            methodology=str(parsed_data.get("methodology", "") or "").strip(),
            model_architecture=str(parsed_data.get("model_architecture", "") or "").strip(),
            key_contributions=parsed_data.get("key_contributions", []) if parsed_data.get("key_contributions") is not None else [],
            datasets_benchmarks=parsed_data.get("datasets_benchmarks", []) if parsed_data.get("datasets_benchmarks") is not None else [],
            key_findings=parsed_data.get("key_findings", []) if parsed_data.get("key_findings") is not None else [],
            quantitative_results=parsed_data.get("quantitative_results", []) if parsed_data.get("quantitative_results") is not None else [],
            limitations=parsed_data.get("limitations", []) if parsed_data.get("limitations") is not None else [],
            open_questions=parsed_data.get("open_questions", []) if parsed_data.get("open_questions") is not None else [],
            technical_concepts=parsed_data.get("technical_concepts", []) if parsed_data.get("technical_concepts") is not None else [],
        )

        validate_structured_analysis(analysis)
        return analysis
