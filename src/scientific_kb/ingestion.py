"""Ingestion pipeline for arXiv scientific papers.

Provides streaming ingestion, schema validation, domain filtering, deduplication,
and error tracking for raw arXiv records and JSONL files.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.models import ScientificPaper  # type: ignore
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.parser import is_target_domain, parse_paper  # type: ignore
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb.models import ScientificPaper  # type: ignore
    # pyrefly: ignore [missing-import]
    from scientific_kb.parser import is_target_domain, parse_paper  # type: ignore


@dataclass
class IngestionResult:
    """Structured result containing ingested papers and execution metrics."""

    papers: List[ScientificPaper] = field(default_factory=list)
    accepted: int = 0
    invalid: int = 0
    out_of_domain: int = 0
    duplicates: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)


def ingest_papers(records: Iterable[Any]) -> IngestionResult:
    """Ingest, validate, filter, and deduplicate an iterable of raw arXiv records.

    Processes records in a streaming fashion without converting the full input
    into a list upfront.

    Args:
        records (Iterable[Any]): Stream/iterable of raw dictionaries.

    Returns:
        IngestionResult: Structured ingestion summary with accepted papers and error log.
    """
    papers: List[ScientificPaper] = []
    seen_arxiv_ids: Set[str] = set()
    errors: List[Dict[str, Any]] = []

    accepted_count = 0
    invalid_count = 0
    out_of_domain_count = 0
    duplicates_count = 0

    for idx, raw_record in enumerate(records):
        # Handle malformed JSON marker from stream generator
        if isinstance(raw_record, dict) and "__json_error__" in raw_record:
            invalid_count += 1
            errors.append({
                "index": idx,
                "arxiv_id": None,
                "status": "INVALID",
                "error": f"Malformed JSON: {raw_record['__json_error__']}",
            })
            continue

        if not isinstance(raw_record, dict):
            invalid_count += 1
            errors.append({
                "index": idx,
                "arxiv_id": None,
                "status": "INVALID",
                "error": f"Expected dictionary record, got {type(raw_record).__name__}",
            })
            continue

        # Attempt to parse and validate the paper
        try:
            paper = parse_paper(raw_record)
        except (TypeError, ValueError) as parse_err:
            invalid_count += 1
            raw_id = raw_record.get("arxiv_id") or raw_record.get("id")
            errors.append({
                "index": idx,
                "arxiv_id": str(raw_id) if raw_id is not None else None,
                "status": "INVALID",
                "error": str(parse_err),
            })
            continue

        # Domain filtering: check if paper has at least one supported category
        if not is_target_domain(paper):
            out_of_domain_count += 1
            errors.append({
                "index": idx,
                "arxiv_id": paper.arxiv_id,
                "status": "OUT_OF_DOMAIN",
                "reason": f"Categories {paper.categories} not in supported domain",
            })
            continue

        # Deduplication check: preserve first-seen ordering
        if paper.arxiv_id in seen_arxiv_ids:
            duplicates_count += 1
            errors.append({
                "index": idx,
                "arxiv_id": paper.arxiv_id,
                "status": "DUPLICATE",
                "reason": f"Duplicate arxiv_id: {paper.arxiv_id}",
            })
            continue

        # Accepted paper
        seen_arxiv_ids.add(paper.arxiv_id)
        papers.append(paper)
        accepted_count += 1

    return IngestionResult(
        papers=papers,
        accepted=accepted_count,
        invalid=invalid_count,
        out_of_domain=out_of_domain_count,
        duplicates=duplicates_count,
        errors=errors,
    )


def ingest_jsonl(file_path: str) -> IngestionResult:
    """Stream and ingest arXiv papers from a local JSONL file.

    Reads line-by-line to avoid loading large files completely into memory.

    Args:
        file_path (str): Path to local UTF-8 JSON Lines file.

    Returns:
        IngestionResult: Structured ingestion summary.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    def _line_generator(target_file: Path) -> Iterable[Dict[str, Any]]:
        with open(target_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    yield data
                except json.JSONDecodeError as decode_err:
                    yield {"__json_error__": str(decode_err), "__raw_line__": line_str}

    return ingest_papers(_line_generator(path))
