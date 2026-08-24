"""arXiv Dataset Acquisition Pipeline.

Fetches authentic scientific paper metadata directly from the official arXiv Query API
(Atom feed), formats records to match canonical arXiv JSONL schema, filters for target
AI/ML/NLP domains (cs.CL, cs.AI, cs.LG, cs.CV, stat.ML), and stores the result in
dataset/arxiv_ai_ml_subset.jsonl.

Adheres strictly to arXiv API rate-limit policy (3s delay between requests).
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.parser import SUPPORTED_CATEGORIES, is_target_domain, parse_paper  # type: ignore
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb.parser import SUPPORTED_CATEGORIES, is_target_domain, parse_paper  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("arxiv_acquisition")

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
BASE_ARXIV_API_URL = "http://export.arxiv.org/api/query"

# Target search categories and representative seminal queries
SEARCH_QUERIES = [
    "cat:cs.CL",
    "cat:cs.AI",
    "cat:cs.LG",
    "cat:cs.CV",
    "cat:stat.ML",
    "all:\"retrieval augmented generation\" OR all:\"rag\"",
    "all:\"large language models\" OR all:\"llm\"",
    "all:\"transformer\" AND (cat:cs.CL OR cat:cs.AI)",
    "all:\"low rank adaptation\" OR all:\"lora\"",
    "all:\"diffusion models\" AND (cat:cs.CV OR cat:cs.LG)",
]


def fetch_arxiv_batch(
    query: str,
    start: int = 0,
    max_results: int = 25,
    timeout: int = 15,
) -> str:
    """Query the official arXiv API and return raw Atom XML response string."""
    params = {
        "search_query": query,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    encoded_url = f"{BASE_ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
    logger.info("Querying arXiv API: %s", encoded_url)

    req = urllib.request.Request(
        encoded_url,
        headers={"User-Agent": "ElevanceSkills-GenAI-Internship-ScientificRAG/1.0 (mailto:internship@elevanceskills.edu)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8")


def parse_atom_entry(entry: ET.Element) -> Optional[Dict[str, Any]]:
    """Parse a single Atom XML <entry> element into canonical arXiv JSON record format."""
    id_elem = entry.find(f"{ATOM_NS}id")
    if id_elem is None or not id_elem.text:
        return None

    raw_id = id_elem.text.strip()
    arxiv_id_match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", raw_id)
    if not arxiv_id_match:
        # Check for legacy arXiv ID e.g. cs/0101001
        arxiv_id_match = re.search(r"abs/(.+)$", raw_id)
        if not arxiv_id_match:
            return None
    arxiv_id = arxiv_id_match.group(1)

    title_elem = entry.find(f"{ATOM_NS}title")
    title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else ""
    title = re.sub(r"\s+", " ", title)

    summary_elem = entry.find(f"{ATOM_NS}summary")
    abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None and summary_elem.text else ""
    abstract = re.sub(r"\s+", " ", abstract)

    published_elem = entry.find(f"{ATOM_NS}published")
    published_date = published_elem.text.strip() if published_elem is not None and published_elem.text else ""

    updated_elem = entry.find(f"{ATOM_NS}updated")
    updated_date = updated_elem.text.strip() if updated_elem is not None and updated_elem.text else ""

    authors: List[str] = []
    for author_elem in entry.findall(f"{ATOM_NS}author"):
        name_elem = author_elem.find(f"{ATOM_NS}name")
        if name_elem is not None and name_elem.text:
            authors.append(name_elem.text.strip())

    categories: List[str] = []
    primary_cat_elem = entry.find(f"{ARXIV_NS}primary_category")
    if primary_cat_elem is not None:
        p_term = primary_cat_elem.get("term")
        if p_term:
            categories.append(p_term)

    for cat_elem in entry.findall(f"{ATOM_NS}category"):
        term = cat_elem.get("term")
        if term and term not in categories:
            categories.append(term)

    doi_elem = entry.find(f"{ARXIV_NS}doi")
    doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else ""

    journal_elem = entry.find(f"{ARXIV_NS}journal_ref")
    journal_ref = journal_elem.text.strip() if journal_elem is not None and journal_elem.text else ""

    if not title or not abstract or not categories:
        return None

    # Check target domain filter
    if not any(c in SUPPORTED_CATEGORIES for c in categories):
        return None

    versions = [{"version": "v1", "created": published_date}]

    return {
        "id": arxiv_id,
        "submitter": authors[0] if authors else "",
        "authors": ", ".join(authors),
        "title": title,
        "comments": "",
        "journal-ref": journal_ref,
        "doi": doi,
        "report-no": None,
        "categories": " ".join(categories),
        "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
        "abstract": abstract,
        "versions": versions,
        "published_date": published_date[:10] if len(published_date) >= 10 else published_date,
        "update_date": updated_date[:10] if len(updated_date) >= 10 else published_date[:10],
    }


def download_arxiv_subset(
    output_path: str = "dataset/arxiv_ai_ml_subset.jsonl",
    target_count: int = 150,
    request_delay: float = 3.0,
) -> int:
    """Download and filter authentic arXiv paper metadata into a JSONL subset.

    Args:
        output_path (str): Filepath where JSONL will be saved.
        target_count (int): Desired minimum number of unique scientific papers.
        request_delay (float): Politeness delay in seconds between API requests (arXiv requires >=3s).

    Returns:
        int: Number of collected unique papers.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: Set[str] = set()
    collected_records: List[Dict[str, Any]] = []

    # If file already exists, read existing IDs
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        p_id = rec.get("id") or rec.get("arxiv_id")
                        if p_id:
                            seen_ids.add(p_id)
                            collected_records.append(rec)
                    except json.JSONDecodeError:
                        continue
        logger.info("Found %d pre-existing records in %s", len(collected_records), output_path)

    query_idx = 0
    while len(collected_records) < target_count and query_idx < len(SEARCH_QUERIES):
        query = SEARCH_QUERIES[query_idx]
        query_idx += 1

        batch_start = 0
        batch_size = 25

        while len(collected_records) < target_count and batch_start < 75:
            try:
                xml_data = fetch_arxiv_batch(
                    query=query,
                    start=batch_start,
                    max_results=batch_size,
                )
                tree = ET.fromstring(xml_data)
                entries = tree.findall(f"{ATOM_NS}entry")
                if not entries:
                    break

                new_in_batch = 0
                for entry in entries:
                    record = parse_atom_entry(entry)
                    if record and record["id"] not in seen_ids:
                        seen_ids.add(record["id"])
                        collected_records.append(record)
                        new_in_batch += 1
                        if len(collected_records) >= target_count:
                            break

                logger.info(
                    "Query '%s' [start=%d]: added %d new papers (total collected: %d/%d)",
                    query,
                    batch_start,
                    new_in_batch,
                    len(collected_records),
                    target_count,
                )

                batch_start += batch_size
                time.sleep(request_delay)

            except Exception as exc:
                logger.error("Error querying arXiv API for '%s': %s", query, exc)
                time.sleep(request_delay * 2)
                break

    # Save to JSONL file
    with open(out_file, "w", encoding="utf-8") as f:
        for rec in collected_records:
            f.write(json.dumps(rec) + "\n")

    logger.info("Successfully wrote %d authentic arXiv records to %s", len(collected_records), output_path)
    return len(collected_records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire authentic arXiv AI/ML subset dataset.")
    parser.add_argument("--output", default="dataset/arxiv_ai_ml_subset.jsonl", help="Output JSONL path.")
    parser.add_argument("--count", type=int, default=150, help="Target number of records.")
    parser.add_argument("--delay", type=float, default=3.0, help="Politeness delay between requests in seconds.")
    args = parser.parse_args()

    download_arxiv_subset(
        output_path=args.output,
        target_count=args.count,
        request_delay=args.delay,
    )
