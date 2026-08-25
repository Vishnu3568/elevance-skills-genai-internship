"""Kaggle Cornell arXiv Dataset Streaming Filter & Acquisition Pipeline.

Processes the official Cornell University arXiv Kaggle dataset snapshot
(arxiv-metadata-oai-snapshot.json) in a memory-safe, streaming line-by-line manner,
filters records for target AI/ML/NLP subcategories (cs.CL, cs.AI, cs.LG, cs.CV, stat.ML),
deduplicates by canonical arXiv ID, validates against the ScientificPaper schema,
and produces a curated, deterministic scientific corpus.
"""

import argparse
import gzip
import json
import logging
import os
import re
import sys
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Set, TextIO, Tuple

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.models import ScientificPaper  # type: ignore
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.parser import SUPPORTED_CATEGORIES, is_target_domain, parse_paper  # type: ignore
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb.models import ScientificPaper  # type: ignore
    # pyrefly: ignore [missing-import]
    from scientific_kb.parser import SUPPORTED_CATEGORIES, is_target_domain, parse_paper  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("kaggle_arxiv_filter")

DEFAULT_KAGGLE_SNAPSHOT_PATH = "dataset/arxiv-metadata-oai-snapshot.json"
DEFAULT_TARGET_OUTPUT_PATH = "dataset/arxiv_ai_ml_subset_kaggle.jsonl"

# Default year window matching the project's intended scientific corpus era
DEFAULT_MIN_YEAR: int = 2019
DEFAULT_MAX_YEAR: int = 2020

# RFC 2822 pattern used by Kaggle arXiv snapshot versions[].created field
# e.g. "Sun, 1 Apr 2007 13:06:50 GMT"
_RFC2822_PATTERN = re.compile(
    r"^\w{3},\s+\d{1,2}\s+\w{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}",
    re.ASCII,
)


def parse_kaggle_date(raw: str) -> str:
    """Normalize a raw Kaggle arXiv date string to ISO 8601 UTC format.

    The Kaggle Cornell arXiv snapshot stores ``versions[].created`` as an
    RFC 2822 string, e.g.::

        "Fri, 18 Dec 2020 05:31:07 GMT"

    This function converts it to::

        "2020-12-18T05:31:07Z"

    If the input is already in ISO format (``YYYY-MM-DD`` or
    ``YYYY-MM-DDTHH:MM:SSZ``) it is returned unchanged.  Any input that
    cannot be parsed is returned as-is so downstream validators can reject
    it explicitly rather than silently swallowing the error.

    Args:
        raw (str): Raw date string from Kaggle snapshot.

    Returns:
        str: Normalized ISO 8601 UTC date string, or original string on
             parse failure.
    """
    if not raw or not isinstance(raw, str):
        return raw
    cleaned = raw.strip()
    # Already ISO: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ
    if re.match(r"^\d{4}-\d{2}-\d{2}", cleaned):
        return cleaned
    # RFC 2822 from Kaggle snapshot
    if _RFC2822_PATTERN.match(cleaned):
        try:
            # parsedate_to_datetime handles timezone-aware parsing
            dt = parsedate_to_datetime(cleaned)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:  # noqa: BLE001
            pass
    return cleaned


def stream_json_lines(file_obj: TextIO) -> Iterator[Tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
    """Yield parsed JSON records line-by-line from a text stream.

    Args:
        file_obj (TextIO): Open text file object.

    Yields:
        Iterator[Tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
            Tuple of (line_number, parsed_record_or_None, error_str_or_None).
    """
    for line_no, raw_line in enumerate(file_obj, 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            if isinstance(record, dict):
                yield line_no, record, None
            else:
                yield line_no, None, f"Line {line_no} is not a JSON object"
        except json.JSONDecodeError as err:
            yield line_no, None, f"JSONDecodeError on line {line_no}: {err}"


def open_snapshot_file(file_path: Path) -> TextIO:
    """Open a plain text or gzip-compressed snapshot file for streaming.

    Args:
        file_path (Path): Path to snapshot file.

    Returns:
        TextIO: Open file handle in read text mode.
    """
    if file_path.suffix == ".gz":
        return gzip.open(file_path, "rt", encoding="utf-8")  # type: ignore
    return open(file_path, "r", encoding="utf-8")


def filter_kaggle_snapshot(
    input_path: str = DEFAULT_KAGGLE_SNAPSHOT_PATH,
    output_path: str = DEFAULT_TARGET_OUTPUT_PATH,
    target_count: int = 100,
    target_categories: Optional[Set[str]] = None,
    min_year: Optional[int] = DEFAULT_MIN_YEAR,
    max_year: Optional[int] = DEFAULT_MAX_YEAR,
) -> Dict[str, Any]:
    """Stream and filter the Kaggle Cornell arXiv metadata snapshot into a curated subset.

    Args:
        input_path (str): Path to local arxiv-metadata-oai-snapshot.json.
        output_path (str): Target output file path (JSONL).
        target_count (int): Maximum number of validated papers to extract.
        target_categories (Optional[Set[str]]): Target category codes (defaults to CS/AI/ML set).
        min_year (Optional[int]): Minimum publication year (inclusive). Defaults to
            ``DEFAULT_MIN_YEAR`` (2019).
        max_year (Optional[int]): Maximum publication year (inclusive). Defaults to
            ``DEFAULT_MAX_YEAR`` (2020).

    Returns:
        Dict[str, Any]: Extraction summary statistics including total scanned, matched, and saved counts.
    """
    in_file = Path(input_path)
    if not in_file.exists():
        raise FileNotFoundError(
            f"Kaggle arXiv metadata snapshot not found at '{input_path}'. "
            f"Please download 'arxiv-metadata-oai-snapshot.json' from "
            f"https://www.kaggle.com/datasets/Cornell-University/arxiv and place it at '{input_path}'."
        )

    categories = target_categories or SUPPORTED_CATEGORIES
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: Set[str] = set()
    collected_papers: List[Dict[str, Any]] = []

    total_scanned = 0
    malformed_lines = 0
    domain_matches = 0

    logger.info("Starting streaming filtration from '%s'...", in_file)

    with open_snapshot_file(in_file) as f_in:
        for line_no, raw_rec, error in stream_json_lines(f_in):
            total_scanned += 1

            if error or raw_rec is None:
                malformed_lines += 1
                continue

            # Quick pre-filter by categories string before full parse
            cat_str = raw_rec.get("categories", "")
            if isinstance(cat_str, str):
                cat_tokens = set(cat_str.strip().split())
                if not cat_tokens.intersection(categories):
                    continue
            else:
                continue

            # Normalize the versions[].created date from RFC 2822 → ISO 8601
            # before passing to parse_paper so that published_date is consistent.
            raw_versions = raw_rec.get("versions", [])
            if isinstance(raw_versions, list) and raw_versions:
                first_ver = raw_versions[0]
                if isinstance(first_ver, dict) and "created" in first_ver:
                    first_ver["created"] = parse_kaggle_date(first_ver["created"])

            # Parse and validate with ScientificPaper schema
            try:
                paper: ScientificPaper = parse_paper(raw_rec)
            except (ValueError, TypeError):
                continue

            # Verify target categories
            if not any(cat in categories for cat in paper.categories):
                continue

            # Normalize published_date in the paper object itself
            normalized_pub = parse_kaggle_date(paper.published_date)

            # Year-bound filter — extract year from the normalized ISO date
            try:
                paper_year = int(normalized_pub[:4])
            except (ValueError, TypeError):
                # Cannot determine year — skip to keep corpus clean
                continue

            if min_year is not None and paper_year < min_year:
                continue
            if max_year is not None and paper_year > max_year:
                continue

            domain_matches += 1

            # Deduplication by canonical arXiv ID
            if paper.arxiv_id in seen_ids:
                continue

            seen_ids.add(paper.arxiv_id)

            # Convert to dictionary matching canonical storage format.
            # published_date is stored as normalized ISO 8601 UTC string.
            record_dict = {
                "id": paper.arxiv_id,
                "submitter": raw_rec.get("submitter", ""),
                "authors": paper.authors,
                "title": paper.title,
                "comments": raw_rec.get("comments", ""),
                "journal-ref": paper.journal_ref or "",
                "doi": paper.doi or "",
                "report-no": raw_rec.get("report-no"),
                "categories": " ".join(paper.categories),
                "license": raw_rec.get("license", "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"),
                "abstract": paper.abstract,
                "versions": raw_rec.get("versions", [{"version": "v1", "created": normalized_pub}]),
                "published_date": normalized_pub,
                "update_date": raw_rec.get("update_date", normalized_pub[:10]),
            }
            collected_papers.append(record_dict)

            if len(collected_papers) >= target_count:
                logger.info("Reached target count of %d papers. Stopping stream.", target_count)
                break

    # Write output JSONL
    with open(out_file, "w", encoding="utf-8") as f_out:
        for p in collected_papers:
            f_out.write(json.dumps(p) + "\n")

    logger.info(
        "Filtration complete: scanned %d lines, %d domain matches, %d unique records saved to '%s'",
        total_scanned,
        domain_matches,
        len(collected_papers),
        output_path,
    )

    return {
        "total_scanned": total_scanned,
        "malformed_lines": malformed_lines,
        "domain_matches": domain_matches,
        "collected_count": len(collected_papers),
        "output_path": str(out_file),
        "unique_ids": len(seen_ids),
    }


# ---------------------------------------------------------------------------
# Balanced two-pass selection
# ---------------------------------------------------------------------------

def _select_balanced_ids(
    candidates: Dict[str, str],
    year_targets: Dict[int, int],
) -> List[str]:
    """Deterministically select IDs from candidates using year/month bucket quotas.

    Args:
        candidates: Mapping of {arxiv_id: iso_published_date} for all eligible records.
        year_targets: Mapping of {year: target_count} e.g. {2019: 50, 2020: 50}.

    Returns:
        Ordered list of selected arxiv_ids (encounter order within buckets is by
        sorted ID for full determinism).
    """
    from collections import defaultdict
    import math

    # Group IDs into YYYY-MM buckets
    buckets: Dict[str, List[str]] = defaultdict(list)
    for arxiv_id, pub_date in candidates.items():
        if len(pub_date) >= 7:
            ym_key = pub_date[:7]  # "YYYY-MM"
        else:
            ym_key = pub_date[:4] + "-01"
        buckets[ym_key].append(arxiv_id)

    # Sort IDs within each bucket deterministically
    for key in buckets:
        buckets[key].sort()

    selected: List[str] = []

    for year, target in sorted(year_targets.items()):
        year_str = str(year)
        year_buckets = sorted(k for k in buckets if k.startswith(year_str))

        if not year_buckets:
            logger.warning("No candidates found for year %d — skipping.", year)
            continue

        n_buckets = len(year_buckets)
        remaining = target

        # Distribute quota across months using ceiling division so we don't
        # leave papers on the table when bucket counts are uneven.
        # Each bucket gets at most ceil(remaining / buckets_left) papers.
        for i, bucket_key in enumerate(year_buckets):
            buckets_left = n_buckets - i
            quota = math.ceil(remaining / buckets_left)
            chosen = buckets[bucket_key][:quota]
            selected.extend(chosen)
            remaining -= len(chosen)
            if remaining <= 0:
                break

    return selected


def filter_kaggle_snapshot_balanced(
    input_path: str = DEFAULT_KAGGLE_SNAPSHOT_PATH,
    output_path: str = DEFAULT_TARGET_OUTPUT_PATH,
    target_count: int = 100,
    target_categories: Optional[Set[str]] = None,
    min_year: Optional[int] = DEFAULT_MIN_YEAR,
    max_year: Optional[int] = DEFAULT_MAX_YEAR,
    year_targets: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """Two-pass balanced corpus builder over the Kaggle Cornell arXiv snapshot.

    Produces a deterministic, year/month-distributed subset of the snapshot.

    Unlike :func:`filter_kaggle_snapshot` (which stops at the first N domain
    matches and therefore clusters at the head of the file), this function:

    1. **Pass 1** — streams the entire file to collect lightweight candidate
       metadata ``{arxiv_id: iso_published_date}`` for every record that passes
       domain, schema, and date-range filters.
    2. **Selection** — deterministically distributes the target count across
       ``year_targets`` using a year/month bucket strategy with no random
       sampling.
    3. **Pass 2** — streams the file a second time, writing only the selected
       records in encounter order (preserving determinism across runs).

    Args:
        input_path (str): Path to local arxiv-metadata-oai-snapshot.json.
        output_path (str): Target JSONL output file path.
        target_count (int): Total number of papers to collect.
        target_categories (Optional[Set[str]]): Domain category codes.
        min_year (Optional[int]): Minimum publication year (inclusive).
        max_year (Optional[int]): Maximum publication year (inclusive).
        year_targets (Optional[Dict[int, int]]): Per-year quotas, e.g.
            ``{2019: 50, 2020: 50}``.  If ``None``, quota is split evenly
            across years in ``[min_year, max_year]``.

    Returns:
        Dict[str, Any]: Summary statistics.
    """
    in_file = Path(input_path)
    if not in_file.exists():
        raise FileNotFoundError(
            f"Kaggle arXiv metadata snapshot not found at '{input_path}'. "
            f"Please download 'arxiv-metadata-oai-snapshot.json' from "
            f"https://www.kaggle.com/datasets/Cornell-University/arxiv and place it at '{input_path}'."
        )

    categories = target_categories or SUPPORTED_CATEGORIES
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Build default year_targets if not provided
    if year_targets is None:
        if min_year is not None and max_year is not None:
            years = list(range(min_year, max_year + 1))
        elif min_year is not None:
            years = [min_year]
        else:
            years = [max_year] if max_year else []
        if years:
            base_quota = target_count // len(years)
            remainder = target_count % len(years)
            year_targets = {
                yr: base_quota + (1 if i < remainder else 0)
                for i, yr in enumerate(years)
            }
        else:
            year_targets = {}

    logger.info(
        "Balanced selection targets: %s",
        ", ".join(f"{yr}:{q}" for yr, q in sorted(year_targets.items())),
    )

    # ── Pass 1: collect lightweight candidates ───────────────────────────────
    logger.info("[Pass 1] Collecting candidates from '%s'...", in_file)
    candidates: Dict[str, str] = {}  # {arxiv_id: iso_published_date}
    total_scanned_p1 = 0
    malformed_lines = 0

    with open_snapshot_file(in_file) as f_in:
        for _line_no, raw_rec, error in stream_json_lines(f_in):
            total_scanned_p1 += 1

            if error or raw_rec is None:
                malformed_lines += 1
                continue

            # Quick category pre-filter
            cat_str = raw_rec.get("categories", "")
            if not isinstance(cat_str, str):
                continue
            cat_tokens = set(cat_str.strip().split())
            if not cat_tokens.intersection(categories):
                continue

            # Normalize versions[].created
            raw_versions = raw_rec.get("versions", [])
            if isinstance(raw_versions, list) and raw_versions:
                first_ver = raw_versions[0]
                if isinstance(first_ver, dict) and "created" in first_ver:
                    first_ver["created"] = parse_kaggle_date(first_ver["created"])

            # Parse and validate
            try:
                paper: ScientificPaper = parse_paper(raw_rec)
            except (ValueError, TypeError):
                continue

            if not any(cat in categories for cat in paper.categories):
                continue

            normalized_pub = parse_kaggle_date(paper.published_date)

            try:
                paper_year = int(normalized_pub[:4])
            except (ValueError, TypeError):
                continue

            if min_year is not None and paper_year < min_year:
                continue
            if max_year is not None and paper_year > max_year:
                continue

            # Store only the lightweight tuple; dedup by canonical ID
            if paper.arxiv_id not in candidates:
                candidates[paper.arxiv_id] = normalized_pub

    logger.info(
        "[Pass 1] Done. Scanned %d lines, found %d unique candidates.",
        total_scanned_p1,
        len(candidates),
    )

    # ── Deterministic bucket selection ───────────────────────────────────────
    selected_ids_ordered = _select_balanced_ids(candidates, year_targets)
    selected_id_set: Set[str] = set(selected_ids_ordered)

    logger.info(
        "Selected %d papers across %d years.",
        len(selected_ids_ordered),
        len(year_targets),
    )

    # ── Pass 2: write selected records in encounter order ────────────────────
    logger.info("[Pass 2] Writing selected records to '%s'...", out_file)
    collected_papers: List[Dict[str, Any]] = []
    written_ids: Set[str] = set()
    total_scanned_p2 = 0

    with open_snapshot_file(in_file) as f_in:
        for _line_no, raw_rec, error in stream_json_lines(f_in):
            total_scanned_p2 += 1

            if error or raw_rec is None:
                continue

            cat_str = raw_rec.get("categories", "")
            if not isinstance(cat_str, str):
                continue
            cat_tokens = set(cat_str.strip().split())
            if not cat_tokens.intersection(categories):
                continue

            # Quick ID pre-check before expensive parse
            raw_id = raw_rec.get("id", "")
            if raw_id not in selected_id_set:
                continue

            # Normalize versions[].created
            raw_versions = raw_rec.get("versions", [])
            if isinstance(raw_versions, list) and raw_versions:
                first_ver = raw_versions[0]
                if isinstance(first_ver, dict) and "created" in first_ver:
                    first_ver["created"] = parse_kaggle_date(first_ver["created"])

            try:
                paper = parse_paper(raw_rec)
            except (ValueError, TypeError):
                continue

            if paper.arxiv_id in written_ids:
                continue

            normalized_pub = parse_kaggle_date(paper.published_date)

            record_dict = {
                "id": paper.arxiv_id,
                "submitter": raw_rec.get("submitter", ""),
                "authors": paper.authors,
                "title": paper.title,
                "comments": raw_rec.get("comments", ""),
                "journal-ref": paper.journal_ref or "",
                "doi": paper.doi or "",
                "report-no": raw_rec.get("report-no"),
                "categories": " ".join(paper.categories),
                "license": raw_rec.get(
                    "license",
                    "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
                ),
                "abstract": paper.abstract,
                "versions": raw_rec.get(
                    "versions", [{"version": "v1", "created": normalized_pub}]
                ),
                "published_date": normalized_pub,
                "update_date": raw_rec.get("update_date", normalized_pub[:10]),
            }
            collected_papers.append(record_dict)
            written_ids.add(paper.arxiv_id)

            if len(written_ids) >= len(selected_id_set):
                break

    # Write output JSONL
    with open(out_file, "w", encoding="utf-8") as f_out:
        for p in collected_papers:
            f_out.write(json.dumps(p) + "\n")

    logger.info(
        "[Pass 2] Done. Wrote %d records to '%s'.",
        len(collected_papers),
        output_path,
    )

    return {
        "total_scanned_pass1": total_scanned_p1,
        "total_scanned_pass2": total_scanned_p2,
        "malformed_lines": malformed_lines,
        "candidates_found": len(candidates),
        "selected_count": len(selected_ids_ordered),
        "collected_count": len(collected_papers),
        "output_path": str(out_file),
        "unique_ids": len(written_ids),
        "year_targets": year_targets,
    }


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(
        description=(
            "Stream and filter the Kaggle Cornell arXiv dataset into a "
            "balanced, curated domain subset using deterministic year/month "
            "bucket selection."
        )
    )
    cli_parser.add_argument(
        "--input",
        default=DEFAULT_KAGGLE_SNAPSHOT_PATH,
        help=f"Path to Kaggle arxiv-metadata-oai-snapshot.json (default: {DEFAULT_KAGGLE_SNAPSHOT_PATH})",
    )
    cli_parser.add_argument(
        "--output",
        default=DEFAULT_TARGET_OUTPUT_PATH,
        help=f"Path to save output JSONL (default: {DEFAULT_TARGET_OUTPUT_PATH})",
    )
    cli_parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Target number of unique records to extract (default: 100)",
    )
    cli_parser.add_argument(
        "--min-year",
        type=int,
        default=DEFAULT_MIN_YEAR,
        help=f"Minimum publication year filter, inclusive (default: {DEFAULT_MIN_YEAR})",
    )
    cli_parser.add_argument(
        "--max-year",
        type=int,
        default=DEFAULT_MAX_YEAR,
        help=f"Maximum publication year filter, inclusive (default: {DEFAULT_MAX_YEAR})",
    )

    args = cli_parser.parse_args()

    try:
        summary = filter_kaggle_snapshot_balanced(
            input_path=args.input,
            output_path=args.output,
            target_count=args.count,
            min_year=args.min_year,
            max_year=args.max_year,
        )
        print("\n=== Kaggle Balanced Filtration Summary ===")
        for k, v in summary.items():
            print(f"{k}: {v}")
    except FileNotFoundError as err:
        logger.error("%s", err)
        sys.exit(1)

