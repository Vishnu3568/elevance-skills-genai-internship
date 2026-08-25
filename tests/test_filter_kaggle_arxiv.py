"""Unit tests for the Kaggle Cornell arXiv streaming filter pipeline.
"""

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add project root and src to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    # pyrefly: ignore [missing-import]
    from scripts.filter_kaggle_arxiv import (  # type: ignore
        DEFAULT_KAGGLE_SNAPSHOT_PATH,
        DEFAULT_MAX_YEAR,
        DEFAULT_MIN_YEAR,
        filter_kaggle_snapshot,
        filter_kaggle_snapshot_balanced,
        _select_balanced_ids,
        parse_kaggle_date,
        stream_json_lines,
    )
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.models import ScientificPaper  # type: ignore
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.parser import SUPPORTED_CATEGORIES, parse_paper  # type: ignore
except ImportError:
    # pyrefly: ignore [missing-import]
    from scripts.filter_kaggle_arxiv import (  # type: ignore
        DEFAULT_KAGGLE_SNAPSHOT_PATH,
        DEFAULT_MAX_YEAR,
        DEFAULT_MIN_YEAR,
        filter_kaggle_snapshot,
        filter_kaggle_snapshot_balanced,
        _select_balanced_ids,
        parse_kaggle_date,
        stream_json_lines,
    )
    # pyrefly: ignore [missing-import]
    from scientific_kb.models import ScientificPaper  # type: ignore
    # pyrefly: ignore [missing-import]
    from scientific_kb.parser import SUPPORTED_CATEGORIES, parse_paper  # type: ignore


class TestParseKaggleDate(unittest.TestCase):
    """Focused unit tests for the RFC 2822 → ISO 8601 date normalization function."""

    def test_rfc2822_canonical_format(self):
        """Standard Kaggle RFC 2822 string converts to ISO 8601 UTC."""
        result = parse_kaggle_date("Fri, 18 Dec 2020 05:31:07 GMT")
        self.assertEqual(result, "2020-12-18T05:31:07Z")

    def test_rfc2822_single_digit_day(self):
        """Single-digit day in RFC 2822 string is handled correctly."""
        result = parse_kaggle_date("Sun, 1 Apr 2007 13:06:50 GMT")
        self.assertEqual(result, "2007-04-01T13:06:50Z")

    def test_rfc2822_all_weekday_abbreviations(self):
        """All weekday abbreviations present in arXiv data are handled."""
        cases = [
            ("Mon, 14 Sep 2020 08:00:00 GMT", "2020-09-14T08:00:00Z"),
            ("Tue, 15 Sep 2020 09:00:00 GMT", "2020-09-15T09:00:00Z"),
            ("Wed, 16 Sep 2020 10:00:00 GMT", "2020-09-16T10:00:00Z"),
            ("Thu, 17 Sep 2020 11:00:00 GMT", "2020-09-17T11:00:00Z"),
            ("Sat, 19 Sep 2020 13:00:00 GMT", "2020-09-19T13:00:00Z"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(parse_kaggle_date(raw), expected)

    def test_already_iso_date_passthrough(self):
        """Already-ISO YYYY-MM-DD strings are returned unchanged."""
        iso = "2020-12-18"
        self.assertEqual(parse_kaggle_date(iso), iso)

    def test_already_iso_datetime_passthrough(self):
        """Already-ISO datetime strings (from API acquisition) are returned unchanged."""
        iso = "2020-12-18T05:31:07Z"
        self.assertEqual(parse_kaggle_date(iso), iso)

    def test_year_extraction_from_rfc2822(self):
        """Year can be correctly extracted from normalized output for filtering."""
        result = parse_kaggle_date("Thu, 5 Mar 2020 12:00:00 GMT")
        self.assertTrue(result.startswith("2020"), f"Expected year 2020, got: {result}")

    def test_empty_string_returned_as_is(self):
        """Empty string input is returned without error."""
        self.assertEqual(parse_kaggle_date(""), "")

    def test_unparseable_string_returned_as_is(self):
        """Completely unparseable input is returned as-is rather than raising."""
        garbage = "not-a-date-at-all"
        self.assertEqual(parse_kaggle_date(garbage), garbage)

    def test_default_year_constants(self):
        """Default year window constants are 2019–2020 matching project intent."""
        self.assertEqual(DEFAULT_MIN_YEAR, 2019)
        self.assertEqual(DEFAULT_MAX_YEAR, 2020)



class TestFilterKaggleArxiv(unittest.TestCase):
    """Test suite for memory-safe Kaggle arXiv snapshot streaming and filtering."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_stream_json_lines_valid_and_malformed(self):
        """Verify that stream_json_lines parses valid lines and isolates malformed ones."""
        sample_stream = io.StringIO(
            '{"id": "0704.0001", "title": "Paper 1"}\n'
            '\n'  # empty line
            '{invalid json line here}\n'
            '{"id": "0704.0002", "title": "Paper 2"}\n'
            '"not a dict"\n'
        )

        results = list(stream_json_lines(sample_stream))
        self.assertEqual(len(results), 4)

        # Line 1: Valid
        self.assertEqual(results[0][0], 1)
        self.assertEqual(results[0][1]["id"], "0704.0001")
        self.assertIsNone(results[0][2])

        # Line 3: Malformed JSON
        self.assertEqual(results[1][0], 3)
        self.assertIsNone(results[1][1])
        self.assertIn("JSONDecodeError", results[1][2])

        # Line 4: Valid
        self.assertEqual(results[2][0], 4)
        self.assertEqual(results[2][1]["id"], "0704.0002")
        self.assertIsNone(results[2][2])

        # Line 5: Non-dict JSON
        self.assertEqual(results[3][0], 5)
        self.assertIsNone(results[3][1])
        self.assertIn("not a JSON object", results[3][2])

    def test_category_filtering_and_parsing(self):
        """Verify that only papers in target domains (cs.CL, cs.AI, cs.LG, cs.CV, stat.ML) are collected."""
        records = [
            # In-scope: cs.CL
            {
                "id": "1901.00001",
                "submitter": "Alice Smith",
                "authors": "Alice Smith, Bob Jones",
                "title": "Natural Language Representation with Transformers",
                "comments": "",
                "journal-ref": "ACL 2019",
                "doi": "10.18653/v1/N19-1423",
                "report-no": None,
                "categories": "cs.CL cs.AI",
                "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
                "abstract": "We present deep contextualized representations using self-attention mechanisms.",
                "versions": [{"version": "v1", "created": "Wed, 2 Jan 2019 12:00:00 GMT"}],
                "update_date": "2019-01-03",
            },
            # Out-of-scope: math.PR (Probability)
            {
                "id": "1901.00002",
                "submitter": "Charlie Brown",
                "authors": "Charlie Brown",
                "title": "On the Convergence of Random Walks",
                "comments": "",
                "journal-ref": "",
                "doi": "",
                "report-no": None,
                "categories": "math.PR math.ST",
                "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
                "abstract": "We prove asymptotic bounds for Markov chains in high dimensions.",
                "versions": [{"version": "v1", "created": "Thu, 3 Jan 2019 10:00:00 GMT"}],
                "update_date": "2019-01-03",
            },
            # In-scope: cs.LG
            {
                "id": "1901.00003",
                "submitter": "David Lee",
                "authors": "David Lee, Eve Wang",
                "title": "Scalable Reinforcement Learning on Graphs",
                "comments": "",
                "journal-ref": "NeurIPS 2019",
                "doi": "",
                "report-no": None,
                "categories": "cs.LG stat.ML",
                "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
                "abstract": "We propose an actor-critic algorithm optimized for sparse graph structured environments.",
                "versions": [{"version": "v1", "created": "Fri, 4 Jan 2019 14:00:00 GMT"}],
                "update_date": "2019-01-05",
            },
        ]

        dummy_snapshot = self.temp_path / "dummy_snapshot.json"
        with open(dummy_snapshot, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        out_jsonl = self.temp_path / "filtered_output.jsonl"
        summary = filter_kaggle_snapshot(
            input_path=str(dummy_snapshot),
            output_path=str(out_jsonl),
            target_count=10,
        )

        self.assertEqual(summary["total_scanned"], 3)
        self.assertEqual(summary["domain_matches"], 2)
        self.assertEqual(summary["collected_count"], 2)
        self.assertEqual(summary["unique_ids"], 2)

        # Read output and verify
        with open(out_jsonl, "r", encoding="utf-8") as f:
            saved = [json.loads(line) for line in f]

        self.assertEqual(len(saved), 2)
        saved_ids = {s["id"] for s in saved}
        self.assertIn("1901.00001", saved_ids)
        self.assertIn("1901.00003", saved_ids)
        self.assertNotIn("1901.00002", saved_ids)

    def test_deduplication_by_arxiv_id(self):
        """Verify that duplicate entries for the same arXiv ID are deduplicated."""
        dup_records = [
            {
                "id": "2005.14165",
                "submitter": "Tom Brown",
                "authors": "Tom Brown et al.",
                "title": "Language Models are Few-Shot Learners",
                "categories": "cs.CL cs.AI",
                "abstract": "Recent work demonstrates substantial gains in NLP through pre-training.",
                "versions": [{"version": "v1", "created": "Thu, 28 May 2020 18:00:00 GMT"}],
            },
            # Duplicate with version update
            {
                "id": "2005.14165",
                "submitter": "Tom Brown",
                "authors": "Tom Brown et al.",
                "title": "Language Models are Few-Shot Learners (v2)",
                "categories": "cs.CL cs.AI",
                "abstract": "Updated abstract for few-shot learners.",
                "versions": [
                    {"version": "v1", "created": "Thu, 28 May 2020 18:00:00 GMT"},
                    {"version": "v2", "created": "Wed, 22 Jul 2020 18:00:00 GMT"},
                ],
            },
        ]

        dummy_snapshot = self.temp_path / "dup_snapshot.json"
        with open(dummy_snapshot, "w", encoding="utf-8") as f:
            for r in dup_records:
                f.write(json.dumps(r) + "\n")

        out_jsonl = self.temp_path / "dedup_output.jsonl"
        summary = filter_kaggle_snapshot(
            input_path=str(dummy_snapshot),
            output_path=str(out_jsonl),
            target_count=10,
        )

        self.assertEqual(summary["total_scanned"], 2)
        self.assertEqual(summary["collected_count"], 1)
        self.assertEqual(summary["unique_ids"], 1)

    def test_missing_required_fields_handling(self):
        """Verify that records missing crucial metadata (title, abstract, or categories) are dropped."""
        invalid_records = [
            {"id": "2001.0001", "authors": "Author", "categories": "cs.AI"},  # missing title & abstract
            {"id": "2001.0002", "title": "Some Title", "categories": "cs.AI"},  # missing abstract
            {"id": "2001.0003", "title": "Some Title", "abstract": "Abstract", "categories": ""},  # empty categories
        ]

        dummy_snapshot = self.temp_path / "invalid_snapshot.json"
        with open(dummy_snapshot, "w", encoding="utf-8") as f:
            for r in invalid_records:
                f.write(json.dumps(r) + "\n")

        out_jsonl = self.temp_path / "invalid_output.jsonl"
        summary = filter_kaggle_snapshot(
            input_path=str(dummy_snapshot),
            output_path=str(out_jsonl),
            target_count=10,
        )
        self.assertEqual(summary["collected_count"], 0)

    def test_missing_file_raises_file_not_found(self):
        """Verify that non-existent input snapshot path raises FileNotFoundError with helpful guidance."""
        with self.assertRaises(FileNotFoundError) as ctx:
            filter_kaggle_snapshot(
                input_path="nonexistent_snapshot_path_9999.json",
                output_path=str(self.temp_path / "out.jsonl"),
            )
        self.assertIn("Kaggle arXiv metadata snapshot not found", str(ctx.exception))
        self.assertIn("https://www.kaggle.com/datasets/Cornell-University/arxiv", str(ctx.exception))

    def test_deterministic_streaming_output(self):
        """Verify that streaming output is deterministic across multiple independent runs on same input."""
        sample_records = [
            {
                "id": f"1905.{1000 + i}",
                "authors": f"Author {i}",
                "title": f"Paper Title {i}",
                "categories": "cs.LG",
                "abstract": f"Abstract content for paper {i}.",
                "versions": [{"version": "v1", "created": f"Wed, {10 + i} Jan 2019 12:00:00 GMT"}],
            }
            for i in range(20)
        ]

        dummy_snapshot = self.temp_path / "deterministic_snapshot.json"
        with open(dummy_snapshot, "w", encoding="utf-8") as f:
            for r in sample_records:
                f.write(json.dumps(r) + "\n")

        out1 = self.temp_path / "run1.jsonl"
        out2 = self.temp_path / "run2.jsonl"

        filter_kaggle_snapshot(
            input_path=str(dummy_snapshot), output_path=str(out1), target_count=10,
            min_year=2019, max_year=2020,
        )
        filter_kaggle_snapshot(
            input_path=str(dummy_snapshot), output_path=str(out2), target_count=10,
            min_year=2019, max_year=2020,
        )

        with open(out1, "r", encoding="utf-8") as f1, open(out2, "r", encoding="utf-8") as f2:
            self.assertEqual(f1.read(), f2.read())

    def test_year_bound_filtering_excludes_out_of_range(self):
        """Verify that min_year/max_year bounds correctly exclude records outside the target era."""
        records = [
            # 2007 record — should be excluded
            {
                "id": "0704.0001",
                "authors": "Old Author",
                "title": "Old Paper from 2007",
                "categories": "cs.AI",
                "abstract": "An old paper that predates the target era.",
                "versions": [{"version": "v1", "created": "Mon, 2 Apr 2007 10:00:00 GMT"}],
            },
            # 2019 record — should be included
            {
                "id": "1901.00010",
                "authors": "New Author A",
                "title": "Recent Advances in Neural Architectures 2019",
                "categories": "cs.LG stat.ML",
                "abstract": "This paper presents advances in neural architecture search for 2019.",
                "versions": [{"version": "v1", "created": "Tue, 1 Jan 2019 12:00:00 GMT"}],
            },
            # 2020 record — should be included
            {
                "id": "2005.14165",
                "authors": "New Author B",
                "title": "Language Models are Few-Shot Learners",
                "categories": "cs.CL cs.AI",
                "abstract": "Recent work demonstrates substantial gains through pre-training on large datasets.",
                "versions": [{"version": "v1", "created": "Thu, 28 May 2020 18:00:00 GMT"}],
            },
            # 2021 record — should be excluded (beyond max_year=2020)
            {
                "id": "2101.00001",
                "authors": "Future Author",
                "title": "Paper from 2021",
                "categories": "cs.AI",
                "abstract": "This paper is from 2021 and should be excluded by max_year filter.",
                "versions": [{"version": "v1", "created": "Fri, 1 Jan 2021 09:00:00 GMT"}],
            },
        ]

        dummy_snapshot = self.temp_path / "year_bound_snapshot.json"
        with open(dummy_snapshot, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        out_jsonl = self.temp_path / "year_bound_output.jsonl"
        summary = filter_kaggle_snapshot(
            input_path=str(dummy_snapshot),
            output_path=str(out_jsonl),
            target_count=10,
            min_year=2019,
            max_year=2020,
        )

        self.assertEqual(summary["collected_count"], 2)

        with open(out_jsonl, "r", encoding="utf-8") as f:
            saved = [json.loads(line) for line in f]

        saved_ids = {s["id"] for s in saved}
        self.assertIn("1901.00010", saved_ids)
        self.assertIn("2005.14165", saved_ids)
        self.assertNotIn("0704.0001", saved_ids)
        self.assertNotIn("2101.00001", saved_ids)

        # Verify published_date is ISO 8601 format
        for rec in saved:
            pub = rec.get("published_date", "")
            self.assertRegex(
                pub,
                r"^\d{4}-\d{2}-\d{2}",
                msg=f"published_date must be ISO 8601, got: {pub}",
            )

    def test_published_date_is_iso_format(self):
        """Verify that all output records have ISO 8601 published_date regardless of input format."""
        records = [
            {
                "id": "1905.12345",
                "authors": "Test Author",
                "title": "Test Paper with RFC 2822 Date",
                "categories": "cs.LG",
                "abstract": "Abstract for testing ISO date normalization output.",
                "versions": [{"version": "v1", "created": "Mon, 6 May 2019 14:30:00 GMT"}],
            },
        ]

        dummy_snapshot = self.temp_path / "iso_date_snapshot.json"
        with open(dummy_snapshot, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        out_jsonl = self.temp_path / "iso_date_output.jsonl"
        filter_kaggle_snapshot(
            input_path=str(dummy_snapshot),
            output_path=str(out_jsonl),
            target_count=10,
            min_year=2019,
            max_year=2020,
        )

        with open(out_jsonl, "r", encoding="utf-8") as f:
            saved = [json.loads(line) for line in f]

        self.assertEqual(len(saved), 1)
        pub = saved[0]["published_date"]
        self.assertEqual(pub, "2019-05-06T14:30:00Z")



class TestBalancedSelection(unittest.TestCase):
    """Tests for the deterministic two-pass balanced corpus selection."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_snapshot(self, records):
        """Write records to a temp JSONL file and return its path."""
        snap = self.temp_path / "snapshot.json"
        with open(snap, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return str(snap)

    def _make_record(self, arxiv_id, created_rfc, categories="cs.LG stat.ML",
                     title=None, abstract=None):
        return {
            "id": arxiv_id,
            "authors": "Test Author",
            "title": title or f"Paper {arxiv_id}",
            "abstract": abstract or f"Abstract for paper {arxiv_id} covering machine learning topics.",
            "categories": categories,
            "versions": [{"version": "v1", "created": created_rfc}],
        }

    def _build_balanced_dataset(self):
        """Build a synthetic snapshot with 6 papers per month across 2019 and 2020."""
        records = []
        months = [
            ("Jan", "01"), ("Feb", "02"), ("Mar", "03"), ("Apr", "04"),
            ("May", "05"), ("Jun", "06"), ("Jul", "07"), ("Aug", "08"),
            ("Sep", "09"), ("Oct", "10"), ("Nov", "11"), ("Dec", "12"),
        ]
        weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        for year in [2019, 2020]:
            for month_name, month_num in months:
                for i in range(6):
                    day = 1 + i
                    wday = weekdays[i % len(weekdays)]
                    arxiv_id = f"{str(year)[2:]}{month_num}.{1000 + i:04d}"
                    created = f"{wday}, {day} {month_name} {year} 12:00:00 GMT"
                    records.append(self._make_record(arxiv_id, created))
        return records

    def test_balanced_deterministic_two_runs_identical(self):
        """Two independent balanced runs on same input produce byte-identical output."""
        records = self._build_balanced_dataset()
        snap = self._make_snapshot(records)
        out1 = str(self.temp_path / "run1.jsonl")
        out2 = str(self.temp_path / "run2.jsonl")

        filter_kaggle_snapshot_balanced(
            input_path=snap, output_path=out1, target_count=20,
            min_year=2019, max_year=2020,
        )
        filter_kaggle_snapshot_balanced(
            input_path=snap, output_path=out2, target_count=20,
            min_year=2019, max_year=2020,
        )

        with open(out1) as f1, open(out2) as f2:
            self.assertEqual(f1.read(), f2.read())

    def test_balanced_exactly_100_records(self):
        """Balanced selection returns exactly target_count unique records."""
        records = self._build_balanced_dataset()  # 2 years * 12 months * 6 = 144 records
        snap = self._make_snapshot(records)
        out = str(self.temp_path / "out.jsonl")

        summary = filter_kaggle_snapshot_balanced(
            input_path=snap, output_path=out, target_count=20,
            min_year=2019, max_year=2020,
        )
        self.assertEqual(summary["collected_count"], 20)
        self.assertEqual(summary["unique_ids"], 20)

        with open(out) as f:
            rows = [json.loads(l) for l in f]
        self.assertEqual(len(rows), 20)
        ids = [r["id"] for r in rows]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate IDs in output")

    def test_balanced_both_years_represented(self):
        """Both 2019 and 2020 are present in the balanced output."""
        records = self._build_balanced_dataset()
        snap = self._make_snapshot(records)
        out = str(self.temp_path / "out.jsonl")

        filter_kaggle_snapshot_balanced(
            input_path=snap, output_path=out, target_count=20,
            min_year=2019, max_year=2020,
        )

        with open(out) as f:
            rows = [json.loads(l) for l in f]

        years = {r["published_date"][:4] for r in rows}
        self.assertIn("2019", years, "2019 missing from balanced output")
        self.assertIn("2020", years, "2020 missing from balanced output")

    def test_balanced_year_distribution_approximately_equal(self):
        """Year counts should be within 1 of each other for even split."""
        records = self._build_balanced_dataset()
        snap = self._make_snapshot(records)
        out = str(self.temp_path / "out.jsonl")

        filter_kaggle_snapshot_balanced(
            input_path=snap, output_path=out, target_count=20,
            min_year=2019, max_year=2020,
        )

        with open(out) as f:
            rows = [json.loads(l) for l in f]

        from collections import Counter
        year_counts = Counter(r["published_date"][:4] for r in rows)
        count_2019 = year_counts.get("2019", 0)
        count_2020 = year_counts.get("2020", 0)
        self.assertGreaterEqual(count_2019, 9, f"2019 count too low: {count_2019}")
        self.assertGreaterEqual(count_2020, 9, f"2020 count too low: {count_2020}")

    def test_balanced_spans_multiple_months(self):
        """Selected records should span at least 4 distinct YYYY-MM buckets."""
        records = self._build_balanced_dataset()
        snap = self._make_snapshot(records)
        out = str(self.temp_path / "out.jsonl")

        filter_kaggle_snapshot_balanced(
            input_path=snap, output_path=out, target_count=20,
            min_year=2019, max_year=2020,
        )

        with open(out) as f:
            rows = [json.loads(l) for l in f]

        months = {r["published_date"][:7] for r in rows}
        self.assertGreaterEqual(len(months), 4, f"Too few months: {sorted(months)}")

    def test_balanced_all_records_pass_schema_validation(self):
        """Every record in balanced output must pass ScientificPaper validation."""
        records = self._build_balanced_dataset()
        snap = self._make_snapshot(records)
        out = str(self.temp_path / "out.jsonl")

        filter_kaggle_snapshot_balanced(
            input_path=snap, output_path=out, target_count=20,
            min_year=2019, max_year=2020,
        )

        with open(out) as f:
            rows = [json.loads(l) for l in f]

        fails = []
        for rec in rows:
            try:
                parse_paper(rec)
            except Exception as e:
                fails.append((rec.get("id"), str(e)))

        self.assertEqual(fails, [], f"Schema failures: {fails}")

    def test_balanced_published_dates_are_iso(self):
        """All output published_date fields must be valid ISO 8601 strings."""
        records = self._build_balanced_dataset()
        snap = self._make_snapshot(records)
        out = str(self.temp_path / "out.jsonl")

        filter_kaggle_snapshot_balanced(
            input_path=snap, output_path=out, target_count=20,
            min_year=2019, max_year=2020,
        )

        import re
        iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}")
        with open(out) as f:
            rows = [json.loads(l) for l in f]

        for rec in rows:
            pub = rec.get("published_date", "")
            self.assertRegex(pub, iso_re, f"Non-ISO date in {rec.get('id')}: {pub}")

    def test_balanced_no_duplicate_ids(self):
        """Balanced output must contain no duplicate arXiv IDs."""
        records = self._build_balanced_dataset()
        snap = self._make_snapshot(records)
        out = str(self.temp_path / "out.jsonl")

        filter_kaggle_snapshot_balanced(
            input_path=snap, output_path=out, target_count=20,
            min_year=2019, max_year=2020,
        )

        with open(out) as f:
            rows = [json.loads(l) for l in f]

        ids = [r["id"] for r in rows]
        self.assertEqual(len(ids), len(set(ids)))

    def test_balanced_domain_filtering_still_works(self):
        """Out-of-domain records (e.g. math.PR) are excluded from balanced output."""
        records = self._build_balanced_dataset()  # all cs.LG stat.ML
        # Add out-of-domain records
        for i in range(5):
            records.append({
                "id": f"1905.{9000 + i}",
                "authors": "Math Author",
                "title": f"Random Walk Theorem {i}",
                "abstract": "A mathematical proof about Markov chains and convergence.",
                "categories": "math.PR math.ST",
                "versions": [{"version": "v1", "created": f"Mon, {1 + i} Jun 2019 12:00:00 GMT"}],
            })

        snap = self._make_snapshot(records)
        out = str(self.temp_path / "out.jsonl")

        filter_kaggle_snapshot_balanced(
            input_path=snap, output_path=out, target_count=10,
            min_year=2019, max_year=2020,
        )

        with open(out) as f:
            rows = [json.loads(l) for l in f]

        all_cats = [r["categories"] for r in rows]
        for cats in all_cats:
            self.assertFalse(
                cats == "math.PR math.ST",
                f"Out-of-domain record found in output: {cats}",
            )

    def test_select_balanced_ids_unit(self):
        """Unit test for _select_balanced_ids with synthetic candidates."""
        candidates = {
            "1901.001": "2019-01-05T12:00:00Z",
            "1901.002": "2019-01-06T12:00:00Z",
            "1906.001": "2019-06-01T12:00:00Z",
            "1906.002": "2019-06-02T12:00:00Z",
            "2001.001": "2020-01-01T12:00:00Z",
            "2001.002": "2020-01-02T12:00:00Z",
            "2006.001": "2020-06-01T12:00:00Z",
            "2006.002": "2020-06-02T12:00:00Z",
        }
        selected = _select_balanced_ids(candidates, {2019: 3, 2020: 3})
        self.assertEqual(len(selected), 6)
        ids_2019 = [s for s in selected if s.startswith("19")]
        ids_2020 = [s for s in selected if s.startswith("20")]
        self.assertEqual(len(ids_2019), 3)
        self.assertEqual(len(ids_2020), 3)
        # Check determinism
        selected2 = _select_balanced_ids(candidates, {2019: 3, 2020: 3})
        self.assertEqual(selected, selected2)


if __name__ == "__main__":
    unittest.main()
