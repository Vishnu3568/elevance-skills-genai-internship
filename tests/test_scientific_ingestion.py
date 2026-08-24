import unittest
import sys
import os
import json
import shutil
import tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb import (  # type: ignore
        ScientificPaper,
        IngestionResult,
        ingest_papers,
        ingest_jsonl,
    )
    # pyrefly: ignore [missing-import]
    from src.knowledge_base.ingestion import (  # type: ignore
        classify_updates as customer_classify_updates,
        NEW,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb import (  # type: ignore
        ScientificPaper,
        IngestionResult,
        ingest_papers,
        ingest_jsonl,
    )
    # pyrefly: ignore [missing-import]
    from knowledge_base.ingestion import (  # type: ignore
        classify_updates as customer_classify_updates,
        NEW,
    )

import pandas as pd


class TestScientificIngestion(unittest.TestCase):
    """Unit tests for the scientific paper streaming ingestion pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base = Path(self.temp_dir)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _sample_valid_record(self, arxiv_id="1706.03762", title="Attention Paper", category="cs.CL"):
        return {
            "id": arxiv_id,
            "title": title,
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "abstract": "The dominant sequence transduction models are based on complex neural networks...",
            "categories": category,
            "published": "2017-06-12",
        }

    def test_batch_with_multiple_valid_target_domain_papers(self):
        records = [
            self._sample_valid_record(arxiv_id="1706.03762", title="Attention Is All You Need", category="cs.CL"),
            self._sample_valid_record(arxiv_id="1810.04805", title="BERT Pre-training", category="cs.CL cs.AI"),
            self._sample_valid_record(arxiv_id="2005.14165", title="GPT-3 Language Models", category="cs.CL stat.ML"),
        ]
        result = ingest_papers(records)

        self.assertEqual(result.accepted, 3)
        self.assertEqual(result.invalid, 0)
        self.assertEqual(result.out_of_domain, 0)
        self.assertEqual(result.duplicates, 0)
        self.assertEqual(len(result.papers), 3)
        self.assertEqual(result.papers[0].arxiv_id, "1706.03762")
        self.assertEqual(result.papers[1].arxiv_id, "1810.04805")
        self.assertEqual(result.papers[2].arxiv_id, "2005.14165")

    def test_out_of_domain_paper_is_skipped(self):
        records = [
            self._sample_valid_record(arxiv_id="1706.03762", title="NLP Paper", category="cs.CL"),
            self._sample_valid_record(arxiv_id="9901.00001", title="Quantum Gravity", category="physics.gen-ph hep-th"),
        ]
        result = ingest_papers(records)

        self.assertEqual(result.accepted, 1)
        self.assertEqual(result.out_of_domain, 1)
        self.assertEqual(len(result.papers), 1)
        self.assertEqual(result.papers[0].arxiv_id, "1706.03762")

    def test_invalid_paper_is_skipped_without_stopping_batch(self):
        records = [
            self._sample_valid_record(arxiv_id="1706.03762", title="Valid 1"),
            {"id": "invalid_1", "title": ""},  # missing abstract, authors, etc.
            "not a dictionary",  # invalid record type
            self._sample_valid_record(arxiv_id="1810.04805", title="Valid 2"),
        ]
        result = ingest_papers(records)

        self.assertEqual(result.accepted, 2)
        self.assertEqual(result.invalid, 2)
        self.assertEqual(len(result.papers), 2)
        self.assertEqual(result.papers[0].arxiv_id, "1706.03762")
        self.assertEqual(result.papers[1].arxiv_id, "1810.04805")

    def test_duplicate_arxiv_id_is_removed_and_preserves_first_seen_ordering(self):
        records = [
            self._sample_valid_record(arxiv_id="A", title="Paper A"),
            self._sample_valid_record(arxiv_id="B", title="Paper B"),
            self._sample_valid_record(arxiv_id="A", title="Paper A Duplicate"),
            self._sample_valid_record(arxiv_id="C", title="Paper C"),
            self._sample_valid_record(arxiv_id="B", title="Paper B Duplicate"),
        ]
        result = ingest_papers(records)

        self.assertEqual(result.accepted, 3)
        self.assertEqual(result.duplicates, 2)
        self.assertEqual([p.arxiv_id for p in result.papers], ["A", "B", "C"])
        self.assertEqual(result.papers[0].title, "Paper A")
        self.assertEqual(result.papers[1].title, "Paper B")

    def test_mixed_batch_and_error_information_retained(self):
        records = [
            self._sample_valid_record(arxiv_id="VALID_1", title="Valid 1", category="cs.AI"),
            {"id": "BAD_1", "title": "Missing Abstract", "authors": ["Author"], "categories": "cs.AI", "published": "2020"},
            self._sample_valid_record(arxiv_id="PHYS_1", title="Physics Paper", category="astro-ph.CO"),
            self._sample_valid_record(arxiv_id="VALID_1", title="Duplicate Paper", category="cs.AI"),
            self._sample_valid_record(arxiv_id="VALID_2", title="Valid 2", category="cs.LG"),
        ]
        result = ingest_papers(records)

        self.assertEqual(result.accepted, 2)
        self.assertEqual(result.invalid, 1)
        self.assertEqual(result.out_of_domain, 1)
        self.assertEqual(result.duplicates, 1)
        self.assertEqual(len(result.errors), 3)

        # Check structured error contents
        invalid_err = next(e for e in result.errors if e["status"] == "INVALID")
        self.assertEqual(invalid_err["arxiv_id"], "BAD_1")

        ood_err = next(e for e in result.errors if e["status"] == "OUT_OF_DOMAIN")
        self.assertEqual(ood_err["arxiv_id"], "PHYS_1")

        dup_err = next(e for e in result.errors if e["status"] == "DUPLICATE")
        self.assertEqual(dup_err["arxiv_id"], "VALID_1")

    def test_generator_streaming_input_without_list_conversion(self):
        def record_stream():
            for i in range(5):
                yield self._sample_valid_record(arxiv_id=f"2023.0000{i}", title=f"Paper {i}")

        result = ingest_papers(record_stream())
        self.assertEqual(result.accepted, 5)
        self.assertEqual(len(result.papers), 5)

    def test_jsonl_ingestion_with_valid_and_malformed_lines(self):
        jsonl_path = self.base / "sample_papers.jsonl"

        lines = [
            json.dumps(self._sample_valid_record(arxiv_id="1706.03762", title="Attention Paper")),
            "{ invalid json line -- syntax error",
            "",  # empty line
            json.dumps(self._sample_valid_record(arxiv_id="1810.04805", title="BERT Paper")),
            json.dumps(self._sample_valid_record(arxiv_id="9901.00001", title="Physics", category="physics.gen-ph")),
        ]

        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        result = ingest_jsonl(str(jsonl_path))

        self.assertEqual(result.accepted, 2)
        self.assertEqual(result.invalid, 1)  # the malformed line
        self.assertEqual(result.out_of_domain, 1)
        self.assertEqual(len(result.papers), 2)
        self.assertEqual([p.arxiv_id for p in result.papers], ["1706.03762", "1810.04805"])

    def test_empty_jsonl_file_produces_empty_result(self):
        empty_path = self.base / "empty.jsonl"
        with open(empty_path, "w", encoding="utf-8") as f:
            f.write("")

        result = ingest_jsonl(str(empty_path))
        self.assertEqual(result.accepted, 0)
        self.assertEqual(result.invalid, 0)
        self.assertEqual(result.out_of_domain, 0)
        self.assertEqual(result.duplicates, 0)
        self.assertEqual(len(result.papers), 0)

    def test_missing_jsonl_file_raises_file_not_found(self):
        missing_path = self.base / "non_existent.jsonl"
        with self.assertRaises(FileNotFoundError):
            ingest_jsonl(str(missing_path))

    def test_customer_knowledge_base_ingestion_remains_unaffected(self):
        """Ensure customer FAQ ingestion behavior and invariants remain completely untouched."""
        existing_df = pd.DataFrame([{"prompt": "hello", "response": "world"}])
        incoming_df = pd.DataFrame([{"prompt": "shipping", "response": "yes worldwide"}])

        classified = customer_classify_updates(existing_df, incoming_df)
        self.assertEqual(len(classified), 1)
        self.assertEqual(classified[0]["status"], NEW)


if __name__ == "__main__":
    unittest.main()
