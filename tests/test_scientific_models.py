import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb import (  # type: ignore
        ScientificPaper,
        validate_paper,
        parse_paper,
        is_target_domain,
        SUPPORTED_CATEGORIES,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb import (  # type: ignore
        ScientificPaper,
        validate_paper,
        parse_paper,
        is_target_domain,
        SUPPORTED_CATEGORIES,
    )


class TestScientificPaperModels(unittest.TestCase):
    """Unit tests for the ScientificPaper data model, parser, and domain validator."""

    def test_valid_scientific_paper_creation(self):
        paper = ScientificPaper(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
            categories=["cs.CL", "cs.LG"],
            primary_category="cs.CL",
            published_date="2017-06-12",
            doi="10.48550/arXiv.1706.03762",
            concepts=["Transformer", "Self-Attention", "Multi-Head Attention"],
        )
        validate_paper(paper)
        self.assertEqual(paper.arxiv_id, "1706.03762")
        self.assertEqual(paper.title, "Attention Is All You Need")
        self.assertEqual(len(paper.authors), 3)
        self.assertEqual(paper.primary_category, "cs.CL")
        self.assertEqual(len(paper.concepts), 3)

    def test_required_field_validation_missing_or_empty_fields(self):
        base_kwargs = {
            "arxiv_id": "2005.11401",
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "authors": ["Patrick Lewis", "Ethan Perez"],
            "abstract": "Large pre-trained language models store factual knowledge in their parameters...",
            "categories": ["cs.CL", "cs.AI"],
            "primary_category": "cs.CL",
            "published_date": "2020-05-22",
        }

        # 1. Empty/missing arxiv_id
        with self.assertRaises(ValueError):
            validate_paper(ScientificPaper(**{**base_kwargs, "arxiv_id": "   "}))

        # 2. Empty title
        with self.assertRaises(ValueError):
            validate_paper(ScientificPaper(**{**base_kwargs, "title": ""}))

        # 3. Empty authors list
        with self.assertRaises(ValueError):
            validate_paper(ScientificPaper(**{**base_kwargs, "authors": []}))

        # 4. Invalid author type
        with self.assertRaises(TypeError):
            validate_paper(ScientificPaper(**{**base_kwargs, "authors": "Patrick Lewis"}))  # type: ignore

        # 5. Empty abstract
        with self.assertRaises(ValueError):
            validate_paper(ScientificPaper(**{**base_kwargs, "abstract": "  \n  "}))

        # 6. Empty categories list
        with self.assertRaises(ValueError):
            validate_paper(ScientificPaper(**{**base_kwargs, "categories": []}))

        # 7. Invalid category type
        with self.assertRaises(TypeError):
            validate_paper(ScientificPaper(**{**base_kwargs, "categories": "cs.CL"}))  # type: ignore

        # 8. Empty primary category
        with self.assertRaises(ValueError):
            validate_paper(ScientificPaper(**{**base_kwargs, "primary_category": " "}))

        # 9. Empty published date
        with self.assertRaises(ValueError):
            validate_paper(ScientificPaper(**{**base_kwargs, "published_date": ""}))

    def test_optional_fields_and_defaults(self):
        paper = ScientificPaper(
            arxiv_id="2106.09685",
            title="LoRA: Low-Rank Adaptation of Large Language Models",
            authors=["Edward J. Hu", "Yelong Shen"],
            abstract="An important paradigm of natural language processing consists of large-scale pre-training...",
            categories=["cs.CL", "cs.AI", "cs.LG"],
            primary_category="cs.CL",
            published_date="2021-06-17",
        )
        validate_paper(paper)
        self.assertIsNone(paper.updated_date)
        self.assertIsNone(paper.doi)
        self.assertIsNone(paper.journal_ref)
        self.assertEqual(paper.concepts, [])
        self.assertIsNone(paper.summary)

    def test_parser_canonical_and_alternate_fields(self):
        # Using canonical arxiv-style fields (id, published, updated, journal-ref)
        raw_canonical = {
            "id": "1706.03762",
            "title": " Attention Is All You Need ",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "abstract": " The dominant sequence transduction models... ",
            "categories": "cs.CL cs.LG",
            "published": "2017-06-12",
            "updated": "2017-12-06",
            "journal-ref": "NeurIPS 2017",
            "doi": "10.48550/arXiv.1706.03762",
        }
        paper1 = parse_paper(raw_canonical)
        self.assertEqual(paper1.arxiv_id, "1706.03762")
        self.assertEqual(paper1.title, "Attention Is All You Need")
        self.assertEqual(paper1.published_date, "2017-06-12")
        self.assertEqual(paper1.updated_date, "2017-12-06")
        self.assertEqual(paper1.journal_ref, "NeurIPS 2017")
        self.assertEqual(paper1.primary_category, "cs.CL")

        # Using alternate fields (arxiv_id, published_date, updated_date, journal_ref)
        raw_alternate = {
            "arxiv_id": "2005.11401",
            "title": "Retrieval-Augmented Generation",
            "authors": "Patrick Lewis, Ethan Perez",
            "abstract": "We explore RAG models...",
            "categories": ["cs.CL", "cs.AI"],
            "published_date": "2020-05-22",
            "updated_date": "2021-04-12",
            "journal_ref": "NeurIPS 2020",
        }
        paper2 = parse_paper(raw_alternate)
        self.assertEqual(paper2.arxiv_id, "2005.11401")
        self.assertEqual(paper2.authors, ["Patrick Lewis", "Ethan Perez"])
        self.assertEqual(paper2.published_date, "2020-05-22")
        self.assertEqual(paper2.journal_ref, "NeurIPS 2020")

    def test_parser_author_formats(self):
        # List of authors
        paper1 = parse_paper({
            "id": "0001",
            "title": "Title 1",
            "authors": [" Author A ", "Author B  "],
            "abstract": "Abstract 1",
            "categories": ["cs.AI"],
            "published": "2023-01-01",
        })
        self.assertEqual(paper1.authors, ["Author A", "Author B"])

        # Comma-separated author string
        paper2 = parse_paper({
            "id": "0002",
            "title": "Title 2",
            "authors": "Author X, Author Y ,  Author Z",
            "abstract": "Abstract 2",
            "categories": "cs.LG",
            "published": "2023-01-01",
        })
        self.assertEqual(paper2.authors, ["Author X", "Author Y", "Author Z"])

    def test_parser_category_formats_and_primary_derivation(self):
        # Space-delimited string with omitted primary_category -> derive first
        paper1 = parse_paper({
            "id": "0001",
            "title": "Title 1",
            "authors": ["Author A"],
            "abstract": "Abstract 1",
            "categories": " cs.CL   cs.AI  cs.LG ",
            "published": "2023-01-01",
        })
        self.assertEqual(paper1.categories, ["cs.CL", "cs.AI", "cs.LG"])
        self.assertEqual(paper1.primary_category, "cs.CL")

        # Explicit primary category preserved
        paper2 = parse_paper({
            "id": "0002",
            "title": "Title 2",
            "authors": ["Author B"],
            "abstract": "Abstract 2",
            "categories": ["cs.AI", "cs.LG"],
            "primary_category": "cs.LG",
            "published": "2023-01-01",
        })
        self.assertEqual(paper2.primary_category, "cs.LG")

    def test_concepts_normalization_and_empty_concept_removal(self):
        raw = {
            "id": "0003",
            "title": "Title 3",
            "authors": ["Author C"],
            "abstract": "Abstract 3",
            "categories": ["stat.ML"],
            "published": "2023-01-01",
            "concepts": [" Transformer ", "", "  ", "Self-Attention", "\n"],
        }
        paper = parse_paper(raw)
        self.assertEqual(paper.concepts, ["Transformer", "Self-Attention"])

    def test_target_domain_filtering(self):
        # Target domains
        paper_cl = parse_paper({
            "id": "0001",
            "title": "NLP Paper",
            "authors": ["Author"],
            "abstract": "Text",
            "categories": "cs.CL",
            "published": "2023",
        })
        self.assertTrue(is_target_domain(paper_cl))

        paper_stat = parse_paper({
            "id": "0002",
            "title": "Stat ML Paper",
            "authors": ["Author"],
            "abstract": "Text",
            "categories": ["stat.ML", "math.ST"],
            "published": "2023",
        })
        self.assertTrue(is_target_domain(paper_stat))

        # Non-target domain (e.g. physics)
        paper_physics = parse_paper({
            "id": "0003",
            "title": "Astrophysics Paper",
            "authors": ["Author"],
            "abstract": "Text",
            "categories": ["astro-ph.CO", "gr-qc"],
            "published": "2023",
        })
        self.assertFalse(is_target_domain(paper_physics))

    def test_parser_does_not_mutate_caller_lists(self):
        orig_authors = [" Author A ", " Author B "]
        orig_categories = [" cs.CL ", " cs.AI "]
        orig_concepts = [" Transformer ", " "]

        raw = {
            "id": "0001",
            "title": "Title",
            "authors": orig_authors,
            "abstract": "Abstract",
            "categories": orig_categories,
            "published": "2023",
            "concepts": orig_concepts,
        }
        paper = parse_paper(raw)

        # Output paper has stripped values
        self.assertEqual(paper.authors, ["Author A", "Author B"])
        self.assertEqual(paper.categories, ["cs.CL", "cs.AI"])
        self.assertEqual(paper.concepts, ["Transformer"])

        # Original caller lists remain unchanged
        self.assertEqual(orig_authors, [" Author A ", " Author B "])
        self.assertEqual(orig_categories, [" cs.CL ", " cs.AI "])
        self.assertEqual(orig_concepts, [" Transformer ", " "])

    def test_invalid_raw_record_raises_exception(self):
        with self.assertRaises(TypeError):
            parse_paper("not a dictionary")  # type: ignore

        with self.assertRaises(TypeError):
            parse_paper({"id": 12345, "title": "Invalid ID Type"})  # type: ignore


if __name__ == "__main__":
    unittest.main()
