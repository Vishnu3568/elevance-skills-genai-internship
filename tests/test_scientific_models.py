import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb import (  # type: ignore
        ScientificPaper,
        StructuredPaperAnalysis,
        validate_paper,
        validate_structured_analysis,
        parse_paper,
        is_target_domain,
        SUPPORTED_CATEGORIES,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb import (  # type: ignore
        ScientificPaper,
        StructuredPaperAnalysis,
        validate_paper,
        validate_structured_analysis,
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


class TestStructuredPaperAnalysisModel(unittest.TestCase):
    """Unit tests for StructuredPaperAnalysis data model, validation, and serialization."""

    def test_valid_full_structured_analysis_creation(self):
        analysis = StructuredPaperAnalysis(
            arxiv_id="2005.11401",
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            research_problem="Pre-trained language models struggle with factual recall and precise knowledge access.",
            motivation="Parametric memory alone is prone to hallucinations and difficult to update dynamically.",
            methodology="Combine pre-trained seq2seq models with a dense neural retriever over Wikipedia documents.",
            model_architecture="Dense Passage Retriever (DPR) encoder + BART generator with marginalization over retrieved documents.",
            key_contributions=[
                "Formulated RAG-Sequence and RAG-Token models for end-to-end knowledge retrieval and generation.",
                "Demonstrated state-of-the-art results on open-domain question answering benchmarks.",
            ],
            datasets_benchmarks=["Natural Questions", "TriviaQA", "WebQuestions", "CuratedTREC", "MS-MARCO"],
            key_findings=[
                "RAG generates more specific, diverse, and factual language than parametric-only models.",
                "Non-parametric index can be swapped without retraining the generator.",
            ],
            quantitative_results=[
                "Achieved 44.5 Exact Match on Natural Questions, outperforming standard BART and T5 baselines.",
            ],
            limitations=[
                "Computational overhead of retrieving and marginalizing over multiple documents during inference.",
            ],
            open_questions=[
                "Extending RAG architectures to multi-modal document retrieval and generation.",
            ],
            technical_concepts=[
                "Retrieval-Augmented Generation",
                "Dense Passage Retrieval",
                "Parametric vs Non-Parametric Memory",
            ],
        )
        validate_structured_analysis(analysis)
        self.assertEqual(analysis.arxiv_id, "2005.11401")
        self.assertEqual(analysis.title, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks")
        self.assertEqual(len(analysis.key_contributions), 2)
        self.assertEqual(len(analysis.datasets_benchmarks), 5)
        self.assertEqual(len(analysis.key_findings), 2)
        self.assertEqual(len(analysis.quantitative_results), 1)
        self.assertEqual(len(analysis.limitations), 1)
        self.assertEqual(len(analysis.open_questions), 1)
        self.assertEqual(len(analysis.technical_concepts), 3)

    def test_minimal_structured_analysis_with_defaults(self):
        analysis = StructuredPaperAnalysis(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
        )
        validate_structured_analysis(analysis)
        self.assertEqual(analysis.arxiv_id, "1706.03762")
        self.assertEqual(analysis.title, "Attention Is All You Need")
        self.assertEqual(analysis.research_problem, "")
        self.assertEqual(analysis.motivation, "")
        self.assertEqual(analysis.methodology, "")
        self.assertEqual(analysis.model_architecture, "")
        self.assertEqual(analysis.key_contributions, [])
        self.assertEqual(analysis.datasets_benchmarks, [])
        self.assertEqual(analysis.key_findings, [])
        self.assertEqual(analysis.quantitative_results, [])
        self.assertEqual(analysis.limitations, [])
        self.assertEqual(analysis.open_questions, [])
        self.assertEqual(analysis.technical_concepts, [])

    def test_empty_arxiv_id_fails_validation(self):
        with self.assertRaises(ValueError):
            validate_structured_analysis(
                StructuredPaperAnalysis(arxiv_id="   ", title="Some Title")
            )
        with self.assertRaises(TypeError):
            validate_structured_analysis(
                StructuredPaperAnalysis(arxiv_id=12345, title="Some Title")  # type: ignore
            )

    def test_empty_title_fails_validation(self):
        with self.assertRaises(ValueError):
            validate_structured_analysis(
                StructuredPaperAnalysis(arxiv_id="2005.11401", title="  \n  ")
            )
        with self.assertRaises(TypeError):
            validate_structured_analysis(
                StructuredPaperAnalysis(arxiv_id="2005.11401", title=None)  # type: ignore
            )

    def test_invalid_string_field_types_fail_validation(self):
        base_kwargs = {"arxiv_id": "2005.11401", "title": "Valid Title"}
        invalid_fields = [
            ("research_problem", 123),
            ("motivation", ["invalid", "list"]),
            ("methodology", {"nested": "dict"}),
            ("model_architecture", True),
        ]
        for field_name, invalid_val in invalid_fields:
            with self.assertRaises(TypeError):
                validate_structured_analysis(
                    StructuredPaperAnalysis(**{**base_kwargs, field_name: invalid_val})
                )

    def test_invalid_collection_types_fail_validation(self):
        base_kwargs = {"arxiv_id": "2005.11401", "title": "Valid Title"}
        collection_field_names = [
            "key_contributions",
            "datasets_benchmarks",
            "key_findings",
            "quantitative_results",
            "limitations",
            "open_questions",
            "technical_concepts",
        ]
        for field_name in collection_field_names:
            # String instead of list
            with self.assertRaises(TypeError):
                validate_structured_analysis(
                    StructuredPaperAnalysis(**{**base_kwargs, field_name: "not-a-list"})
                )
            # Integer instead of list
            with self.assertRaises(TypeError):
                validate_structured_analysis(
                    StructuredPaperAnalysis(**{**base_kwargs, field_name: 999})
                )

    def test_collection_item_types_and_empty_item_validation(self):
        base_kwargs = {"arxiv_id": "2005.11401", "title": "Valid Title"}
        # List with non-string item
        with self.assertRaises(TypeError):
            validate_structured_analysis(
                StructuredPaperAnalysis(**{**base_kwargs, "key_contributions": ["Valid", 123]})
            )
        # List with empty/whitespace item
        with self.assertRaises(ValueError):
            validate_structured_analysis(
                StructuredPaperAnalysis(**{**base_kwargs, "technical_concepts": ["Concept 1", "  "]})
            )

    def test_validate_structured_analysis_type_check(self):
        with self.assertRaises(TypeError):
            validate_structured_analysis("not a StructuredPaperAnalysis instance")  # type: ignore
        with self.assertRaises(TypeError):
            validate_structured_analysis({"arxiv_id": "2005.11401", "title": "Dict"})  # type: ignore

    def test_serialization_to_dict_and_from_dict_roundtrip(self):
        original = StructuredPaperAnalysis(
            arxiv_id="2005.11401",
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            research_problem="Improving factual knowledge generation in neural LLMs.",
            motivation="Reducing hallucinations.",
            methodology="RAG neural seq2seq architecture.",
            model_architecture="DPR + BART.",
            key_contributions=["End-to-end RAG training."],
            datasets_benchmarks=["Natural Questions"],
            key_findings=["Higher factual precision."],
            quantitative_results=["44.5 EM on NQ."],
            limitations=["Inference latency."],
            open_questions=["Multi-modal extensions."],
            technical_concepts=["RAG", "DPR"],
        )
        as_dict = original.to_dict()
        self.assertIsInstance(as_dict, dict)
        self.assertEqual(as_dict["arxiv_id"], "2005.11401")
        self.assertEqual(as_dict["key_contributions"], ["End-to-end RAG training."])

        reconstructed = StructuredPaperAnalysis.from_dict(as_dict)
        self.assertEqual(reconstructed.arxiv_id, original.arxiv_id)
        self.assertEqual(reconstructed.title, original.title)
        self.assertEqual(reconstructed.research_problem, original.research_problem)
        self.assertEqual(reconstructed.key_contributions, original.key_contributions)
        self.assertEqual(reconstructed.datasets_benchmarks, original.datasets_benchmarks)
        self.assertEqual(reconstructed.technical_concepts, original.technical_concepts)

    def test_from_dict_with_invalid_payload_raises_errors(self):
        with self.assertRaises(TypeError):
            StructuredPaperAnalysis.from_dict("not-a-dict")  # type: ignore
        with self.assertRaises(ValueError):
            StructuredPaperAnalysis.from_dict({"arxiv_id": "", "title": "Some Title"})


if __name__ == "__main__":
    unittest.main()

