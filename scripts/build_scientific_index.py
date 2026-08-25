"""Build script for the Production Scientific FAISS Vector Store.

Reads authentic arXiv records from dataset/arxiv_ai_ml_subset_kaggle.jsonl (Kaggle snapshot),
validates each record, creates LangChain documents with scientific metadata, builds the FAISS
index using instructor embeddings, and saves exclusively to faiss_index_scientific/.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root and src to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    # pyrefly: ignore [missing-import]
    from src.langchain_helper import get_instructor_embeddings  # type: ignore
    # pyrefly: ignore [missing-import]
    from src.scientific_kb import (  # type: ignore
        DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
        ScientificRetriever,
        build_scientific_vector_store,
        ingest_jsonl,
        load_scientific_vector_store,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from langchain_helper import get_instructor_embeddings  # type: ignore
    # pyrefly: ignore [missing-import]
    from scientific_kb import (  # type: ignore
        DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
        ScientificRetriever,
        build_scientific_vector_store,
        ingest_jsonl,
        load_scientific_vector_store,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("build_scientific_index")

DEFAULT_DATASET_PATH = "dataset/arxiv_ai_ml_subset_kaggle.jsonl"


def build_production_index(
    dataset_path: str = DEFAULT_DATASET_PATH,
    store_path: str = DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
) -> int:
    """Ingest authentic arXiv JSONL dataset and build production FAISS vector store.

    Args:
        dataset_path (str): Path to arXiv JSONL file.
        store_path (str): Target output directory for scientific FAISS store.

    Returns:
        int: Number of vectors in the built FAISS index.
    """
    logger.info("Step 1: Reading and validating arXiv dataset from %s", dataset_path)
    ingestion_result = ingest_jsonl(dataset_path)

    logger.info(
        "Ingestion Summary: Accepted=%d, Invalid=%d, Out-of-Domain=%d, Duplicates=%d, Errors=%d",
        ingestion_result.accepted,
        ingestion_result.invalid,
        ingestion_result.out_of_domain,
        ingestion_result.duplicates,
        len(ingestion_result.errors),
    )

    if ingestion_result.accepted == 0:
        raise ValueError(f"No valid papers found in {dataset_path}")

    papers = ingestion_result.papers
    logger.info("Step 2: Loading embedding model (hkunlp/instructor-large)...")
    embeddings = get_instructor_embeddings()

    logger.info("Step 3: Building isolated scientific FAISS index at %s...", store_path)
    vector_store = build_scientific_vector_store(
        papers=papers,
        store_path=store_path,
        embeddings=embeddings,
    )

    # Verification
    docstore_count = len(vector_store.docstore._dict) if hasattr(vector_store, "docstore") else len(papers)
    vector_count = vector_store.index.ntotal if hasattr(vector_store, "index") else docstore_count

    logger.info("Step 4: Vector store verification:")
    logger.info(" - Total valid source papers: %d", len(papers))
    logger.info(" - FAISS vector count: %d", vector_count)
    logger.info(" - Docstore document count: %d", docstore_count)

    if vector_count != len(papers) or docstore_count != len(papers):
        raise ValueError(f"Count mismatch! Papers: {len(papers)}, Vectors: {vector_count}, Docstore: {docstore_count}")

    # Test load and sample semantic retrieval
    logger.info("Step 5: Testing load and sample retrieval from newly built index...")
    loaded_store = load_scientific_vector_store(store_path=store_path, embeddings=embeddings)
    retriever = ScientificRetriever(vector_store=loaded_store, default_k=3)

    sample_queries = [
        "natural language processing and deep learning",
        "neural network optimization and machine learning representations",
        "computer vision and convolutional architectures",
    ]

    for q in sample_queries:
        results = retriever.retrieve(q, k=2)
        logger.info("Query: '%s' -> Found %d matches:", q, len(results))
        for r in results:
            logger.info("   * [%s] %s (Score: %.4f, Cat: %s)", r.arxiv_id, r.title[:60], r.score, r.primary_category)

    logger.info("Scientific FAISS index build and verification SUCCESSFUL at %s!", store_path)
    return vector_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build production scientific FAISS vector store.")
    parser.add_argument(
        "--dataset",
        "--input",
        dest="dataset",
        default=DEFAULT_DATASET_PATH,
        help=f"Path to input JSONL dataset (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument("--store", default=DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH, help="Target FAISS directory.")
    args = parser.parse_args()

    build_production_index(
        dataset_path=args.dataset,
        store_path=args.store,
    )

