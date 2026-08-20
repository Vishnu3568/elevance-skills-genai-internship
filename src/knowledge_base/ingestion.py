from pathlib import Path
from typing import Dict, List

import pandas as pd


REQUIRED_COLUMNS = {"prompt", "response"}

NEW = "NEW"
UPDATED = "UPDATED"
DUPLICATE = "DUPLICATE"
INVALID = "INVALID"


def normalize_text(value: object) -> str:
    """Normalize text for deterministic comparison."""
    if value is None:
        return ""

    return str(value).strip().lower()


def load_knowledge_csv(file_path: str) -> pd.DataFrame:
    """Load and validate a knowledge CSV file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Knowledge source not found: {file_path}")

    df = pd.read_csv(path, encoding="latin1")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return df[["prompt", "response"]].copy()


def classify_record(
    incoming_prompt: object,
    incoming_response: object,
    existing_records: Dict[str, str],
) -> str:
    """Classify one incoming knowledge record."""

    prompt = normalize_text(incoming_prompt)
    response = normalize_text(incoming_response)

    if not prompt or not response:
        return INVALID

    if prompt not in existing_records:
        return NEW

    if existing_records[prompt] == response:
        return DUPLICATE

    return UPDATED


def classify_updates(
    existing_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
) -> List[Dict[str, str]]:
    """Classify all incoming records against the existing knowledge base."""

    existing_records = {}

    for _, row in existing_df.iterrows():
        prompt = normalize_text(row["prompt"])
        response = normalize_text(row["response"])

        if prompt and response:
            existing_records[prompt] = response

    results = []

    for _, row in incoming_df.iterrows():
        classification = classify_record(
            row["prompt"],
            row["response"],
            existing_records,
        )

        results.append(
            {
                "prompt": row["prompt"],
                "response": row["response"],
                "status": classification,
            }
        )

    return results
