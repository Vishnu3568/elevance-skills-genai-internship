from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    from src.knowledge_base.ingestion import (
        INVALID,
        NEW,
        UPDATED,
        normalize_text,
    )
except ImportError:
    from knowledge_base.ingestion import (
        INVALID,
        NEW,
        UPDATED,
        normalize_text,
    )


REQUIRED_COLUMNS = {"prompt", "response"}


def load_knowledge_base(file_path: str) -> pd.DataFrame:
    """Load the managed knowledge base."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {file_path}"
        )

    df = pd.read_csv(path, encoding="utf-8")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return df[["prompt", "response"]].copy()


def apply_updates(
    knowledge_base: pd.DataFrame,
    classified_updates: List[Dict[str, str]],
) -> pd.DataFrame:
    """
    Apply NEW and UPDATED records to the knowledge base.

    DUPLICATE and INVALID records are ignored.
    """

    result = knowledge_base.copy()

    # Map normalized prompts to their row index.
    prompt_index = {}

    for index, row in result.iterrows():
        prompt = normalize_text(row["prompt"])

        if prompt:
            prompt_index[prompt] = index

    for update in classified_updates:
        status = update["status"]

        if status in (INVALID,):
            continue

        prompt = update["prompt"]
        response = update["response"]

        normalized_prompt = normalize_text(prompt)

        if status == NEW:
            result.loc[len(result)] = {
                "prompt": prompt,
                "response": response,
            }

            prompt_index[normalized_prompt] = len(result) - 1

        elif status == UPDATED:
            index = prompt_index.get(normalized_prompt)

            if index is not None:
                result.loc[index, "prompt"] = prompt
                result.loc[index, "response"] = response

    return result.reset_index(drop=True)


def save_knowledge_base(
    knowledge_base: pd.DataFrame,
    file_path: str,
) -> None:
    """Persist the managed knowledge base."""

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    knowledge_base.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )
