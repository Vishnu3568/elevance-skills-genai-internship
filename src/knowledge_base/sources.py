import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from src.knowledge_base.audit import DEFAULT_HISTORY_PATH
    from src.knowledge_base.updater import update_knowledge_base
except ImportError:
    from knowledge_base.audit import DEFAULT_HISTORY_PATH
    from knowledge_base.updater import update_knowledge_base

DEFAULT_SOURCES_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "dataset",
    "knowledge_sources.json"
)

REQUIRED_SOURCE_FIELDS = {"name", "path", "format", "enabled"}
SUPPORTED_FORMATS = {"csv"}


def validate_source_entry(entry: Dict[str, Any]) -> bool:
    """Validate that a single source configuration entry has all required fields.

    Args:
        entry (Dict[str, Any]): The source entry dictionary.

    Returns:
        bool: True if valid.

    Raises:
        ValueError: If any required field is missing or invalid.
    """
    if not isinstance(entry, dict):
        raise ValueError("Each source entry must be a dictionary.")

    missing = REQUIRED_SOURCE_FIELDS - set(entry.keys())
    if missing:
        raise ValueError(f"Source entry missing required fields: {sorted(missing)}")

    if not isinstance(entry["name"], str) or not entry["name"].strip():
        raise ValueError("Source field 'name' must be a non-empty string.")

    if not isinstance(entry["path"], str) or not entry["path"].strip():
        raise ValueError("Source field 'path' must be a non-empty string.")

    if entry["format"] not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{entry['format']}'. Supported formats: {sorted(SUPPORTED_FORMATS)}"
        )

    if not isinstance(entry["enabled"], bool):
        raise ValueError("Source field 'enabled' must be a boolean.")

    return True


def load_source_config(config_path: str) -> List[Dict[str, Any]]:
    """Load and validate the knowledge sources configuration file.

    Args:
        config_path (str): Path to the JSON configuration file.

    Returns:
        List[Dict[str, Any]]: List of validated source dictionaries.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If JSON is invalid or schema validation fails.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge sources config not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in source config: {e}")

    if not isinstance(data, list):
        raise ValueError("Sources configuration must be a list of source entries.")

    for entry in data:
        validate_source_entry(entry)

    return data


def get_enabled_sources(config_path: str) -> List[Dict[str, Any]]:
    """Retrieve only the enabled sources from the configuration in deterministic order.

    Args:
        config_path (str): Path to the JSON configuration file.

    Returns:
        List[Dict[str, Any]]: List of enabled source entries.
    """
    sources = load_source_config(config_path)
    return [source for source in sources if source.get("enabled") is True]


def process_configured_sources(
    config_path: str,
    knowledge_base_path: str,
    vector_store_path: str,
    history_path: Optional[str] = DEFAULT_HISTORY_PATH,
) -> List[Dict[str, Any]]:
    """Process all enabled knowledge sources sequentially and update the knowledge base.

    Args:
        config_path (str): Path to knowledge sources JSON configuration.
        knowledge_base_path (str): Path to the managed CSV knowledge base.
        vector_store_path (str): Path to the persisted FAISS vector store.
        history_path (Optional[str]): Path to audit log history file.

    Returns:
        List[Dict[str, Any]]: List of execution outcomes for each enabled source.
    """
    enabled_sources = get_enabled_sources(config_path)
    results = []

    for source in enabled_sources:
        source_name = source["name"]
        source_path = source["path"]

        # If path is relative and doesn't exist relative to CWD, check relative to config_path's base
        resolved_path = source_path
        if not os.path.isabs(resolved_path) and not os.path.exists(resolved_path):
            config_dir = os.path.dirname(os.path.abspath(config_path))
            candidate = os.path.join(config_dir, source_path)
            if os.path.exists(candidate):
                resolved_path = candidate

        if not os.path.exists(resolved_path):
            results.append({
                "source_name": source_name,
                "path": source_path,
                "status": "ERROR",
                "error": f"Source file not found: {source_path}",
            })
            continue

        try:
            update_result = update_knowledge_base(
                knowledge_base_path=knowledge_base_path,
                update_source_path=resolved_path,
                vector_store_path=vector_store_path,
                history_path=history_path,
            )
            results.append({
                "source_name": source_name,
                "path": source_path,
                "status": "SUCCESS",
                "result": update_result,
            })
        except Exception as e:
            results.append({
                "source_name": source_name,
                "path": source_path,
                "status": "FAILED",
                "error": str(e),
            })

    return results
