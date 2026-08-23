import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default path for persistent update history
DEFAULT_HISTORY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "dataset",
    "knowledge_update_history.jsonl"
)


def record_update(
    history_path: str,
    update_summary: Optional[Dict[str, Any]] = None,
    source: str = "",
    status: str = "SUCCESS",
    timestamp: Optional[str] = None,
    error: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Append a structured update entry to the persistent JSON Lines audit log.

    Supports SUCCESS, FAILED, and SKIPPED audit entries.

    Args:
        history_path (str): Filepath to the .jsonl audit file.
        update_summary (Optional[Dict[str, Any]]): Execution summary from updater (for SUCCESS).
        source (str): Source identifier / filepath of incoming updates.
        status (str): Status of the update event ('SUCCESS', 'FAILED', 'SKIPPED').
        timestamp (Optional[str]): Explicit ISO timestamp (defaults to current UTC timestamp).
        error (Optional[str]): Error description (for FAILED status).
        reason (Optional[str]): Skip reason (for SKIPPED status).

    Returns:
        Dict[str, Any]: The recorded audit log entry.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    status_upper = str(status).upper()

    if status_upper in ("FAILED", "REBUILD_FAILED"):
        entry = {
            "timestamp": timestamp,
            "source": str(source),
            "status": status_upper,
            "error": str(error or "Unknown error"),
        }
    elif status_upper == "SKIPPED":
        entry = {
            "timestamp": timestamp,
            "source": str(source),
            "status": "SKIPPED",
            "reason": str(reason or "Update skipped"),
        }
    else:
        summary = update_summary or {}
        entry = {
            "timestamp": timestamp,
            "source": str(source),
            "existing_records": int(summary.get("existing_records", 0)),
            "incoming_records": int(summary.get("incoming_records", 0)),
            "final_records": int(summary.get("final_records", 0)),
            "new": int(summary.get("new", 0)),
            "updated": int(summary.get("updated", 0)),
            "duplicate": int(summary.get("duplicate", 0)),
            "invalid": int(summary.get("invalid", 0)),
            "status": status_upper if status_upper in ("REBUILD_SUCCESS", "SUCCESS") else "SUCCESS",
        }

    target_path = Path(history_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def record_failure(
    history_path: str,
    source: str,
    error: str,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience helper to append a FAILED audit log entry."""
    return record_update(
        history_path=history_path,
        source=source,
        status="FAILED",
        error=error,
        timestamp=timestamp,
    )


def record_skip(
    history_path: str,
    source: str,
    reason: str,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience helper to append a SKIPPED audit log entry."""
    return record_update(
        history_path=history_path,
        source=source,
        status="SKIPPED",
        reason=reason,
        timestamp=timestamp,
    )


def load_update_history(history_path: str) -> List[Dict[str, Any]]:
    """Load all audit log entries from the JSON Lines file.

    Args:
        history_path (str): Path to the .jsonl audit file.

    Returns:
        List[Dict[str, Any]]: List of parsed update history entries. Returns empty list if file does not exist.
    """
    target_path = Path(history_path)
    if not target_path.exists():
        return []

    entries = []
    with open(target_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                entries.append(data)
            except json.JSONDecodeError:
                continue

    return entries


def get_last_successful_update(history_path: str) -> Optional[Dict[str, Any]]:
    """Retrieve the most recent successful update entry from the audit log.

    Args:
        history_path (str): Path to the .jsonl audit file.

    Returns:
        Optional[Dict[str, Any]]: The latest entry where status == 'SUCCESS', or None if no entries exist.
    """
    history = load_update_history(history_path)
    for entry in reversed(history):
        if entry.get("status") in ("SUCCESS", "REBUILD_SUCCESS"):
            return entry
    return None    