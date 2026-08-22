import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    from src.knowledge_base.audit import DEFAULT_HISTORY_PATH
    from src.knowledge_base.sources import (
        DEFAULT_SOURCES_CONFIG_PATH,
        process_configured_sources,
    )
    from src.langchain_helper import vectordb_file_path
except ImportError:
    from knowledge_base.audit import DEFAULT_HISTORY_PATH
    from knowledge_base.sources import (
        DEFAULT_SOURCES_CONFIG_PATH,
        process_configured_sources,
    )
    from langchain_helper import vectordb_file_path

DEFAULT_KB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "dataset",
    "knowledge_base.csv",
)
DEFAULT_INTERVAL_SECONDS = 300.0  # 5 minutes


class KnowledgeBaseScheduler:
    """Thread-safe periodic scheduler for multi-source dynamic knowledge base updates."""

    def __init__(
        self,
        config_path: str = DEFAULT_SOURCES_CONFIG_PATH,
        knowledge_base_path: str = DEFAULT_KB_PATH,
        vector_store_path: str = vectordb_file_path,
        history_path: Optional[str] = DEFAULT_HISTORY_PATH,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    ):
        self.config_path = config_path
        self.knowledge_base_path = knowledge_base_path
        self.vector_store_path = vector_store_path
        self.history_path = history_path
        self.interval_seconds = float(interval_seconds)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._is_executing = False
        self._last_run_timestamp: Optional[str] = None
        self._last_status: Optional[str] = None
        self._last_result: Optional[Any] = None
        self._last_error: Optional[str] = None
        self._total_runs = 0

    def is_running(self) -> bool:
        """Check if background worker thread is currently active."""
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> bool:
        """Start the background periodic scheduler thread.

        Returns:
            bool: True if started, False if already running.
        """
        if self.is_running():
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="KnowledgeBaseSchedulerWorker",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self, timeout: float = 5.0) -> bool:
        """Signal the scheduler to terminate and join the background worker thread.

        Args:
            timeout (float): Maximum seconds to wait for worker thread shutdown.

        Returns:
            bool: True if stopped cleanly.
        """
        if not self.is_running():
            return True

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            is_dead = not self._thread.is_alive()
            if is_dead:
                self._thread = None
            return is_dead
        return True

    def run_once(self) -> Dict[str, Any]:
        """Execute a single knowledge update cycle protected by a non-blocking mutex.

        Returns:
            Dict[str, Any]: Execution outcome status dictionary.
        """
        # Non-blocking lock acquisition to prevent concurrent overlapping updates
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            return {
                "status": "SKIPPED",
                "reason": "Another update execution is currently in progress.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        try:
            self._is_executing = True
            timestamp = datetime.now(timezone.utc).isoformat()
            self._last_run_timestamp = timestamp

            results = process_configured_sources(
                config_path=self.config_path,
                knowledge_base_path=self.knowledge_base_path,
                vector_store_path=self.vector_store_path,
                history_path=self.history_path,
            )

            self._last_status = "SUCCESS"
            self._last_result = results
            self._last_error = None
            self._total_runs += 1

            return {
                "status": "SUCCESS",
                "timestamp": timestamp,
                "sources_processed": len(results),
                "results": results,
            }

        except Exception as e:
            self._last_status = "FAILED"
            self._last_error = str(e)
            self._total_runs += 1
            return {
                "status": "FAILED",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }

        finally:
            self._is_executing = False
            self._lock.release()

    def _run_loop(self) -> None:
        """Internal background loop executing periodic update runs."""
        while not self._stop_event.is_set():
            self.run_once()
            # Wait for interval or immediate exit on stop_event signal
            if self._stop_event.wait(self.interval_seconds):
                break

    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler runtime and execution metadata."""
        return {
            "is_running": self.is_running(),
            "is_executing": self._is_executing,
            "interval_seconds": self.interval_seconds,
            "last_run_timestamp": self._last_run_timestamp,
            "last_status": self._last_status,
            "last_result": self._last_result,
            "last_error": self._last_error,
            "total_runs": self._total_runs,
            "config_path": self.config_path,
            "knowledge_base_path": self.knowledge_base_path,
            "vector_store_path": self.vector_store_path,
            "history_path": self.history_path,
        }
