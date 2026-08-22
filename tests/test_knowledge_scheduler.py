import unittest
import sys
import os
import time
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src.knowledge_base.scheduler import KnowledgeBaseScheduler


class TestKnowledgeScheduler(unittest.TestCase):

    def setUp(self):
        self.scheduler = KnowledgeBaseScheduler(
            config_path="dummy_config.json",
            knowledge_base_path="dummy_kb.csv",
            vector_store_path="dummy_faiss",
            history_path="dummy_history.jsonl",
            interval_seconds=0.1,
        )

    def tearDown(self):
        self.scheduler.stop(timeout=1.0)

    @patch("src.knowledge_base.scheduler.process_configured_sources")
    def test_scheduler_start_and_stop_cleanly(self, mock_process):
        mock_process.return_value = [{"source_name": "s1", "status": "SUCCESS"}]

        started = self.scheduler.start()
        self.assertTrue(started)
        self.assertTrue(self.scheduler.is_running())

        # Allow quick cycle
        time.sleep(0.05)

        stopped = self.scheduler.stop(timeout=1.0)
        self.assertTrue(stopped)
        self.assertFalse(self.scheduler.is_running())
        self.assertGreaterEqual(mock_process.call_count, 1)

    @patch("src.knowledge_base.scheduler.process_configured_sources")
    def test_calling_start_twice_does_not_create_duplicate_worker(self, mock_process):
        mock_process.return_value = []

        self.assertTrue(self.scheduler.start())
        worker_thread = self.scheduler._thread

        # Second start call should return False
        self.assertFalse(self.scheduler.start())
        # Thread identity must remain unchanged
        self.assertIs(self.scheduler._thread, worker_thread)

    @patch("src.knowledge_base.scheduler.process_configured_sources")
    def test_run_once_executes_configured_sources(self, mock_process):
        mock_process.return_value = [
            {"source_name": "s1", "status": "SUCCESS", "result": {"new": 1}}
        ]

        result = self.scheduler.run_once()

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["sources_processed"], 1)
        self.assertEqual(self.scheduler._total_runs, 1)
        self.assertEqual(self.scheduler._last_status, "SUCCESS")
        mock_process.assert_called_once_with(
            config_path="dummy_config.json",
            knowledge_base_path="dummy_kb.csv",
            vector_store_path="dummy_faiss",
            history_path="dummy_history.jsonl",
        )

    def test_concurrent_run_once_is_skipped(self):
        # Manually hold the scheduler lock to simulate an active execution
        self.scheduler._lock.acquire()
        try:
            result = self.scheduler.run_once()
            self.assertEqual(result["status"], "SKIPPED")
            self.assertIn("Another update execution is currently in progress", result["reason"])
        finally:
            self.scheduler._lock.release()

    @patch("src.knowledge_base.scheduler.process_configured_sources")
    def test_exception_in_process_configured_sources_is_contained(self, mock_process):
        mock_process.side_effect = RuntimeError("Disk full simulation")

        result = self.scheduler.run_once()

        self.assertEqual(result["status"], "FAILED")
        self.assertIn("Disk full simulation", result["error"])
        self.assertEqual(self.scheduler._last_status, "FAILED")
        self.assertEqual(self.scheduler._last_error, "Disk full simulation")
        self.assertFalse(self.scheduler._is_executing)

    @patch("src.knowledge_base.scheduler.process_configured_sources")
    def test_scheduler_can_restart_after_stopping(self, mock_process):
        mock_process.return_value = []

        self.assertTrue(self.scheduler.start())
        self.assertTrue(self.scheduler.is_running())

        self.assertTrue(self.scheduler.stop(timeout=1.0))
        self.assertFalse(self.scheduler.is_running())

        # Restart
        self.assertTrue(self.scheduler.start())
        self.assertTrue(self.scheduler.is_running())

    def test_get_status_returns_complete_metadata(self):
        status = self.scheduler.get_status()

        required_keys = [
            "is_running", "is_executing", "interval_seconds",
            "last_run_timestamp", "last_status", "last_result",
            "last_error", "total_runs", "config_path",
            "knowledge_base_path", "vector_store_path", "history_path"
        ]
        for key in required_keys:
            self.assertIn(key, status)

        self.assertEqual(status["interval_seconds"], 0.1)
        self.assertFalse(status["is_running"])
        self.assertFalse(status["is_executing"])


# Standalone functions for pytest compatibility
def test_scheduler_start_and_stop_cleanly():
    t = TestKnowledgeScheduler()
    t.setUp()
    try:
        t.test_scheduler_start_and_stop_cleanly()
    finally:
        t.tearDown()


if __name__ == "__main__":
    unittest.main()
