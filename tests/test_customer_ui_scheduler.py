import unittest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    # pyrefly: ignore [missing-import]
    import src.main as customer_main  # type: ignore
except ImportError:
    # pyrefly: ignore [missing-import]
    import main as customer_main  # type: ignore

KnowledgeBaseScheduler = customer_main.KnowledgeBaseScheduler


class TestCustomerUIScheduler(unittest.TestCase):

    def setUp(self):
        customer_main._GLOBAL_SCHEDULER_INSTANCE = None
        clear_cache = getattr(customer_main.get_knowledge_scheduler, "clear", None)
        if clear_cache:
            clear_cache()

    @patch.object(KnowledgeBaseScheduler, "start", return_value=True)
    def test_get_knowledge_scheduler_returns_singleton(self, mock_start):
        scheduler1 = customer_main.get_knowledge_scheduler()
        self.assertIsInstance(scheduler1, KnowledgeBaseScheduler)

        scheduler2 = customer_main.get_knowledge_scheduler()
        self.assertIs(scheduler1, scheduler2)

    @patch("streamlit.expander")
    @patch("streamlit.markdown")
    @patch("streamlit.button", return_value=False)
    def test_render_sync_status_display(self, mock_button, mock_markdown, mock_expander):
        mock_scheduler = MagicMock()
        mock_scheduler.get_status.return_value = {
            "is_running": True,
            "interval_seconds": 300.0,
            "last_run_timestamp": "2026-08-22T12:00:00Z",
            "last_status": "SUCCESS",
            "last_error": None,
            "history_path": "dummy_history.jsonl",
        }

        mock_expander.return_value.__enter__.return_value = MagicMock()

        customer_main.render_sync_status(mock_scheduler)

        # Verify markdown was called for status & interval
        self.assertGreater(mock_markdown.call_count, 0)
        rendered_texts = [call[0][0] for call in mock_markdown.call_args_list]
        self.assertTrue(any("🟢 Active" in t for t in rendered_texts))
        self.assertTrue(any("300s" in t for t in rendered_texts))

    @patch("streamlit.expander")
    @patch("streamlit.markdown")
    @patch("streamlit.button", return_value=True)
    @patch("streamlit.spinner")
    @patch("streamlit.success")
    def test_render_sync_status_manual_trigger_success(
        self, mock_success, mock_spinner, mock_button, mock_markdown, mock_expander
    ):
        mock_scheduler = MagicMock()
        mock_scheduler.get_status.return_value = {
            "is_running": True,
            "interval_seconds": 300.0,
            "history_path": "dummy_history.jsonl",
        }
        mock_scheduler.run_once.return_value = {"status": "SUCCESS"}

        mock_expander.return_value.__enter__.return_value = MagicMock()
        mock_spinner.return_value.__enter__.return_value = MagicMock()

        customer_main.render_sync_status(mock_scheduler)

        mock_scheduler.run_once.assert_called_once()
        mock_success.assert_called_once_with("Knowledge sources synchronized successfully!")


# Standalone functions for pytest compatibility
def test_get_knowledge_scheduler_returns_singleton():
    t = TestCustomerUIScheduler()
    t.setUp()
    try:
        t.test_get_knowledge_scheduler_returns_singleton()
    finally:
        pass


if __name__ == "__main__":
    unittest.main()
