"""Unit Tests for Unified Streamlit UI Helpers (Day 36).

Validates that UI helper functions, session state initializers, and metadata renderers
execute cleanly without errors or side effects.
"""

import unittest
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src.cross_task.models import CitationItem, EvidenceItem, UnifiedResponse
import src.unified_main as ui_module


class TestCrossTaskUI(unittest.TestCase):
    """Test suite for unified UI component rendering and session helpers."""

    def test_ui_import_and_exports(self):
        """Verify unified_main imports cleanly and exposes expected functions."""
        self.assertTrue(callable(ui_module.configure_page))
        self.assertTrue(callable(ui_module.initialize_session_state))
        self.assertTrue(callable(ui_module.render_sidebar))
        self.assertTrue(callable(ui_module.render_response_metadata))
        self.assertTrue(callable(ui_module.main))

    @patch("src.unified_main.st")
    def test_render_response_metadata(self, mock_st):
        """Verify render_response_metadata formats metrics and expanders without crashing."""
        mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

        resp = UnifiedResponse(
            query="What are Transformer attention heads?",
            final_text_response="Attention heads project queries, keys, and values into parallel subspaces.",
            domain="scientific",
            detected_language="en",
            detected_language_name="English",
            confidence_score=0.96,
            confidence_tier="HIGH",
            evidence=[EvidenceItem(description="Multi-head attention mechanism", source="arxiv:1706.03762")],
            citations=[CitationItem(title="Attention Is All You Need", source_id="1706.03762", url="https://arxiv.org/abs/1706.03762")],
            is_grounded=True,
            sentiment="NEUTRAL",
            execution_time_ms=120.4,
        )

        ui_module.render_response_metadata(resp)
        mock_st.columns.assert_called_once_with(4)
        col3_mock = mock_st.columns.return_value[2]
        col3_mock.metric.assert_called_once_with("Grounding", "GROUNDED (HIGH)")
        mock_st.info.assert_called_once_with("📊 Detected Sentiment Tone: **NEUTRAL**")
        self.assertTrue(mock_st.expander.called)

    @patch("src.unified_main.st")
    def test_render_response_metadata_ood_and_sentiment_confidence(self, mock_st):
        """Verify render_response_metadata displays INSUFFICIENT grounding and formatted sentiment confidence."""
        mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

        resp = UnifiedResponse(
            query="Can I get a certificate upon completion?",
            final_text_response="I don't know.",
            domain="customer_support",
            detected_language="en",
            detected_language_name="English",
            confidence_score=0.0,
            confidence_tier="INSUFFICIENT",
            evidence=[],
            citations=[],
            is_grounded=False,
            sentiment="POSITIVE",
            execution_time_ms=85.2,
            metadata={"sentiment_confidence": 0.9898},
        )

        ui_module.render_response_metadata(resp)
        col3_mock = mock_st.columns.return_value[2]
        col3_mock.metric.assert_called_once_with("Grounding", "INSUFFICIENT")
        mock_st.info.assert_called_once_with("📊 Detected Sentiment Tone: **POSITIVE** (99.0%)")
        self.assertFalse(mock_st.expander.called)


if __name__ == "__main__":
    unittest.main()
