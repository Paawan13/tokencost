"""Tests for the CostTracker class."""

from unittest.mock import MagicMock, patch
import io
import sys

import pytest

from llm_cost import BudgetExceededError, CostTracker


class TestCostTracker:
    """Tests for CostTracker functionality."""

    def test_initial_state(self):
        """Tracker should start with zero cost and no history."""
        tracker = CostTracker(print_summary=False)

        assert tracker.total_cost == 0.0
        assert tracker.request_count == 0
        assert tracker.history == []
        assert tracker.budget is None
        assert tracker.budget_exceeded is False

    def test_budget_configuration(self):
        """Budget should be configurable via constructor."""
        tracker = CostTracker(budget=10.0, print_summary=False)

        assert tracker.budget == 10.0
        assert tracker.budget_exceeded is False

    def test_reset_clears_all_data(self):
        """Reset should clear all tracked data."""
        tracker = CostTracker(budget=1.0, print_summary=False)
        tracker._total_cost = 5.0
        tracker._request_count = 10
        tracker._history.append({"model": "test", "cost": 1.0})
        tracker._budget_exceeded = True

        tracker.reset()

        assert tracker.total_cost == 0.0
        assert tracker.request_count == 0
        assert tracker.history == []
        assert tracker.budget_exceeded is False

    def test_history_returns_copy(self):
        """History property should return a copy to prevent mutation."""
        tracker = CostTracker(print_summary=False)
        tracker._history.append({"model": "test", "cost": 0.01})

        history = tracker.history
        history.append({"model": "modified", "cost": 999})

        assert len(tracker.history) == 1
        assert tracker.history[0]["model"] == "test"

    @patch("llm_cost.tracker.completion_cost")
    def test_log_success_event_tracks_cost(self, mock_completion_cost):
        """Successful events should be tracked with cost."""
        mock_completion_cost.return_value = 0.05
        tracker = CostTracker(print_summary=False)

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        tracker.log_success_event(
            kwargs={"model": "gpt-4"},
            response_obj=mock_response,
            start_time=None,
            end_time=None,
        )

        assert tracker.total_cost == 0.05
        assert tracker.request_count == 1
        assert len(tracker.history) == 1
        assert tracker.history[0]["model"] == "gpt-4"
        assert tracker.history[0]["prompt_tokens"] == 100
        assert tracker.history[0]["completion_tokens"] == 50
        assert tracker.history[0]["cost"] == 0.05
        assert "timestamp" in tracker.history[0]

    @patch("llm_cost.tracker.completion_cost")
    def test_budget_exceeded_callback(self, mock_completion_cost):
        """Callback should fire when budget is exceeded."""
        mock_completion_cost.return_value = 2.0
        callback = MagicMock()
        tracker = CostTracker(budget=1.0, on_budget_exceeded=callback, print_summary=False)

        mock_response = MagicMock()
        mock_response.usage = None

        tracker.log_success_event(
            kwargs={"model": "gpt-4"},
            response_obj=mock_response,
            start_time=None,
            end_time=None,
        )

        assert tracker.budget_exceeded is True
        callback.assert_called_once_with(tracker)

    @patch("llm_cost.tracker.completion_cost")
    def test_budget_exceeded_raises_exception(self, mock_completion_cost):
        """Exception should be raised when budget exceeded and raise_on_budget=True."""
        mock_completion_cost.return_value = 2.0
        tracker = CostTracker(budget=1.0, raise_on_budget=True, print_summary=False)

        mock_response = MagicMock()
        mock_response.usage = None

        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.log_success_event(
                kwargs={"model": "gpt-4"},
                response_obj=mock_response,
                start_time=None,
                end_time=None,
            )

        assert exc_info.value.budget == 1.0
        assert exc_info.value.total_cost == 2.0

    @patch("llm_cost.tracker.completion_cost")
    def test_callback_fires_only_once(self, mock_completion_cost):
        """Callback should fire only once even with multiple exceeding requests."""
        mock_completion_cost.return_value = 2.0
        callback = MagicMock()
        tracker = CostTracker(budget=1.0, on_budget_exceeded=callback, print_summary=False)

        mock_response = MagicMock()
        mock_response.usage = None

        tracker.log_success_event(
            kwargs={"model": "gpt-4"},
            response_obj=mock_response,
            start_time=None,
            end_time=None,
        )

        tracker.log_success_event(
            kwargs={"model": "gpt-4"},
            response_obj=mock_response,
            start_time=None,
            end_time=None,
        )

        callback.assert_called_once()

    def test_log_failure_event_does_not_track_cost(self):
        """Failed events should not incur cost."""
        tracker = CostTracker(print_summary=False)

        tracker.log_failure_event(
            kwargs={"model": "gpt-4"},
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        assert tracker.total_cost == 0.0
        assert tracker.request_count == 0
        assert len(tracker.history) == 0

    @patch("llm_cost.tracker.completion_cost")
    def test_cost_calculation_error_defaults_to_zero(self, mock_completion_cost):
        """If cost calculation fails, default to zero cost."""
        mock_completion_cost.side_effect = Exception("Unknown model")
        tracker = CostTracker(print_summary=False)

        mock_response = MagicMock()
        mock_response.usage = None

        tracker.log_success_event(
            kwargs={"model": "unknown-model"},
            response_obj=mock_response,
            start_time=None,
            end_time=None,
        )

        assert tracker.total_cost == 0.0
        assert tracker.request_count == 1


    @patch("llm_cost.tracker.completion_cost")
    def test_print_exit_summary(self, mock_completion_cost):
        """Exit summary should print formatted cost report."""
        mock_completion_cost.return_value = 0.05
        tracker = CostTracker(budget=1.0, print_summary=False)

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        tracker.log_success_event(
            kwargs={"model": "gpt-4"},
            response_obj=mock_response,
            start_time=None,
            end_time=None,
        )

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        tracker._print_exit_summary()
        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        assert "LLM COST SUMMARY" in output
        assert "Total Cost:" in output
        assert "Total Requests: 1" in output
        assert "gpt-4" in output
        assert "0.05" in output


class TestBudgetExceededError:
    """Tests for BudgetExceededError exception."""

    def test_error_message(self):
        """Error message should contain budget and total cost."""
        error = BudgetExceededError(budget=5.0, total_cost=7.5)

        assert error.budget == 5.0
        assert error.total_cost == 7.5
        assert "5.00" in str(error)
        assert "7.5" in str(error)
