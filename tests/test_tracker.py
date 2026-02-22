"""Tests for the CostTracker class."""

import concurrent.futures
import io
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

from tokencost import BudgetExceededError, CostTracker


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

    @patch("tokencost.tracker.completion_cost")
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

    @patch("tokencost.tracker.completion_cost")
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

    @patch("tokencost.tracker.completion_cost")
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

    @patch("tokencost.tracker.completion_cost")
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

    @patch("tokencost.tracker.completion_cost")
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


    @patch("tokencost.tracker.completion_cost")
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


class TestThreadSafety:
    """Tests for thread-safe operations."""

    @patch("tokencost.tracker.completion_cost")
    def test_concurrent_log_success_events(self, mock_completion_cost):
        """Multiple threads logging events should not corrupt data."""
        mock_completion_cost.return_value = 0.01
        tracker = CostTracker(print_summary=False)

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        num_threads = 10
        requests_per_thread = 100

        def log_requests():
            for _ in range(requests_per_thread):
                tracker.log_success_event(
                    kwargs={"model": "gpt-4"},
                    response_obj=mock_response,
                    start_time=None,
                    end_time=None,
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(log_requests) for _ in range(num_threads)]
            concurrent.futures.wait(futures)

        expected_count = num_threads * requests_per_thread
        expected_cost = expected_count * 0.01

        assert tracker.request_count == expected_count
        assert tracker.total_cost == pytest.approx(expected_cost)
        assert len(tracker.history) == expected_count

    @patch("tokencost.tracker.completion_cost")
    def test_concurrent_reset_and_log(self, mock_completion_cost):
        """Reset during logging should not cause errors."""
        mock_completion_cost.return_value = 0.01
        tracker = CostTracker(print_summary=False)

        mock_response = MagicMock()
        mock_response.usage = None

        errors = []

        def log_requests():
            try:
                for _ in range(50):
                    tracker.log_success_event(
                        kwargs={"model": "gpt-4"},
                        response_obj=mock_response,
                        start_time=None,
                        end_time=None,
                    )
            except Exception as e:
                errors.append(e)

        def reset_periodically():
            try:
                for _ in range(10):
                    tracker.reset()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=log_requests),
            threading.Thread(target=log_requests),
            threading.Thread(target=reset_periodically),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"


class TestCostByModel:
    """Tests for per-model cost aggregation."""

    @patch("tokencost.tracker.completion_cost")
    def test_cost_by_model_empty(self, mock_completion_cost):
        """cost_by_model should return empty dict when no requests."""
        tracker = CostTracker(print_summary=False)

        assert tracker.cost_by_model == {}

    @patch("tokencost.tracker.completion_cost")
    def test_cost_by_model_single_model(self, mock_completion_cost):
        """cost_by_model should aggregate costs for a single model."""
        mock_completion_cost.return_value = 0.05
        tracker = CostTracker(print_summary=False)

        mock_response = MagicMock()
        mock_response.usage = None

        for _ in range(3):
            tracker.log_success_event(
                kwargs={"model": "gpt-4"},
                response_obj=mock_response,
                start_time=None,
                end_time=None,
            )

        costs = tracker.cost_by_model
        assert len(costs) == 1
        assert costs["gpt-4"] == pytest.approx(0.15)

    @patch("tokencost.tracker.completion_cost")
    def test_cost_by_model_multiple_models(self, mock_completion_cost):
        """cost_by_model should aggregate costs per model."""
        tracker = CostTracker(print_summary=False)

        mock_response = MagicMock()
        mock_response.usage = None

        # gpt-4: 2 requests at $0.10 each
        mock_completion_cost.return_value = 0.10
        for _ in range(2):
            tracker.log_success_event(
                kwargs={"model": "gpt-4"},
                response_obj=mock_response,
                start_time=None,
                end_time=None,
            )

        # gpt-3.5-turbo: 3 requests at $0.01 each
        mock_completion_cost.return_value = 0.01
        for _ in range(3):
            tracker.log_success_event(
                kwargs={"model": "gpt-3.5-turbo"},
                response_obj=mock_response,
                start_time=None,
                end_time=None,
            )

        # claude-3: 1 request at $0.05
        mock_completion_cost.return_value = 0.05
        tracker.log_success_event(
            kwargs={"model": "claude-3"},
            response_obj=mock_response,
            start_time=None,
            end_time=None,
        )

        costs = tracker.cost_by_model
        assert len(costs) == 3
        assert costs["gpt-4"] == pytest.approx(0.20)
        assert costs["gpt-3.5-turbo"] == pytest.approx(0.03)
        assert costs["claude-3"] == pytest.approx(0.05)


class TestAsyncSupport:
    """Tests for async litellm support."""

    @patch("tokencost.tracker.completion_cost")
    @pytest.mark.asyncio
    async def test_async_log_success_event(self, mock_completion_cost):
        """async_log_success_event should track cost like sync version."""
        mock_completion_cost.return_value = 0.05
        tracker = CostTracker(print_summary=False)

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        await tracker.async_log_success_event(
            kwargs={"model": "gpt-4"},
            response_obj=mock_response,
            start_time=None,
            end_time=None,
        )

        assert tracker.total_cost == 0.05
        assert tracker.request_count == 1
        assert len(tracker.history) == 1
        assert tracker.history[0]["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_async_log_stream_event_does_not_track_cost(self):
        """async_log_stream_event should not track cost."""
        tracker = CostTracker(print_summary=False)

        await tracker.async_log_stream_event(
            kwargs={"model": "gpt-4"},
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        assert tracker.total_cost == 0.0
        assert tracker.request_count == 0

    @pytest.mark.asyncio
    async def test_async_log_failure_event_does_not_track_cost(self):
        """async_log_failure_event should not track cost."""
        tracker = CostTracker(print_summary=False)

        await tracker.async_log_failure_event(
            kwargs={"model": "gpt-4"},
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        assert tracker.total_cost == 0.0
        assert tracker.request_count == 0


class TestStreamingSupport:
    """Tests for streaming response support."""

    def test_log_stream_event_does_not_track_cost(self):
        """Stream events should not track cost (handled by final log_success_event)."""
        tracker = CostTracker(print_summary=False)

        tracker.log_stream_event(
            kwargs={"model": "gpt-4"},
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        assert tracker.total_cost == 0.0
        assert tracker.request_count == 0
        assert len(tracker.history) == 0

    @patch("tokencost.tracker.completion_cost")
    def test_streaming_completes_with_success_event(self, mock_completion_cost):
        """Streaming should still track cost via final log_success_event."""
        mock_completion_cost.return_value = 0.05
        tracker = CostTracker(print_summary=False)

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        # Simulate stream events (no cost tracking)
        for _ in range(5):
            tracker.log_stream_event(
                kwargs={"model": "gpt-4"},
                response_obj=None,
                start_time=None,
                end_time=None,
            )

        # Final success event tracks the complete cost
        tracker.log_success_event(
            kwargs={"model": "gpt-4"},
            response_obj=mock_response,
            start_time=None,
            end_time=None,
        )

        assert tracker.total_cost == 0.05
        assert tracker.request_count == 1
        assert len(tracker.history) == 1
