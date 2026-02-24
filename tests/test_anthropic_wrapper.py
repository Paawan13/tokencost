"""Tests for the Anthropic wrapper module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tokencost import CostTracker
from tokencost.anthropic_wrapper import (
    _extract_streaming_usage,
    _extract_usage,
    _record_to_tracker,
    patch_anthropic,
    track_anthropic,
    unpatch_anthropic,
)


class MockUsage:
    """Mock Anthropic usage object."""

    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockResponse:
    """Mock Anthropic message response."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        input_tokens: int = 100,
        output_tokens: int = 50,
    ):
        self.model = model
        self.usage = MockUsage(input_tokens, output_tokens)
        self.id = "msg_123"
        self.type = "message"
        self.role = "assistant"
        self.content = [{"type": "text", "text": "Hello!"}]
        self.stop_reason = "end_turn"


class MockStreamEvent:
    """Mock Anthropic streaming event."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        usage: MockUsage | None = None,
        message: MockResponse | None = None,
    ):
        self.model = model
        self.usage = usage
        self.message = message
        self.type = "content_block_delta"


class TestExtractUsage:
    """Tests for _extract_usage function."""

    def test_extract_usage_normal_response(self):
        """Test extracting usage from a normal response."""
        response = MockResponse(
            model="claude-3-5-sonnet-20241022", input_tokens=100, output_tokens=50
        )
        model, input_tokens, output_tokens = _extract_usage(response)
        assert model == "claude-3-5-sonnet-20241022"
        assert input_tokens == 100
        assert output_tokens == 50

    def test_extract_usage_no_usage(self):
        """Test extracting usage when usage is None."""
        response = MockResponse()
        response.usage = None
        model, input_tokens, output_tokens = _extract_usage(response)
        assert model == "claude-3-5-sonnet-20241022"
        assert input_tokens == 0
        assert output_tokens == 0

    def test_extract_usage_missing_model(self):
        """Test extracting usage when model is missing."""
        response = MagicMock()
        del response.model
        response.usage = MockUsage(100, 50)
        model, input_tokens, output_tokens = _extract_usage(response)
        assert model == "unknown"
        assert input_tokens == 100
        assert output_tokens == 50


class TestExtractStreamingUsage:
    """Tests for _extract_streaming_usage function."""

    def test_extract_streaming_usage_with_usage(self):
        """Test extracting usage from final streaming event."""
        event = MockStreamEvent(
            model="claude-3-5-sonnet-20241022", usage=MockUsage(100, 50)
        )
        result = _extract_streaming_usage(event)
        assert result is not None
        model, input_tokens, output_tokens = result
        assert model == "claude-3-5-sonnet-20241022"
        assert input_tokens == 100
        assert output_tokens == 50

    def test_extract_streaming_usage_no_usage(self):
        """Test extracting usage from intermediate event (no usage)."""
        event = MockStreamEvent(model="claude-3-5-sonnet-20241022", usage=None)
        result = _extract_streaming_usage(event)
        assert result is None

    def test_extract_streaming_usage_from_message(self):
        """Test extracting usage from message in event."""
        message = MockResponse(
            model="claude-3-5-sonnet-20241022", input_tokens=150, output_tokens=75
        )
        event = MockStreamEvent(model=None, usage=None, message=message)
        # Remove model from event to test fallback to message.model
        del event.model
        result = _extract_streaming_usage(event)
        assert result is not None
        model, input_tokens, output_tokens = result
        assert model == "claude-3-5-sonnet-20241022"
        assert input_tokens == 150
        assert output_tokens == 75


class TestRecordToTracker:
    """Tests for _record_to_tracker function."""

    def test_record_to_tracker_with_tracker(self):
        """Test recording cost to tracker."""
        tracker = CostTracker(print_summary=False)
        _record_to_tracker(tracker, "claude-3-5-sonnet-20241022", 1000, 500)
        assert tracker.total_cost > 0
        assert tracker.request_count == 1

    def test_record_to_tracker_no_tracker(self):
        """Test that None tracker doesn't cause errors."""
        # Should not raise
        _record_to_tracker(None, "claude-3-5-sonnet-20241022", 1000, 500)

    def test_record_to_tracker_unknown_model(self):
        """Test that unknown model doesn't record."""
        tracker = CostTracker(print_summary=False)
        _record_to_tracker(tracker, "unknown-model-xyz", 1000, 500)
        # Should not record anything for unknown model
        assert tracker.total_cost == 0
        assert tracker.request_count == 0


class TestTrackAnthropic:
    """Tests for track_anthropic function."""

    def test_track_anthropic_sync_client(self):
        """Test wrapping a sync Anthropic client."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockResponse()

        tracker = CostTracker(print_summary=False)
        wrapped = track_anthropic(mock_client, tracker)

        # Call create
        response = wrapped.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response.model == "claude-3-5-sonnet-20241022"
        assert tracker.total_cost > 0
        assert tracker.request_count == 1

    def test_track_anthropic_sync_streaming(self):
        """Test wrapping sync streaming response."""
        # Create mock streaming events
        events = [
            MockStreamEvent(model="claude-3-5-sonnet-20241022", usage=None),
            MockStreamEvent(model="claude-3-5-sonnet-20241022", usage=None),
            MockStreamEvent(
                model="claude-3-5-sonnet-20241022", usage=MockUsage(100, 50)
            ),  # Final event
        ]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = iter(events)

        tracker = CostTracker(print_summary=False)
        wrapped = track_anthropic(mock_client, tracker)

        # Call create with streaming
        stream = wrapped.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        # Consume the stream
        received_events = list(stream)
        assert len(received_events) == 3

        # Cost should be recorded after final event
        assert tracker.total_cost > 0
        assert tracker.request_count == 1

    def test_track_anthropic_without_tracker(self):
        """Test wrapping client without tracker (just pass-through)."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockResponse()

        wrapped = track_anthropic(mock_client)  # No tracker

        response = wrapped.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response.model == "claude-3-5-sonnet-20241022"

    def test_track_anthropic_preserves_other_attrs(self):
        """Test that wrapped client preserves other attributes."""
        mock_client = MagicMock()
        mock_client.api_key = "test-key"
        mock_client.base_url = "https://api.anthropic.com"

        wrapped = track_anthropic(mock_client)

        assert wrapped.api_key == "test-key"
        assert wrapped.base_url == "https://api.anthropic.com"


class TestTrackAnthropicAsync:
    """Tests for track_anthropic with async clients."""

    @pytest.mark.asyncio
    async def test_track_anthropic_async_client(self):
        """Test wrapping an async Anthropic client."""
        mock_client = MagicMock()
        mock_client.__class__.__name__ = "AsyncAnthropic"
        mock_client.messages.create = AsyncMock(return_value=MockResponse())

        tracker = CostTracker(print_summary=False)
        wrapped = track_anthropic(mock_client, tracker)

        # Call create
        response = await wrapped.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response.model == "claude-3-5-sonnet-20241022"
        assert tracker.total_cost > 0
        assert tracker.request_count == 1

    @pytest.mark.asyncio
    async def test_track_anthropic_async_streaming(self):
        """Test wrapping async streaming response."""
        events = [
            MockStreamEvent(model="claude-3-5-sonnet-20241022", usage=None),
            MockStreamEvent(model="claude-3-5-sonnet-20241022", usage=None),
            MockStreamEvent(
                model="claude-3-5-sonnet-20241022", usage=MockUsage(100, 50)
            ),
        ]

        async def async_iter():
            for event in events:
                yield event

        mock_client = MagicMock()
        mock_client.__class__.__name__ = "AsyncAnthropic"
        mock_client.messages.create = AsyncMock(return_value=async_iter())

        tracker = CostTracker(print_summary=False)
        wrapped = track_anthropic(mock_client, tracker)

        # Call create with streaming
        stream = await wrapped.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        # Consume the stream
        received_events = []
        async for event in stream:
            received_events.append(event)

        assert len(received_events) == 3
        assert tracker.total_cost > 0
        assert tracker.request_count == 1


class TestPatchUnpatchAnthropic:
    """Tests for patch_anthropic and unpatch_anthropic functions."""

    def test_patch_anthropic_changes_tracker(self):
        """Test that patch_anthropic can update the tracker on subsequent calls."""
        import tokencost.anthropic_wrapper as wrapper

        # Ensure we start fresh
        unpatch_anthropic()

        tracker1 = CostTracker(print_summary=False)
        tracker2 = CostTracker(print_summary=False)

        patch_anthropic(tracker1)
        assert wrapper._global_tracker is tracker1

        # Calling again should update the tracker
        patch_anthropic(tracker2)
        assert wrapper._global_tracker is tracker2

        unpatch_anthropic()

    def test_unpatch_anthropic_when_not_patched(self):
        """Test that unpatch_anthropic is safe to call when not patched."""
        import tokencost.anthropic_wrapper as wrapper

        wrapper._original_create = None
        wrapper._original_async_create = None

        # Should not raise
        unpatch_anthropic()


class TestBudgetIntegration:
    """Tests for budget integration with Anthropic wrapper."""

    def test_budget_exceeded_during_tracking(self):
        """Test that budget is checked during tracking."""
        from tokencost import BudgetExceededError

        mock_client = MagicMock()
        # Return a response with high token count
        mock_client.messages.create.return_value = MockResponse(
            model="claude-3-5-sonnet-20241022", input_tokens=100000, output_tokens=50000
        )

        tracker = CostTracker(budget=0.001, raise_on_budget=True, print_summary=False)
        wrapped = track_anthropic(mock_client, tracker)

        with pytest.raises(BudgetExceededError):
            wrapped.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
            )

    def test_multiple_requests_accumulate(self):
        """Test that multiple requests accumulate cost."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockResponse(
            model="claude-3-5-sonnet-20241022", input_tokens=100, output_tokens=50
        )

        tracker = CostTracker(print_summary=False)
        wrapped = track_anthropic(mock_client, tracker)

        # Make multiple requests
        for _ in range(5):
            wrapped.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert tracker.request_count == 5
        # Cost should be 5x single request cost
        single_cost = tracker.history[0]["cost"]
        assert abs(tracker.total_cost - single_cost * 5) < 0.0001


class TestClaudeModels:
    """Tests for various Claude models."""

    def test_claude_opus_model(self):
        """Test tracking Claude Opus model."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockResponse(
            model="claude-3-opus-20240229", input_tokens=100, output_tokens=50
        )

        tracker = CostTracker(print_summary=False)
        wrapped = track_anthropic(mock_client, tracker)

        wrapped.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert tracker.total_cost > 0
        assert "claude-3-opus-20240229" in tracker.cost_by_model

    def test_claude_sonnet_model(self):
        """Test tracking Claude Sonnet model."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockResponse(
            model="claude-3-5-sonnet-20241022", input_tokens=100, output_tokens=50
        )

        tracker = CostTracker(print_summary=False)
        wrapped = track_anthropic(mock_client, tracker)

        wrapped.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert tracker.total_cost > 0
        assert "claude-3-5-sonnet-20241022" in tracker.cost_by_model

    def test_claude_haiku_model(self):
        """Test tracking Claude Haiku model."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockResponse(
            model="claude-3-haiku-20240307", input_tokens=100, output_tokens=50
        )

        tracker = CostTracker(print_summary=False)
        wrapped = track_anthropic(mock_client, tracker)

        wrapped.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert tracker.total_cost > 0
        assert "claude-3-haiku-20240307" in tracker.cost_by_model

    def test_relative_pricing(self):
        """Test that Claude Opus costs more than Haiku."""
        mock_client = MagicMock()

        tracker = CostTracker(print_summary=False)
        wrapped = track_anthropic(mock_client, tracker)

        # Request with Opus
        mock_client.messages.create.return_value = MockResponse(
            model="claude-3-opus-20240229", input_tokens=1000, output_tokens=500
        )
        wrapped.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )
        opus_cost = tracker.history[0]["cost"]

        # Request with Haiku
        mock_client.messages.create.return_value = MockResponse(
            model="claude-3-haiku-20240307", input_tokens=1000, output_tokens=500
        )
        wrapped.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )
        haiku_cost = tracker.history[1]["cost"]

        # Opus should be significantly more expensive than Haiku
        assert opus_cost > haiku_cost * 5  # Opus is ~20x more expensive


class TestLatestClaudeModels:
    """Tests for the latest Claude 4.x models."""

    def test_claude_opus_4_6(self):
        """Test tracking Claude Opus 4.6 model."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockResponse(
            model="claude-opus-4-6", input_tokens=100, output_tokens=50
        )

        tracker = CostTracker(print_summary=False)
        wrapped = track_anthropic(mock_client, tracker)

        wrapped.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert tracker.total_cost > 0
        assert "claude-opus-4-6" in tracker.cost_by_model

    def test_claude_opus_4_5(self):
        """Test tracking Claude Opus 4.5 model."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockResponse(
            model="claude-opus-4-5-20251101", input_tokens=100, output_tokens=50
        )

        tracker = CostTracker(print_summary=False)
        wrapped = track_anthropic(mock_client, tracker)

        wrapped.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert tracker.total_cost > 0
        assert "claude-opus-4-5-20251101" in tracker.cost_by_model

    def test_claude_sonnet_4_6(self):
        """Test tracking Claude Sonnet 4.6 model."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockResponse(
            model="claude-sonnet-4-6", input_tokens=100, output_tokens=50
        )

        tracker = CostTracker(print_summary=False)
        wrapped = track_anthropic(mock_client, tracker)

        wrapped.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert tracker.total_cost > 0
        assert "claude-sonnet-4-6" in tracker.cost_by_model

    def test_claude_sonnet_4_5(self):
        """Test tracking Claude Sonnet 4.5 model."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockResponse(
            model="claude-sonnet-4-5-20250929", input_tokens=100, output_tokens=50
        )

        tracker = CostTracker(print_summary=False)
        wrapped = track_anthropic(mock_client, tracker)

        wrapped.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert tracker.total_cost > 0
        assert "claude-sonnet-4-5-20250929" in tracker.cost_by_model

    def test_claude_haiku_4_5(self):
        """Test tracking Claude Haiku 4.5 model."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockResponse(
            model="claude-haiku-4-5-20251001", input_tokens=100, output_tokens=50
        )

        tracker = CostTracker(print_summary=False)
        wrapped = track_anthropic(mock_client, tracker)

        wrapped.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert tracker.total_cost > 0
        assert "claude-haiku-4-5-20251001" in tracker.cost_by_model

    def test_claude_4_relative_pricing(self):
        """Test that Claude 4.x Opus costs more than Sonnet which costs more than Haiku."""
        mock_client = MagicMock()

        tracker = CostTracker(print_summary=False)
        wrapped = track_anthropic(mock_client, tracker)

        # Request with Opus 4.5
        mock_client.messages.create.return_value = MockResponse(
            model="claude-opus-4-5-20251101", input_tokens=1000, output_tokens=500
        )
        wrapped.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )
        opus_cost = tracker.history[0]["cost"]

        # Request with Sonnet 4.5
        mock_client.messages.create.return_value = MockResponse(
            model="claude-sonnet-4-5-20250929", input_tokens=1000, output_tokens=500
        )
        wrapped.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )
        sonnet_cost = tracker.history[1]["cost"]

        # Request with Haiku 4.5
        mock_client.messages.create.return_value = MockResponse(
            model="claude-haiku-4-5-20251001", input_tokens=1000, output_tokens=500
        )
        wrapped.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )
        haiku_cost = tracker.history[2]["cost"]

        # Opus > Sonnet > Haiku
        assert opus_cost > sonnet_cost
        assert sonnet_cost > haiku_cost
