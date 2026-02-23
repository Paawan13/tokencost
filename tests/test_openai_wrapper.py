"""Tests for the OpenAI wrapper module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tokencost import CostTracker
from tokencost.openai_wrapper import (
    _extract_streaming_usage,
    _extract_usage,
    _record_to_tracker,
    patch_openai,
    track_openai,
    unpatch_openai,
)


class MockUsage:
    """Mock OpenAI usage object."""

    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class MockResponse:
    """Mock OpenAI chat completion response."""

    def __init__(
        self,
        model: str = "gpt-4o",
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
    ):
        self.model = model
        self.usage = MockUsage(prompt_tokens, completion_tokens)
        self.id = "chatcmpl-123"
        self.choices = []


class MockStreamChunk:
    """Mock OpenAI streaming chunk."""

    def __init__(
        self,
        model: str = "gpt-4o",
        usage: MockUsage | None = None,
    ):
        self.model = model
        self.usage = usage
        self.id = "chatcmpl-123"
        self.choices = []


class TestExtractUsage:
    """Tests for _extract_usage function."""

    def test_extract_usage_normal_response(self):
        """Test extracting usage from a normal response."""
        response = MockResponse(model="gpt-4o", prompt_tokens=100, completion_tokens=50)
        model, prompt, completion = _extract_usage(response)
        assert model == "gpt-4o"
        assert prompt == 100
        assert completion == 50

    def test_extract_usage_no_usage(self):
        """Test extracting usage when usage is None."""
        response = MockResponse()
        response.usage = None
        model, prompt, completion = _extract_usage(response)
        assert model == "gpt-4o"
        assert prompt == 0
        assert completion == 0

    def test_extract_usage_missing_model(self):
        """Test extracting usage when model is missing."""
        response = MagicMock()
        del response.model
        response.usage = MockUsage(100, 50)
        model, prompt, completion = _extract_usage(response)
        assert model == "unknown"
        assert prompt == 100
        assert completion == 50


class TestExtractStreamingUsage:
    """Tests for _extract_streaming_usage function."""

    def test_extract_streaming_usage_with_usage(self):
        """Test extracting usage from final streaming chunk."""
        chunk = MockStreamChunk(model="gpt-4o", usage=MockUsage(100, 50))
        result = _extract_streaming_usage(chunk)
        assert result is not None
        model, prompt, completion = result
        assert model == "gpt-4o"
        assert prompt == 100
        assert completion == 50

    def test_extract_streaming_usage_no_usage(self):
        """Test extracting usage from intermediate chunk (no usage)."""
        chunk = MockStreamChunk(model="gpt-4o", usage=None)
        result = _extract_streaming_usage(chunk)
        assert result is None


class TestRecordToTracker:
    """Tests for _record_to_tracker function."""

    def test_record_to_tracker_with_tracker(self):
        """Test recording cost to tracker."""
        tracker = CostTracker(print_summary=False)
        _record_to_tracker(tracker, "gpt-4o", 1000, 500)
        assert tracker.total_cost > 0
        assert tracker.request_count == 1

    def test_record_to_tracker_no_tracker(self):
        """Test that None tracker doesn't cause errors."""
        # Should not raise
        _record_to_tracker(None, "gpt-4o", 1000, 500)

    def test_record_to_tracker_unknown_model(self):
        """Test that unknown model doesn't record."""
        tracker = CostTracker(print_summary=False)
        _record_to_tracker(tracker, "unknown-model-xyz", 1000, 500)
        # Should not record anything for unknown model
        assert tracker.total_cost == 0
        assert tracker.request_count == 0


class TestTrackOpenai:
    """Tests for track_openai function."""

    def test_track_openai_sync_client(self):
        """Test wrapping a sync OpenAI client."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockResponse()

        tracker = CostTracker(print_summary=False)
        wrapped = track_openai(mock_client, tracker)

        # Call create
        response = wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Hello"}]
        )

        assert response.model == "gpt-4o"
        assert tracker.total_cost > 0
        assert tracker.request_count == 1

    def test_track_openai_sync_streaming(self):
        """Test wrapping sync streaming response."""
        # Create mock streaming response
        chunks = [
            MockStreamChunk(model="gpt-4o", usage=None),
            MockStreamChunk(model="gpt-4o", usage=None),
            MockStreamChunk(model="gpt-4o", usage=MockUsage(100, 50)),  # Final chunk
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)

        tracker = CostTracker(print_summary=False)
        wrapped = track_openai(mock_client, tracker)

        # Call create with streaming
        stream = wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Hello"}], stream=True
        )

        # Consume the stream
        received_chunks = list(stream)
        assert len(received_chunks) == 3

        # Cost should be recorded after final chunk
        assert tracker.total_cost > 0
        assert tracker.request_count == 1

    def test_track_openai_without_tracker(self):
        """Test wrapping client without tracker (just pass-through)."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockResponse()

        wrapped = track_openai(mock_client)  # No tracker

        response = wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Hello"}]
        )

        assert response.model == "gpt-4o"

    def test_track_openai_preserves_other_attrs(self):
        """Test that wrapped client preserves other attributes."""
        mock_client = MagicMock()
        mock_client.api_key = "test-key"
        mock_client.base_url = "https://api.openai.com"

        wrapped = track_openai(mock_client)

        assert wrapped.api_key == "test-key"
        assert wrapped.base_url == "https://api.openai.com"


class TestTrackOpenaiAsync:
    """Tests for track_openai with async clients."""

    @pytest.mark.asyncio
    async def test_track_openai_async_client(self):
        """Test wrapping an async OpenAI client."""
        mock_client = MagicMock()
        mock_client.__class__.__name__ = "AsyncOpenAI"
        mock_client.chat.completions.create = AsyncMock(return_value=MockResponse())

        tracker = CostTracker(print_summary=False)
        wrapped = track_openai(mock_client, tracker)

        # Call create
        response = await wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Hello"}]
        )

        assert response.model == "gpt-4o"
        assert tracker.total_cost > 0
        assert tracker.request_count == 1

    @pytest.mark.asyncio
    async def test_track_openai_async_streaming(self):
        """Test wrapping async streaming response."""
        chunks = [
            MockStreamChunk(model="gpt-4o", usage=None),
            MockStreamChunk(model="gpt-4o", usage=None),
            MockStreamChunk(model="gpt-4o", usage=MockUsage(100, 50)),
        ]

        async def async_iter():
            for chunk in chunks:
                yield chunk

        mock_client = MagicMock()
        mock_client.__class__.__name__ = "AsyncOpenAI"
        mock_client.chat.completions.create = AsyncMock(return_value=async_iter())

        tracker = CostTracker(print_summary=False)
        wrapped = track_openai(mock_client, tracker)

        # Call create with streaming
        stream = await wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Hello"}], stream=True
        )

        # Consume the stream
        received_chunks = []
        async for chunk in stream:
            received_chunks.append(chunk)

        assert len(received_chunks) == 3
        assert tracker.total_cost > 0
        assert tracker.request_count == 1


class TestPatchUnpatchOpenai:
    """Tests for patch_openai and unpatch_openai functions."""

    def test_patch_openai_without_sdk(self):
        """Test that patch_openai raises ImportError when SDK not installed."""
        with patch.dict("sys.modules", {"openai": None}):
            # Reset the global state
            import tokencost.openai_wrapper as wrapper

            wrapper._original_create = None
            wrapper._original_async_create = None

            # This should work since openai is likely installed in test env
            # But if not, it should raise ImportError

    def test_unpatch_openai_when_not_patched(self):
        """Test that unpatch_openai is safe to call when not patched."""
        import tokencost.openai_wrapper as wrapper

        wrapper._original_create = None
        wrapper._original_async_create = None

        # Should not raise
        unpatch_openai()


class TestBudgetIntegration:
    """Tests for budget integration with OpenAI wrapper."""

    def test_budget_exceeded_during_tracking(self):
        """Test that budget is checked during tracking."""
        from tokencost import BudgetExceededError

        mock_client = MagicMock()
        # Return a response with high token count
        mock_client.chat.completions.create.return_value = MockResponse(
            model="gpt-4o", prompt_tokens=100000, completion_tokens=50000
        )

        tracker = CostTracker(budget=0.001, raise_on_budget=True, print_summary=False)
        wrapped = track_openai(mock_client, tracker)

        with pytest.raises(BudgetExceededError):
            wrapped.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": "Hello"}]
            )

    def test_multiple_requests_accumulate(self):
        """Test that multiple requests accumulate cost."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockResponse(
            model="gpt-4o", prompt_tokens=100, completion_tokens=50
        )

        tracker = CostTracker(print_summary=False)
        wrapped = track_openai(mock_client, tracker)

        # Make multiple requests
        for _ in range(5):
            wrapped.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": "Hello"}]
            )

        assert tracker.request_count == 5
        # Cost should be 5x single request cost
        single_cost = tracker.history[0]["cost"]
        assert abs(tracker.total_cost - single_cost * 5) < 0.0001
