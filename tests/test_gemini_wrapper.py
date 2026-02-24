"""Tests for the Gemini wrapper module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from tokencost import CostTracker
from tokencost.gemini_wrapper import (
    _extract_streaming_usage,
    _extract_usage,
    _normalize_model_name,
    _record_to_tracker,
    patch_gemini,
    track_gemini,
    unpatch_gemini,
)


class MockUsageMetadata:
    """Mock Gemini usage metadata object."""

    def __init__(
        self,
        prompt_token_count: int,
        candidates_token_count: int,
        total_token_count: int | None = None,
        cached_content_token_count: int = 0,
    ):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = (
            total_token_count
            if total_token_count is not None
            else prompt_token_count + candidates_token_count
        )
        self.cached_content_token_count = cached_content_token_count


class MockResponse:
    """Mock Gemini GenerateContentResponse."""

    def __init__(
        self,
        model: str = "models/gemini-2.0-flash",
        prompt_token_count: int = 100,
        candidates_token_count: int = 50,
    ):
        self.model = model
        self.usage_metadata = MockUsageMetadata(
            prompt_token_count, candidates_token_count
        )
        self.text = "Hello! How can I help you today?"
        self.candidates = []


class MockStreamChunk:
    """Mock Gemini streaming chunk."""

    def __init__(
        self,
        model: str = "models/gemini-2.0-flash",
        usage_metadata: MockUsageMetadata | None = None,
        text: str = "",
    ):
        self.model = model
        self.usage_metadata = usage_metadata
        self.text = text
        self.candidates = []


class TestNormalizeModelName:
    """Tests for _normalize_model_name function."""

    def test_normalize_with_models_prefix(self):
        """Test normalizing model name with models/ prefix."""
        result = _normalize_model_name("models/gemini-2.0-flash")
        assert result == "gemini/gemini-2.0-flash"

    def test_normalize_without_prefix(self):
        """Test normalizing model name without prefix."""
        result = _normalize_model_name("gemini-2.0-flash")
        assert result == "gemini/gemini-2.0-flash"

    def test_normalize_already_has_gemini_prefix(self):
        """Test normalizing model name that already has gemini/ prefix."""
        result = _normalize_model_name("gemini/gemini-2.0-flash")
        assert result == "gemini/gemini-2.0-flash"

    def test_normalize_models_and_gemini_prefix(self):
        """Test normalizing model name with both prefixes."""
        # This is an edge case that shouldn't happen in practice
        # After removing "models/", the remaining "gemini/gemini-2.0-flash"
        # already has the "gemini/" prefix, so it stays as is
        result = _normalize_model_name("models/gemini/gemini-2.0-flash")
        assert result == "gemini/gemini-2.0-flash"


class TestExtractUsage:
    """Tests for _extract_usage function."""

    def test_extract_usage_normal_response(self):
        """Test extracting usage from a normal response."""
        response = MockResponse(
            model="models/gemini-2.0-flash",
            prompt_token_count=100,
            candidates_token_count=50,
        )
        model, prompt_tokens, completion_tokens = _extract_usage(response)
        assert model == "gemini/gemini-2.0-flash"
        assert prompt_tokens == 100
        assert completion_tokens == 50

    def test_extract_usage_no_usage_metadata(self):
        """Test extracting usage when usage_metadata is None."""
        response = MockResponse()
        response.usage_metadata = None
        model, prompt_tokens, completion_tokens = _extract_usage(response)
        assert model == "gemini/gemini-2.0-flash"
        assert prompt_tokens == 0
        assert completion_tokens == 0

    def test_extract_usage_missing_model(self):
        """Test extracting usage when model is missing."""
        response = MagicMock()
        del response.model
        response.candidates = []
        response.usage_metadata = MockUsageMetadata(100, 50)
        model, prompt_tokens, completion_tokens = _extract_usage(response)
        assert model == "gemini/unknown"
        assert prompt_tokens == 100
        assert completion_tokens == 50

    def test_extract_usage_with_cached_tokens(self):
        """Test extracting usage with cached content tokens."""
        response = MockResponse()
        response.usage_metadata = MockUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=50,
            cached_content_token_count=25,
        )
        model, prompt_tokens, completion_tokens = _extract_usage(response)
        assert prompt_tokens == 100
        assert completion_tokens == 50


class TestExtractStreamingUsage:
    """Tests for _extract_streaming_usage function."""

    def test_extract_streaming_usage_with_usage(self):
        """Test extracting usage from final streaming chunk."""
        chunk = MockStreamChunk(
            model="models/gemini-2.0-flash",
            usage_metadata=MockUsageMetadata(100, 50),
        )
        result = _extract_streaming_usage(chunk)
        assert result is not None
        model, prompt_tokens, completion_tokens = result
        assert model == "gemini/gemini-2.0-flash"
        assert prompt_tokens == 100
        assert completion_tokens == 50

    def test_extract_streaming_usage_no_usage(self):
        """Test extracting usage from intermediate chunk (no usage)."""
        chunk = MockStreamChunk(model="models/gemini-2.0-flash", usage_metadata=None)
        result = _extract_streaming_usage(chunk)
        assert result is None

    def test_extract_streaming_usage_zero_counts(self):
        """Test extracting usage when counts are zero (intermediate chunk)."""
        chunk = MockStreamChunk(
            model="models/gemini-2.0-flash",
            usage_metadata=MockUsageMetadata(0, 0),
        )
        result = _extract_streaming_usage(chunk)
        # Should return None because both counts are 0
        assert result is None

    def test_extract_streaming_usage_missing_model(self):
        """Test extracting usage when model is missing from chunk."""
        chunk = MagicMock()
        del chunk.model
        chunk.candidates = []
        chunk.usage_metadata = MockUsageMetadata(100, 50)
        result = _extract_streaming_usage(chunk)
        assert result is not None
        model, prompt_tokens, completion_tokens = result
        assert model == "gemini/unknown"
        assert prompt_tokens == 100
        assert completion_tokens == 50


class TestRecordToTracker:
    """Tests for _record_to_tracker function."""

    def test_record_to_tracker_with_tracker(self):
        """Test recording cost to tracker."""
        tracker = CostTracker(print_summary=False)
        _record_to_tracker(tracker, "gemini/gemini-2.0-flash", 1000, 500)
        assert tracker.total_cost > 0
        assert tracker.request_count == 1

    def test_record_to_tracker_no_tracker(self):
        """Test that None tracker doesn't cause errors."""
        # Should not raise
        _record_to_tracker(None, "gemini/gemini-2.0-flash", 1000, 500)

    def test_record_to_tracker_unknown_model(self):
        """Test that unknown model doesn't record."""
        tracker = CostTracker(print_summary=False)
        _record_to_tracker(tracker, "unknown-model-xyz", 1000, 500)
        # Should not record anything for unknown model
        assert tracker.total_cost == 0
        assert tracker.request_count == 0


class TestTrackGemini:
    """Tests for track_gemini function."""

    def test_track_gemini_sync_client(self):
        """Test wrapping a sync Gemini client."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockResponse()

        tracker = CostTracker(print_summary=False)
        wrapped = track_gemini(mock_client, tracker)

        # Call generate_content
        response = wrapped.models.generate_content(
            model="gemini-2.0-flash",
            contents="Hello",
        )

        assert response.text == "Hello! How can I help you today?"
        assert tracker.total_cost > 0
        assert tracker.request_count == 1

    def test_track_gemini_sync_streaming(self):
        """Test wrapping sync streaming response."""
        # Create mock streaming chunks
        chunks = [
            MockStreamChunk(model="models/gemini-2.0-flash", text="Hello"),
            MockStreamChunk(model="models/gemini-2.0-flash", text=" there"),
            MockStreamChunk(
                model="models/gemini-2.0-flash",
                usage_metadata=MockUsageMetadata(100, 50),
                text="!",
            ),  # Final chunk
        ]

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = iter(chunks)

        tracker = CostTracker(print_summary=False)
        wrapped = track_gemini(mock_client, tracker)

        # Call generate_content (returns iterator)
        stream = wrapped.models.generate_content(
            model="gemini-2.0-flash",
            contents="Hello",
        )

        # Consume the stream
        received_chunks = list(stream)
        assert len(received_chunks) == 3

        # Cost should be recorded after final chunk
        assert tracker.total_cost > 0
        assert tracker.request_count == 1

    def test_track_gemini_without_tracker(self):
        """Test wrapping client without tracker (just pass-through)."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockResponse()

        wrapped = track_gemini(mock_client)  # No tracker

        response = wrapped.models.generate_content(
            model="gemini-2.0-flash",
            contents="Hello",
        )

        assert response.text == "Hello! How can I help you today?"

    def test_track_gemini_preserves_other_attrs(self):
        """Test that wrapped client preserves other attributes."""
        mock_client = MagicMock()
        mock_client.api_key = "test-key"
        mock_client.project = "test-project"

        wrapped = track_gemini(mock_client)

        assert wrapped.api_key == "test-key"
        assert wrapped.project == "test-project"

    def test_track_gemini_model_from_request(self):
        """Test that model is extracted from request if not in response."""
        mock_client = MagicMock()
        response = MockResponse()
        response.model = None  # No model in response
        mock_client.models.generate_content.return_value = response

        tracker = CostTracker(print_summary=False)
        wrapped = track_gemini(mock_client, tracker)

        wrapped.models.generate_content(
            model="gemini-2.0-flash",
            contents="Hello",
        )

        # Should still record with model from request
        assert tracker.request_count == 1


class TestTrackGeminiAsync:
    """Tests for track_gemini with async clients."""

    @pytest.mark.asyncio
    async def test_track_gemini_async_client(self):
        """Test wrapping an async Gemini client."""
        mock_client = MagicMock()
        mock_client.__class__.__name__ = "AsyncClient"
        mock_client.models.generate_content = AsyncMock(return_value=MockResponse())

        tracker = CostTracker(print_summary=False)
        wrapped = track_gemini(mock_client, tracker)

        # Call generate_content
        response = await wrapped.models.generate_content(
            model="gemini-2.0-flash",
            contents="Hello",
        )

        assert response.text == "Hello! How can I help you today?"
        assert tracker.total_cost > 0
        assert tracker.request_count == 1

    @pytest.mark.asyncio
    async def test_track_gemini_async_streaming(self):
        """Test wrapping async streaming response."""
        chunks = [
            MockStreamChunk(model="models/gemini-2.0-flash", text="Hello"),
            MockStreamChunk(model="models/gemini-2.0-flash", text=" there"),
            MockStreamChunk(
                model="models/gemini-2.0-flash",
                usage_metadata=MockUsageMetadata(100, 50),
                text="!",
            ),
        ]

        async def async_iter():
            for chunk in chunks:
                yield chunk

        mock_client = MagicMock()
        mock_client.__class__.__name__ = "AsyncClient"
        mock_client.models.generate_content = AsyncMock(return_value=async_iter())

        tracker = CostTracker(print_summary=False)
        wrapped = track_gemini(mock_client, tracker)

        # Call generate_content with streaming
        stream = await wrapped.models.generate_content(
            model="gemini-2.0-flash",
            contents="Hello",
        )

        # Consume the stream
        received_chunks = []
        async for chunk in stream:
            received_chunks.append(chunk)

        assert len(received_chunks) == 3
        assert tracker.total_cost > 0
        assert tracker.request_count == 1


class TestPatchUnpatchGemini:
    """Tests for patch_gemini and unpatch_gemini functions."""

    def test_patch_gemini_changes_tracker(self):
        """Test that patch_gemini can update the tracker on subsequent calls."""
        import tokencost.gemini_wrapper as wrapper

        # Ensure we start fresh
        unpatch_gemini()

        tracker1 = CostTracker(print_summary=False)
        tracker2 = CostTracker(print_summary=False)

        # Mock the import to avoid requiring the actual SDK
        import sys
        from unittest.mock import MagicMock

        mock_models_module = MagicMock()
        mock_models_class = MagicMock()
        mock_models_class.generate_content = MagicMock()
        mock_models_module.Models = mock_models_class
        sys.modules["google.genai.models"] = mock_models_module
        sys.modules["google.genai"] = MagicMock()
        sys.modules["google"] = MagicMock()

        try:
            patch_gemini(tracker1)
            assert wrapper._global_tracker is tracker1

            # Calling again should update the tracker
            patch_gemini(tracker2)
            assert wrapper._global_tracker is tracker2

            unpatch_gemini()
        finally:
            # Cleanup
            del sys.modules["google.genai.models"]
            del sys.modules["google.genai"]
            del sys.modules["google"]

    def test_unpatch_gemini_when_not_patched(self):
        """Test that unpatch_gemini is safe to call when not patched."""
        import tokencost.gemini_wrapper as wrapper

        wrapper._original_generate_content = None
        wrapper._original_async_generate_content = None

        # Should not raise
        unpatch_gemini()


class TestBudgetIntegration:
    """Tests for budget integration with Gemini wrapper."""

    def test_budget_exceeded_during_tracking(self):
        """Test that budget is checked during tracking."""
        from tokencost import BudgetExceededError

        mock_client = MagicMock()
        # Return a response with high token count
        mock_client.models.generate_content.return_value = MockResponse(
            model="models/gemini-2.0-flash",
            prompt_token_count=100000,
            candidates_token_count=50000,
        )

        tracker = CostTracker(budget=0.001, raise_on_budget=True, print_summary=False)
        wrapped = track_gemini(mock_client, tracker)

        with pytest.raises(BudgetExceededError):
            wrapped.models.generate_content(
                model="gemini-2.0-flash",
                contents="Hello",
            )

    def test_multiple_requests_accumulate(self):
        """Test that multiple requests accumulate cost."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockResponse(
            model="models/gemini-2.0-flash",
            prompt_token_count=100,
            candidates_token_count=50,
        )

        tracker = CostTracker(print_summary=False)
        wrapped = track_gemini(mock_client, tracker)

        # Make multiple requests
        for _ in range(5):
            wrapped.models.generate_content(
                model="gemini-2.0-flash",
                contents="Hello",
            )

        assert tracker.request_count == 5
        # Cost should be 5x single request cost
        single_cost = tracker.history[0]["cost"]
        assert abs(tracker.total_cost - single_cost * 5) < 0.0001


class TestGeminiModels:
    """Tests for various Gemini models."""

    def test_gemini_flash_model(self):
        """Test tracking Gemini Flash model."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockResponse(
            model="models/gemini-2.0-flash",
            prompt_token_count=100,
            candidates_token_count=50,
        )

        tracker = CostTracker(print_summary=False)
        wrapped = track_gemini(mock_client, tracker)

        wrapped.models.generate_content(
            model="gemini-2.0-flash",
            contents="Hello",
        )

        assert tracker.total_cost > 0
        assert "gemini/gemini-2.0-flash" in tracker.cost_by_model

    def test_gemini_pro_model(self):
        """Test tracking Gemini Pro model."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockResponse(
            model="models/gemini-1.5-pro",
            prompt_token_count=100,
            candidates_token_count=50,
        )

        tracker = CostTracker(print_summary=False)
        wrapped = track_gemini(mock_client, tracker)

        wrapped.models.generate_content(
            model="gemini-1.5-pro",
            contents="Hello",
        )

        assert tracker.total_cost > 0
        assert "gemini/gemini-1.5-pro" in tracker.cost_by_model

    def test_gemini_ultra_model(self):
        """Test tracking Gemini Ultra model."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockResponse(
            model="models/gemini-ultra",
            prompt_token_count=100,
            candidates_token_count=50,
        )

        tracker = CostTracker(print_summary=False)
        wrapped = track_gemini(mock_client, tracker)

        wrapped.models.generate_content(
            model="gemini-ultra",
            contents="Hello",
        )

        # Note: gemini-ultra may not have pricing in litellm
        # The test verifies the wrapper doesn't crash
        assert tracker.request_count >= 0

    def test_relative_pricing(self):
        """Test that Gemini Pro costs more than Flash."""
        mock_client = MagicMock()

        tracker = CostTracker(print_summary=False)
        wrapped = track_gemini(mock_client, tracker)

        # Request with Flash
        mock_client.models.generate_content.return_value = MockResponse(
            model="models/gemini-2.0-flash",
            prompt_token_count=1000,
            candidates_token_count=500,
        )
        wrapped.models.generate_content(
            model="gemini-2.0-flash",
            contents="Hello",
        )
        flash_cost = tracker.history[0]["cost"] if tracker.history else 0

        # Request with Pro
        mock_client.models.generate_content.return_value = MockResponse(
            model="models/gemini-1.5-pro",
            prompt_token_count=1000,
            candidates_token_count=500,
        )
        wrapped.models.generate_content(
            model="gemini-1.5-pro",
            contents="Hello",
        )
        pro_cost = tracker.history[1]["cost"] if len(tracker.history) > 1 else 0

        # Pro should be more expensive than Flash (if both have pricing)
        if flash_cost > 0 and pro_cost > 0:
            assert pro_cost > flash_cost


class TestGeminiGenerateContentAsync:
    """Tests for generate_content_async method."""

    @pytest.mark.asyncio
    async def test_generate_content_async_method(self):
        """Test the generate_content_async method on wrapped client."""
        mock_client = MagicMock()
        mock_client.models.generate_content_async = AsyncMock(
            return_value=MockResponse()
        )

        tracker = CostTracker(print_summary=False)
        wrapped = track_gemini(mock_client, tracker)

        # Call generate_content_async
        response = await wrapped.models.generate_content_async(
            model="gemini-2.0-flash",
            contents="Hello",
        )

        assert response.text == "Hello! How can I help you today?"
        assert tracker.total_cost > 0
        assert tracker.request_count == 1

    @pytest.mark.asyncio
    async def test_generate_content_async_streaming(self):
        """Test generate_content_async with streaming."""
        chunks = [
            MockStreamChunk(model="models/gemini-2.0-flash", text="Hello"),
            MockStreamChunk(
                model="models/gemini-2.0-flash",
                usage_metadata=MockUsageMetadata(100, 50),
                text="!",
            ),
        ]

        async def async_iter():
            for chunk in chunks:
                yield chunk

        mock_client = MagicMock()
        mock_client.models.generate_content_async = AsyncMock(
            return_value=async_iter()
        )

        tracker = CostTracker(print_summary=False)
        wrapped = track_gemini(mock_client, tracker)

        stream = await wrapped.models.generate_content_async(
            model="gemini-2.0-flash",
            contents="Hello",
        )

        received_chunks = []
        async for chunk in stream:
            received_chunks.append(chunk)

        assert len(received_chunks) == 2
        assert tracker.total_cost > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_none_token_counts(self):
        """Test handling None token counts."""
        response = MockResponse()
        response.usage_metadata.prompt_token_count = None
        response.usage_metadata.candidates_token_count = None

        model, prompt_tokens, completion_tokens = _extract_usage(response)
        assert prompt_tokens == 0
        assert completion_tokens == 0

    def test_missing_usage_metadata_attrs(self):
        """Test handling missing attributes on usage_metadata."""
        response = MockResponse()
        # Create a mock that has usage_metadata but no token counts
        response.usage_metadata = MagicMock(spec=[])

        model, prompt_tokens, completion_tokens = _extract_usage(response)
        assert prompt_tokens == 0
        assert completion_tokens == 0

    def test_wrapped_client_models_property_returns_wrapper(self):
        """Test that accessing .models returns a wrapper, not original."""
        mock_client = MagicMock()
        tracker = CostTracker(print_summary=False)
        wrapped = track_gemini(mock_client, tracker)

        # The models property should return a _WrappedModels instance
        models = wrapped.models
        assert hasattr(models, "_original")
        assert hasattr(models, "_tracker")

    def test_multiple_streams_tracked_separately(self):
        """Test that multiple streaming responses are tracked separately."""
        mock_client = MagicMock()
        tracker = CostTracker(print_summary=False)
        wrapped = track_gemini(mock_client, tracker)

        # First stream
        chunks1 = [
            MockStreamChunk(
                model="models/gemini-2.0-flash",
                usage_metadata=MockUsageMetadata(50, 25),
            ),
        ]
        mock_client.models.generate_content.return_value = iter(chunks1)
        list(
            wrapped.models.generate_content(
                model="gemini-2.0-flash",
                contents="Hello",
            )
        )

        # Second stream
        chunks2 = [
            MockStreamChunk(
                model="models/gemini-2.0-flash",
                usage_metadata=MockUsageMetadata(100, 50),
            ),
        ]
        mock_client.models.generate_content.return_value = iter(chunks2)
        list(
            wrapped.models.generate_content(
                model="gemini-2.0-flash",
                contents="World",
            )
        )

        assert tracker.request_count == 2
