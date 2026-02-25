"""Wrapper for Google Gemini SDK (google-genai) integration to track costs automatically.

This wrapper supports the new google-genai SDK which uses a client-based API
similar to OpenAI/Anthropic:

    from google import genai
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Hello, world!"
    )

Note: The old google-generativeai SDK is deprecated. This wrapper only supports
the new google-genai SDK.
"""

import functools
import threading
from typing import TYPE_CHECKING, Any, Callable

from .pricing import calculate_cost

if TYPE_CHECKING:
    from .tracker import CostTracker

# Global state for patching (protected by _global_lock)
_original_generate_content = None
_original_async_generate_content = None
_global_tracker: "CostTracker | None" = None
_global_lock = threading.Lock()


def _get_global_tracker() -> "CostTracker | None":
    """Thread-safe getter for the global tracker."""
    with _global_lock:
        return _global_tracker


def _normalize_model_name(model: str) -> str:
    """Normalize Gemini model name to match litellm format.

    Args:
        model: Model name from response (e.g., "models/gemini-2.0-flash").

    Returns:
        Normalized model name (e.g., "gemini/gemini-2.0-flash").
    """
    # Remove "models/" prefix if present
    if model.startswith("models/"):
        model = model[7:]

    # Add "gemini/" prefix for litellm compatibility if not present
    if not model.startswith("gemini/"):
        model = f"gemini/{model}"

    return model


def _extract_usage(response: Any) -> tuple[str, int, int]:
    """Extract model and token usage from a Gemini response.

    Args:
        response: Gemini GenerateContentResponse object.

    Returns:
        Tuple of (model, prompt_tokens, completion_tokens).
    """
    # Get model name from response
    model = getattr(response, "model", None)
    if model is None:
        # Try to get from candidates
        candidates = getattr(response, "candidates", None)
        if candidates and len(candidates) > 0:
            model = getattr(candidates[0], "model", "unknown")
        else:
            model = "unknown"

    # Extract usage metadata
    usage_metadata = getattr(response, "usage_metadata", None)

    if usage_metadata is None:
        return _normalize_model_name(model), 0, 0

    # Gemini uses prompt_token_count and candidates_token_count
    prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
    completion_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0

    return _normalize_model_name(model), prompt_tokens, completion_tokens


def _extract_streaming_usage(chunk: Any) -> tuple[str, int, int] | None:
    """Extract usage from a streaming chunk (only present in final chunk).

    Args:
        chunk: Gemini streaming chunk object.

    Returns:
        Tuple of (model, prompt_tokens, completion_tokens) if usage present,
        None otherwise.
    """
    usage_metadata = getattr(chunk, "usage_metadata", None)
    if usage_metadata is None:
        return None

    # Check if we have actual token counts (not just metadata object)
    prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
    completion_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0

    # If both are 0, likely not the final chunk with usage info
    if prompt_tokens == 0 and completion_tokens == 0:
        return None

    # Get model name
    model = getattr(chunk, "model", None)
    if model is None:
        candidates = getattr(chunk, "candidates", None)
        if candidates and len(candidates) > 0:
            model = getattr(candidates[0], "model", "unknown")
        else:
            model = "unknown"

    return _normalize_model_name(model), prompt_tokens, completion_tokens


def _record_to_tracker(
    tracker: "CostTracker | None",
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Record cost to tracker if available."""
    if tracker is None:
        return

    try:
        cost = calculate_cost(model, prompt_tokens, completion_tokens)
    except ValueError:
        # Unknown model, skip recording
        return

    tracker.record_cost(model, prompt_tokens, completion_tokens, cost)


def _wrap_streaming_response(stream: Any, tracker: "CostTracker | None") -> Any:
    """Wrap a streaming response to capture usage from final chunk.

    Args:
        stream: Gemini streaming response iterator.
        tracker: CostTracker instance to record costs.

    Yields:
        Original chunks from the stream.
    """
    for chunk in stream:
        yield chunk
        # Check for usage in each chunk (only present in final chunk)
        usage_info = _extract_streaming_usage(chunk)
        if usage_info is not None:
            model, prompt_tokens, completion_tokens = usage_info
            _record_to_tracker(tracker, model, prompt_tokens, completion_tokens)


async def _wrap_async_streaming_response(
    stream: Any, tracker: "CostTracker | None"
) -> Any:
    """Wrap an async streaming response to capture usage from final chunk.

    Args:
        stream: Gemini async streaming response iterator.
        tracker: CostTracker instance to record costs.

    Yields:
        Original chunks from the stream.
    """
    async for chunk in stream:
        yield chunk
        # Check for usage in each chunk (only present in final chunk)
        usage_info = _extract_streaming_usage(chunk)
        if usage_info is not None:
            model, prompt_tokens, completion_tokens = usage_info
            _record_to_tracker(tracker, model, prompt_tokens, completion_tokens)


def _make_sync_wrapped_generate_content_base(
    original_generate_content: Any,
    get_tracker: Callable[[], "CostTracker | None"],
    get_model: Callable[[tuple, dict], str | None] = None,
) -> Any:
    """Base factory for sync generate_content wrappers.

    Args:
        original_generate_content: Original generate_content method.
        get_tracker: Callable that returns the tracker to use.
        get_model: Optional callable to extract model from args/kwargs.

    Returns:
        Wrapped generate_content function.
    """

    @functools.wraps(original_generate_content)
    def wrapped_generate_content(*args: Any, **kwargs: Any) -> Any:
        # Check if streaming is enabled
        config = kwargs.get("config", None)
        is_streaming = False
        if config is not None:
            # Config could be a dict or GenerateContentConfig object
            if isinstance(config, dict):
                is_streaming = config.get("response_modalities") == "stream"
            else:
                is_streaming = getattr(config, "response_modalities", None) == "stream"

        response = original_generate_content(*args, **kwargs)

        tracker = get_tracker()

        # Check if response is iterable (streaming)
        if hasattr(response, "__iter__") and not hasattr(response, "usage_metadata"):
            return _wrap_streaming_response(response, tracker)
        else:
            model, prompt_tokens, completion_tokens = _extract_usage(response)
            # If model is unknown, try to get it from the request
            if model == "gemini/unknown" and get_model is not None:
                request_model = get_model(args, kwargs)
                if request_model:
                    model = _normalize_model_name(request_model)
            _record_to_tracker(tracker, model, prompt_tokens, completion_tokens)
            return response

    return wrapped_generate_content


def _make_async_wrapped_generate_content_base(
    original_generate_content: Any,
    get_tracker: Callable[[], "CostTracker | None"],
    get_model: Callable[[tuple, dict], str | None] = None,
) -> Any:
    """Base factory for async generate_content wrappers."""

    @functools.wraps(original_generate_content)
    async def wrapped_generate_content(*args: Any, **kwargs: Any) -> Any:
        response = await original_generate_content(*args, **kwargs)

        tracker = get_tracker()

        # Check if response is async iterable (streaming)
        if hasattr(response, "__aiter__"):
            return _wrap_async_streaming_response(response, tracker)
        else:
            model, prompt_tokens, completion_tokens = _extract_usage(response)
            # If model is unknown, try to get it from the request
            if model == "gemini/unknown" and get_model is not None:
                request_model = get_model(args, kwargs)
                if request_model:
                    model = _normalize_model_name(request_model)
            _record_to_tracker(tracker, model, prompt_tokens, completion_tokens)
            return response

    return wrapped_generate_content


def _get_model_from_args(args: tuple, kwargs: dict) -> str | None:
    """Extract model name from generate_content args/kwargs.

    The model parameter is the first positional arg or a keyword arg.
    """
    if len(args) > 0:
        return args[0]
    return kwargs.get("model")


def _make_wrapped_generate_content(
    original_generate_content: Any, tracker: "CostTracker | None"
) -> Any:
    """Create a wrapped version of models.generate_content (sync)."""
    return _make_sync_wrapped_generate_content_base(
        original_generate_content,
        lambda: tracker,
        _get_model_from_args,
    )


def _make_wrapped_async_generate_content(
    original_generate_content: Any, tracker: "CostTracker | None"
) -> Any:
    """Create a wrapped version of models.generate_content (async)."""
    return _make_async_wrapped_generate_content_base(
        original_generate_content,
        lambda: tracker,
        _get_model_from_args,
    )


def _make_global_wrapped_generate_content(original_generate_content: Any) -> Any:
    """Create a wrapped version that uses the global tracker (for patch_gemini)."""
    return _make_sync_wrapped_generate_content_base(
        original_generate_content,
        _get_global_tracker,
        _get_model_from_args,
    )


def _make_global_wrapped_async_generate_content(original_generate_content: Any) -> Any:
    """Create a wrapped async version that uses the global tracker."""
    return _make_async_wrapped_generate_content_base(
        original_generate_content,
        _get_global_tracker,
        _get_model_from_args,
    )


class _WrappedModels:
    """Wrapper for client.models resource that tracks costs."""

    def __init__(self, original_models: Any, tracker: "CostTracker | None") -> None:
        self._original = original_models
        self._tracker = tracker

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        return _make_wrapped_generate_content(
            self._original.generate_content, self._tracker
        )(*args, **kwargs)

    async def generate_content_async(self, *args: Any, **kwargs: Any) -> Any:
        return await _make_wrapped_async_generate_content(
            self._original.generate_content_async, self._tracker
        )(*args, **kwargs)


class _WrappedAsyncModels:
    """Wrapper for async client.models resource that tracks costs."""

    def __init__(self, original_models: Any, tracker: "CostTracker | None") -> None:
        self._original = original_models
        self._tracker = tracker

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    async def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        return await _make_wrapped_async_generate_content(
            self._original.generate_content, self._tracker
        )(*args, **kwargs)


class _WrappedClient:
    """Wrapper for genai.Client that tracks costs."""

    def __init__(
        self, client: Any, tracker: "CostTracker | None", is_async: bool = False
    ) -> None:
        self._client = client
        self._tracker = tracker
        self._is_async = is_async

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    @property
    def models(self) -> Any:
        if self._is_async:
            return _WrappedAsyncModels(self._client.models, self._tracker)
        return _WrappedModels(self._client.models, self._tracker)


def track_gemini(client: Any, tracker: "CostTracker | None" = None) -> Any:
    """Wrap Google Gemini genai.Client to track costs.

    Returns a wrapped client with the same interface that automatically
    tracks costs for models.generate_content() calls.

    Note: This wrapper supports the new google-genai SDK (not the deprecated
    google-generativeai SDK).

    Args:
        client: genai.Client instance.
        tracker: CostTracker instance to record costs. If None, costs are
            calculated but not recorded anywhere.

    Returns:
        Wrapped client that tracks costs.

    Example:
        >>> from google import genai
        >>> from tokencost import CostTracker, track_gemini
        >>>
        >>> tracker = CostTracker(budget=1.0)
        >>> client = track_gemini(genai.Client(), tracker)
        >>>
        >>> # Track content generation
        >>> response = client.models.generate_content(
        ...     model="gemini-2.0-flash",
        ...     contents="Hello, world!"
        ... )
        >>> print(f"Total: ${tracker.total_cost:.6f}")
    """
    # Detect if client is async by checking class name
    is_async = "Async" in type(client).__name__
    return _WrappedClient(client, tracker, is_async)


def patch_gemini(tracker: "CostTracker | None" = None) -> None:
    """Globally patch Google Gemini SDK to track all costs.

    After calling this function, all genai.Client instances will
    automatically have their models.generate_content() calls tracked.

    This function can be called multiple times with different trackers.
    Each call updates the active tracker used for cost recording.

    This function is thread-safe.

    Note: This patches the google-genai SDK (not the deprecated
    google-generativeai SDK).

    Args:
        tracker: CostTracker instance to record costs. If None, costs are
            calculated but not recorded.

    Example:
        >>> from tokencost import CostTracker, patch_gemini
        >>> from google import genai
        >>>
        >>> tracker = CostTracker()
        >>> patch_gemini(tracker)
        >>>
        >>> # All Gemini clients now track costs
        >>> client = genai.Client()
        >>> response = client.models.generate_content(
        ...     model="gemini-2.0-flash",
        ...     contents="Hello!"
        ... )
        >>> print(tracker.total_cost)
    """
    global _original_generate_content, _original_async_generate_content, _global_tracker

    try:
        from google.genai import models as genai_models
    except ImportError as e:
        raise ImportError(
            "Google Gemini SDK not installed. Install it with: pip install google-genai"
        ) from e

    with _global_lock:
        # Update the global tracker (can be changed on subsequent calls)
        _global_tracker = tracker

        # Get the Models class
        Models = genai_models.Models

        # Patch sync generate_content (only once)
        if _original_generate_content is None:
            _original_generate_content = Models.generate_content
            Models.generate_content = _make_global_wrapped_generate_content(
                _original_generate_content
            )

        # Patch async generate_content if available (only once)
        if _original_async_generate_content is None:
            if hasattr(Models, "generate_content_async"):
                _original_async_generate_content = Models.generate_content_async
                Models.generate_content_async = (
                    _make_global_wrapped_async_generate_content(
                        _original_async_generate_content
                    )
                )


def unpatch_gemini() -> None:
    """Remove global patches from Google Gemini SDK.

    Restores the original models.generate_content() methods.
    This function is thread-safe.
    """
    global _original_generate_content, _original_async_generate_content, _global_tracker

    with _global_lock:
        if (
            _original_generate_content is None
            and _original_async_generate_content is None
        ):
            return

        try:
            from google.genai import models as genai_models
        except ImportError:
            return

        Models = genai_models.Models

        # Restore methods
        if _original_generate_content is not None:
            Models.generate_content = _original_generate_content
            _original_generate_content = None

        if _original_async_generate_content is not None:
            Models.generate_content_async = _original_async_generate_content
            _original_async_generate_content = None

        _global_tracker = None
