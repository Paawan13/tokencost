"""Wrapper for OpenAI SDK integration to track costs automatically."""

import functools
from typing import TYPE_CHECKING, Any

from .pricing import calculate_cost

if TYPE_CHECKING:
    from .tracker import CostTracker

# Global state for patching
_original_create = None
_original_async_create = None
_global_tracker: "CostTracker | None" = None


def _extract_usage(response: Any) -> tuple[str, int, int]:
    """Extract model and token usage from an OpenAI response.

    Args:
        response: OpenAI ChatCompletion response object.

    Returns:
        Tuple of (model, prompt_tokens, completion_tokens).
    """
    model = getattr(response, "model", "unknown")
    usage = getattr(response, "usage", None)

    if usage is None:
        return model, 0, 0

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    return model, prompt_tokens, completion_tokens


def _extract_streaming_usage(chunk: Any) -> tuple[str, int, int] | None:
    """Extract usage from a streaming chunk (only present in final chunk).

    Args:
        chunk: OpenAI ChatCompletionChunk object.

    Returns:
        Tuple of (model, prompt_tokens, completion_tokens) if usage present,
        None otherwise.
    """
    usage = getattr(chunk, "usage", None)
    if usage is None:
        return None

    model = getattr(chunk, "model", "unknown")
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    return model, prompt_tokens, completion_tokens


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
        stream: OpenAI streaming response iterator.
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
        stream: OpenAI async streaming response iterator.
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


def _make_wrapped_create(original_create: Any, tracker: "CostTracker | None") -> Any:
    """Create a wrapped version of chat.completions.create (sync)."""

    @functools.wraps(original_create)
    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        # For streaming, inject stream_options to include usage
        is_streaming = kwargs.get("stream", False)
        if is_streaming:
            # Copy to avoid modifying caller's dict
            stream_options = dict(kwargs.get("stream_options", {}) or {})
            stream_options["include_usage"] = True
            kwargs["stream_options"] = stream_options

        response = original_create(*args, **kwargs)

        if is_streaming:
            return _wrap_streaming_response(response, tracker)
        else:
            model, prompt_tokens, completion_tokens = _extract_usage(response)
            _record_to_tracker(tracker, model, prompt_tokens, completion_tokens)
            return response

    return wrapped_create


def _make_wrapped_async_create(
    original_create: Any, tracker: "CostTracker | None"
) -> Any:
    """Create a wrapped version of chat.completions.create (async)."""

    @functools.wraps(original_create)
    async def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        # For streaming, inject stream_options to include usage
        is_streaming = kwargs.get("stream", False)
        if is_streaming:
            # Copy to avoid modifying caller's dict
            stream_options = dict(kwargs.get("stream_options", {}) or {})
            stream_options["include_usage"] = True
            kwargs["stream_options"] = stream_options

        response = await original_create(*args, **kwargs)

        if is_streaming:
            return _wrap_async_streaming_response(response, tracker)
        else:
            model, prompt_tokens, completion_tokens = _extract_usage(response)
            _record_to_tracker(tracker, model, prompt_tokens, completion_tokens)
            return response

    return wrapped_create


def _make_global_wrapped_create(original_create: Any) -> Any:
    """Create a wrapped version that uses the global tracker (for patch_openai)."""

    @functools.wraps(original_create)
    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        # For streaming, inject stream_options to include usage
        is_streaming = kwargs.get("stream", False)
        if is_streaming:
            # Copy to avoid modifying caller's dict
            stream_options = dict(kwargs.get("stream_options", {}) or {})
            stream_options["include_usage"] = True
            kwargs["stream_options"] = stream_options

        response = original_create(*args, **kwargs)

        # Use global tracker (looked up at call time, not closure time)
        tracker = _global_tracker
        if is_streaming:
            return _wrap_streaming_response(response, tracker)
        else:
            model, prompt_tokens, completion_tokens = _extract_usage(response)
            _record_to_tracker(tracker, model, prompt_tokens, completion_tokens)
            return response

    return wrapped_create


def _make_global_wrapped_async_create(original_create: Any) -> Any:
    """Create a wrapped async version that uses the global tracker (for patch_openai)."""

    @functools.wraps(original_create)
    async def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        # For streaming, inject stream_options to include usage
        is_streaming = kwargs.get("stream", False)
        if is_streaming:
            # Copy to avoid modifying caller's dict
            stream_options = dict(kwargs.get("stream_options", {}) or {})
            stream_options["include_usage"] = True
            kwargs["stream_options"] = stream_options

        response = await original_create(*args, **kwargs)

        # Use global tracker (looked up at call time, not closure time)
        tracker = _global_tracker
        if is_streaming:
            return _wrap_async_streaming_response(response, tracker)
        else:
            model, prompt_tokens, completion_tokens = _extract_usage(response)
            _record_to_tracker(tracker, model, prompt_tokens, completion_tokens)
            return response

    return wrapped_create


class _WrappedCompletions:
    """Wrapper for client.chat.completions that tracks costs."""

    def __init__(self, original_completions: Any, tracker: "CostTracker | None") -> None:
        self._original = original_completions
        self._tracker = tracker

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return _make_wrapped_create(self._original.create, self._tracker)(
            *args, **kwargs
        )


class _WrappedAsyncCompletions:
    """Wrapper for async client.chat.completions that tracks costs."""

    def __init__(self, original_completions: Any, tracker: "CostTracker | None") -> None:
        self._original = original_completions
        self._tracker = tracker

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        return await _make_wrapped_async_create(self._original.create, self._tracker)(
            *args, **kwargs
        )


class _WrappedChat:
    """Wrapper for client.chat that provides wrapped completions."""

    def __init__(
        self, original_chat: Any, tracker: "CostTracker | None", is_async: bool = False
    ) -> None:
        self._original = original_chat
        self._tracker = tracker
        self._is_async = is_async

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    @property
    def completions(self) -> Any:
        if self._is_async:
            return _WrappedAsyncCompletions(self._original.completions, self._tracker)
        return _WrappedCompletions(self._original.completions, self._tracker)


class _WrappedClient:
    """Wrapper for OpenAI client that tracks costs."""

    def __init__(
        self, client: Any, tracker: "CostTracker | None", is_async: bool = False
    ) -> None:
        self._client = client
        self._tracker = tracker
        self._is_async = is_async

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    @property
    def chat(self) -> _WrappedChat:
        return _WrappedChat(self._client.chat, self._tracker, self._is_async)


def track_openai(client: Any, tracker: "CostTracker | None" = None) -> Any:
    """Wrap OpenAI/AsyncOpenAI client to track costs.

    Returns a wrapped client with the same interface that automatically
    tracks costs for chat.completions.create() calls.

    Args:
        client: OpenAI or AsyncOpenAI client instance.
        tracker: CostTracker instance to record costs. If None, costs are
            calculated but not recorded anywhere.

    Returns:
        Wrapped client that tracks costs.

    Example:
        >>> from openai import OpenAI
        >>> from tokencost import CostTracker, track_openai
        >>>
        >>> tracker = CostTracker(budget=1.0)
        >>> client = track_openai(OpenAI(), tracker)
        >>>
        >>> response = client.chat.completions.create(
        ...     model="gpt-4o",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
        >>> print(tracker.total_cost)
    """
    # Detect if client is async by checking class name
    is_async = "Async" in type(client).__name__
    return _WrappedClient(client, tracker, is_async)


def patch_openai(tracker: "CostTracker | None" = None) -> None:
    """Globally patch OpenAI SDK to track all costs.

    After calling this function, all OpenAI client instances will
    automatically have their chat.completions.create() calls tracked.

    This function can be called multiple times with different trackers.
    Each call updates the active tracker used for cost recording.

    Args:
        tracker: CostTracker instance to record costs. If None, costs are
            calculated but not recorded.

    Example:
        >>> from tokencost import CostTracker, patch_openai
        >>> from openai import OpenAI
        >>>
        >>> tracker = CostTracker()
        >>> patch_openai(tracker)
        >>>
        >>> # All OpenAI clients now track costs
        >>> client = OpenAI()
        >>> response = client.chat.completions.create(...)
        >>> print(tracker.total_cost)
    """
    global _original_create, _original_async_create, _global_tracker

    try:
        from openai.resources.chat.completions import (
            AsyncCompletions,
            Completions,
        )
    except ImportError as e:
        raise ImportError(
            "OpenAI SDK not installed. Install it with: pip install openai"
        ) from e

    # Update the global tracker (can be changed on subsequent calls)
    _global_tracker = tracker

    # Patch sync completions (only once)
    if _original_create is None:
        _original_create = Completions.create
        Completions.create = _make_global_wrapped_create(_original_create)

    # Patch async completions (only once)
    if _original_async_create is None:
        _original_async_create = AsyncCompletions.create
        AsyncCompletions.create = _make_global_wrapped_async_create(
            _original_async_create
        )


def unpatch_openai() -> None:
    """Remove global patches from OpenAI SDK.

    Restores the original chat.completions.create() methods.
    """
    global _original_create, _original_async_create, _global_tracker

    if _original_create is None and _original_async_create is None:
        return

    try:
        from openai.resources.chat.completions import (
            AsyncCompletions,
            Completions,
        )
    except ImportError:
        return

    if _original_create is not None:
        Completions.create = _original_create
        _original_create = None

    if _original_async_create is not None:
        AsyncCompletions.create = _original_async_create
        _original_async_create = None

    _global_tracker = None
