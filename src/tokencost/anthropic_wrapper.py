"""Wrapper for Anthropic SDK integration to track costs automatically.

Note: Anthropic does not provide embedding models. This wrapper only supports
the Messages API for Claude LLMs. For embeddings, consider using OpenAI,
Voyage AI, or other embedding providers.
"""

import functools
import threading
from typing import TYPE_CHECKING, Any, Callable

from .pricing import calculate_cost

if TYPE_CHECKING:
    from .tracker import CostTracker

# Global state for patching (protected by _global_lock)
_original_create = None
_original_async_create = None
_global_tracker: "CostTracker | None" = None
_global_lock = threading.Lock()


def _get_global_tracker() -> "CostTracker | None":
    """Thread-safe getter for the global tracker."""
    with _global_lock:
        return _global_tracker


def _extract_usage(response: Any) -> tuple[str, int, int]:
    """Extract model and token usage from an Anthropic response.

    Args:
        response: Anthropic Message response object.

    Returns:
        Tuple of (model, input_tokens, output_tokens).
    """
    model = getattr(response, "model", "unknown")
    usage = getattr(response, "usage", None)

    if usage is None:
        return model, 0, 0

    # Anthropic uses input_tokens and output_tokens
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0

    return model, input_tokens, output_tokens


def _extract_streaming_usage(event: Any) -> tuple[str, int, int] | None:
    """Extract usage from a streaming event (only present in message_stop or final event).

    Args:
        event: Anthropic streaming event object.

    Returns:
        Tuple of (model, input_tokens, output_tokens) if usage present,
        None otherwise.
    """
    # For Anthropic streaming, usage is in the final message_stop event
    # or can be accessed via the accumulated message
    usage = getattr(event, "usage", None)
    if usage is None:
        # Check if this is a MessageStopEvent with a message
        message = getattr(event, "message", None)
        if message is not None:
            usage = getattr(message, "usage", None)

    if usage is None:
        return None

    model = getattr(event, "model", None)
    if model is None:
        message = getattr(event, "message", None)
        if message is not None:
            model = getattr(message, "model", "unknown")
        else:
            model = "unknown"

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0

    return model, input_tokens, output_tokens


def _record_to_tracker(
    tracker: "CostTracker | None",
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Record cost to tracker if available."""
    if tracker is None:
        return

    try:
        cost = calculate_cost(model, input_tokens, output_tokens)
    except ValueError:
        # Unknown model, skip recording
        return

    tracker.record_cost(model, input_tokens, output_tokens, cost)


def _wrap_streaming_response(stream: Any, tracker: "CostTracker | None") -> Any:
    """Wrap a streaming response to capture usage from final event.

    Args:
        stream: Anthropic streaming response iterator.
        tracker: CostTracker instance to record costs.

    Yields:
        Original events from the stream.
    """
    for event in stream:
        yield event
        # Check for usage in each event (only present in final event)
        usage_info = _extract_streaming_usage(event)
        if usage_info is not None:
            model, input_tokens, output_tokens = usage_info
            _record_to_tracker(tracker, model, input_tokens, output_tokens)


async def _wrap_async_streaming_response(
    stream: Any, tracker: "CostTracker | None"
) -> Any:
    """Wrap an async streaming response to capture usage from final event.

    Args:
        stream: Anthropic async streaming response iterator.
        tracker: CostTracker instance to record costs.

    Yields:
        Original events from the stream.
    """
    async for event in stream:
        yield event
        # Check for usage in each event (only present in final event)
        usage_info = _extract_streaming_usage(event)
        if usage_info is not None:
            model, input_tokens, output_tokens = usage_info
            _record_to_tracker(tracker, model, input_tokens, output_tokens)


def _make_sync_wrapped_create_base(
    original_create: Any, get_tracker: Callable[[], "CostTracker | None"]
) -> Any:
    """Base factory for sync create wrappers."""

    @functools.wraps(original_create)
    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        is_streaming = kwargs.get("stream", False)

        response = original_create(*args, **kwargs)

        tracker = get_tracker()
        if is_streaming:
            return _wrap_streaming_response(response, tracker)
        else:
            model, input_tokens, output_tokens = _extract_usage(response)
            _record_to_tracker(tracker, model, input_tokens, output_tokens)
            return response

    return wrapped_create


def _make_async_wrapped_create_base(
    original_create: Any, get_tracker: Callable[[], "CostTracker | None"]
) -> Any:
    """Base factory for async create wrappers."""

    @functools.wraps(original_create)
    async def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        is_streaming = kwargs.get("stream", False)

        response = await original_create(*args, **kwargs)

        tracker = get_tracker()
        if is_streaming:
            return _wrap_async_streaming_response(response, tracker)
        else:
            model, input_tokens, output_tokens = _extract_usage(response)
            _record_to_tracker(tracker, model, input_tokens, output_tokens)
            return response

    return wrapped_create


def _make_wrapped_create(original_create: Any, tracker: "CostTracker | None") -> Any:
    """Create a wrapped version of messages.create (sync)."""
    return _make_sync_wrapped_create_base(original_create, lambda: tracker)


def _make_wrapped_async_create(
    original_create: Any, tracker: "CostTracker | None"
) -> Any:
    """Create a wrapped version of messages.create (async)."""
    return _make_async_wrapped_create_base(original_create, lambda: tracker)


def _make_global_wrapped_create(original_create: Any) -> Any:
    """Create a wrapped version that uses the global tracker (for patch_anthropic)."""
    return _make_sync_wrapped_create_base(original_create, _get_global_tracker)


def _make_global_wrapped_async_create(original_create: Any) -> Any:
    """Create a wrapped async version that uses the global tracker."""
    return _make_async_wrapped_create_base(original_create, _get_global_tracker)


class _WrappedMessages:
    """Wrapper for client.messages that tracks costs."""

    def __init__(
        self, original_messages: Any, tracker: "CostTracker | None"
    ) -> None:
        self._original = original_messages
        self._tracker = tracker

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return _make_wrapped_create(self._original.create, self._tracker)(
            *args, **kwargs
        )


class _WrappedAsyncMessages:
    """Wrapper for async client.messages that tracks costs."""

    def __init__(self, original_messages: Any, tracker: "CostTracker | None") -> None:
        self._original = original_messages
        self._tracker = tracker

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        return await _make_wrapped_async_create(self._original.create, self._tracker)(
            *args, **kwargs
        )


class _WrappedClient:
    """Wrapper for Anthropic client that tracks costs."""

    def __init__(
        self, client: Any, tracker: "CostTracker | None", is_async: bool = False
    ) -> None:
        self._client = client
        self._tracker = tracker
        self._is_async = is_async

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    @property
    def messages(self) -> Any:
        if self._is_async:
            return _WrappedAsyncMessages(self._client.messages, self._tracker)
        return _WrappedMessages(self._client.messages, self._tracker)


def track_anthropic(client: Any, tracker: "CostTracker | None" = None) -> Any:
    """Wrap Anthropic/AsyncAnthropic client to track costs.

    Returns a wrapped client with the same interface that automatically
    tracks costs for messages.create() calls.

    Note: Anthropic does not provide embedding models. For embeddings,
    use OpenAI, Voyage AI, or other embedding providers.

    Args:
        client: Anthropic or AsyncAnthropic client instance.
        tracker: CostTracker instance to record costs. If None, costs are
            calculated but not recorded anywhere.

    Returns:
        Wrapped client that tracks costs.

    Example:
        >>> from anthropic import Anthropic
        >>> from tokencost import CostTracker, track_anthropic
        >>>
        >>> tracker = CostTracker(budget=1.0)
        >>> client = track_anthropic(Anthropic(), tracker)
        >>>
        >>> # Track message completions
        >>> response = client.messages.create(
        ...     model="claude-3-5-sonnet-20241022",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
        >>> print(f"Total: ${tracker.total_cost:.6f}")
    """
    # Detect if client is async by checking class name
    is_async = "Async" in type(client).__name__
    return _WrappedClient(client, tracker, is_async)


def patch_anthropic(tracker: "CostTracker | None" = None) -> None:
    """Globally patch Anthropic SDK to track all costs.

    After calling this function, all Anthropic client instances will
    automatically have their messages.create() calls tracked.

    This function can be called multiple times with different trackers.
    Each call updates the active tracker used for cost recording.

    This function is thread-safe.

    Args:
        tracker: CostTracker instance to record costs. If None, costs are
            calculated but not recorded.

    Example:
        >>> from tokencost import CostTracker, patch_anthropic
        >>> from anthropic import Anthropic
        >>>
        >>> tracker = CostTracker()
        >>> patch_anthropic(tracker)
        >>>
        >>> # All Anthropic clients now track costs
        >>> client = Anthropic()
        >>> response = client.messages.create(...)
        >>> print(tracker.total_cost)
    """
    global _original_create, _original_async_create, _global_tracker

    try:
        from anthropic.resources.messages import (
            AsyncMessages,
            Messages,
        )
    except ImportError as e:
        raise ImportError(
            "Anthropic SDK not installed. Install it with: pip install anthropic"
        ) from e

    with _global_lock:
        # Update the global tracker (can be changed on subsequent calls)
        _global_tracker = tracker

        # Patch sync messages (only once)
        if _original_create is None:
            _original_create = Messages.create
            Messages.create = _make_global_wrapped_create(_original_create)

        # Patch async messages (only once)
        if _original_async_create is None:
            _original_async_create = AsyncMessages.create
            AsyncMessages.create = _make_global_wrapped_async_create(
                _original_async_create
            )


def unpatch_anthropic() -> None:
    """Remove global patches from Anthropic SDK.

    Restores the original messages.create() methods.
    This function is thread-safe.
    """
    global _original_create, _original_async_create, _global_tracker

    with _global_lock:
        if _original_create is None and _original_async_create is None:
            return

        try:
            from anthropic.resources.messages import (
                AsyncMessages,
                Messages,
            )
        except ImportError:
            return

        # Restore methods
        if _original_create is not None:
            Messages.create = _original_create
            _original_create = None

        if _original_async_create is not None:
            AsyncMessages.create = _original_async_create
            _original_async_create = None

        _global_tracker = None
