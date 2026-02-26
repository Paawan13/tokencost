"""Wrapper for OpenAI SDK integration to track costs automatically."""

import functools
import threading
from typing import TYPE_CHECKING, Any, Callable

from .pricing import (
    _normalize_moonshot_model,
    calculate_cost,
    calculate_embedding_cost,
)

if TYPE_CHECKING:
    from .tracker import CostTracker

# Global state for patching (protected by _global_lock)
_original_create = None
_original_async_create = None
_original_embedding_create = None
_original_async_embedding_create = None
_global_tracker: "CostTracker | None" = None
_global_lock = threading.Lock()


def _get_global_tracker() -> "CostTracker | None":
    """Thread-safe getter for the global tracker."""
    with _global_lock:
        return _global_tracker


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

    # Normalize model name for pricing lookup (e.g., Moonshot/Kimi models)
    normalized_model = _normalize_moonshot_model(model)

    try:
        cost = calculate_cost(normalized_model, prompt_tokens, completion_tokens)
    except ValueError:
        # Unknown model, skip recording
        return

    tracker.record_cost(model, prompt_tokens, completion_tokens, cost)


def _extract_embedding_usage(response: Any) -> tuple[str, int]:
    """Extract model and token usage from an OpenAI embedding response.

    Args:
        response: OpenAI CreateEmbeddingResponse object.

    Returns:
        Tuple of (model, total_tokens).
    """
    model = getattr(response, "model", "unknown")
    usage = getattr(response, "usage", None)

    if usage is None:
        return model, 0

    # Embedding usage has prompt_tokens and total_tokens (no completion_tokens)
    total_tokens = getattr(usage, "total_tokens", 0)
    if total_tokens == 0:
        total_tokens = getattr(usage, "prompt_tokens", 0)

    return model, total_tokens


def _record_embedding_to_tracker(
    tracker: "CostTracker | None",
    model: str,
    input_tokens: int,
) -> None:
    """Record embedding cost to tracker if available."""
    if tracker is None:
        return

    try:
        cost = calculate_embedding_cost(model, input_tokens)
    except ValueError:
        # Unknown model, skip recording
        return

    tracker.record_cost(model, input_tokens, 0, cost, request_type="embedding")


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


def _make_sync_wrapped_create_base(
    original_create: Any, get_tracker: Callable[[], "CostTracker | None"]
) -> Any:
    """Base factory for sync create wrappers."""

    @functools.wraps(original_create)
    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        is_streaming = kwargs.get("stream", False)
        if is_streaming:
            # Copy to avoid modifying caller's dict
            stream_options = dict(kwargs.get("stream_options", {}) or {})
            stream_options["include_usage"] = True
            kwargs["stream_options"] = stream_options

        response = original_create(*args, **kwargs)

        tracker = get_tracker()
        if is_streaming:
            return _wrap_streaming_response(response, tracker)
        else:
            model, prompt_tokens, completion_tokens = _extract_usage(response)
            _record_to_tracker(tracker, model, prompt_tokens, completion_tokens)
            return response

    return wrapped_create


def _make_async_wrapped_create_base(
    original_create: Any, get_tracker: Callable[[], "CostTracker | None"]
) -> Any:
    """Base factory for async create wrappers."""

    @functools.wraps(original_create)
    async def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        is_streaming = kwargs.get("stream", False)
        if is_streaming:
            # Copy to avoid modifying caller's dict
            stream_options = dict(kwargs.get("stream_options", {}) or {})
            stream_options["include_usage"] = True
            kwargs["stream_options"] = stream_options

        response = await original_create(*args, **kwargs)

        tracker = get_tracker()
        if is_streaming:
            return _wrap_async_streaming_response(response, tracker)
        else:
            model, prompt_tokens, completion_tokens = _extract_usage(response)
            _record_to_tracker(tracker, model, prompt_tokens, completion_tokens)
            return response

    return wrapped_create


def _make_wrapped_create(original_create: Any, tracker: "CostTracker | None") -> Any:
    """Create a wrapped version of chat.completions.create (sync)."""
    return _make_sync_wrapped_create_base(original_create, lambda: tracker)


def _make_wrapped_async_create(
    original_create: Any, tracker: "CostTracker | None"
) -> Any:
    """Create a wrapped version of chat.completions.create (async)."""
    return _make_async_wrapped_create_base(original_create, lambda: tracker)


def _make_global_wrapped_create(original_create: Any) -> Any:
    """Create a wrapped version that uses the global tracker (for patch_openai)."""
    return _make_sync_wrapped_create_base(original_create, _get_global_tracker)


def _make_global_wrapped_async_create(original_create: Any) -> Any:
    """Create a wrapped async version that uses the global tracker (for patch_openai)."""
    return _make_async_wrapped_create_base(original_create, _get_global_tracker)


# Embedding wrapper factory functions


def _make_sync_wrapped_embedding_create_base(
    original_create: Any, get_tracker: Callable[[], "CostTracker | None"]
) -> Any:
    """Base factory for sync embedding create wrappers."""

    @functools.wraps(original_create)
    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        response = original_create(*args, **kwargs)

        tracker = get_tracker()
        model, input_tokens = _extract_embedding_usage(response)
        _record_embedding_to_tracker(tracker, model, input_tokens)
        return response

    return wrapped_create


def _make_async_wrapped_embedding_create_base(
    original_create: Any, get_tracker: Callable[[], "CostTracker | None"]
) -> Any:
    """Base factory for async embedding create wrappers."""

    @functools.wraps(original_create)
    async def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        response = await original_create(*args, **kwargs)

        tracker = get_tracker()
        model, input_tokens = _extract_embedding_usage(response)
        _record_embedding_to_tracker(tracker, model, input_tokens)
        return response

    return wrapped_create


def _make_wrapped_embedding_create(
    original_create: Any, tracker: "CostTracker | None"
) -> Any:
    """Create a wrapped version of embeddings.create (sync)."""
    return _make_sync_wrapped_embedding_create_base(original_create, lambda: tracker)


def _make_wrapped_async_embedding_create(
    original_create: Any, tracker: "CostTracker | None"
) -> Any:
    """Create a wrapped version of embeddings.create (async)."""
    return _make_async_wrapped_embedding_create_base(original_create, lambda: tracker)


def _make_global_wrapped_embedding_create(original_create: Any) -> Any:
    """Create a wrapped version that uses the global tracker (for patch_openai)."""
    return _make_sync_wrapped_embedding_create_base(original_create, _get_global_tracker)


def _make_global_wrapped_async_embedding_create(original_create: Any) -> Any:
    """Create a wrapped async version that uses the global tracker (for patch_openai)."""
    return _make_async_wrapped_embedding_create_base(
        original_create, _get_global_tracker
    )


class _WrappedEmbeddings:
    """Wrapper for client.embeddings that tracks costs."""

    def __init__(self, original_embeddings: Any, tracker: "CostTracker | None") -> None:
        self._original = original_embeddings
        self._tracker = tracker
        self._create = _make_wrapped_embedding_create(
            self._original.create, self._tracker
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return self._create(*args, **kwargs)


class _WrappedAsyncEmbeddings:
    """Wrapper for async client.embeddings that tracks costs."""

    def __init__(self, original_embeddings: Any, tracker: "CostTracker | None") -> None:
        self._original = original_embeddings
        self._tracker = tracker
        self._create = _make_wrapped_async_embedding_create(
            self._original.create, self._tracker
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        return await self._create(*args, **kwargs)


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

    @property
    def embeddings(self) -> Any:
        if self._is_async:
            return _WrappedAsyncEmbeddings(self._client.embeddings, self._tracker)
        return _WrappedEmbeddings(self._client.embeddings, self._tracker)


def track_openai(client: Any, tracker: "CostTracker | None" = None) -> Any:
    """Wrap OpenAI/AsyncOpenAI client to track costs.

    Returns a wrapped client with the same interface that automatically
    tracks costs for chat.completions.create() and embeddings.create() calls.

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
        >>> # Track chat completions
        >>> response = client.chat.completions.create(
        ...     model="gpt-4o",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
        >>> # Track embeddings
        >>> embeddings = client.embeddings.create(
        ...     model="text-embedding-3-small",
        ...     input=["Hello world"]
        ... )
        >>> print(f"Total: ${tracker.total_cost:.6f}")
        >>> print(f"Embeddings: ${tracker.embedding_cost:.6f}")
        >>> print(f"Completions: ${tracker.completion_cost:.6f}")
    """
    # Detect if client is async by checking class name
    is_async = "Async" in type(client).__name__
    return _WrappedClient(client, tracker, is_async)


def patch_openai(tracker: "CostTracker | None" = None) -> None:
    """Globally patch OpenAI SDK to track all costs.

    After calling this function, all OpenAI client instances will
    automatically have their chat.completions.create() and embeddings.create()
    calls tracked.

    This function can be called multiple times with different trackers.
    Each call updates the active tracker used for cost recording.

    This function is thread-safe.

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
        >>> embeddings = client.embeddings.create(...)
        >>> print(tracker.total_cost)
    """
    global _original_create, _original_async_create, _global_tracker
    global _original_embedding_create, _original_async_embedding_create

    try:
        from openai.resources.chat.completions import (
            AsyncCompletions,
            Completions,
        )
        from openai.resources.embeddings import (
            AsyncEmbeddings,
            Embeddings,
        )
    except ImportError as e:
        raise ImportError(
            "OpenAI SDK not installed. Install it with: pip install openai"
        ) from e

    with _global_lock:
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

        # Patch sync embeddings (only once)
        if _original_embedding_create is None:
            _original_embedding_create = Embeddings.create
            Embeddings.create = _make_global_wrapped_embedding_create(
                _original_embedding_create
            )

        # Patch async embeddings (only once)
        if _original_async_embedding_create is None:
            _original_async_embedding_create = AsyncEmbeddings.create
            AsyncEmbeddings.create = _make_global_wrapped_async_embedding_create(
                _original_async_embedding_create
            )


def unpatch_openai() -> None:
    """Remove global patches from OpenAI SDK.

    Restores the original chat.completions.create() and embeddings.create() methods.
    This function is thread-safe.
    """
    global _original_create, _original_async_create, _global_tracker
    global _original_embedding_create, _original_async_embedding_create

    with _global_lock:
        has_completion_patches = (
            _original_create is not None or _original_async_create is not None
        )
        has_embedding_patches = (
            _original_embedding_create is not None
            or _original_async_embedding_create is not None
        )

        if not has_completion_patches and not has_embedding_patches:
            return

        try:
            from openai.resources.chat.completions import (
                AsyncCompletions,
                Completions,
            )
            from openai.resources.embeddings import (
                AsyncEmbeddings,
                Embeddings,
            )
        except ImportError:
            return

        # Restore completion methods
        if _original_create is not None:
            Completions.create = _original_create
            _original_create = None

        if _original_async_create is not None:
            AsyncCompletions.create = _original_async_create
            _original_async_create = None

        # Restore embedding methods
        if _original_embedding_create is not None:
            Embeddings.create = _original_embedding_create
            _original_embedding_create = None

        if _original_async_embedding_create is not None:
            AsyncEmbeddings.create = _original_async_embedding_create
            _original_async_embedding_create = None

        _global_tracker = None
