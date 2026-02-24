"""Cost tracker implementation using litellm's CustomLogger."""

import atexit
import threading
from datetime import datetime, timezone
from typing import Callable

from litellm import completion_cost
from litellm.integrations.custom_logger import CustomLogger

from .exceptions import (
    BudgetExceededError,
    CompletionBudgetExceededError,
    EmbeddingBudgetExceededError,
)
from .pricing import is_embedding_model


class CostTracker(CustomLogger):
    """Tracks LLM API costs via litellm's callback system.

    Supports tracking both completion and embedding costs separately,
    with optional type-specific budgets for granular cost control.

    Attributes:
        total_cost: Running total cost in USD.
        request_count: Number of successful requests tracked.
        history: List of all logged request entries.
        budget: Configured spending limit in USD (None if unlimited).
        budget_exceeded: Whether the budget has been exceeded.
        embedding_cost: Total embedding cost in USD.
        completion_cost: Total completion cost in USD.
        embedding_count: Number of embedding requests.
        completion_count: Number of completion requests.
        cost_by_request_type: Cost breakdown by type {"embedding": ..., "completion": ...}.
        embedding_budget_exceeded: Whether embedding budget exceeded.
        completion_budget_exceeded: Whether completion budget exceeded.
    """

    def __init__(
        self,
        budget: float | None = None,
        embedding_budget: float | None = None,
        completion_budget: float | None = None,
        on_budget_exceeded: Callable[["CostTracker"], None] | None = None,
        on_embedding_budget_exceeded: Callable[["CostTracker"], None] | None = None,
        on_completion_budget_exceeded: Callable[["CostTracker"], None] | None = None,
        raise_on_budget: bool = False,
        print_summary: bool = True,
    ) -> None:
        """Initialize the cost tracker.

        Args:
            budget: Total spending limit in USD. None means unlimited.
            embedding_budget: Embedding-specific budget in USD. None means unlimited.
            completion_budget: Completion-specific budget in USD. None means unlimited.
            on_budget_exceeded: Callback when total budget is exceeded.
                Receives the tracker instance as argument.
            on_embedding_budget_exceeded: Callback when embedding budget is exceeded.
            on_completion_budget_exceeded: Callback when completion budget is exceeded.
            raise_on_budget: If True, raises BudgetExceededError when any budget exceeded.
            print_summary: If True, prints cost summary when program exits.
        """
        super().__init__()
        self._budget = budget
        self._embedding_budget = embedding_budget
        self._completion_budget = completion_budget
        self._on_budget_exceeded = on_budget_exceeded
        self._on_embedding_budget_exceeded = on_embedding_budget_exceeded
        self._on_completion_budget_exceeded = on_completion_budget_exceeded
        self._raise_on_budget = raise_on_budget
        self._print_summary = print_summary
        self._total_cost: float = 0.0
        self._request_count: int = 0
        self._history: list[dict] = []
        self._cost_by_model: dict[str, float] = {}
        self._budget_exceeded: bool = False
        self._callback_fired: bool = False
        self._lock = threading.Lock()

        # Embedding-specific tracking
        self._embedding_cost: float = 0.0
        self._embedding_count: int = 0
        self._embedding_budget_exceeded: bool = False
        self._embedding_callback_fired: bool = False

        # Completion-specific tracking
        self._completion_cost: float = 0.0
        self._completion_count: int = 0
        self._completion_budget_exceeded: bool = False
        self._completion_callback_fired: bool = False

        # Cost by request type
        self._cost_by_request_type: dict[str, float] = {
            "embedding": 0.0,
            "completion": 0.0,
        }

        if self._print_summary:
            atexit.register(self._print_exit_summary)

    @property
    def total_cost(self) -> float:
        """Running total cost in USD."""
        with self._lock:
            return self._total_cost

    @property
    def request_count(self) -> int:
        """Number of successful requests tracked."""
        with self._lock:
            return self._request_count

    @property
    def history(self) -> list[dict]:
        """List of all logged request entries."""
        with self._lock:
            return self._history.copy()

    @property
    def budget(self) -> float | None:
        """Configured spending limit in USD."""
        return self._budget

    @property
    def budget_exceeded(self) -> bool:
        """Whether the budget has been exceeded."""
        with self._lock:
            return self._budget_exceeded

    @property
    def cost_by_model(self) -> dict[str, float]:
        """Cost aggregated by model name."""
        with self._lock:
            return self._cost_by_model.copy()

    @property
    def embedding_cost(self) -> float:
        """Total embedding cost in USD."""
        with self._lock:
            return self._embedding_cost

    @property
    def completion_cost(self) -> float:
        """Total completion cost in USD."""
        with self._lock:
            return self._completion_cost

    @property
    def embedding_count(self) -> int:
        """Number of embedding requests."""
        with self._lock:
            return self._embedding_count

    @property
    def completion_count(self) -> int:
        """Number of completion requests."""
        with self._lock:
            return self._completion_count

    @property
    def cost_by_request_type(self) -> dict[str, float]:
        """Cost breakdown by request type."""
        with self._lock:
            return self._cost_by_request_type.copy()

    @property
    def embedding_budget(self) -> float | None:
        """Configured embedding budget in USD."""
        return self._embedding_budget

    @property
    def completion_budget(self) -> float | None:
        """Configured completion budget in USD."""
        return self._completion_budget

    @property
    def embedding_budget_exceeded(self) -> bool:
        """Whether the embedding budget has been exceeded."""
        with self._lock:
            return self._embedding_budget_exceeded

    @property
    def completion_budget_exceeded(self) -> bool:
        """Whether the completion budget has been exceeded."""
        with self._lock:
            return self._completion_budget_exceeded

    def reset(self) -> None:
        """Clear all tracked data and reset budget exceeded state."""
        with self._lock:
            self._total_cost = 0.0
            self._request_count = 0
            self._history.clear()
            self._cost_by_model.clear()
            self._budget_exceeded = False
            self._callback_fired = False

            # Reset embedding-specific tracking
            self._embedding_cost = 0.0
            self._embedding_count = 0
            self._embedding_budget_exceeded = False
            self._embedding_callback_fired = False

            # Reset completion-specific tracking
            self._completion_cost = 0.0
            self._completion_count = 0
            self._completion_budget_exceeded = False
            self._completion_callback_fired = False

            # Reset cost by request type
            self._cost_by_request_type = {"embedding": 0.0, "completion": 0.0}

    def record_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        request_type: str = "completion",
    ) -> None:
        """Record a cost entry manually (thread-safe).

        This is the public API for recording costs from external integrations
        like the OpenAI wrapper.

        Args:
            model: The model name.
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
            cost: Cost in USD.
            request_type: Type of request, either "completion" or "embedding".
        """
        self._record_cost(model, prompt_tokens, completion_tokens, cost, request_type)

    def _record_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        request_type: str = "completion",
    ) -> None:
        """Record a cost entry (thread-safe, internal).

        Args:
            model: The model name.
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
            cost: Cost in USD.
            request_type: Type of request, either "completion" or "embedding".
        """
        entry = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_type": request_type,
        }

        with self._lock:
            self._history.append(entry)
            self._total_cost += cost
            self._request_count += 1
            self._cost_by_model[model] = self._cost_by_model.get(model, 0.0) + cost

            # Track by request type
            if request_type == "embedding":
                self._embedding_cost += cost
                self._embedding_count += 1
                self._cost_by_request_type["embedding"] += cost
            else:
                self._completion_cost += cost
                self._completion_count += 1
                self._cost_by_request_type["completion"] += cost

        self._check_budget()

    def _extract_and_record(self, kwargs, response_obj) -> None:
        """Extract cost info from response and record it."""
        try:
            cost = completion_cost(completion_response=response_obj)
        except Exception:
            cost = 0.0

        model = kwargs.get("model", "unknown")

        usage = getattr(response_obj, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        # Detect request type from call_type or model
        call_type = kwargs.get("call_type", "")
        if call_type == "embedding" or is_embedding_model(model):
            request_type = "embedding"
        else:
            request_type = "completion"

        self._record_cost(model, prompt_tokens, completion_tokens, cost, request_type)

    def log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        """Log a successful LLM completion event.

        This method is called by litellm after a successful sync completion.
        """
        self._extract_and_record(kwargs, response_obj)

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ) -> None:
        """Log a successful LLM completion event (async version).

        This method is called by litellm after a successful async completion.
        """
        # Debug: uncomment to trace callback invocations
        # print(f"[DEBUG] async_log_success_event called for {kwargs.get('model')}")
        self._extract_and_record(kwargs, response_obj)

    def log_stream_event(self, kwargs, response_obj, start_time, end_time) -> None:
        """Log a streaming chunk event.

        This method is called by litellm for each chunk during streaming.
        Individual chunks don't have complete usage info, so we don't track
        cost here. The final accumulated response is tracked via
        log_success_event which is called after streaming completes.

        Note: litellm accumulates streaming responses and calls log_success_event
        with the complete response at the end of the stream.
        """
        pass

    async def async_log_stream_event(
        self, kwargs, response_obj, start_time, end_time
    ) -> None:
        """Log a streaming chunk event (async version).

        See log_stream_event for details.
        """
        pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        """Log a failed LLM completion event.

        Failed requests do not incur cost.
        """
        pass

    async def async_log_failure_event(
        self, kwargs, response_obj, start_time, end_time
    ) -> None:
        """Log a failed LLM completion event (async version).

        Failed requests do not incur cost.
        """
        pass

    def _check_budget(self) -> None:
        """Check if any budget is exceeded and trigger alerts if so (thread-safe)."""
        # Collect actions to execute outside the lock
        actions: list[tuple[str, float, float]] = []

        with self._lock:
            # Check embedding budget first (more specific)
            if (
                self._embedding_budget is not None
                and self._embedding_cost > self._embedding_budget
            ):
                self._embedding_budget_exceeded = True
                if not self._embedding_callback_fired:
                    self._embedding_callback_fired = True
                    actions.append(
                        ("embedding", self._embedding_budget, self._embedding_cost)
                    )

            # Check completion budget (more specific)
            if (
                self._completion_budget is not None
                and self._completion_cost > self._completion_budget
            ):
                self._completion_budget_exceeded = True
                if not self._completion_callback_fired:
                    self._completion_callback_fired = True
                    actions.append(
                        ("completion", self._completion_budget, self._completion_cost)
                    )

            # Check total budget last (most general)
            if self._budget is not None and self._total_cost > self._budget:
                self._budget_exceeded = True
                if not self._callback_fired:
                    self._callback_fired = True
                    actions.append(("total", self._budget, self._total_cost))

        # Execute callbacks outside the lock to prevent deadlocks
        exception_to_raise = None

        for budget_type, budget, cost in actions:
            if budget_type == "total":
                if self._on_budget_exceeded is not None:
                    self._on_budget_exceeded(self)
                if self._raise_on_budget and exception_to_raise is None:
                    exception_to_raise = BudgetExceededError(budget, cost)

            elif budget_type == "embedding":
                if self._on_embedding_budget_exceeded is not None:
                    self._on_embedding_budget_exceeded(self)
                if self._raise_on_budget and exception_to_raise is None:
                    exception_to_raise = EmbeddingBudgetExceededError(budget, cost)

            elif budget_type == "completion":
                if self._on_completion_budget_exceeded is not None:
                    self._on_completion_budget_exceeded(self)
                if self._raise_on_budget and exception_to_raise is None:
                    exception_to_raise = CompletionBudgetExceededError(budget, cost)

        if exception_to_raise is not None:
            raise exception_to_raise

    def _print_exit_summary(self) -> None:
        """Print cost summary on program exit (thread-safe)."""
        # Snapshot the data under lock to ensure consistency
        with self._lock:
            if self._request_count == 0:
                return
            total_cost = self._total_cost
            request_count = self._request_count
            budget_exceeded = self._budget_exceeded
            embedding_cost = self._embedding_cost
            embedding_count = self._embedding_count
            embedding_budget_exceeded = self._embedding_budget_exceeded
            completion_cost = self._completion_cost
            completion_count = self._completion_count
            completion_budget_exceeded = self._completion_budget_exceeded
            history = self._history.copy()

        print("\n" + "=" * 50)
        print("LLM COST SUMMARY")
        print("=" * 50)
        print(f"Total Cost:     ${total_cost:.6f}")
        print(f"Total Requests: {request_count}")

        if self._budget is not None:
            remaining = self._budget - total_cost
            status = "EXCEEDED" if budget_exceeded else "OK"
            print(f"Total Budget:   ${self._budget:.2f} ({status})")
            if not budget_exceeded:
                print(f"Remaining:      ${remaining:.6f}")

        # Show breakdown by type if there are both embeddings and completions
        if embedding_count > 0 or completion_count > 0:
            print("-" * 50)
            print("By Type:")

            if embedding_count > 0:
                emb_budget_str = ""
                if self._embedding_budget is not None:
                    emb_status = "EXCEEDED" if embedding_budget_exceeded else "OK"
                    emb_budget_str = f" | Budget: ${self._embedding_budget:.2f} ({emb_status})"
                print(
                    f"  Embeddings:  {embedding_count} requests = ${embedding_cost:.6f}{emb_budget_str}"
                )

            if completion_count > 0:
                comp_budget_str = ""
                if self._completion_budget is not None:
                    comp_status = "EXCEEDED" if completion_budget_exceeded else "OK"
                    comp_budget_str = f" | Budget: ${self._completion_budget:.2f} ({comp_status})"
                print(
                    f"  Completions: {completion_count} requests = ${completion_cost:.6f}{comp_budget_str}"
                )

        if history:
            print("-" * 50)
            print("Requests:")
            for i, entry in enumerate(history, 1):
                tokens = f"{entry['prompt_tokens']}+{entry['completion_tokens']}"
                req_type = entry.get("request_type", "completion")
                type_indicator = "[E]" if req_type == "embedding" else "[C]"
                print(
                    f"  {i}. {type_indicator} {entry['model']}: {tokens} tokens = ${entry['cost']:.6f}"
                )

        print("=" * 50)
