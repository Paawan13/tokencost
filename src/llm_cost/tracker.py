"""Cost tracker implementation using litellm's CustomLogger."""

import atexit
import threading
from datetime import datetime, timezone
from typing import Callable

from litellm import completion_cost
from litellm.integrations.custom_logger import CustomLogger

from .exceptions import BudgetExceededError


class CostTracker(CustomLogger):
    """Tracks LLM API costs via litellm's callback system.

    Attributes:
        total_cost: Running total cost in USD.
        request_count: Number of successful requests tracked.
        history: List of all logged request entries.
        budget: Configured spending limit in USD (None if unlimited).
        budget_exceeded: Whether the budget has been exceeded.
    """

    def __init__(
        self,
        budget: float | None = None,
        on_budget_exceeded: Callable[["CostTracker"], None] | None = None,
        raise_on_budget: bool = False,
        print_summary: bool = True,
    ) -> None:
        """Initialize the cost tracker.

        Args:
            budget: Spending limit in USD. None means unlimited.
            on_budget_exceeded: Callback function called when budget is exceeded.
                Receives the tracker instance as argument.
            raise_on_budget: If True, raises BudgetExceededError when budget exceeded.
            print_summary: If True, prints cost summary when program exits.
        """
        super().__init__()
        self._budget = budget
        self._on_budget_exceeded = on_budget_exceeded
        self._raise_on_budget = raise_on_budget
        self._print_summary = print_summary
        self._total_cost: float = 0.0
        self._request_count: int = 0
        self._history: list[dict] = []
        self._budget_exceeded: bool = False
        self._callback_fired: bool = False
        self._lock = threading.Lock()

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
            costs: dict[str, float] = {}
            for entry in self._history:
                model = entry["model"]
                costs[model] = costs.get(model, 0.0) + entry["cost"]
            return costs

    def reset(self) -> None:
        """Clear all tracked data and reset budget exceeded state."""
        with self._lock:
            self._total_cost = 0.0
            self._request_count = 0
            self._history.clear()
            self._budget_exceeded = False
            self._callback_fired = False

    def _record_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
    ) -> None:
        """Record a cost entry (thread-safe).

        Args:
            model: The model name.
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
            cost: Cost in USD.
        """
        entry = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with self._lock:
            self._history.append(entry)
            self._total_cost += cost
            self._request_count += 1

        self._check_budget()

    def log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        """Log a successful LLM completion event.

        This method is called by litellm after a successful completion.
        """
        try:
            cost = completion_cost(completion_response=response_obj)
        except Exception:
            cost = 0.0

        model = kwargs.get("model", "unknown")

        usage = getattr(response_obj, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        self._record_cost(model, prompt_tokens, completion_tokens, cost)

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ) -> None:
        """Log a successful LLM completion event (async version).

        This method is called by litellm after a successful async completion.
        Uses the same logic as log_success_event since the actual recording
        is CPU-bound and thread-safe.
        """
        self.log_success_event(kwargs, response_obj, start_time, end_time)

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
        """Check if budget is exceeded and trigger alerts if so (thread-safe)."""
        if self._budget is None:
            return

        should_fire_callback = False
        should_raise = False
        total_cost = 0.0

        with self._lock:
            if self._total_cost > self._budget:
                self._budget_exceeded = True

                if not self._callback_fired:
                    self._callback_fired = True
                    should_fire_callback = True
                    should_raise = self._raise_on_budget
                    total_cost = self._total_cost

        # Execute callbacks outside the lock to prevent deadlocks
        if should_fire_callback:
            if self._on_budget_exceeded is not None:
                self._on_budget_exceeded(self)

            if should_raise:
                raise BudgetExceededError(self._budget, total_cost)

    def _print_exit_summary(self) -> None:
        """Print cost summary on program exit (thread-safe)."""
        # Snapshot the data under lock to ensure consistency
        with self._lock:
            if self._request_count == 0:
                return
            total_cost = self._total_cost
            request_count = self._request_count
            budget_exceeded = self._budget_exceeded
            history = self._history.copy()

        print("\n" + "=" * 50)
        print("LLM COST SUMMARY")
        print("=" * 50)
        print(f"Total Cost:     ${total_cost:.6f}")
        print(f"Total Requests: {request_count}")

        if self._budget is not None:
            remaining = self._budget - total_cost
            status = "EXCEEDED" if budget_exceeded else "OK"
            print(f"Budget:         ${self._budget:.4f} ({status})")
            if not budget_exceeded:
                print(f"Remaining:      ${remaining:.6f}")

        if history:
            print("-" * 50)
            print("Requests:")
            for i, entry in enumerate(history, 1):
                tokens = f"{entry['prompt_tokens']}+{entry['completion_tokens']}"
                print(f"  {i}. {entry['model']}: {tokens} tokens = ${entry['cost']:.6f}")

        print("=" * 50)
