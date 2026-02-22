# tokencost

A lightweight Python library for tracking LLM API costs via litellm's callback system, with budget alerts and spending limits.

## Installation

```bash
pip install tokencost
```

## Quick Start

```python
import litellm
from tokencost import CostTracker, BudgetExceededError

# Create a tracker with a $5 budget
def alert(tracker):
    print(f"Budget exceeded! Spent ${tracker.total_cost:.2f}")

tracker = CostTracker(
    budget=5.00,
    on_budget_exceeded=alert,
    raise_on_budget=True
)

# Register with litellm
litellm.callbacks = [tracker]

# Make LLM calls as usual
try:
    response = litellm.completion(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
except BudgetExceededError as e:
    print(f"Stopped at ${e.total_cost:.2f} (budget: ${e.budget:.2f})")

# Check usage
print(f"Total: ${tracker.total_cost:.4f} across {tracker.request_count} requests")
```

## Features

- **Real-time cost tracking** during LLM calls
- **Budget alerts** via callback and/or exception
- **Async support** for `litellm.acompletion()`
- **Per-model cost aggregation** via `cost_by_model` property
- **Automatic exit summary** — prints cost report when program ends
- **Thread-safe** for concurrent usage
- **Uses litellm's pricing data** for accurate costs (1600+ models)

## Async Support

```python
import asyncio
import litellm
from tokencost import CostTracker

async def main():
    tracker = CostTracker()
    litellm.callbacks = [tracker]

    # Works with async completions
    response = await litellm.acompletion(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    print(f"Cost: ${tracker.total_cost:.6f}")

asyncio.run(main())
```

## Per-Model Cost Breakdown

```python
tracker = CostTracker()
litellm.callbacks = [tracker]

# Make calls to different models...
litellm.completion(model="gpt-4", messages=[...])
litellm.completion(model="gpt-3.5-turbo", messages=[...])
litellm.completion(model="claude-3-sonnet", messages=[...])

# Get cost breakdown by model
for model, cost in tracker.cost_by_model.items():
    print(f"{model}: ${cost:.6f}")
```

## API Reference

### CostTracker

```python
CostTracker(
    budget: float | None = None,           # Spending limit in USD
    on_budget_exceeded: Callable | None = None,  # Callback when exceeded
    raise_on_budget: bool = False,         # Raise exception when exceeded
    print_summary: bool = True             # Print summary on program exit
)
```

**Properties:**

- `total_cost: float` — Running total in USD
- `request_count: int` — Number of successful requests
- `history: list[dict]` — All logged requests
- `budget: float | None` — Configured budget
- `budget_exceeded: bool` — Whether budget has been exceeded
- `cost_by_model: dict[str, float]` — Cost aggregated by model name

**Methods:**

- `reset()` — Clear all tracked data

### BudgetExceededError

Raised when budget is exceeded (if `raise_on_budget=True`).

```python
class BudgetExceededError(Exception):
    budget: float       # Configured budget
    total_cost: float   # Actual spend when exceeded
```

## Exit Summary

When your program ends, a cost summary is automatically printed:

```
==================================================
LLM COST SUMMARY
==================================================
Total Cost:     $0.001959
Total Requests: 4
Budget:         $0.0100 (OK)
Remaining:      $0.008041
--------------------------------------------------
Requests:
  1. gpt-4: 7+18 tokens = $0.000750
  2. gpt-4: 13+17 tokens = $0.000900
  3. gpt-3.5-turbo: 8+82 tokens = $0.000166
  4. gpt-3.5-turbo: 10+58 tokens = $0.000143
==================================================
```

Disable with `print_summary=False`.

## History Entry Format

Each request is logged with:

```python
{
    "model": "gpt-4",
    "prompt_tokens": 150,
    "completion_tokens": 50,
    "cost": 0.0123,
    "timestamp": "2026-02-22T10:30:00Z"
}
```

## Development

```bash
git clone https://github.com/Paawan13/tokencost.git
cd tokencost
pip install -e ".[dev]"
pytest
```

## License

MIT
