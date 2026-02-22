# LLM Cost Tracker

A lightweight Python library for tracking LLM API costs via litellm's callback system, with budget alerts and spending limits.

## Installation

```bash
pip install litellm-cost-tracker
```

## Quick Start

```python
import litellm
from llm_cost import CostTracker, BudgetExceededError

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
- **Automatic exit summary** — prints cost report when program ends
- **In-memory storage** (no persistence needed)
- **Uses litellm's pricing data** for accurate costs (1600+ models)

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
  1. gpt-5-mini: 7+18 tokens = $0.000038
  2. gpt-5-mini: 13+17 tokens = $0.000037
  3. gpt-5-mini: 8+82 tokens = $0.000166
  4. gpt-5-mini: 10+858 tokens = $0.001718
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
git clone https://github.com/Paawan13/litellm-cost-tracker.git
cd litellm-cost-tracker
pip install -e ".[dev]"
pytest
```

## License

MIT
