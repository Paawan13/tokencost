# LLM Cost Tracker Design

## Overview

A lightweight Python library for tracking LLM API costs via litellm's callback system, with budget alerts and spending limits.

## Requirements

- Track costs in real-time during LLM calls
- Budget alerts via callback and/or exception
- In-memory storage (no persistence)
- Global totals only (no per-user/project grouping)
- Use litellm's built-in pricing data

## Architecture

Single-module library implementing litellm's `CustomLogger` interface.

```
llm_cost/
├── __init__.py       # Exports CostTracker, BudgetExceededError
├── tracker.py        # CostTracker implementation
└── exceptions.py     # BudgetExceededError
```

## Core Components

### CostTracker

Inherits from `litellm.integrations.custom_logger.CustomLogger`.

**Constructor:**
```python
CostTracker(
    budget: float | None = None,           # Spending limit in USD
    on_budget_exceeded: Callable | None = None,  # Callback when exceeded
    raise_on_budget: bool = False          # Raise exception when exceeded
)
```

**Properties:**
- `total_cost: float` — running total in USD
- `request_count: int` — number of successful requests
- `history: list[dict]` — all logged requests
- `budget: float | None` — configured budget
- `budget_exceeded: bool` — whether budget has been exceeded

**Methods:**
- `reset()` — clear all tracked data

**Litellm hooks:**
- `log_success_event()` — logs cost after successful completion
- `log_failure_event()` — tracks failed attempts (no cost)

### BudgetExceededError

```python
class BudgetExceededError(Exception):
    budget: float       # Configured budget
    total_cost: float   # Actual spend when exceeded
```

## Data Structures

**History entry:**
```python
{
    "model": "gpt-4",
    "prompt_tokens": 150,
    "completion_tokens": 50,
    "cost": 0.0123,
    "timestamp": "2026-02-22T10:30:00Z"
}
```

## Budget Alert Behavior

1. Budget check runs after each request is logged
2. `on_budget_exceeded(tracker)` called once when first exceeded
3. If `raise_on_budget=True`, raises `BudgetExceededError` after callback
4. Both callback and exception can be used together

## Dependencies

- `litellm` — CustomLogger base class and `completion_cost()` function

## Usage Example

```python
import litellm
from llm_cost import CostTracker, BudgetExceededError

def alert(tracker):
    print(f"Budget exceeded! Spent ${tracker.total_cost:.2f}")

tracker = CostTracker(budget=5.00, on_budget_exceeded=alert, raise_on_budget=True)
litellm.callbacks = [tracker]

try:
    response = litellm.completion(model="gpt-4", messages=[{"role": "user", "content": "Hi"}])
except BudgetExceededError as e:
    print(f"Stopped at ${e.total_cost:.2f} (budget: ${e.budget:.2f})")

print(f"Total: ${tracker.total_cost:.4f} across {tracker.request_count} requests")
```
