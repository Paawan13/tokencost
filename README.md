# tokencost

A lightweight Python library for tracking LLM API costs via litellm's callback system, with budget alerts and spending limits.

## Installation

```bash
pip install llm-tokencost
```

## Quick Start

### With litellm

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

### With OpenAI SDK

```python
from openai import OpenAI
from tokencost import CostTracker, track_openai

tracker = CostTracker(budget=1.0)
client = track_openai(OpenAI(), tracker)

# Use the client as normal - costs are tracked automatically
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(f"Cost: ${tracker.total_cost:.6f}")
```

## Features

- **Real-time cost tracking** during LLM calls
- **Budget alerts** via callback and/or exception
- **Async support** for `litellm.acompletion()` and `AsyncOpenAI`
- **Per-model cost aggregation** via `cost_by_model` property
- **RAG cost tracking** — separate budgets for embeddings vs completions
- **OpenAI SDK support** — native integration with `openai` package
- **Automatic exit summary** — prints cost report when program ends
- **Thread-safe** for concurrent usage
- **Uses litellm's pricing data** for accurate costs (1600+ models)

## OpenAI SDK Integration

### Wrapping a Client

Use `track_openai()` to wrap an OpenAI client instance:

```python
from openai import OpenAI, AsyncOpenAI
from tokencost import CostTracker, track_openai

tracker = CostTracker(budget=1.0)

# Wrap sync client
client = track_openai(OpenAI(), tracker)

# Or wrap async client
async_client = track_openai(AsyncOpenAI(), tracker)

# Both chat completions and embeddings are tracked
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)

embeddings = client.embeddings.create(
    model="text-embedding-3-small",
    input=["Hello world"]
)

print(f"Total: ${tracker.total_cost:.6f}")
print(f"Completions: ${tracker.completion_cost:.6f}")
print(f"Embeddings: ${tracker.embedding_cost:.6f}")
```

### Global Patching

Use `patch_openai()` to automatically track all OpenAI client instances:

```python
from openai import OpenAI
from tokencost import CostTracker, patch_openai, unpatch_openai

tracker = CostTracker()
patch_openai(tracker)

# All clients now track costs automatically
client = OpenAI()
response = client.chat.completions.create(...)

print(f"Cost: ${tracker.total_cost:.6f}")

# Remove patches when done
unpatch_openai()
```

### Streaming Support

Streaming responses are fully supported with automatic cost tracking:

```python
client = track_openai(OpenAI(), tracker)

stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")

# Cost is tracked after stream completes
print(f"\nCost: ${tracker.total_cost:.6f}")
```

## RAG Cost Tracking

For RAG applications, you can set separate budgets for embeddings and completions:

```python
from tokencost import (
    CostTracker,
    EmbeddingBudgetExceededError,
    CompletionBudgetExceededError,
)

tracker = CostTracker(
    budget=1.00,              # Total budget
    embedding_budget=0.10,    # Limit embedding costs
    completion_budget=0.90,   # Limit completion costs
    raise_on_budget=True
)

# With separate callbacks
tracker = CostTracker(
    embedding_budget=0.10,
    completion_budget=0.50,
    on_embedding_budget_exceeded=lambda t: print("Embedding budget exceeded!"),
    on_completion_budget_exceeded=lambda t: print("Completion budget exceeded!"),
)

# Track costs by type
print(f"Embedding cost: ${tracker.embedding_cost:.6f} ({tracker.embedding_count} requests)")
print(f"Completion cost: ${tracker.completion_cost:.6f} ({tracker.completion_count} requests)")

# Check budget status
print(f"Embedding budget exceeded: {tracker.embedding_budget_exceeded}")
print(f"Completion budget exceeded: {tracker.completion_budget_exceeded}")
```

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
    budget: float | None = None,           # Total spending limit in USD
    embedding_budget: float | None = None, # Embedding-specific budget
    completion_budget: float | None = None,# Completion-specific budget
    on_budget_exceeded: Callable | None = None,  # Callback when total exceeded
    on_embedding_budget_exceeded: Callable | None = None,  # Callback for embeddings
    on_completion_budget_exceeded: Callable | None = None, # Callback for completions
    raise_on_budget: bool = False,         # Raise exception when exceeded
    print_summary: bool = True             # Print summary on program exit
)
```

**Properties:**

- `total_cost: float` — Running total in USD
- `request_count: int` — Number of successful requests
- `history: list[dict]` — All logged requests
- `budget: float | None` — Configured total budget
- `budget_exceeded: bool` — Whether total budget has been exceeded
- `cost_by_model: dict[str, float]` — Cost aggregated by model name
- `embedding_cost: float` — Total embedding cost in USD
- `completion_cost: float` — Total completion cost in USD
- `embedding_count: int` — Number of embedding requests
- `completion_count: int` — Number of completion requests
- `embedding_budget: float | None` — Configured embedding budget
- `completion_budget: float | None` — Configured completion budget
- `embedding_budget_exceeded: bool` — Whether embedding budget exceeded
- `completion_budget_exceeded: bool` — Whether completion budget exceeded
- `cost_by_request_type: dict[str, float]` — Cost breakdown by type

**Methods:**

- `reset()` — Clear all tracked data

### OpenAI Integration

```python
# Wrap a client instance
track_openai(client, tracker) -> WrappedClient

# Global patching
patch_openai(tracker)   # Patch all OpenAI clients
unpatch_openai()        # Remove patches
```

### Exceptions

```python
class BudgetExceededError(Exception):
    budget: float       # Configured budget
    total_cost: float   # Actual spend when exceeded

class EmbeddingBudgetExceededError(BudgetExceededError):
    # Raised when embedding budget is exceeded

class CompletionBudgetExceededError(BudgetExceededError):
    # Raised when completion budget is exceeded
```

### Pricing Utilities

```python
from tokencost import (
    calculate_cost,
    calculate_embedding_cost,
    get_model_pricing,
    is_embedding_model,
    list_models,
)

# Calculate cost for a completion
cost = calculate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)

# Calculate cost for embeddings
cost = calculate_embedding_cost("text-embedding-3-small", input_tokens=1000)

# Get pricing info for a model
pricing = get_model_pricing("gpt-4o")
print(pricing["input_cost_per_token"])

# Check if model is an embedding model
is_embedding_model("text-embedding-3-small")  # True

# List all supported models
models = list_models()
```

## Exit Summary

When your program ends, a cost summary is automatically printed:

```
==================================================
LLM COST SUMMARY
==================================================
Total Cost:     $0.002459
Total Requests: 5
Total Budget:   $1.00 (OK)
Remaining:      $0.997541
--------------------------------------------------
By Type:
  Embeddings:  1 requests = $0.000500 | Budget: $0.10 (OK)
  Completions: 4 requests = $0.001959 | Budget: $0.90 (OK)
--------------------------------------------------
Requests:
  1. [C] gpt-4: 7+18 tokens = $0.000750
  2. [C] gpt-4: 13+17 tokens = $0.000900
  3. [E] text-embedding-3-small: 100+0 tokens = $0.000500
  4. [C] gpt-3.5-turbo: 8+82 tokens = $0.000166
  5. [C] gpt-3.5-turbo: 10+58 tokens = $0.000143
==================================================
```

`[C]` = Completion, `[E]` = Embedding. Disable with `print_summary=False`.

## History Entry Format

Each request is logged with:

```python
{
    "model": "gpt-4",
    "prompt_tokens": 150,
    "completion_tokens": 50,
    "cost": 0.0123,
    "timestamp": "2026-02-22T10:30:00Z",
    "request_type": "completion"  # or "embedding"
}
```

## Development

```bash
git clone https://github.com/Paawan13/llm-tokencost.git
cd tokencost
pip install -e ".[dev]"
pytest
```

## License

MIT
