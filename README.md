# tokencost

A lightweight Python library for tracking LLM API costs with budget alerts and spending limits. Works directly with **OpenAI** and **Anthropic** SDKs.

## Installation

```bash
pip install llm-tokencost
```

With provider SDKs:

```bash
# For OpenAI SDK integration
pip install llm-tokencost[openai]

# For Anthropic SDK integration
pip install llm-tokencost[anthropic]

# For all providers
pip install llm-tokencost[all]
```

## Quick Start

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

### With Anthropic SDK

```python
from anthropic import Anthropic
from tokencost import CostTracker, track_anthropic

tracker = CostTracker(budget=1.0)
client = track_anthropic(Anthropic(), tracker)

# Use the client as normal - costs are tracked automatically
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

print(f"Cost: ${tracker.total_cost:.6f}")
```

### With Budget Alerts

```python
from openai import OpenAI
from tokencost import CostTracker, BudgetExceededError, track_openai

def alert(tracker):
    print(f"Budget exceeded! Spent ${tracker.total_cost:.2f}")

tracker = CostTracker(
    budget=5.00,
    on_budget_exceeded=alert,
    raise_on_budget=True
)

client = track_openai(OpenAI(), tracker)

try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
except BudgetExceededError as e:
    print(f"Stopped at ${e.total_cost:.2f} (budget: ${e.budget:.2f})")

print(f"Total: ${tracker.total_cost:.4f} across {tracker.request_count} requests")
```

## Features

- **Real-time cost tracking** during LLM API calls
- **Budget alerts** via callback and/or exception
- **OpenAI SDK support** — track `chat.completions` and `embeddings`
- **Anthropic SDK support** — track `messages` API
- **Async support** — works with `AsyncOpenAI` and `AsyncAnthropic`
- **Streaming support** — costs tracked after stream completes
- **Per-model cost aggregation** via `cost_by_model` property
- **RAG cost tracking** — separate budgets for embeddings vs completions
- **Automatic exit summary** — prints cost report when program ends
- **Thread-safe** for concurrent usage
- **Accurate pricing** for 1600+ models via litellm's pricing database

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

## Anthropic SDK Integration

### Wrapping a Client

Use `track_anthropic()` to wrap an Anthropic client instance:

```python
from anthropic import Anthropic, AsyncAnthropic
from tokencost import CostTracker, track_anthropic

tracker = CostTracker(budget=1.0)

# Wrap sync client
client = track_anthropic(Anthropic(), tracker)

# Or wrap async client
async_client = track_anthropic(AsyncAnthropic(), tracker)

# Messages are tracked automatically
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

print(f"Cost: ${tracker.total_cost:.6f}")
```

### Global Patching

Use `patch_anthropic()` to automatically track all Anthropic client instances:

```python
from anthropic import Anthropic
from tokencost import CostTracker, patch_anthropic, unpatch_anthropic

tracker = CostTracker()
patch_anthropic(tracker)

# All clients now track costs automatically
client = Anthropic()
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

print(f"Cost: ${tracker.total_cost:.6f}")

# Remove patches when done
unpatch_anthropic()
```

### Streaming Support

Streaming responses are fully supported:

```python
client = track_anthropic(Anthropic(), tracker)

with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="")

# Cost is tracked after stream completes
print(f"\nCost: ${tracker.total_cost:.6f}")
```

> **Note:** Anthropic does not provide embedding models. For embeddings, use OpenAI, Voyage AI, or other embedding providers.

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
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from tokencost import CostTracker, track_openai, track_anthropic

async def main():
    tracker = CostTracker()

    # Async OpenAI
    openai_client = track_openai(AsyncOpenAI(), tracker)
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    # Async Anthropic
    anthropic_client = track_anthropic(AsyncAnthropic(), tracker)
    response = await anthropic_client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}]
    )

    print(f"Cost: ${tracker.total_cost:.6f}")

asyncio.run(main())
```

## Per-Model Cost Breakdown

```python
from openai import OpenAI
from anthropic import Anthropic
from tokencost import CostTracker, track_openai, track_anthropic

tracker = CostTracker()

openai_client = track_openai(OpenAI(), tracker)
anthropic_client = track_anthropic(Anthropic(), tracker)

# Make calls to different models...
openai_client.chat.completions.create(model="gpt-4o", messages=[...])
openai_client.chat.completions.create(model="gpt-4o-mini", messages=[...])
anthropic_client.messages.create(model="claude-opus-4-6", max_tokens=1024, messages=[...])

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

### Anthropic Integration

```python
# Wrap a client instance
track_anthropic(client, tracker) -> WrappedClient

# Global patching
patch_anthropic(tracker)   # Patch all Anthropic clients
unpatch_anthropic()        # Remove patches
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
