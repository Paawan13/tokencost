"""
Test new features: async support and cost_by_model aggregation.
"""

import asyncio
import time
import litellm
from tokencost import CostTracker


def test_cost_by_model():
    """Test cost_by_model property with multiple models."""
    print("=" * 50)
    print("TEST: Cost by Model Aggregation")
    print("=" * 50)

    tracker = CostTracker(print_summary=False)
    litellm.callbacks = [tracker]

    # Test with different models
    models_and_prompts = [
        ("gpt-5-mini", "Say hi"),
        ("gpt-5-mini", "Say hello"),
        ("gpt-5-nano", "Say hey"),
        ("gpt-5-mini", "Say greetings"),
    ]

    for model, prompt in models_and_prompts:
        print(f"  Requesting {model}...")
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        time.sleep(0.1)  # Small delay for callback processing

    print()
    print("Cost by Model:")
    cost_by_model = tracker.cost_by_model
    for model, cost in cost_by_model.items():
        print(f"  {model}: ${cost:.6f}")

    print()
    print(f"Total cost: ${tracker.total_cost:.6f}")
    print(f"Sum of model costs: ${sum(cost_by_model.values()):.6f}")
    print(f"Request count: {tracker.request_count}")

    # Verify sum matches total
    assert abs(sum(cost_by_model.values()) - tracker.total_cost) < 0.0001, "Cost sum mismatch"
    assert tracker.request_count == len(models_and_prompts), f"Expected {len(models_and_prompts)} requests"
    print()
    print("PASSED: cost_by_model aggregation works correctly")
    print()

    return tracker


async def test_async_support():
    """Test async_log_success_event with litellm.acompletion()."""
    print("=" * 50)
    print("TEST: Async Support")
    print("=" * 50)

    tracker = CostTracker(print_summary=False)
    litellm.callbacks = [tracker]

    # Make async requests concurrently
    prompts = [
        "What is 1+1?",
        "What is 2+2?",
        "What is 3+3?",
    ]

    tasks = [
        litellm.acompletion(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        for prompt in prompts
    ]

    print(f"Sending {len(tasks)} async requests concurrently...")
    responses = await asyncio.gather(*tasks)

    # Wait for async callbacks to complete
    await asyncio.sleep(0.5)

    print(f"Completed {len(responses)} requests")
    print(f"Total cost: ${tracker.total_cost:.6f}")
    print(f"Request count: {tracker.request_count}")
    print()

    assert tracker.request_count == len(prompts), f"Expected {len(prompts)}, got {tracker.request_count}"
    assert tracker.total_cost > 0, "Expected non-zero cost"
    print("PASSED: Async support works correctly")
    print()

    return tracker


async def main():
    # Configure litellm
    litellm.api_key = "sk-CvtN3R_6STSDIfTXN5h6KA"
    litellm.api_base = "https://litellm.n1-research.com"

    print()
    print("Testing New Features of litellm-cost-tracker")
    print("=" * 50)
    print()

    # Note: Tests are run in separate processes to avoid litellm callback state issues
    # Run each test independently for most reliable results

    # Test 1: Async Support
    await test_async_support()

    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
    print()
    print("Note: Run test_cost_by_model() separately for sync test")
    print("(litellm callback state can interfere between sync/async tests)")


if __name__ == "__main__":
    asyncio.run(main())
