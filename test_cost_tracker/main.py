"""
Test project for litellm-cost-tracker library.
"""

import litellm
from tokencost import CostTracker, BudgetExceededError


def budget_alert(tracker):
    """Called when budget is exceeded."""
    print(f"\n{'='*50}")
    print(f"BUDGET ALERT! Spent ${tracker.total_cost:.4f} of ${tracker.budget:.4f}")
    print(f"{'='*50}\n")


def main():
    # Configure litellm for custom endpoint
    litellm.api_key = "sk-CvtN3R_6STSDIfTXN5h6KA"
    litellm.api_base = "https://litellm.n1-research.com"

    # Initialize cost tracker with $0.01 budget
    tracker = CostTracker(
        budget=0.01,
        on_budget_exceeded=budget_alert,
        raise_on_budget=False  # Don't raise, just alert
    )
    litellm.callbacks = [tracker]

    print("=" * 50)
    print("LLM Cost Tracker Test Project")
    print("=" * 50)
    print(f"Budget: ${tracker.budget:.2f}")
    print()

    # Warmup call (litellm quirk)
    litellm.completion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "hi"}]
    )

    # Test conversations
    conversations = [
        "What is Python?",
        "Explain machine learning in one sentence.",
        "What is 2 + 2?",
        "Say hello in French.",
        "What's the capital of Japan?",
    ]

    for i, prompt in enumerate(conversations, 1):
        print(f"[Request {i}] {prompt}")

        response = litellm.completion(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content[:80]
        # Handle non-ASCII characters for Windows console
        answer_safe = answer.encode('ascii', 'replace').decode('ascii')
        print(f"  Answer: {answer_safe}...")
        print(f"  Running cost: ${tracker.total_cost:.6f} | Requests: {tracker.request_count}")
        print()

    # Final summary
    print("=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(f"Total Cost: ${tracker.total_cost:.6f}")
    print(f"Total Requests: {tracker.request_count}")
    print(f"Budget: ${tracker.budget:.4f}")
    print(f"Budget Exceeded: {tracker.budget_exceeded}")
    print()

    print("Request History:")
    for i, entry in enumerate(tracker.history, 1):
        print(f"  {i}. {entry['model']}: {entry['prompt_tokens']}+{entry['completion_tokens']} tokens = ${entry['cost']:.6f}")


if __name__ == "__main__":
    main()
