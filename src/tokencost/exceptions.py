"""Exception classes for LLM cost tracking."""


class BudgetExceededError(Exception):
    """Raised when the configured budget has been exceeded.

    Attributes:
        budget: The configured budget limit in USD.
        total_cost: The actual spend when the budget was exceeded.
    """

    def __init__(self, budget: float, total_cost: float) -> None:
        self.budget = budget
        self.total_cost = total_cost
        super().__init__(
            f"Budget exceeded: spent ${total_cost:.4f} (budget: ${budget:.2f})"
        )
