"""LLM Cost Tracker - Track LLM API costs via litellm's callback system."""

from .exceptions import BudgetExceededError
from .tracker import CostTracker

__version__ = "0.2.0"
__all__ = ["CostTracker", "BudgetExceededError"]
