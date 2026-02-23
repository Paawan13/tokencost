"""LLM Cost Tracker - Track LLM API costs via litellm's callback system."""

from .exceptions import BudgetExceededError
from .openai_wrapper import patch_openai, track_openai, unpatch_openai
from .pricing import calculate_cost, get_model_pricing, list_models
from .tracker import CostTracker

__version__ = "0.4.0"
__all__ = [
    "CostTracker",
    "BudgetExceededError",
    "calculate_cost",
    "get_model_pricing",
    "list_models",
    "track_openai",
    "patch_openai",
    "unpatch_openai",
]
