"""LLM Cost Tracker - Track LLM API costs via litellm's callback system."""

from .exceptions import (
    BudgetExceededError,
    CompletionBudgetExceededError,
    EmbeddingBudgetExceededError,
)
from .openai_wrapper import patch_openai, track_openai, unpatch_openai
from .pricing import (
    calculate_cost,
    calculate_embedding_cost,
    get_model_pricing,
    is_embedding_model,
    list_models,
)
from .tracker import CostTracker

__version__ = "0.5.0"
__all__ = [
    "CostTracker",
    "BudgetExceededError",
    "EmbeddingBudgetExceededError",
    "CompletionBudgetExceededError",
    "calculate_cost",
    "calculate_embedding_cost",
    "get_model_pricing",
    "is_embedding_model",
    "list_models",
    "track_openai",
    "patch_openai",
    "unpatch_openai",
]
