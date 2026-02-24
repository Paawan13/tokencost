"""LLM Cost Tracker - Track LLM API costs via litellm's callback system."""

from .anthropic_wrapper import patch_anthropic, track_anthropic, unpatch_anthropic
from .exceptions import (
    BudgetExceededError,
    CompletionBudgetExceededError,
    EmbeddingBudgetExceededError,
)
from .gemini_wrapper import patch_gemini, track_gemini, unpatch_gemini
from .openai_wrapper import patch_openai, track_openai, unpatch_openai
from .pricing import (
    calculate_cost,
    calculate_embedding_cost,
    get_model_pricing,
    is_embedding_model,
    list_models,
)
from .tracker import CostTracker

__version__ = "0.6.0"
__all__ = [
    "BudgetExceededError",
    "calculate_cost",
    "calculate_embedding_cost",
    "CompletionBudgetExceededError",
    "CostTracker",
    "EmbeddingBudgetExceededError",
    "get_model_pricing",
    "is_embedding_model",
    "list_models",
    "patch_anthropic",
    "patch_gemini",
    "patch_openai",
    "track_anthropic",
    "track_gemini",
    "track_openai",
    "unpatch_anthropic",
    "unpatch_gemini",
    "unpatch_openai",
]
