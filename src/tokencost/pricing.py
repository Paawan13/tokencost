"""Standalone cost calculation using litellm's pricing database."""

from litellm import cost_per_token, model_cost

# Custom pricing for Moonshot/Kimi models (not reliably in litellm)
# Prices are in USD per token
MOONSHOT_PRICING = {
    # Kimi K2.5 (Multimodal)
    "moonshot/kimi-k2.5": {"input": 0.60e-6, "output": 3.00e-6, "max_tokens": 262144},
    # Kimi K2 Generation
    "moonshot/kimi-k2-0905-preview": {
        "input": 0.60e-6,
        "output": 2.50e-6,
        "max_tokens": 262144,
    },
    "moonshot/kimi-k2-0711-preview": {
        "input": 0.60e-6,
        "output": 2.50e-6,
        "max_tokens": 131072,
    },
    "moonshot/kimi-k2-turbo-preview": {
        "input": 1.15e-6,
        "output": 8.00e-6,
        "max_tokens": 262144,
    },
    "moonshot/kimi-k2-thinking": {
        "input": 0.60e-6,
        "output": 2.50e-6,
        "max_tokens": 262144,
    },
    "moonshot/kimi-k2-thinking-turbo": {
        "input": 1.15e-6,
        "output": 8.00e-6,
        "max_tokens": 262144,
    },
    # Moonshot V1 Legacy
    "moonshot/moonshot-v1-8k": {"input": 0.20e-6, "output": 2.00e-6, "max_tokens": 8192},
    "moonshot/moonshot-v1-32k": {
        "input": 1.00e-6,
        "output": 3.00e-6,
        "max_tokens": 32768,
    },
    "moonshot/moonshot-v1-128k": {
        "input": 2.00e-6,
        "output": 5.00e-6,
        "max_tokens": 131072,
    },
}


def _normalize_moonshot_model(model: str) -> str:
    """Normalize Moonshot/Kimi model name to canonical form.

    Adds 'moonshot/' prefix if the model is a Kimi or Moonshot model
    without the prefix.

    Args:
        model: The model name.

    Returns:
        Normalized model name with 'moonshot/' prefix if applicable.
    """
    if model.startswith("moonshot/"):
        return model
    if model.startswith("kimi-") or model.startswith("moonshot-"):
        return f"moonshot/{model}"
    return model


def _get_moonshot_pricing(model: str) -> dict | None:
    """Get pricing for a Moonshot/Kimi model from custom pricing dict.

    Args:
        model: The model name (will be normalized).

    Returns:
        Pricing dict with 'input' and 'output' cost per token, or None if not found.
    """
    normalized = _normalize_moonshot_model(model)
    return MOONSHOT_PRICING.get(normalized)


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Calculate cost from model name and token counts.

    Uses litellm's pricing database to compute the total cost for a given
    model and token usage. Also supports custom pricing for Moonshot/Kimi models.

    Args:
        model: The model name (e.g., "gpt-4o", "claude-3-opus-20240229", "kimi-k2.5").
        prompt_tokens: Number of input/prompt tokens.
        completion_tokens: Number of output/completion tokens.

    Returns:
        Total cost in USD.

    Raises:
        ValueError: If the model is not found in pricing database.

    Example:
        >>> cost = calculate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
        >>> print(f"${cost:.6f}")
    """
    # Check custom Moonshot/Kimi pricing first
    moonshot_pricing = _get_moonshot_pricing(model)
    if moonshot_pricing is not None:
        prompt_cost = prompt_tokens * moonshot_pricing["input"]
        completion_cost = completion_tokens * moonshot_pricing["output"]
        return prompt_cost + completion_cost

    try:
        prompt_cost, completion_cost = cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return prompt_cost + completion_cost
    except Exception as e:
        raise ValueError(f"Unknown model: {model}. Error: {e}") from e


def get_model_pricing(model: str) -> dict:
    """Get pricing info for a model.

    Args:
        model: The model name (e.g., "gpt-4o", "claude-3-opus-20240229", "kimi-k2.5").

    Returns:
        Dictionary containing pricing information with keys like:
        - input_cost_per_token: Cost per input token in USD
        - output_cost_per_token: Cost per output token in USD
        - max_tokens: Maximum context length
        - And other model-specific metadata

    Raises:
        ValueError: If the model is not found in pricing database.

    Example:
        >>> pricing = get_model_pricing("gpt-4o")
        >>> print(f"Input: ${pricing['input_cost_per_token']}/token")
    """
    # Check custom Moonshot/Kimi pricing first
    moonshot_pricing = _get_moonshot_pricing(model)
    if moonshot_pricing is not None:
        return {
            "input_cost_per_token": moonshot_pricing["input"],
            "output_cost_per_token": moonshot_pricing["output"],
            "max_tokens": moonshot_pricing["max_tokens"],
        }

    # litellm's model_cost is a dict of model_name -> pricing info
    if model in model_cost:
        return model_cost[model].copy()

    # Try with provider prefix (e.g., "openai/gpt-4o" for "gpt-4o")
    for key in model_cost:
        if key == model or key.endswith("/" + model):
            return model_cost[key].copy()

    raise ValueError(f"Unknown model: {model}")


def list_models() -> list[str]:
    """List all supported models.

    Returns:
        List of all model names supported by litellm's pricing database.

    Example:
        >>> models = list_models()
        >>> gpt_models = [m for m in models if "gpt" in m]
    """
    return list(model_cost.keys())


def is_embedding_model(model: str) -> bool:
    """Check if a model is an embedding model.

    Args:
        model: The model name (e.g., "text-embedding-3-small", "gpt-4o").

    Returns:
        True if the model is an embedding model, False otherwise.

    Example:
        >>> is_embedding_model("text-embedding-3-small")
        True
        >>> is_embedding_model("gpt-4o")
        False
    """
    # Normalize model name and check for embedding indicators
    model_lower = model.lower()

    # Common embedding model patterns
    embedding_patterns = [
        "embedding",
        "embed",
        "ada-002",
        "voyage",
        "e5-",
        "bge-",
        "gte-",
    ]

    for pattern in embedding_patterns:
        if pattern in model_lower:
            return True

    # Also check litellm's model_cost for mode="embedding"
    model_info = None
    if model in model_cost:
        model_info = model_cost[model]
    else:
        # Try with provider prefix
        for key in model_cost:
            if key == model or key.endswith("/" + model):
                model_info = model_cost[key]
                break

    if model_info and model_info.get("mode") == "embedding":
        return True

    return False


def calculate_embedding_cost(model: str, input_tokens: int) -> float:
    """Calculate cost for embedding requests (input tokens only).

    Embedding models only use input tokens, so this function calculates
    the cost based on input token count alone.

    Args:
        model: The embedding model name (e.g., "text-embedding-3-small").
        input_tokens: Number of input tokens.

    Returns:
        Total cost in USD.

    Raises:
        ValueError: If the model is not found in litellm's pricing database.

    Example:
        >>> cost = calculate_embedding_cost("text-embedding-3-small", input_tokens=1000)
        >>> print(f"${cost:.6f}")
    """
    try:
        # Embeddings only have input tokens, no completion tokens
        prompt_cost, _ = cost_per_token(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=0,
        )
        return prompt_cost
    except Exception as e:
        raise ValueError(f"Unknown model: {model}. Error: {e}") from e
