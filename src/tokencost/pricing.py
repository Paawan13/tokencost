"""Standalone cost calculation using litellm's pricing database."""

from litellm import cost_per_token, model_cost


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Calculate cost from model name and token counts.

    Uses litellm's pricing database to compute the total cost for a given
    model and token usage.

    Args:
        model: The model name (e.g., "gpt-4o", "claude-3-opus-20240229").
        prompt_tokens: Number of input/prompt tokens.
        completion_tokens: Number of output/completion tokens.

    Returns:
        Total cost in USD.

    Raises:
        ValueError: If the model is not found in litellm's pricing database.

    Example:
        >>> cost = calculate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
        >>> print(f"${cost:.6f}")
    """
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
        model: The model name (e.g., "gpt-4o", "claude-3-opus-20240229").

    Returns:
        Dictionary containing pricing information with keys like:
        - input_cost_per_token: Cost per input token in USD
        - output_cost_per_token: Cost per output token in USD
        - max_tokens: Maximum context length
        - And other model-specific metadata

    Raises:
        ValueError: If the model is not found in litellm's pricing database.

    Example:
        >>> pricing = get_model_pricing("gpt-4o")
        >>> print(f"Input: ${pricing['input_cost_per_token']}/token")
    """
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
