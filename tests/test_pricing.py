"""Tests for the pricing module."""

import pytest

from tokencost.pricing import (
    MOONSHOT_PRICING,
    _get_moonshot_pricing,
    _normalize_moonshot_model,
    calculate_cost,
    calculate_embedding_cost,
    get_model_pricing,
    is_embedding_model,
    list_models,
)

# All expected Moonshot/Kimi models for parametrized tests
EXPECTED_MOONSHOT_MODELS = [
    "moonshot/kimi-k2.5",
    "moonshot/kimi-k2-0905-preview",
    "moonshot/kimi-k2-0711-preview",
    "moonshot/kimi-k2-turbo-preview",
    "moonshot/kimi-k2-thinking",
    "moonshot/kimi-k2-thinking-turbo",
    "moonshot/moonshot-v1-8k",
    "moonshot/moonshot-v1-32k",
    "moonshot/moonshot-v1-128k",
]


class TestCalculateCost:
    """Tests for calculate_cost function."""

    def test_calculate_cost_gpt4o(self):
        """Test cost calculation for gpt-4o."""
        cost = calculate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
        assert cost > 0
        assert isinstance(cost, float)

    def test_calculate_cost_gpt35_turbo(self):
        """Test cost calculation for gpt-3.5-turbo."""
        cost = calculate_cost("gpt-3.5-turbo", prompt_tokens=1000, completion_tokens=500)
        assert cost > 0
        assert isinstance(cost, float)

    def test_calculate_cost_claude(self):
        """Test cost calculation for Claude models."""
        cost = calculate_cost(
            "claude-3-opus-20240229", prompt_tokens=1000, completion_tokens=500
        )
        assert cost > 0
        assert isinstance(cost, float)

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        cost = calculate_cost("gpt-4o", prompt_tokens=0, completion_tokens=0)
        assert cost == 0.0

    def test_calculate_cost_prompt_only(self):
        """Test cost calculation with only prompt tokens."""
        cost = calculate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=0)
        assert cost > 0

    def test_calculate_cost_completion_only(self):
        """Test cost calculation with only completion tokens."""
        cost = calculate_cost("gpt-4o", prompt_tokens=0, completion_tokens=500)
        assert cost > 0

    def test_calculate_cost_unknown_model(self):
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            calculate_cost(
                "unknown-model-xyz-123", prompt_tokens=1000, completion_tokens=500
            )

    def test_calculate_cost_relative_pricing(self):
        """Test that GPT-4o costs less than GPT-4 but more than GPT-3.5."""
        gpt4_cost = calculate_cost("gpt-4", prompt_tokens=1000, completion_tokens=500)
        gpt4o_cost = calculate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
        gpt35_cost = calculate_cost(
            "gpt-3.5-turbo", prompt_tokens=1000, completion_tokens=500
        )

        # GPT-4 should be more expensive than GPT-4o
        assert gpt4_cost > gpt4o_cost
        # GPT-3.5 should be cheaper than GPT-4o
        assert gpt35_cost < gpt4o_cost


class TestGetModelPricing:
    """Tests for get_model_pricing function."""

    def test_get_model_pricing_gpt4o(self):
        """Test getting pricing info for gpt-4o."""
        pricing = get_model_pricing("gpt-4o")
        assert isinstance(pricing, dict)
        # Should have some pricing keys
        assert len(pricing) > 0

    def test_get_model_pricing_unknown_model(self):
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_pricing("unknown-model-xyz-123")

    def test_get_model_pricing_returns_copy(self):
        """Test that get_model_pricing returns a copy."""
        pricing1 = get_model_pricing("gpt-4o")
        pricing2 = get_model_pricing("gpt-4o")
        # Should be equal but not the same object
        assert pricing1 == pricing2
        assert pricing1 is not pricing2


class TestListModels:
    """Tests for list_models function."""

    def test_list_models_returns_list(self):
        """Test that list_models returns a list."""
        models = list_models()
        assert isinstance(models, list)

    def test_list_models_not_empty(self):
        """Test that list_models returns a non-empty list."""
        models = list_models()
        assert len(models) > 0

    def test_list_models_contains_common_models(self):
        """Test that common models are in the list."""
        models = list_models()
        # Check for at least some common models
        model_str = " ".join(models)
        assert "gpt-4" in model_str or "gpt4" in model_str
        assert "gpt-3.5" in model_str or "gpt35" in model_str

    def test_list_models_all_strings(self):
        """Test that all model names are strings."""
        models = list_models()
        for model in models:
            assert isinstance(model, str)


class TestIsEmbeddingModel:
    """Tests for is_embedding_model function."""

    def test_is_embedding_model_openai_embeddings(self):
        """Test OpenAI embedding models are detected."""
        assert is_embedding_model("text-embedding-3-small") is True
        assert is_embedding_model("text-embedding-3-large") is True
        assert is_embedding_model("text-embedding-ada-002") is True

    def test_is_embedding_model_not_embedding(self):
        """Test that completion models are not detected as embedding."""
        assert is_embedding_model("gpt-4o") is False
        assert is_embedding_model("gpt-3.5-turbo") is False
        assert is_embedding_model("claude-3-opus-20240229") is False

    def test_is_embedding_model_case_insensitive(self):
        """Test that detection is case insensitive."""
        assert is_embedding_model("TEXT-EMBEDDING-3-SMALL") is True
        assert is_embedding_model("Text-Embedding-3-Large") is True

    def test_is_embedding_model_voyage_pattern(self):
        """Test Voyage embedding models are detected."""
        assert is_embedding_model("voyage-2") is True
        assert is_embedding_model("voyage-large-2") is True


class TestCalculateEmbeddingCost:
    """Tests for calculate_embedding_cost function."""

    def test_calculate_embedding_cost_basic(self):
        """Test basic embedding cost calculation."""
        cost = calculate_embedding_cost("text-embedding-3-small", input_tokens=1000)
        assert cost > 0
        assert isinstance(cost, float)

    def test_calculate_embedding_cost_zero_tokens(self):
        """Test embedding cost with zero tokens."""
        cost = calculate_embedding_cost("text-embedding-3-small", input_tokens=0)
        assert cost == 0.0

    def test_calculate_embedding_cost_unknown_model(self):
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            calculate_embedding_cost("unknown-embedding-model", input_tokens=1000)

    def test_calculate_embedding_cost_relative_pricing(self):
        """Test that small embedding model is cheaper than large."""
        small_cost = calculate_embedding_cost(
            "text-embedding-3-small", input_tokens=1000
        )
        large_cost = calculate_embedding_cost(
            "text-embedding-3-large", input_tokens=1000
        )

        # Large should be more expensive than small
        assert large_cost > small_cost

    def test_calculate_embedding_cost_much_cheaper_than_completion(self):
        """Test that embedding is much cheaper than completion."""
        embedding_cost = calculate_embedding_cost(
            "text-embedding-3-small", input_tokens=1000
        )
        completion_cost = calculate_cost(
            "gpt-4o", prompt_tokens=1000, completion_tokens=0
        )

        # Embedding should be significantly cheaper
        assert embedding_cost < completion_cost


class TestMoonshotNormalization:
    """Tests for Moonshot/Kimi model name normalization."""

    def test_normalize_kimi_model(self):
        """Test normalizing kimi- prefixed models."""
        assert _normalize_moonshot_model("kimi-k2.5") == "moonshot/kimi-k2.5"
        assert _normalize_moonshot_model("kimi-k2-0905-preview") == "moonshot/kimi-k2-0905-preview"
        assert _normalize_moonshot_model("kimi-k2-thinking") == "moonshot/kimi-k2-thinking"

    def test_normalize_moonshot_model(self):
        """Test normalizing moonshot- prefixed models."""
        assert _normalize_moonshot_model("moonshot-v1-8k") == "moonshot/moonshot-v1-8k"
        assert _normalize_moonshot_model("moonshot-v1-128k") == "moonshot/moonshot-v1-128k"

    def test_normalize_already_prefixed(self):
        """Test that already prefixed models are unchanged."""
        assert _normalize_moonshot_model("moonshot/kimi-k2.5") == "moonshot/kimi-k2.5"
        assert _normalize_moonshot_model("moonshot/moonshot-v1-8k") == "moonshot/moonshot-v1-8k"

    def test_normalize_non_moonshot(self):
        """Test that non-Moonshot models are unchanged."""
        assert _normalize_moonshot_model("gpt-4o") == "gpt-4o"
        assert _normalize_moonshot_model("claude-3-opus") == "claude-3-opus"


class TestMoonshotPricingLookup:
    """Tests for Moonshot pricing dictionary lookup."""

    def test_get_moonshot_pricing_kimi_k2(self):
        """Test getting pricing for Kimi K2 models."""
        pricing = _get_moonshot_pricing("kimi-k2.5")
        assert pricing is not None
        assert pricing["input"] == 0.60e-6
        assert pricing["output"] == 3.00e-6

    def test_get_moonshot_pricing_with_prefix(self):
        """Test getting pricing with moonshot/ prefix."""
        pricing = _get_moonshot_pricing("moonshot/kimi-k2.5")
        assert pricing is not None
        assert pricing["input"] == 0.60e-6

    def test_get_moonshot_pricing_legacy_model(self):
        """Test getting pricing for legacy moonshot-v1 models."""
        pricing = _get_moonshot_pricing("moonshot-v1-8k")
        assert pricing is not None
        assert pricing["input"] == 0.20e-6
        assert pricing["output"] == 2.00e-6

    def test_get_moonshot_pricing_unknown(self):
        """Test that unknown models return None."""
        assert _get_moonshot_pricing("gpt-4o") is None
        assert _get_moonshot_pricing("unknown-model") is None


class TestMoonshotCalculateCost:
    """Tests for calculate_cost with Moonshot/Kimi models."""

    def test_calculate_cost_kimi_k25(self):
        """Test cost calculation for kimi-k2.5."""
        # 1M input tokens should cost $0.60
        cost = calculate_cost("kimi-k2.5", prompt_tokens=1_000_000, completion_tokens=0)
        assert cost == pytest.approx(0.60)

        # 1M output tokens should cost $3.00
        cost = calculate_cost("kimi-k2.5", prompt_tokens=0, completion_tokens=1_000_000)
        assert cost == pytest.approx(3.00)

    def test_calculate_cost_kimi_k2_preview(self):
        """Test cost calculation for kimi-k2-0905-preview."""
        cost = calculate_cost("kimi-k2-0905-preview", prompt_tokens=1_000_000, completion_tokens=0)
        assert cost == pytest.approx(0.60)

        cost = calculate_cost("kimi-k2-0905-preview", prompt_tokens=0, completion_tokens=1_000_000)
        assert cost == pytest.approx(2.50)

    def test_calculate_cost_kimi_turbo_more_expensive(self):
        """Test that turbo models cost more than standard."""
        standard_cost = calculate_cost(
            "kimi-k2-0905-preview", prompt_tokens=1000, completion_tokens=500
        )
        turbo_cost = calculate_cost(
            "kimi-k2-turbo-preview", prompt_tokens=1000, completion_tokens=500
        )
        assert turbo_cost > standard_cost

    def test_calculate_cost_moonshot_v1_tiers(self):
        """Test moonshot-v1 context tier pricing."""
        # 8k should be cheapest
        cost_8k = calculate_cost("moonshot-v1-8k", prompt_tokens=1000, completion_tokens=500)
        cost_32k = calculate_cost("moonshot-v1-32k", prompt_tokens=1000, completion_tokens=500)
        cost_128k = calculate_cost("moonshot-v1-128k", prompt_tokens=1000, completion_tokens=500)

        assert cost_8k < cost_32k < cost_128k

    def test_calculate_cost_with_prefix(self):
        """Test cost calculation with moonshot/ prefix."""
        cost1 = calculate_cost("kimi-k2.5", prompt_tokens=1000, completion_tokens=500)
        cost2 = calculate_cost("moonshot/kimi-k2.5", prompt_tokens=1000, completion_tokens=500)
        assert cost1 == cost2

    def test_calculate_cost_zero_tokens(self):
        """Test zero token cost calculation."""
        cost = calculate_cost("kimi-k2.5", prompt_tokens=0, completion_tokens=0)
        assert cost == 0.0


class TestMoonshotGetModelPricing:
    """Tests for get_model_pricing with Moonshot/Kimi models."""

    def test_get_model_pricing_kimi(self):
        """Test getting pricing info for Kimi models."""
        pricing = get_model_pricing("kimi-k2.5")
        assert "input_cost_per_token" in pricing
        assert "output_cost_per_token" in pricing
        assert "max_tokens" in pricing
        assert pricing["max_tokens"] == 262144

    def test_get_model_pricing_moonshot_legacy(self):
        """Test getting pricing info for legacy moonshot models."""
        pricing = get_model_pricing("moonshot-v1-8k")
        assert pricing["max_tokens"] == 8192

        pricing = get_model_pricing("moonshot-v1-128k")
        assert pricing["max_tokens"] == 131072


class TestMoonshotPricingCompleteness:
    """Tests to verify all expected models are in pricing dict."""

    @pytest.mark.parametrize("model_name", EXPECTED_MOONSHOT_MODELS)
    def test_model_is_present_in_pricing_dict(self, model_name: str):
        """Test that an expected model is in the pricing dict."""
        assert model_name in MOONSHOT_PRICING, f"Missing model: {model_name}"
