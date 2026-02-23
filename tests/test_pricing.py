"""Tests for the pricing module."""

import pytest

from tokencost.pricing import calculate_cost, get_model_pricing, list_models


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
