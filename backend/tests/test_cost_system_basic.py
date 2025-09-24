"""
Basic test suite for the Cost Optimization System
Tests the actual implementation as it exists
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import actual modules and classes
from src.cost.token_counter import TokenCounter, TokenCount, ModelProvider, ModelConfig
from src.cost.cost_calculator import (
    CostCalculator,
    CostBreakdown,
    BudgetConfig,
    BudgetAlert,
    BudgetPeriod,
    CostCategory,
    CostForecast,
)
from src.cost.budget_manager import BudgetManager, BudgetStatus, BudgetEnforcement


# ============= Fixtures =============


@pytest.fixture
def token_counter():
    """Create a TokenCounter instance"""
    return TokenCounter()


@pytest.fixture
def budget_config():
    """Create a test budget configuration"""
    return BudgetConfig(
        daily_limit=Decimal("10.0"),
        monthly_limit=Decimal("100.0"),
        hard_stop_enabled=True,
    )


@pytest.fixture
def cost_calculator(budget_config):
    """Create a CostCalculator instance"""
    return CostCalculator(budget_config)


@pytest.fixture
def budget_manager():
    """Create a BudgetManager instance"""
    return BudgetManager()


@pytest.fixture
def sample_token_count():
    """Create a sample token count"""
    return TokenCount(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        model="gpt-4",
        provider=ModelProvider.OPENAI,
    )


# ============= TokenCounter Tests =============


class TestTokenCounter:
    """Test suite for TokenCounter class"""

    def test_initialization(self, token_counter):
        """Test TokenCounter initialization"""
        assert token_counter is not None
        assert hasattr(token_counter, "count_tokens")
        assert hasattr(token_counter, "estimate_tokens")

    def test_model_configs_loaded(self, token_counter):
        """Test that model configurations are loaded"""
        assert len(token_counter.model_configs) > 0
        assert "gpt-4" in token_counter.model_configs
        assert "gpt-3.5-turbo" in token_counter.model_configs

    def test_count_tokens_basic(self, token_counter):
        """Test basic token counting"""
        text = "This is a test sentence for token counting."
        result = token_counter.count_tokens(text, model="gpt-4")

        assert isinstance(result, TokenCount)
        assert result.total_tokens > 0
        assert result.model == "gpt-4"
        assert result.provider == ModelProvider.OPENAI

    def test_estimate_tokens(self, token_counter):
        """Test token estimation"""
        text = (
            "This is a longer test sentence that will be used to estimate token count."
        )
        estimate = token_counter.estimate_tokens(text)

        assert isinstance(estimate, int)
        assert estimate > 0
        # Rough estimate: ~4 chars per token
        assert estimate < len(text)

    def test_count_tokens_with_empty_text(self, token_counter):
        """Test token counting with empty text"""
        result = token_counter.count_tokens("", model="gpt-4")

        assert result.total_tokens == 0
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_get_model_config(self, token_counter):
        """Test getting model configuration"""
        config = token_counter.get_model_config("gpt-4")

        assert isinstance(config, ModelConfig)
        assert config.name == "gpt-4"
        assert config.provider == ModelProvider.OPENAI
        assert config.input_cost_per_1k > 0
        assert config.output_cost_per_1k > 0

    def test_count_tokens_for_messages(self, token_counter):
        """Test token counting for message format"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        result = token_counter.count_tokens_for_messages(messages, model="gpt-4")

        assert result.total_tokens > 0
        assert result.input_tokens > 0


# ============= CostCalculator Tests =============


class TestCostCalculator:
    """Test suite for CostCalculator class"""

    def test_initialization(self, cost_calculator):
        """Test CostCalculator initialization"""
        assert cost_calculator is not None
        assert hasattr(cost_calculator, "calculate_cost")
        assert hasattr(cost_calculator, "check_budget")

    def test_calculate_cost_basic(self, cost_calculator, sample_token_count):
        """Test basic cost calculation"""
        cost = cost_calculator.calculate_cost(
            token_count=sample_token_count, category=CostCategory.PROCESSING
        )

        assert isinstance(cost, CostBreakdown)
        assert cost.total_cost > 0
        assert cost.input_cost > 0
        assert cost.output_cost > 0
        assert cost.model == "gpt-4"

    def test_budget_tracking(self, cost_calculator):
        """Test budget tracking functionality"""
        # Track some spending
        cost_calculator.track_cost(
            amount=Decimal("1.0"), category=CostCategory.PROCESSING, model="gpt-4"
        )

        usage = cost_calculator.get_current_usage(BudgetPeriod.DAILY)
        assert usage > 0
        assert usage == Decimal("1.0")

    def test_check_budget_within_limits(self, cost_calculator):
        """Test budget checking when within limits"""
        result = cost_calculator.check_budget(Decimal("5.0"))

        assert result["allowed"] == True
        assert "remaining" in result
        assert result["remaining"]["daily"] == Decimal("10.0")

    def test_check_budget_exceeds_limit(self, cost_calculator):
        """Test budget checking when exceeding limits"""
        # Track spending close to limit
        cost_calculator.track_cost(
            amount=Decimal("9.5"), category=CostCategory.PROCESSING, model="gpt-4"
        )

        result = cost_calculator.check_budget(Decimal("1.0"))

        # Should be blocked by hard stop
        assert result["allowed"] == False
        assert "reason" in result

    def test_get_cost_summary(self, cost_calculator):
        """Test getting cost summary"""
        # Track some costs
        cost_calculator.track_cost(
            amount=Decimal("2.0"), category=CostCategory.PROCESSING, model="gpt-4"
        )
        cost_calculator.track_cost(
            amount=Decimal("1.0"),
            category=CostCategory.EMBEDDING,
            model="text-embedding-3-small",
        )

        summary = cost_calculator.get_cost_summary(BudgetPeriod.DAILY)

        assert "total" in summary
        assert summary["total"] == Decimal("3.0")
        assert "by_category" in summary
        assert "by_model" in summary

    @pytest.mark.asyncio
    async def test_async_calculate_cost(self, cost_calculator, sample_token_count):
        """Test async cost calculation"""
        cost = await cost_calculator.calculate_cost_async(
            token_count=sample_token_count, category=CostCategory.PROCESSING
        )

        assert isinstance(cost, CostBreakdown)
        assert cost.total_cost > 0


# ============= BudgetManager Tests =============


class TestBudgetManager:
    """Test suite for BudgetManager class"""

    def test_initialization(self, budget_manager):
        """Test BudgetManager initialization"""
        assert budget_manager is not None
        assert hasattr(budget_manager, "set_budget")
        assert hasattr(budget_manager, "check_budget_status")

    def test_set_budget(self, budget_manager):
        """Test setting budget limits"""
        budget_manager.set_budget(period=BudgetPeriod.DAILY, limit=Decimal("20.0"))

        status = budget_manager.check_budget_status(BudgetPeriod.DAILY)
        assert status.limit == Decimal("20.0")
        assert status.spent == Decimal("0.0")
        assert status.remaining == Decimal("20.0")

    def test_track_spending(self, budget_manager):
        """Test tracking spending"""
        budget_manager.set_budget(BudgetPeriod.DAILY, Decimal("10.0"))
        budget_manager.track_spending(amount=Decimal("3.0"), period=BudgetPeriod.DAILY)

        status = budget_manager.check_budget_status(BudgetPeriod.DAILY)
        assert status.spent == Decimal("3.0")
        assert status.remaining == Decimal("7.0")
        assert status.percentage_used == 0.3

    def test_budget_alert_trigger(self, budget_manager):
        """Test budget alert triggering"""
        budget_manager.set_budget(BudgetPeriod.DAILY, Decimal("10.0"))
        budget_manager.add_alert(
            name="80% Warning", threshold_percentage=0.8, period=BudgetPeriod.DAILY
        )

        # Track spending to trigger alert
        alerts = budget_manager.track_spending(
            amount=Decimal("8.5"), period=BudgetPeriod.DAILY
        )

        assert len(alerts) > 0
        assert alerts[0].triggered == True

    def test_budget_enforcement(self, budget_manager):
        """Test budget enforcement"""
        budget_manager.set_budget(BudgetPeriod.DAILY, Decimal("10.0"))
        budget_manager.enable_enforcement(enforcement_type=BudgetEnforcement.HARD_STOP)

        # Should allow within budget
        assert budget_manager.can_spend(Decimal("5.0"), BudgetPeriod.DAILY) == True

        # Track spending
        budget_manager.track_spending(Decimal("9.0"), BudgetPeriod.DAILY)

        # Should block over budget
        assert budget_manager.can_spend(Decimal("2.0"), BudgetPeriod.DAILY) == False

    def test_reset_budget(self, budget_manager):
        """Test budget reset"""
        budget_manager.set_budget(BudgetPeriod.DAILY, Decimal("10.0"))
        budget_manager.track_spending(Decimal("5.0"), BudgetPeriod.DAILY)

        # Reset budget
        budget_manager.reset_period(BudgetPeriod.DAILY)

        status = budget_manager.check_budget_status(BudgetPeriod.DAILY)
        assert status.spent == Decimal("0.0")
        assert status.remaining == Decimal("10.0")


# ============= Integration Tests =============


class TestCostSystemIntegration:
    """Integration tests for the complete cost system"""

    def test_end_to_end_cost_tracking(self, token_counter, cost_calculator):
        """Test complete flow from token counting to cost tracking"""
        # Count tokens
        text = "This is a test prompt for the AI model to process."
        token_count = token_counter.count_tokens(text, model="gpt-4")

        # Calculate cost
        cost = cost_calculator.calculate_cost(
            token_count=token_count, category=CostCategory.PROCESSING
        )

        assert cost.total_cost > 0

        # Check budget allows it
        budget_check = cost_calculator.check_budget(cost.total_cost)
        assert budget_check["allowed"] == True

        # Track the cost
        cost_calculator.track_cost(
            amount=cost.total_cost, category=CostCategory.PROCESSING, model="gpt-4"
        )

        # Verify tracking
        usage = cost_calculator.get_current_usage(BudgetPeriod.DAILY)
        assert usage == cost.total_cost

    def test_budget_enforcement_flow(self, cost_calculator, budget_manager):
        """Test budget enforcement preventing overspending"""
        # Set strict budget
        budget_manager.set_budget(BudgetPeriod.DAILY, Decimal("1.0"))
        budget_manager.enable_enforcement(BudgetEnforcement.HARD_STOP)

        # Simulate multiple operations
        total_spent = Decimal("0.0")
        operations_completed = 0

        for i in range(10):
            cost = Decimal("0.3")  # Each operation costs $0.30

            if budget_manager.can_spend(cost, BudgetPeriod.DAILY):
                budget_manager.track_spending(cost, BudgetPeriod.DAILY)
                total_spent += cost
                operations_completed += 1
            else:
                break

        # Should have stopped before exceeding budget
        assert total_spent <= Decimal("1.0")
        assert operations_completed == 3  # 3 * 0.3 = 0.9, 4th would exceed

    @pytest.mark.asyncio
    async def test_concurrent_cost_calculations(self, cost_calculator):
        """Test concurrent cost calculations"""
        token_counts = [
            TokenCount(100, 50, 150, "gpt-4", ModelProvider.OPENAI),
            TokenCount(200, 100, 300, "gpt-3.5-turbo", ModelProvider.OPENAI),
            TokenCount(150, 75, 225, "gpt-4", ModelProvider.OPENAI),
        ]

        tasks = [
            cost_calculator.calculate_cost_async(tc, CostCategory.PROCESSING)
            for tc in token_counts
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all(isinstance(r, CostBreakdown) for r in results)
        assert all(r.total_cost > 0 for r in results)


# ============= Error Handling Tests =============


class TestErrorHandling:
    """Test error handling in the cost system"""

    def test_invalid_model_name(self, token_counter):
        """Test handling of invalid model names"""
        with pytest.raises(ValueError, match="Unknown model"):
            token_counter.count_tokens("test", model="invalid-model")

    def test_negative_token_count(self, cost_calculator):
        """Test handling of negative token counts"""
        invalid_count = TokenCount(
            input_tokens=-100,
            output_tokens=50,
            total_tokens=-50,
            model="gpt-4",
            provider=ModelProvider.OPENAI,
        )

        with pytest.raises(ValueError, match="negative"):
            cost_calculator.calculate_cost(invalid_count, CostCategory.PROCESSING)

    def test_invalid_budget_period(self, budget_manager):
        """Test handling of invalid budget periods"""
        with pytest.raises(ValueError):
            budget_manager.set_budget("invalid_period", Decimal("10.0"))

    def test_exceeding_decimal_precision(self, cost_calculator):
        """Test handling of decimal precision issues"""
        # Very small cost that might cause precision issues
        tiny_cost = Decimal("0.0000000001")

        # Should handle without error
        cost_calculator.track_cost(
            amount=tiny_cost, category=CostCategory.PROCESSING, model="gpt-4"
        )

        usage = cost_calculator.get_current_usage(BudgetPeriod.DAILY)
        assert usage >= 0  # Should not be negative or error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
