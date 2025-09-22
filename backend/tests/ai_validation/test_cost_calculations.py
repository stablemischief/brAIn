"""
AI Validation Tests for Cost Calculation Accuracy
Tests the cost calculation system against known scenarios and edge cases.
"""
import pytest
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Import cost calculation modules
from src.cost.cost_calculator import (
    CostCalculator, CostBreakdown, CostCategory, BudgetPeriod
)
from src.cost.token_counter import TokenCounter, ModelProvider
from src.cost.budget_manager import BudgetManager


@pytest.mark.ai
class TestCostCalculationAccuracy:
    """Test suite for cost calculation accuracy validation."""

    @pytest.fixture
    def cost_test_scenarios(self):
        """Load cost test scenarios from fixtures."""
        fixture_path = Path(__file__).parent.parent / "fixtures" / "ai_test_data" / "cost_test_scenarios.json"
        with open(fixture_path, 'r') as f:
            return json.load(f)

    @pytest.fixture
    def cost_calculator(self):
        """Create CostCalculator instance for testing."""
        return CostCalculator()

    @pytest.fixture
    def budget_manager(self):
        """Create BudgetManager instance for testing."""
        return BudgetManager(
            daily_limit=Decimal("50.00"),
            monthly_limit=Decimal("1000.00")
        )

    def test_basic_cost_calculations(self, cost_calculator, cost_test_scenarios):
        """Test basic cost calculations against known scenarios."""
        test_cases = cost_test_scenarios["test_cases"]

        for case in test_cases:
            if case["id"] == "basic_gpt4_calculation":
                # Calculate cost for GPT-4 scenario
                breakdown = cost_calculator.calculate_operation_cost(
                    model=case["model"],
                    provider=ModelProvider.OPENAI,
                    input_tokens=case["input_tokens"],
                    output_tokens=case["output_tokens"],
                    category=CostCategory.PROCESSING
                )

                # Validate calculations within tolerance
                tolerance = Decimal("0.001")
                assert abs(breakdown.input_cost - Decimal(str(case["expected_input_cost"]))) < tolerance
                assert abs(breakdown.output_cost - Decimal(str(case["expected_output_cost"]))) < tolerance
                assert abs(breakdown.total_cost - Decimal(str(case["expected_total_cost"]))) < tolerance

    def test_embedding_cost_calculations(self, cost_calculator, cost_test_scenarios):
        """Test embedding-specific cost calculations."""
        test_cases = cost_test_scenarios["test_cases"]

        for case in test_cases:
            if case["id"] == "embedding_calculation":
                breakdown = cost_calculator.calculate_operation_cost(
                    model=case["model"],
                    provider=ModelProvider.OPENAI,
                    input_tokens=case["input_tokens"],
                    output_tokens=case["output_tokens"],
                    category=CostCategory.EMBEDDING
                )

                # Embedding models typically don't have output costs
                assert breakdown.output_cost == Decimal("0")
                assert abs(breakdown.total_cost - Decimal(str(case["expected_total_cost"]))) < Decimal("0.0001")

    def test_large_batch_cost_calculations(self, cost_calculator, cost_test_scenarios):
        """Test cost calculations for large batch operations."""
        test_cases = cost_test_scenarios["test_cases"]

        for case in test_cases:
            if case["id"] == "large_batch_processing":
                breakdown = cost_calculator.calculate_operation_cost(
                    model=case["model"],
                    provider=ModelProvider.OPENAI,
                    input_tokens=case["input_tokens"],
                    output_tokens=case["output_tokens"],
                    category=CostCategory.PROCESSING
                )

                # Verify batch processing doesn't introduce rounding errors
                expected_total = Decimal(str(case["expected_total_cost"]))
                tolerance = Decimal("0.001")
                assert abs(breakdown.total_cost - expected_total) < tolerance

    def test_anthropic_claude_calculations(self, cost_calculator, cost_test_scenarios):
        """Test Claude model cost calculations."""
        test_cases = cost_test_scenarios["test_cases"]

        for case in test_cases:
            if case["id"] == "claude_opus_calculation":
                breakdown = cost_calculator.calculate_operation_cost(
                    model=case["model"],
                    provider=ModelProvider.ANTHROPIC,
                    input_tokens=case["input_tokens"],
                    output_tokens=case["output_tokens"],
                    category=CostCategory.CHAT
                )

                # Claude typically has different input/output pricing ratios
                expected_total = Decimal(str(case["expected_total_cost"]))
                tolerance = Decimal("0.001")
                assert abs(breakdown.total_cost - expected_total) < tolerance

    def test_zero_token_edge_cases(self, cost_calculator, cost_test_scenarios):
        """Test edge cases with zero tokens."""
        test_cases = cost_test_scenarios["test_cases"]

        for case in test_cases:
            if case["id"] == "zero_token_edge_case":
                breakdown = cost_calculator.calculate_operation_cost(
                    model=case["model"],
                    provider=ModelProvider.OPENAI,
                    input_tokens=case["input_tokens"],
                    output_tokens=case["output_tokens"],
                    category=CostCategory.PROCESSING
                )

                assert breakdown.input_cost == Decimal("0")
                assert breakdown.output_cost == Decimal("0")
                assert breakdown.total_cost == Decimal("0")

    def test_budget_limit_validation(self, cost_calculator, budget_manager, cost_test_scenarios):
        """Test budget limit validation scenarios."""
        test_cases = cost_test_scenarios["test_cases"]

        for case in test_cases:
            if case.get("should_exceed_budget"):
                # Set a lower budget limit
                budget_manager.set_daily_limit(Decimal(str(case["budget_limit"])))

                breakdown = cost_calculator.calculate_operation_cost(
                    model=case["model"],
                    provider=ModelProvider.OPENAI,
                    input_tokens=case["input_tokens"],
                    output_tokens=case["output_tokens"],
                    category=CostCategory.PROCESSING
                )

                # Verify the operation would exceed budget
                would_exceed = budget_manager.would_exceed_budget(
                    breakdown.total_cost,
                    BudgetPeriod.DAILY
                )

                assert would_exceed == case["should_exceed_budget"]

    def test_monthly_budget_scenarios(self, budget_manager, cost_test_scenarios):
        """Test monthly budget tracking scenarios."""
        monthly_scenarios = cost_test_scenarios["monthly_budget_scenarios"]

        for scenario in monthly_scenarios:
            # Reset budget manager
            budget_manager = BudgetManager(
                monthly_limit=Decimal(str(scenario["monthly_budget"]))
            )

            # Add daily costs
            for daily_cost in scenario["daily_costs"]:
                budget_manager.add_cost(
                    Decimal(str(daily_cost)),
                    datetime.now(),
                    "test_operation"
                )

            # Check remaining budget
            remaining = budget_manager.get_remaining_budget(BudgetPeriod.MONTHLY)
            expected_remaining = Decimal(str(scenario["expected_remaining"]))

            tolerance = Decimal("0.01")
            assert abs(remaining - expected_remaining) < tolerance

    def test_cost_aggregation_accuracy(self, cost_calculator):
        """Test accuracy of cost aggregation over multiple operations."""
        operations = [
            {"model": "gpt-4", "input_tokens": 1000, "output_tokens": 500},
            {"model": "gpt-4", "input_tokens": 1500, "output_tokens": 750},
            {"model": "gpt-3.5-turbo", "input_tokens": 2000, "output_tokens": 1000}
        ]

        total_cost = Decimal("0")
        for op in operations:
            breakdown = cost_calculator.calculate_operation_cost(
                model=op["model"],
                provider=ModelProvider.OPENAI,
                input_tokens=op["input_tokens"],
                output_tokens=op["output_tokens"],
                category=CostCategory.PROCESSING
            )
            total_cost += breakdown.total_cost

        # Verify aggregation doesn't introduce rounding errors
        assert total_cost > Decimal("0")

        # Test that individual costs sum to total (within tolerance)
        recalculated_total = cost_calculator.aggregate_costs([
            cost_calculator.calculate_operation_cost(**op, provider=ModelProvider.OPENAI, category=CostCategory.PROCESSING)
            for op in operations
        ])

        tolerance = Decimal("0.001")
        assert abs(total_cost - recalculated_total) < tolerance

    def test_cost_precision_and_rounding(self, cost_calculator):
        """Test cost calculation precision and rounding behavior."""
        # Use a scenario that would produce fractional costs
        breakdown = cost_calculator.calculate_operation_cost(
            model="gpt-4",
            provider=ModelProvider.OPENAI,
            input_tokens=333,  # Odd number to test precision
            output_tokens=167,
            category=CostCategory.PROCESSING
        )

        # Verify costs are properly rounded to cents
        assert breakdown.input_cost.as_tuple().exponent >= -2
        assert breakdown.output_cost.as_tuple().exponent >= -2
        assert breakdown.total_cost.as_tuple().exponent >= -2

    def test_different_model_providers(self, cost_calculator):
        """Test cost calculations across different model providers."""
        test_scenarios = [
            {
                "provider": ModelProvider.OPENAI,
                "model": "gpt-4",
                "input_tokens": 1000,
                "output_tokens": 500
            },
            {
                "provider": ModelProvider.ANTHROPIC,
                "model": "claude-3-opus-20240229",
                "input_tokens": 1000,
                "output_tokens": 500
            }
        ]

        for scenario in test_scenarios:
            breakdown = cost_calculator.calculate_operation_cost(
                model=scenario["model"],
                provider=scenario["provider"],
                input_tokens=scenario["input_tokens"],
                output_tokens=scenario["output_tokens"],
                category=CostCategory.PROCESSING
            )

            # Verify calculation completed without errors
            assert breakdown.total_cost >= Decimal("0")
            assert breakdown.input_tokens == scenario["input_tokens"]
            assert breakdown.output_tokens == scenario["output_tokens"]

    @pytest.mark.performance
    def test_cost_calculation_performance(self, cost_calculator):
        """Test performance of cost calculations under load."""
        import time

        start_time = time.time()

        # Perform 1000 cost calculations
        for i in range(1000):
            cost_calculator.calculate_operation_cost(
                model="gpt-4",
                provider=ModelProvider.OPENAI,
                input_tokens=1000 + i,
                output_tokens=500 + i,
                category=CostCategory.PROCESSING
            )

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete 1000 calculations in under 1 second
        assert execution_time < 1.0, f"Cost calculations took {execution_time:.2f}s, expected < 1.0s"

    def test_cost_breakdown_validation(self, cost_calculator):
        """Test detailed cost breakdown validation."""
        breakdown = cost_calculator.calculate_operation_cost(
            model="gpt-4",
            provider=ModelProvider.OPENAI,
            input_tokens=1000,
            output_tokens=500,
            category=CostCategory.PROCESSING
        )

        # Verify all breakdown fields are populated correctly
        assert breakdown.operation_id is not None
        assert breakdown.model == "gpt-4"
        assert breakdown.provider == ModelProvider.OPENAI
        assert breakdown.category == CostCategory.PROCESSING
        assert breakdown.input_tokens == 1000
        assert breakdown.output_tokens == 500
        assert breakdown.total_tokens == 1500
        assert breakdown.total_cost == breakdown.input_cost + breakdown.output_cost

    def test_cost_estimation_accuracy(self, cost_calculator):
        """Test cost estimation accuracy for planning purposes."""
        # Estimate cost for a document processing operation
        estimated_cost = cost_calculator.estimate_document_processing_cost(
            document_length=5000,  # characters
            processing_type="full_analysis",
            model="gpt-4"
        )

        # Verify estimation is reasonable
        assert estimated_cost > Decimal("0")
        assert estimated_cost < Decimal("10.0")  # Should be reasonable for a document

        # Test batch estimation
        batch_estimate = cost_calculator.estimate_batch_processing_cost(
            document_count=100,
            average_document_length=3000,
            processing_type="extraction",
            model="gpt-3.5-turbo"
        )

        assert batch_estimate > estimated_cost  # Batch should cost more than single doc