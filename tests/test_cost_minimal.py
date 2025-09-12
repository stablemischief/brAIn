"""
Minimal test suite for the Cost System - tests actual implementation
"""

import pytest
import os
import sys
from decimal import Decimal
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import what actually exists
from src.cost.token_counter import TokenCounter, TokenCount, ModelProvider
from src.cost.cost_calculator import CostCalculator, CostBreakdown, BudgetConfig, CostCategory


class TestTokenCounter:
    """Test TokenCounter functionality"""
    
    def test_token_counter_initialization(self):
        """Test that TokenCounter can be initialized"""
        counter = TokenCounter()
        assert counter is not None
        assert hasattr(counter, 'count_tokens_text')
        assert hasattr(counter, 'count_tokens_messages')
    
    def test_count_tokens_text_basic(self):
        """Test basic text token counting"""
        counter = TokenCounter()
        text = "Hello, this is a test."
        
        result = counter.count_tokens_text(text, model="gpt-4")
        
        assert isinstance(result, TokenCount)
        assert result.total_tokens > 0
        assert result.model == "gpt-4"
        assert result.provider == ModelProvider.OPENAI
    
    def test_count_tokens_messages(self):
        """Test message token counting"""
        counter = TokenCounter()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ]
        
        result = counter.count_tokens_messages(messages, model="gpt-4")
        
        assert isinstance(result, TokenCount)
        assert result.total_tokens > 0
        assert result.input_tokens > 0
    
    def test_estimate_tokens_anthropic(self):
        """Test Anthropic token estimation"""
        counter = TokenCounter()
        text = "This is a test sentence for estimation."
        
        estimate = counter.estimate_tokens_anthropic(text)
        
        assert isinstance(estimate, int)
        assert estimate > 0
        assert estimate < len(text)  # Should be less than character count
    
    def test_empty_text_handling(self):
        """Test handling of empty text"""
        counter = TokenCounter()
        result = counter.count_tokens_text("", model="gpt-4")
        
        assert result.total_tokens == 0
        assert result.input_tokens == 0


class TestCostCalculator:
    """Test CostCalculator functionality"""
    
    def test_cost_calculator_initialization(self):
        """Test that CostCalculator can be initialized"""
        calc = CostCalculator()
        assert calc is not None
        assert hasattr(calc, 'calculate_operation_cost')
        assert hasattr(calc, 'calculate_batch_cost')
    
    def test_calculate_operation_cost(self):
        """Test operation cost calculation"""
        calc = CostCalculator()
        
        token_count = TokenCount(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            model="gpt-4",
            provider=ModelProvider.OPENAI
        )
        
        cost = calc.calculate_operation_cost(
            token_count=token_count,
            category=CostCategory.PROCESSING,
            operation_id="test-operation"
        )
        
        assert isinstance(cost, CostBreakdown)
        assert cost.total_cost > 0
        assert cost.input_cost >= 0
        assert cost.output_cost >= 0
        assert cost.model == "gpt-4"
    
    def test_budget_config(self):
        """Test budget configuration"""
        config = BudgetConfig(
            daily_limit=Decimal("10.0"),
            monthly_limit=Decimal("100.0")
        )
        
        calc = CostCalculator(budget_config=config)
        assert calc.budget_config is not None
        assert calc.budget_config.daily_limit == Decimal("10.0")
    
    def test_check_budget_limits(self):
        """Test budget limit checking"""
        config = BudgetConfig(
            daily_limit=Decimal("10.0"),
            hard_stop_enabled=True
        )
        
        calc = CostCalculator(budget_config=config)
        
        # Should be allowed within budget - returns (bool, list)
        allowed, messages = calc.check_budget_limits(Decimal("5.0"))
        assert allowed == True
        
        # Add some costs and check again
        calc.daily_costs[datetime.now().strftime("%Y-%m-%d")] = Decimal("8.0")
        
        allowed, messages = calc.check_budget_limits(Decimal("5.0"))
        assert allowed == False  # Would exceed daily limit
        assert len(messages) > 0
    
    def test_estimate_document_processing_cost(self):
        """Test document processing cost estimation"""
        calc = CostCalculator()
        
        # Use actual method signature
        estimates = calc.estimate_document_processing_cost(
            document_content="This is a test document content for processing.",
            processing_types=["extraction", "summarization"],
            models=["gpt-4", "gpt-3.5-turbo"]
        )
        
        assert isinstance(estimates, dict)
        assert len(estimates) > 0
        # Structure is {model: {processing_type: CostBreakdown}}
        # Check we have at least one model
        first_model = list(estimates.keys())[0]
        assert first_model in ["gpt-4", "gpt-3.5-turbo"]
        # Check we have processing types
        model_estimates = estimates[first_model]
        assert isinstance(model_estimates, dict)
        assert len(model_estimates) > 0
    
    def test_get_cost_analytics_empty(self):
        """Test getting cost analytics with no data"""
        calc = CostCalculator()
        
        analytics = calc.get_cost_analytics()
        
        # Should return error message for empty data
        assert "error" in analytics
        assert "No cost data" in analytics["error"]
    
    def test_calculate_batch_cost(self):
        """Test batch cost calculation"""
        calc = CostCalculator()
        
        token_counts = [
            TokenCount(100, 50, 150, "gpt-4", ModelProvider.OPENAI),
            TokenCount(200, 100, 300, "gpt-3.5-turbo", ModelProvider.OPENAI)
        ]
        
        batch_result = calc.calculate_batch_cost(
            token_counts, 
            batch_id="test-batch"
        )
        
        # Returns list of CostBreakdown objects
        assert isinstance(batch_result, list)
        assert len(batch_result) == 2
        
        # Check individual costs
        for cost in batch_result:
            assert isinstance(cost, CostBreakdown)
            assert cost.total_cost > 0
            assert cost.model in ["gpt-4", "gpt-3.5-turbo"]
            
        # Check total
        total_cost = sum(cost.total_cost for cost in batch_result)
        assert total_cost > 0


class TestCostOptimizer:
    """Test cost optimization functionality"""
    
    def test_optimizer_import(self):
        """Test that optimizer can be imported"""
        from src.cost.optimizer import CostOptimizer
        
        optimizer = CostOptimizer()
        assert optimizer is not None
        assert hasattr(optimizer, 'analyze_costs')
    
    def test_analyze_costs_basic(self):
        """Test basic cost analysis"""
        from src.cost.optimizer import CostOptimizer
        
        optimizer = CostOptimizer()
        
        # Sample cost data
        costs = [
            {"model": "gpt-4", "tokens": 1000, "cost": 0.03},
            {"model": "gpt-3.5-turbo", "tokens": 2000, "cost": 0.004}
        ]
        
        analysis = optimizer.analyze_costs(costs)
        
        assert "total_cost" in analysis
        assert "by_model" in analysis
        assert analysis["total_cost"] > 0
    
    def test_get_recommendations(self):
        """Test getting optimization recommendations"""
        from src.cost.optimizer import CostOptimizer
        
        optimizer = CostOptimizer()
        
        usage_data = {
            "daily_cost": 50.0,
            "primary_model": "gpt-4",
            "average_tokens": 1000
        }
        
        recommendations = optimizer.get_recommendations(usage_data)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0


class TestCostAnalytics:
    """Test cost analytics functionality"""
    
    def test_analytics_import(self):
        """Test that analytics can be imported"""
        from src.cost.analytics import CostAnalytics
        
        analytics = CostAnalytics()
        assert analytics is not None
        assert hasattr(analytics, 'generate_report')
    
    def test_generate_report_basic(self):
        """Test basic report generation"""
        from src.cost.analytics import CostAnalytics
        
        analytics = CostAnalytics()
        
        # Generate daily report
        report = analytics.generate_report(period="daily")
        
        assert report is not None
        assert "period" in report
        assert report["period"] == "daily"
    
    def test_calculate_trends(self):
        """Test trend calculation"""
        from src.cost.analytics import CostAnalytics
        
        analytics = CostAnalytics()
        
        # Sample historical data
        history = [
            {"date": "2024-01-01", "cost": 10.0},
            {"date": "2024-01-02", "cost": 12.0},
            {"date": "2024-01-03", "cost": 15.0}
        ]
        
        trends = analytics.calculate_trends(history)
        
        assert "direction" in trends
        assert trends["direction"] in ["increasing", "decreasing", "stable"]


class TestCostSystemIntegration:
    """Integration tests for the cost system"""
    
    def test_end_to_end_flow(self):
        """Test complete flow from token counting to cost tracking"""
        # Initialize components
        counter = TokenCounter()
        calculator = CostCalculator()
        
        # Count tokens
        text = "This is a test prompt for cost calculation."
        token_count = counter.count_tokens(text, model="gpt-4")
        
        # Calculate cost
        cost = calculator.calculate_cost(
            token_count=token_count,
            category=CostCategory.PROCESSING
        )
        
        assert cost.total_cost > 0
        
        # Track the cost
        calculator.track_cost(
            amount=cost.total_cost,
            category=CostCategory.PROCESSING,
            model="gpt-4"
        )
        
        # Verify tracking
        usage = calculator.get_current_usage("daily")
        assert usage >= cost.total_cost
    
    def test_budget_enforcement(self):
        """Test that budget enforcement works"""
        config = BudgetConfig(
            daily_limit=Decimal("1.0"),
            hard_stop_enabled=True
        )
        
        calculator = CostCalculator(budget_config=config)
        
        # Track costs up to limit
        calculator.track_cost(Decimal("0.5"), CostCategory.PROCESSING, "gpt-4")
        calculator.track_cost(Decimal("0.4"), CostCategory.PROCESSING, "gpt-4")
        
        # Check if we can spend more
        can_spend = calculator.check_budget(Decimal("0.5"))
        
        # Should be blocked
        assert can_spend["allowed"] == False
        assert "budget" in can_spend.get("reason", "").lower()


class TestErrorHandling:
    """Test error handling in the cost system"""
    
    def test_invalid_model_handling(self):
        """Test handling of invalid model names"""
        counter = TokenCounter()
        
        with pytest.raises(ValueError):
            counter.count_tokens("test", model="invalid-model-xyz")
    
    def test_negative_cost_handling(self):
        """Test handling of negative costs"""
        calculator = CostCalculator()
        
        invalid_count = TokenCount(
            input_tokens=-100,
            output_tokens=50,
            total_tokens=-50,
            model="gpt-4",
            provider=ModelProvider.OPENAI
        )
        
        with pytest.raises(ValueError):
            calculator.calculate_cost(invalid_count, CostCategory.PROCESSING)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])