"""
Comprehensive test suite for the Cost Optimization System
Tests token counting, cost calculation, budget management, and optimization
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import json
from typing import Dict, List, Optional

# Import all cost system modules
from src.cost.token_counter import (
    TokenCounter, TokenCount, ModelProvider, ModelConfig
)
from src.cost.cost_calculator import (
    CostCalculator, CostBreakdown, UsageMetrics, CostAlert
)
from src.cost.budget_manager import (
    BudgetManager, BudgetConfig, BudgetPeriod, BudgetAlert, BudgetStatus
)
from src.cost.optimizer import (
    CostOptimizer, OptimizationStrategy, OptimizationRecommendation,
    ModelComparison, BatchingStrategy
)
from src.cost.analytics import (
    CostAnalytics, CostTrend, UsagePattern, CostReport,
    AggregationPeriod, CostMetric
)


# ============= Fixtures =============

@pytest.fixture
def token_counter():
    """Create a TokenCounter instance"""
    return TokenCounter()


@pytest.fixture
def cost_calculator():
    """Create a CostCalculator instance"""
    return CostCalculator()


@pytest.fixture
def budget_manager():
    """Create a BudgetManager instance with test config"""
    config = BudgetConfig(
        daily_limit=10.0,
        monthly_limit=100.0,
        alert_threshold=0.8,
        enforce_limits=True
    )
    return BudgetManager(config)


@pytest.fixture
def cost_optimizer():
    """Create a CostOptimizer instance"""
    return CostOptimizer()


@pytest.fixture
def cost_analytics():
    """Create a CostAnalytics instance"""
    return CostAnalytics()


@pytest.fixture
def sample_text():
    """Sample text for token counting"""
    return "This is a sample text for testing token counting functionality."


@pytest.fixture
def long_text():
    """Longer text for testing"""
    return " ".join(["This is a longer text sample."] * 100)


# ============= TokenCounter Tests =============

class TestTokenCounter:
    """Test suite for TokenCounter class"""
    
    def test_initialization(self, token_counter):
        """Test TokenCounter initialization"""
        assert token_counter is not None
        assert hasattr(token_counter, 'count_tokens')
        assert hasattr(token_counter, 'estimate_tokens')
    
    def test_count_tokens_openai(self, token_counter, sample_text):
        """Test token counting for OpenAI models"""
        result = token_counter.count_tokens(
            text=sample_text,
            model="gpt-4",
            provider=ModelProvider.OPENAI
        )
        
        assert isinstance(result, TokenCount)
        assert result.total_tokens > 0
        assert result.model == "gpt-4"
        assert result.provider == ModelProvider.OPENAI
    
    def test_count_tokens_anthropic(self, token_counter, sample_text):
        """Test token counting for Anthropic models"""
        result = token_counter.count_tokens(
            text=sample_text,
            model="claude-3-opus",
            provider=ModelProvider.ANTHROPIC
        )
        
        assert isinstance(result, TokenCount)
        assert result.total_tokens > 0
        assert result.model == "claude-3-opus"
        assert result.provider == ModelProvider.ANTHROPIC
    
    def test_count_tokens_with_messages(self, token_counter):
        """Test token counting with message format"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"}
        ]
        
        result = token_counter.count_tokens_for_messages(
            messages=messages,
            model="gpt-4"
        )
        
        assert result.total_tokens > 0
        assert result.input_tokens > 0
        assert result.output_tokens > 0
    
    def test_estimate_tokens(self, token_counter, long_text):
        """Test token estimation for quick calculations"""
        estimate = token_counter.estimate_tokens(long_text)
        actual = token_counter.count_tokens(
            long_text, 
            model="gpt-4", 
            provider=ModelProvider.OPENAI
        )
        
        # Estimation should be within 20% of actual
        assert abs(estimate - actual.total_tokens) / actual.total_tokens < 0.2
    
    def test_model_config(self, token_counter):
        """Test model configuration retrieval"""
        config = token_counter.get_model_config("gpt-4")
        
        assert isinstance(config, ModelConfig)
        assert config.name == "gpt-4"
        assert config.provider == ModelProvider.OPENAI
        assert config.max_tokens > 0
        assert config.input_cost_per_1k > 0
    
    def test_unsupported_model(self, token_counter):
        """Test handling of unsupported models"""
        with pytest.raises(ValueError, match="Unsupported model"):
            token_counter.count_tokens(
                "test text",
                model="unsupported-model",
                provider=ModelProvider.CUSTOM
            )
    
    def test_empty_text(self, token_counter):
        """Test token counting with empty text"""
        result = token_counter.count_tokens(
            text="",
            model="gpt-4",
            provider=ModelProvider.OPENAI
        )
        
        assert result.total_tokens == 0
        assert result.input_tokens == 0
    
    def test_token_truncation(self, token_counter):
        """Test token truncation for max limits"""
        very_long_text = " ".join(["word"] * 10000)
        
        result = token_counter.count_tokens_with_truncation(
            text=very_long_text,
            model="gpt-4",
            max_tokens=1000
        )
        
        assert result.total_tokens <= 1000
        assert result.metadata.get("truncated") is True


# ============= CostCalculator Tests =============

class TestCostCalculator:
    """Test suite for CostCalculator class"""
    
    def test_initialization(self, cost_calculator):
        """Test CostCalculator initialization"""
        assert cost_calculator is not None
        assert hasattr(cost_calculator, 'calculate_cost')
        assert hasattr(cost_calculator, 'get_pricing')
    
    def test_calculate_cost_simple(self, cost_calculator):
        """Test simple cost calculation"""
        token_count = TokenCount(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            model="gpt-4",
            provider=ModelProvider.OPENAI
        )
        
        cost = cost_calculator.calculate_cost(token_count)
        
        assert isinstance(cost, CostBreakdown)
        assert cost.input_cost > 0
        assert cost.output_cost > 0
        assert cost.total_cost == cost.input_cost + cost.output_cost
    
    def test_calculate_cost_batch(self, cost_calculator):
        """Test batch cost calculation"""
        token_counts = [
            TokenCount(100, 50, 150, "gpt-4", ModelProvider.OPENAI),
            TokenCount(200, 100, 300, "gpt-4", ModelProvider.OPENAI),
            TokenCount(150, 75, 225, "gpt-4", ModelProvider.OPENAI)
        ]
        
        total_cost = cost_calculator.calculate_batch_cost(token_counts)
        
        assert total_cost > 0
        assert isinstance(total_cost, Decimal)
    
    def test_get_pricing(self, cost_calculator):
        """Test pricing retrieval"""
        pricing = cost_calculator.get_pricing("gpt-4")
        
        assert "input_per_1k" in pricing
        assert "output_per_1k" in pricing
        assert pricing["input_per_1k"] > 0
        assert pricing["output_per_1k"] > 0
    
    def test_update_pricing(self, cost_calculator):
        """Test pricing updates"""
        new_pricing = {
            "input_per_1k": 0.05,
            "output_per_1k": 0.10
        }
        
        cost_calculator.update_pricing("gpt-4", new_pricing)
        updated = cost_calculator.get_pricing("gpt-4")
        
        assert updated["input_per_1k"] == 0.05
        assert updated["output_per_1k"] == 0.10
    
    def test_cost_with_multiplier(self, cost_calculator):
        """Test cost calculation with multiplier"""
        token_count = TokenCount(100, 50, 150, "gpt-4", ModelProvider.OPENAI)
        
        normal_cost = cost_calculator.calculate_cost(token_count)
        premium_cost = cost_calculator.calculate_cost(
            token_count, 
            multiplier=1.5
        )
        
        assert premium_cost.total_cost == normal_cost.total_cost * Decimal("1.5")
    
    def test_cost_alert_trigger(self, cost_calculator):
        """Test cost alert triggering"""
        token_count = TokenCount(10000, 5000, 15000, "gpt-4", ModelProvider.OPENAI)
        
        with patch.object(cost_calculator, 'trigger_alert') as mock_alert:
            cost = cost_calculator.calculate_cost(
                token_count,
                alert_threshold=0.01  # Low threshold to trigger
            )
            mock_alert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_cost_calculation(self, cost_calculator):
        """Test asynchronous cost calculation"""
        token_count = TokenCount(100, 50, 150, "gpt-4", ModelProvider.OPENAI)
        
        cost = await cost_calculator.calculate_cost_async(token_count)
        
        assert isinstance(cost, CostBreakdown)
        assert cost.total_cost > 0


# ============= BudgetManager Tests =============

class TestBudgetManager:
    """Test suite for BudgetManager class"""
    
    def test_initialization(self, budget_manager):
        """Test BudgetManager initialization"""
        assert budget_manager is not None
        assert budget_manager.config.daily_limit == 10.0
        assert budget_manager.config.monthly_limit == 100.0
    
    def test_check_budget_within_limits(self, budget_manager):
        """Test budget check when within limits"""
        result = budget_manager.check_budget(5.0)
        
        assert result.within_budget is True
        assert result.daily_remaining == 5.0
        assert result.monthly_remaining == 95.0
    
    def test_check_budget_exceeds_daily(self, budget_manager):
        """Test budget check when exceeding daily limit"""
        result = budget_manager.check_budget(15.0)
        
        assert result.within_budget is False
        assert result.daily_remaining == 10.0
        assert "daily limit" in result.message.lower()
    
    def test_track_spending(self, budget_manager):
        """Test spending tracking"""
        budget_manager.track_spending(3.0, "gpt-4")
        budget_manager.track_spending(2.0, "gpt-4")
        
        status = budget_manager.get_budget_status()
        
        assert status.daily_spent == 5.0
        assert status.monthly_spent == 5.0
        assert status.daily_remaining == 5.0
    
    def test_budget_alert_threshold(self, budget_manager):
        """Test budget alert at threshold"""
        with patch.object(budget_manager, 'send_alert') as mock_alert:
            budget_manager.track_spending(8.5, "gpt-4")  # 85% of daily
            
            mock_alert.assert_called()
            alert = mock_alert.call_args[0][0]
            assert isinstance(alert, BudgetAlert)
            assert alert.level == "warning"
    
    def test_reset_daily_budget(self, budget_manager):
        """Test daily budget reset"""
        budget_manager.track_spending(5.0, "gpt-4")
        budget_manager.reset_daily_budget()
        
        status = budget_manager.get_budget_status()
        
        assert status.daily_spent == 0.0
        assert status.monthly_spent == 5.0  # Monthly not reset
    
    def test_reset_monthly_budget(self, budget_manager):
        """Test monthly budget reset"""
        budget_manager.track_spending(50.0, "gpt-4")
        budget_manager.reset_monthly_budget()
        
        status = budget_manager.get_budget_status()
        
        assert status.daily_spent == 0.0
        assert status.monthly_spent == 0.0
    
    def test_budget_enforcement(self, budget_manager):
        """Test budget enforcement blocking"""
        budget_manager.track_spending(9.0, "gpt-4")
        
        # Should allow small amount
        assert budget_manager.can_proceed(0.5) is True
        
        # Should block large amount
        assert budget_manager.can_proceed(2.0) is False
    
    def test_budget_history(self, budget_manager):
        """Test budget history tracking"""
        budget_manager.track_spending(2.0, "gpt-4")
        budget_manager.track_spending(3.0, "claude-3")
        
        history = budget_manager.get_spending_history()
        
        assert len(history) == 2
        assert history[0]["amount"] == 2.0
        assert history[0]["model"] == "gpt-4"
    
    def test_budget_projection(self, budget_manager):
        """Test budget projection"""
        budget_manager.track_spending(5.0, "gpt-4")
        
        projection = budget_manager.project_monthly_spending()
        
        assert projection > 5.0  # Should project based on current rate
        assert projection <= 100.0  # Shouldn't exceed monthly limit in projection


# ============= CostOptimizer Tests =============

class TestCostOptimizer:
    """Test suite for CostOptimizer class"""
    
    def test_initialization(self, cost_optimizer):
        """Test CostOptimizer initialization"""
        assert cost_optimizer is not None
        assert hasattr(cost_optimizer, 'analyze_usage')
        assert hasattr(cost_optimizer, 'recommend_optimizations')
    
    def test_analyze_usage_patterns(self, cost_optimizer):
        """Test usage pattern analysis"""
        usage_data = [
            {"model": "gpt-4", "tokens": 1000, "cost": 0.03},
            {"model": "gpt-4", "tokens": 1500, "cost": 0.045},
            {"model": "gpt-3.5-turbo", "tokens": 2000, "cost": 0.004},
        ]
        
        analysis = cost_optimizer.analyze_usage(usage_data)
        
        assert "total_cost" in analysis
        assert "average_cost" in analysis
        assert "model_breakdown" in analysis
        assert analysis["total_cost"] == 0.079
    
    def test_recommend_model_switch(self, cost_optimizer):
        """Test model switching recommendations"""
        usage_metrics = UsageMetrics(
            total_tokens=10000,
            total_cost=0.30,
            primary_model="gpt-4",
            average_tokens_per_request=500
        )
        
        recommendations = cost_optimizer.recommend_optimizations(usage_metrics)
        
        assert len(recommendations) > 0
        assert any(r.strategy == OptimizationStrategy.MODEL_SWITCH for r in recommendations)
    
    def test_recommend_batching(self, cost_optimizer):
        """Test batching recommendations"""
        request_patterns = [
            {"timestamp": datetime.now(), "tokens": 100},
            {"timestamp": datetime.now(), "tokens": 150},
            {"timestamp": datetime.now(), "tokens": 120},
        ]
        
        batching = cost_optimizer.recommend_batching(request_patterns)
        
        assert isinstance(batching, BatchingStrategy)
        assert batching.recommended_batch_size > 1
        assert batching.estimated_savings > 0
    
    def test_compare_models(self, cost_optimizer):
        """Test model comparison"""
        comparison = cost_optimizer.compare_models(
            current_model="gpt-4",
            alternative_model="gpt-3.5-turbo",
            monthly_tokens=1000000
        )
        
        assert isinstance(comparison, ModelComparison)
        assert comparison.current_cost > comparison.alternative_cost
        assert comparison.savings_percentage > 0
    
    def test_optimize_prompt_length(self, cost_optimizer):
        """Test prompt optimization recommendations"""
        long_prompt = "This is a very long and detailed prompt " * 50
        
        optimization = cost_optimizer.optimize_prompt(long_prompt)
        
        assert optimization.optimized_length < len(long_prompt)
        assert optimization.token_savings > 0
        assert optimization.maintains_quality is True
    
    def test_caching_recommendations(self, cost_optimizer):
        """Test caching strategy recommendations"""
        repeated_queries = [
            {"query": "What is AI?", "count": 50},
            {"query": "How does ML work?", "count": 30},
            {"query": "Explain deep learning", "count": 25},
        ]
        
        caching = cost_optimizer.recommend_caching(repeated_queries)
        
        assert len(caching.queries_to_cache) > 0
        assert caching.estimated_savings > 0
        assert caching.cache_hit_rate > 0
    
    def test_optimization_impact(self, cost_optimizer):
        """Test optimization impact calculation"""
        recommendations = [
            OptimizationRecommendation(
                strategy=OptimizationStrategy.MODEL_SWITCH,
                estimated_savings=0.10,
                implementation_effort="low"
            ),
            OptimizationRecommendation(
                strategy=OptimizationStrategy.BATCHING,
                estimated_savings=0.05,
                implementation_effort="medium"
            ),
        ]
        
        impact = cost_optimizer.calculate_total_impact(recommendations)
        
        assert impact.total_savings == 0.15
        assert impact.roi_percentage > 0


# ============= CostAnalytics Tests =============

class TestCostAnalytics:
    """Test suite for CostAnalytics class"""
    
    def test_initialization(self, cost_analytics):
        """Test CostAnalytics initialization"""
        assert cost_analytics is not None
        assert hasattr(cost_analytics, 'generate_report')
        assert hasattr(cost_analytics, 'analyze_trends')
    
    def test_generate_daily_report(self, cost_analytics):
        """Test daily report generation"""
        report = cost_analytics.generate_report(
            period=AggregationPeriod.DAILY,
            date=datetime.now()
        )
        
        assert isinstance(report, CostReport)
        assert report.period == AggregationPeriod.DAILY
        assert report.metrics is not None
    
    def test_analyze_cost_trends(self, cost_analytics):
        """Test cost trend analysis"""
        historical_data = [
            {"date": datetime.now() - timedelta(days=i), "cost": 10 + i}
            for i in range(7)
        ]
        
        trend = cost_analytics.analyze_trends(historical_data)
        
        assert isinstance(trend, CostTrend)
        assert trend.direction in ["increasing", "decreasing", "stable"]
        assert trend.percentage_change is not None
    
    def test_identify_usage_patterns(self, cost_analytics):
        """Test usage pattern identification"""
        usage_logs = [
            {"hour": i, "tokens": 1000 * (1 + i % 3)} 
            for i in range(24)
        ]
        
        patterns = cost_analytics.identify_patterns(usage_logs)
        
        assert len(patterns) > 0
        assert all(isinstance(p, UsagePattern) for p in patterns)
        assert patterns[0].peak_hours is not None
    
    def test_cost_breakdown_by_model(self, cost_analytics):
        """Test cost breakdown by model"""
        usage_data = [
            {"model": "gpt-4", "cost": 0.50},
            {"model": "gpt-3.5-turbo", "cost": 0.20},
            {"model": "claude-3", "cost": 0.30},
        ]
        
        breakdown = cost_analytics.breakdown_by_model(usage_data)
        
        assert "gpt-4" in breakdown
        assert breakdown["gpt-4"]["percentage"] == 50.0
        assert sum(b["percentage"] for b in breakdown.values()) == 100.0
    
    def test_forecast_costs(self, cost_analytics):
        """Test cost forecasting"""
        historical = [
            {"date": datetime.now() - timedelta(days=i), "cost": 10.0}
            for i in range(30)
        ]
        
        forecast = cost_analytics.forecast_costs(
            historical_data=historical,
            days_ahead=7
        )
        
        assert len(forecast) == 7
        assert all("estimated_cost" in f for f in forecast)
        assert all("confidence" in f for f in forecast)
    
    def test_anomaly_detection(self, cost_analytics):
        """Test cost anomaly detection"""
        normal_costs = [10.0] * 20
        anomaly_costs = normal_costs + [100.0]  # Spike
        
        anomalies = cost_analytics.detect_anomalies(anomaly_costs)
        
        assert len(anomalies) > 0
        assert anomalies[0]["index"] == 20
        assert anomalies[0]["severity"] == "high"
    
    def test_export_analytics(self, cost_analytics):
        """Test analytics export"""
        report = CostReport(
            period=AggregationPeriod.MONTHLY,
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            metrics={"total_cost": 100.0}
        )
        
        # Test JSON export
        json_export = cost_analytics.export_to_json(report)
        assert isinstance(json_export, str)
        assert json.loads(json_export)["metrics"]["total_cost"] == 100.0
        
        # Test CSV export
        csv_export = cost_analytics.export_to_csv([report])
        assert isinstance(csv_export, str)
        assert "total_cost" in csv_export


# ============= Integration Tests =============

class TestCostSystemIntegration:
    """Integration tests for the complete cost system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_cost_tracking(
        self, 
        token_counter, 
        cost_calculator, 
        budget_manager,
        cost_analytics
    ):
        """Test complete flow from token counting to analytics"""
        # Count tokens
        text = "This is a test prompt for the AI model."
        token_count = token_counter.count_tokens(
            text=text,
            model="gpt-4",
            provider=ModelProvider.OPENAI
        )
        
        # Calculate cost
        cost = cost_calculator.calculate_cost(token_count)
        
        # Check budget
        budget_status = budget_manager.check_budget(cost.total_cost)
        assert budget_status.within_budget is True
        
        # Track spending
        budget_manager.track_spending(float(cost.total_cost), "gpt-4")
        
        # Generate analytics
        report = cost_analytics.generate_report(
            period=AggregationPeriod.DAILY,
            date=datetime.now()
        )
        
        assert report is not None
        assert isinstance(report, CostReport)
    
    @pytest.mark.asyncio
    async def test_optimization_workflow(
        self,
        cost_optimizer,
        cost_analytics
    ):
        """Test optimization recommendation workflow"""
        # Simulate usage data
        usage_data = [
            {"model": "gpt-4", "tokens": 10000, "cost": 0.30},
            {"model": "gpt-4", "tokens": 15000, "cost": 0.45},
            {"model": "gpt-4", "tokens": 12000, "cost": 0.36},
        ]
        
        # Analyze usage
        analysis = cost_optimizer.analyze_usage(usage_data)
        
        # Get recommendations
        metrics = UsageMetrics(
            total_tokens=37000,
            total_cost=1.11,
            primary_model="gpt-4",
            average_tokens_per_request=12333
        )
        recommendations = cost_optimizer.recommend_optimizations(metrics)
        
        # Calculate impact
        impact = cost_optimizer.calculate_total_impact(recommendations)
        
        assert impact.total_savings > 0
        assert len(recommendations) > 0
    
    def test_budget_enforcement_integration(
        self,
        budget_manager,
        cost_calculator,
        token_counter
    ):
        """Test budget enforcement preventing overspending"""
        # Set strict budget
        budget_manager.config.daily_limit = 1.0
        budget_manager.config.enforce_limits = True
        
        # Simulate multiple requests
        for i in range(10):
            tokens = TokenCount(
                input_tokens=1000,
                output_tokens=500,
                total_tokens=1500,
                model="gpt-4",
                provider=ModelProvider.OPENAI
            )
            
            cost = cost_calculator.calculate_cost(tokens)
            
            if budget_manager.can_proceed(float(cost.total_cost)):
                budget_manager.track_spending(float(cost.total_cost), "gpt-4")
            else:
                # Should hit this after a few iterations
                assert i > 0  # Should process at least one request
                break
        
        status = budget_manager.get_budget_status()
        assert status.daily_spent <= 1.0


# ============= Error Handling Tests =============

class TestErrorHandling:
    """Test error handling across the cost system"""
    
    def test_invalid_model_handling(self, token_counter):
        """Test handling of invalid model names"""
        with pytest.raises(ValueError):
            token_counter.count_tokens(
                "test",
                model="invalid-model",
                provider=ModelProvider.OPENAI
            )
    
    def test_negative_cost_handling(self, cost_calculator):
        """Test handling of negative costs"""
        with pytest.raises(ValueError):
            invalid_tokens = TokenCount(
                input_tokens=-100,
                output_tokens=50,
                total_tokens=-50,
                model="gpt-4",
                provider=ModelProvider.OPENAI
            )
            cost_calculator.calculate_cost(invalid_tokens)
    
    def test_budget_overflow_handling(self, budget_manager):
        """Test handling of budget overflow"""
        huge_amount = float('inf')
        result = budget_manager.check_budget(huge_amount)
        
        assert result.within_budget is False
        assert "exceeds" in result.message.lower()
    
    def test_corrupted_data_handling(self, cost_analytics):
        """Test handling of corrupted analytics data"""
        corrupted_data = [
            {"date": "invalid-date", "cost": "not-a-number"},
            {"date": datetime.now(), "cost": None},
        ]
        
        with pytest.raises((ValueError, TypeError)):
            cost_analytics.analyze_trends(corrupted_data)


# ============= Performance Tests =============

class TestPerformance:
    """Performance tests for the cost system"""
    
    def test_token_counting_performance(self, token_counter):
        """Test token counting performance with large text"""
        large_text = " ".join(["word"] * 100000)  # 100k words
        
        import time
        start = time.time()
        result = token_counter.count_tokens(
            large_text,
            model="gpt-4",
            provider=ModelProvider.OPENAI
        )
        duration = time.time() - start
        
        assert duration < 5.0  # Should complete within 5 seconds
        assert result.total_tokens > 0
    
    def test_batch_calculation_performance(self, cost_calculator):
        """Test batch cost calculation performance"""
        large_batch = [
            TokenCount(100, 50, 150, "gpt-4", ModelProvider.OPENAI)
            for _ in range(10000)
        ]
        
        import time
        start = time.time()
        total = cost_calculator.calculate_batch_cost(large_batch)
        duration = time.time() - start
        
        assert duration < 1.0  # Should complete within 1 second
        assert total > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_budget_checks(self, budget_manager):
        """Test concurrent budget checking"""
        tasks = []
        for _ in range(100):
            tasks.append(
                asyncio.create_task(
                    asyncio.to_thread(budget_manager.check_budget, 0.01)
                )
            )
        
        results = await asyncio.gather(*tasks)
        
        assert all(r.within_budget for r in results)
        assert len(results) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])