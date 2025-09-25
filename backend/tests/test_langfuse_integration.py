"""
brAIn v2.0 Langfuse Integration Tests
Comprehensive test suite for Langfuse monitoring and cost tracking.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from uuid import uuid4, UUID

# Import modules under test
from app.monitoring.langfuse_client import (
    LangfuseConfig,
    LangfuseClientManager,
    get_langfuse_client,
    initialize_langfuse,
    validate_langfuse_environment,
)
from app.monitoring.llm_tracker import (
    TraceMetadata,
    OperationResult,
    LLMTracker,
    track_llm_operation,
    track_embedding_generation,
    add_user_feedback,
)
from app.monitoring.cost_calculator import (
    CostCalculator,
    TokenUsage,
    ModelConfig,
    ModelPricing,
)
from app.monitoring.budget_manager import (
    BudgetManager,
    BudgetAlert,
    BudgetStatus,
    BudgetPeriod,
    AlertSeverity,
)
from analytics.cost_analytics import (
    CostAnalytics,
    AnalyticsQuery,
    TimeGranularity,
    CostMetric,
    generate_daily_cost_summary,
)


class TestLangfuseConfig:
    """Test Langfuse configuration validation"""

    def test_valid_config_creation(self):
        """Test creating valid Langfuse configuration"""
        config = LangfuseConfig(
            public_key="pk_test_123",
            secret_key="sk_test_456",
            host="https://cloud.langfuse.com",
            environment="test",
        )

        assert config.public_key == "pk_test_123"
        assert config.secret_key == "sk_test_456"
        assert config.host == "https://cloud.langfuse.com"
        assert config.environment == "test"
        assert config.enabled is True
        assert config.flush_at == 15
        assert config.timeout == 10.0

    def test_default_values(self):
        """Test default configuration values"""
        config = LangfuseConfig(public_key="pk_test", secret_key="sk_test")

        assert config.host == "https://cloud.langfuse.com"
        assert config.environment == "production"
        assert config.enabled is True
        assert config.flush_at == 15
        assert config.flush_interval == 0.5
        assert config.max_retries == 3
        assert config.timeout == 10.0
        assert config.debug is False


class TestLangfuseClientManager:
    """Test Langfuse client management"""

    @pytest.fixture
    def mock_langfuse(self):
        """Mock Langfuse client"""
        with patch("monitoring.langfuse_client.Langfuse") as mock:
            mock_instance = Mock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def test_config(self):
        """Test configuration"""
        return LangfuseConfig(
            public_key="pk_test_123", secret_key="sk_test_456", environment="test"
        )

    def test_singleton_pattern(self):
        """Test that LangfuseClientManager follows singleton pattern"""
        manager1 = LangfuseClientManager()
        manager2 = LangfuseClientManager()

        assert manager1 is manager2

    def test_initialization_success(self, mock_langfuse, test_config):
        """Test successful client initialization"""
        manager = LangfuseClientManager.initialize(test_config)

        assert manager.is_enabled is True
        assert manager.config == test_config
        assert manager.client is not None

    def test_initialization_disabled(self):
        """Test initialization with disabled tracking"""
        disabled_config = LangfuseConfig(
            public_key="pk_test", secret_key="sk_test", enabled=False
        )

        manager = LangfuseClientManager.initialize(disabled_config)

        assert manager.is_enabled is False
        assert manager.client is None

    def test_health_check_success(self, mock_langfuse, test_config):
        """Test successful health check"""
        mock_trace = Mock()
        mock_langfuse.trace.return_value = mock_trace

        manager = LangfuseClientManager.initialize(test_config)
        health_result = manager.health_check()

        assert health_result["status"] == "healthy"
        assert health_result["client_available"] is True
        mock_langfuse.trace.assert_called_once()
        mock_trace.update.assert_called_once_with(output="Health check successful")

    def test_health_check_disabled(self):
        """Test health check with disabled client"""
        disabled_config = LangfuseConfig(
            public_key="pk_test", secret_key="sk_test", enabled=False
        )

        manager = LangfuseClientManager.initialize(disabled_config)
        health_result = manager.health_check()

        assert health_result["status"] == "disabled"
        assert health_result["client_available"] is False

    @patch.dict(
        "os.environ",
        {
            "LANGFUSE_PUBLIC_KEY": "pk_env_123",
            "LANGFUSE_SECRET_KEY": "sk_env_456",
            "LANGFUSE_HOST": "https://custom.langfuse.com",
            "LANGFUSE_ENVIRONMENT": "staging",
        },
    )
    def test_load_config_from_env(self):
        """Test loading configuration from environment variables"""
        config = LangfuseClientManager._load_config_from_env()

        assert config.public_key == "pk_env_123"
        assert config.secret_key == "sk_env_456"
        assert config.host == "https://custom.langfuse.com"
        assert config.environment == "staging"


class TestCostCalculator:
    """Test cost calculation functionality"""

    @pytest.fixture
    def calculator(self):
        """Cost calculator instance"""
        return CostCalculator()

    def test_token_counting_basic(self, calculator):
        """Test basic token counting"""
        text = "Hello, world! This is a test."
        token_count = calculator.count_tokens(text, "gpt-4-turbo")

        assert token_count > 0
        assert isinstance(token_count, int)

    def test_token_counting_empty_text(self, calculator):
        """Test token counting with empty text"""
        token_count = calculator.count_tokens("", "gpt-4-turbo")
        assert token_count == 0

    def test_cost_calculation_gpt4(self, calculator):
        """Test cost calculation for GPT-4"""
        token_usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = calculator.calculate_cost("gpt-4-turbo", token_usage)

        # GPT-4 Turbo: $0.01 input, $0.03 output per 1K tokens
        expected_cost = (1000 / 1000 * 0.01) + (500 / 1000 * 0.03)
        assert abs(cost - expected_cost) < 0.000001

    def test_cost_calculation_embedding(self, calculator):
        """Test cost calculation for embedding model"""
        token_usage = TokenUsage(input_tokens=1000, output_tokens=0)
        cost = calculator.calculate_cost("text-embedding-3-small", token_usage)

        # Embedding models only charge for input tokens
        expected_cost = 1000 / 1000 * 0.00002
        assert abs(cost - expected_cost) < 0.000001

    def test_estimate_tokens(self, calculator):
        """Test token estimation"""
        input_text = "This is a test input text."
        output_text = "This is the generated response."

        token_usage = calculator.estimate_tokens(input_text, output_text, "gpt-4-turbo")

        assert token_usage.input_tokens > 0
        assert token_usage.output_tokens > 0
        assert (
            token_usage.total_tokens
            == token_usage.input_tokens + token_usage.output_tokens
        )

    def test_batch_cost_calculation(self, calculator):
        """Test batch cost calculation"""
        operations = [
            ("gpt-4-turbo", TokenUsage(input_tokens=1000, output_tokens=500)),
            ("text-embedding-3-small", TokenUsage(input_tokens=500, output_tokens=0)),
            ("gpt-3.5-turbo", TokenUsage(input_tokens=2000, output_tokens=1000)),
        ]

        result = calculator.calculate_batch_cost(operations)

        assert "total_cost" in result
        assert "model_breakdown" in result
        assert "operation_count" in result
        assert result["operation_count"] == 3
        assert result["total_cost"] > 0

    def test_model_comparison(self, calculator):
        """Test model cost comparison"""
        token_usage = TokenUsage(input_tokens=1000, output_tokens=500)
        models = ["gpt-4-turbo", "gpt-3.5-turbo", "claude-3-sonnet-20240229"]

        comparison = calculator.compare_model_costs(token_usage, models)

        assert "models" in comparison
        assert "cheapest" in comparison
        assert "most_expensive" in comparison
        assert len(comparison["models"]) == 3

        # GPT-3.5 should be cheaper than GPT-4
        assert (
            comparison["models"]["gpt-3.5-turbo"]["cost"]
            < comparison["models"]["gpt-4-turbo"]["cost"]
        )


class TestLLMTracker:
    """Test LLM operation tracking"""

    @pytest.fixture
    def tracker(self):
        """LLM tracker instance"""
        return LLMTracker()

    @pytest.fixture
    def sample_metadata(self):
        """Sample trace metadata"""
        return TraceMetadata(
            operation_type="test_operation",
            user_id=uuid4(),
            model_name="gpt-4-turbo",
            session_id="test_session",
            tags=["test"],
            custom_metadata={"test": True},
        )

    def test_trace_metadata_creation(self, sample_metadata):
        """Test trace metadata creation"""
        assert sample_metadata.operation_type == "test_operation"
        assert sample_metadata.model_name == "gpt-4-turbo"
        assert sample_metadata.session_id == "test_session"
        assert "test" in sample_metadata.tags
        assert sample_metadata.custom_metadata["test"] is True

    def test_operation_result_creation(self):
        """Test operation result creation"""
        token_usage = TokenUsage(input_tokens=1000, output_tokens=500)

        result = OperationResult(
            success=True,
            output="Test output",
            token_usage=token_usage,
            cost=0.05,
            duration_ms=1500.0,
        )

        assert result.success is True
        assert result.output == "Test output"
        assert result.token_usage.total_tokens == 1500
        assert result.cost == 0.05
        assert result.duration_ms == 1500.0

    @patch("monitoring.llm_tracker.is_langfuse_enabled")
    @patch("monitoring.llm_tracker.get_langfuse_client")
    def test_create_trace_enabled(
        self, mock_get_client, mock_is_enabled, tracker, sample_metadata
    ):
        """Test trace creation when Langfuse is enabled"""
        mock_is_enabled.return_value = True
        mock_client = Mock()
        mock_trace = Mock()
        mock_trace.id = "trace_123"
        mock_client.trace.return_value = mock_trace
        mock_get_client.return_value = mock_client

        trace = tracker.create_trace("test_operation", sample_metadata, "test input")

        assert trace is not None
        mock_client.trace.assert_called_once()
        assert tracker._session_operations.get("test_session") == ["trace_123"]

    @patch("monitoring.llm_tracker.is_langfuse_enabled")
    def test_create_trace_disabled(self, mock_is_enabled, tracker, sample_metadata):
        """Test trace creation when Langfuse is disabled"""
        mock_is_enabled.return_value = False

        trace = tracker.create_trace("test_operation", sample_metadata, "test input")

        assert trace is None

    def test_session_summary(self, tracker):
        """Test session summary generation"""
        session_id = "test_session"
        tracker._session_operations[session_id] = ["trace_1", "trace_2", "trace_3"]

        summary = tracker.get_session_summary(session_id)

        assert summary is not None
        assert summary["session_id"] == session_id
        assert summary["operation_count"] == 3
        assert summary["operation_ids"] == ["trace_1", "trace_2", "trace_3"]


class TestTrackingDecorator:
    """Test LLM operation tracking decorator"""

    @pytest.mark.asyncio
    async def test_async_function_tracking(self):
        """Test tracking async functions"""

        @track_llm_operation(
            operation_type="test", model_name="gpt-4-turbo", tags=["test"]
        )
        async def async_llm_function(prompt: str) -> str:
            await asyncio.sleep(0.1)  # Simulate async operation
            return f"Response to: {prompt}"

        result = await async_llm_function("Test prompt")

        assert result == "Response to: Test prompt"

    def test_sync_function_tracking(self):
        """Test tracking sync functions"""

        @track_llm_operation(
            operation_type="test", model_name="gpt-4-turbo", tags=["test"]
        )
        def sync_llm_function(prompt: str) -> str:
            return f"Response to: {prompt}"

        # Note: sync functions are wrapped to run in asyncio
        result = sync_llm_function("Test prompt")

        assert result == "Response to: Test prompt"

    def test_embedding_tracking_decorator(self):
        """Test embedding generation tracking decorator"""
        user_id = uuid4()
        document_id = uuid4()

        @track_embedding_generation(user_id=user_id, document_id=document_id)
        def generate_embedding(text: str) -> list:
            return [0.1, 0.2, 0.3] * 512  # Mock embedding

        result = generate_embedding("Test text")

        assert len(result) == 1536  # Standard embedding dimension

    @patch("monitoring.llm_tracker.add_user_feedback")
    def test_user_feedback(self, mock_add_feedback):
        """Test user feedback functionality"""
        mock_add_feedback.return_value = True

        result = add_user_feedback("trace_123", 0.8, "Good response")

        assert result is True
        mock_add_feedback.assert_called_once_with("trace_123", 0.8, "Good response")


class TestBudgetManager:
    """Test budget management functionality"""

    @pytest.fixture
    def budget_manager(self):
        """Budget manager instance"""
        return BudgetManager()

    @pytest.fixture
    def test_user_id(self):
        """Test user ID"""
        return uuid4()

    def test_set_user_budget(self, budget_manager, test_user_id):
        """Test setting user budget"""
        result = budget_manager.set_user_budget(
            test_user_id, 100.0, BudgetPeriod.MONTHLY
        )

        assert result is True
        assert budget_manager._user_budgets[test_user_id][BudgetPeriod.MONTHLY] == 100.0

    def test_create_budget_alert(self, budget_manager, test_user_id):
        """Test creating budget alert"""
        alert = budget_manager.create_budget_alert(
            alert_id="test_alert",
            user_id=test_user_id,
            name="Test Alert",
            threshold_percentage=0.8,
            budget_period=BudgetPeriod.MONTHLY,
            severity=AlertSeverity.WARNING,
            notification_channels=["email", "dashboard"],
        )

        assert alert.id == "test_alert"
        assert alert.user_id == test_user_id
        assert alert.threshold_percentage == 0.8
        assert alert.severity == AlertSeverity.WARNING

    def test_budget_alert_triggering(self, budget_manager, test_user_id):
        """Test budget alert triggering"""
        # Create alert
        budget_manager.create_budget_alert(
            alert_id="test_alert",
            user_id=test_user_id,
            name="80% Alert",
            threshold_percentage=0.8,
            budget_period=BudgetPeriod.MONTHLY,
            severity=AlertSeverity.WARNING,
            notification_channels=["dashboard"],
        )

        # Test alert triggering
        triggered_alerts = budget_manager.check_budget_alerts(
            test_user_id, 85.0, 100.0, BudgetPeriod.MONTHLY
        )

        assert len(triggered_alerts) == 1
        assert triggered_alerts[0].id == "test_alert"

    def test_spending_forecast(self, budget_manager, test_user_id):
        """Test spending forecast generation"""
        # Set budget
        budget_manager.set_user_budget(test_user_id, 100.0, BudgetPeriod.MONTHLY)

        # Generate forecast
        forecast = budget_manager.generate_spending_forecast(
            test_user_id, BudgetPeriod.MONTHLY
        )

        assert forecast is not None
        assert forecast.period == BudgetPeriod.MONTHLY
        assert forecast.projected_spending >= 0
        assert len(forecast.confidence_interval) == 2

    def test_cost_optimization_recommendations(self, budget_manager, test_user_id):
        """Test cost optimization recommendations"""
        recommendations = budget_manager.get_cost_optimization_recommendations(
            test_user_id
        )

        assert isinstance(recommendations, list)
        # Recommendations may be empty or contain items based on spending patterns
        for rec in recommendations:
            assert hasattr(rec, "category")
            assert hasattr(rec, "title")
            assert hasattr(rec, "potential_savings")
            assert hasattr(rec, "priority")


class TestCostAnalytics:
    """Test cost analytics functionality"""

    @pytest.fixture
    def analytics(self):
        """Cost analytics instance"""
        return CostAnalytics()

    @pytest.fixture
    def sample_query(self):
        """Sample analytics query"""
        return AnalyticsQuery(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            granularity=TimeGranularity.DAILY,
            metrics=[CostMetric.TOTAL_COST, CostMetric.TOKEN_COUNT],
        )

    def test_dashboard_data_generation(self, analytics, sample_query):
        """Test dashboard data generation"""
        dashboard_data = analytics.generate_dashboard_data(sample_query)

        assert "summary" in dashboard_data.model_dump()
        assert "time_series" in dashboard_data.model_dump()
        assert "breakdowns" in dashboard_data.model_dump()
        assert "trends" in dashboard_data.model_dump()
        assert "alerts" in dashboard_data.model_dump()
        assert "recommendations" in dashboard_data.model_dump()

    def test_cost_report_generation(self, analytics, sample_query):
        """Test cost report generation"""
        report = analytics.generate_cost_report(sample_query, include_details=True)

        assert "report_metadata" in report
        assert "summary" in report
        assert "detailed_breakdown" in report
        assert "trends" in report
        assert "operation_details" in report

    def test_roi_metrics_calculation(self, analytics, sample_query):
        """Test ROI metrics calculation"""
        roi_metrics = analytics.calculate_roi_metrics(sample_query)

        assert "cost_efficiency" in roi_metrics
        assert "productivity_metrics" in roi_metrics
        assert "quality_metrics" in roi_metrics
        assert "roi_analysis" in roi_metrics

    def test_daily_cost_summary(self):
        """Test daily cost summary generation"""
        summary = generate_daily_cost_summary(days=7)

        assert isinstance(summary, dict)
        assert "summary" in summary
        assert "time_series" in summary
        assert "breakdowns" in summary


class TestEnvironmentValidation:
    """Test environment validation"""

    @patch.dict(
        "os.environ",
        {"LANGFUSE_PUBLIC_KEY": "pk_test_123", "LANGFUSE_SECRET_KEY": "sk_test_456"},
    )
    def test_valid_environment(self):
        """Test validation with valid environment"""
        result = validate_langfuse_environment()

        assert result["valid"] is True
        assert len(result["missing_required"]) == 0

    @patch.dict(
        "os.environ",
        {
            "LANGFUSE_PUBLIC_KEY": "pk_test_123"
            # Missing LANGFUSE_SECRET_KEY
        },
        clear=True,
    )
    def test_invalid_environment(self):
        """Test validation with invalid environment"""
        result = validate_langfuse_environment()

        assert result["valid"] is False
        assert "LANGFUSE_SECRET_KEY" in result["missing_required"]

    def test_initialize_with_custom_config(self):
        """Test initialization with custom configuration"""
        manager = initialize_langfuse(
            public_key="pk_custom_123", secret_key="sk_custom_456", environment="test"
        )

        assert manager is not None
        assert manager.config.public_key == "pk_custom_123"
        assert manager.config.secret_key == "sk_custom_456"
        assert manager.config.environment == "test"


class TestIntegrationScenarios:
    """Test complete integration scenarios"""

    @pytest.mark.asyncio
    async def test_complete_document_processing_flow(self):
        """Test complete document processing with tracking"""
        user_id = uuid4()
        document_id = uuid4()

        @track_llm_operation(
            operation_type="document_processing",
            model_name="gpt-4-turbo",
            user_id=user_id,
            document_id=document_id,
            tags=["integration_test"],
        )
        async def process_document(content: str) -> dict:
            # Simulate document processing
            await asyncio.sleep(0.1)
            return {
                "summary": "Processed document content",
                "key_points": ["Point 1", "Point 2"],
                "sentiment": "positive",
            }

        result = await process_document("Test document content")

        assert result["summary"] == "Processed document content"
        assert len(result["key_points"]) == 2

    def test_cost_tracking_with_budget_alerts(self):
        """Test cost tracking with budget alerts"""
        user_id = uuid4()
        budget_manager = BudgetManager()

        # Set budget
        budget_manager.set_user_budget(user_id, 50.0, BudgetPeriod.MONTHLY)

        # Create alert
        budget_manager.create_budget_alert(
            alert_id="integration_alert",
            user_id=user_id,
            name="Integration Test Alert",
            threshold_percentage=0.8,
            budget_period=BudgetPeriod.MONTHLY,
            severity=AlertSeverity.WARNING,
            notification_channels=["dashboard"],
        )

        # Simulate spending that triggers alert
        triggered_alerts = budget_manager.check_budget_alerts(
            user_id, 42.0, 50.0, BudgetPeriod.MONTHLY
        )

        assert len(triggered_alerts) == 1
        assert triggered_alerts[0].name == "Integration Test Alert"

    def test_analytics_with_real_data_flow(self):
        """Test analytics generation with simulated data flow"""
        analytics = CostAnalytics()

        # Create query for last week
        query = AnalyticsQuery(
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
            granularity=TimeGranularity.DAILY,
            metrics=[
                CostMetric.TOTAL_COST,
                CostMetric.TOKEN_COUNT,
                CostMetric.OPERATION_COUNT,
            ],
        )

        # Generate dashboard data
        dashboard_data = analytics.generate_dashboard_data(query)

        # Verify data structure
        assert dashboard_data.summary["total_cost"] > 0
        assert len(dashboard_data.time_series) > 0
        assert dashboard_data.breakdowns.total_cost > 0
        assert len(dashboard_data.breakdowns.by_model) > 0


# Pytest fixtures for test database and mocking
@pytest.fixture(scope="session")
def mock_database():
    """Mock database for testing"""
    return {}


@pytest.fixture
def clean_environment():
    """Clean environment for each test"""
    # Reset singleton instances
    LangfuseClientManager._instance = None
    LangfuseClientManager._client = None
    LangfuseClientManager._config = None
    yield
    # Cleanup after test
    LangfuseClientManager._instance = None
    LangfuseClientManager._client = None
    LangfuseClientManager._config = None


# Test configuration for pytest
def pytest_configure(config):
    """Configure pytest for Langfuse integration tests"""
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


if __name__ == "__main__":
    pytest.main([__file__])
