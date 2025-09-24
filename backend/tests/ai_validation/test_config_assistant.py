"""
AI Validation Tests for Pydantic AI Configuration Assistant
Tests the AI-powered configuration wizard and validation systems.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any

# Import configuration modules
from app.api.config import ConfigurationAssistant, ConfigurationValidator
from app.config.settings import SettingsManager
from pydantic import BaseModel, ValidationError


@pytest.mark.ai
class TestConfigurationAssistant:
    """Test suite for AI configuration assistant validation."""

    @pytest.fixture
    def config_assistant(self):
        """Create ConfigurationAssistant instance for testing."""
        return ConfigurationAssistant(
            model="gpt-4", validation_enabled=True, max_retries=3
        )

    @pytest.fixture
    def config_validator(self):
        """Create ConfigurationValidator instance for testing."""
        return ConfigurationValidator()

    @pytest.fixture
    def settings_manager(self):
        """Create SettingsManager instance for testing."""
        return SettingsManager()

    @pytest.fixture
    def config_scenarios(self):
        """Configuration scenarios for testing."""
        return {
            "database_config": {
                "user_request": "Configure PostgreSQL database for production Django app with high availability",
                "expected_fields": [
                    "database_name",
                    "host",
                    "port",
                    "user",
                    "password",
                    "connection_pooling",
                    "ssl_mode",
                    "replica_hosts",
                    "max_connections",
                    "backup_settings",
                ],
                "validation_rules": {
                    "port": {"type": "integer", "range": [1, 65535]},
                    "ssl_mode": {
                        "type": "string",
                        "allowed_values": ["require", "prefer", "disable"],
                    },
                    "max_connections": {"type": "integer", "minimum": 1},
                },
            },
            "api_config": {
                "user_request": "Set up OpenAI API configuration with rate limiting and error handling",
                "expected_fields": [
                    "api_key",
                    "model",
                    "max_tokens",
                    "temperature",
                    "rate_limit_rpm",
                    "timeout",
                    "retry_strategy",
                    "error_handling",
                    "logging_level",
                ],
                "validation_rules": {
                    "temperature": {"type": "float", "range": [0.0, 2.0]},
                    "max_tokens": {"type": "integer", "minimum": 1, "maximum": 32000},
                    "rate_limit_rpm": {"type": "integer", "minimum": 1},
                },
            },
            "search_config": {
                "user_request": "Configure hybrid search with vector similarity and keyword matching",
                "expected_fields": [
                    "vector_weight",
                    "keyword_weight",
                    "similarity_threshold",
                    "max_results",
                    "boost_factors",
                    "filters",
                    "index_settings",
                    "cache_config",
                ],
                "validation_rules": {
                    "vector_weight": {"type": "float", "range": [0.0, 1.0]},
                    "keyword_weight": {"type": "float", "range": [0.0, 1.0]},
                    "similarity_threshold": {"type": "float", "range": [0.0, 1.0]},
                },
            },
        }

    def test_configuration_generation_accuracy(
        self, config_assistant, config_scenarios
    ):
        """Test accuracy of AI-generated configurations."""
        for scenario_name, scenario in config_scenarios.items():
            with patch.object(config_assistant, "_call_ai_model") as mock_ai:
                # Mock AI response with realistic configuration
                mock_ai.return_value = self._generate_mock_config(scenario)

                # Generate configuration
                generated_config = config_assistant.generate_configuration(
                    scenario["user_request"]
                )

                # Validate completeness
                completeness_score = self._calculate_completeness(
                    generated_config, scenario["expected_fields"]
                )

                assert (
                    completeness_score >= 0.85
                ), f"Configuration completeness {completeness_score:.3f} below threshold for {scenario_name}"

                # Validate structure
                assert isinstance(
                    generated_config, dict
                ), f"Generated configuration should be a dictionary for {scenario_name}"

    def test_configuration_validation_accuracy(
        self, config_validator, config_scenarios
    ):
        """Test configuration validation accuracy."""
        for scenario_name, scenario in config_scenarios.items():
            # Test valid configuration
            valid_config = self._generate_valid_config(scenario)
            validation_result = config_validator.validate_configuration(
                valid_config, scenario["validation_rules"]
            )

            assert (
                validation_result.is_valid
            ), f"Valid configuration rejected for {scenario_name}: {validation_result.errors}"

            # Test invalid configuration
            invalid_config = self._generate_invalid_config(scenario)
            validation_result = config_validator.validate_configuration(
                invalid_config, scenario["validation_rules"]
            )

            assert (
                not validation_result.is_valid
            ), f"Invalid configuration accepted for {scenario_name}"
            assert (
                len(validation_result.errors) > 0
            ), f"No validation errors reported for invalid {scenario_name} config"

    def test_ai_suggestion_quality(self, config_assistant):
        """Test quality of AI configuration suggestions."""
        test_requests = [
            {
                "request": "I need a secure database configuration",
                "expected_security_features": [
                    "ssl",
                    "encryption",
                    "authentication",
                    "access_control",
                ],
                "minimum_features": 3,
            },
            {
                "request": "Configure for high performance processing",
                "expected_performance_features": [
                    "caching",
                    "pooling",
                    "optimization",
                    "parallel",
                ],
                "minimum_features": 2,
            },
            {
                "request": "Set up monitoring and logging",
                "expected_monitoring_features": [
                    "metrics",
                    "alerts",
                    "logging",
                    "health_checks",
                ],
                "minimum_features": 3,
            },
        ]

        for test in test_requests:
            with patch.object(config_assistant, "_call_ai_model") as mock_ai:
                mock_ai.return_value = self._generate_contextual_config(test)

                config = config_assistant.generate_configuration(test["request"])

                # Check for expected features
                feature_count = self._count_matching_features(
                    config,
                    test.get("expected_security_features", [])
                    + test.get("expected_performance_features", [])
                    + test.get("expected_monitoring_features", []),
                )

                assert (
                    feature_count >= test["minimum_features"]
                ), f"Configuration missing expected features for request: {test['request']}"

    def test_configuration_template_accuracy(self, config_assistant):
        """Test configuration template generation accuracy."""
        template_requests = [
            {
                "template_type": "microservices",
                "expected_components": [
                    "service_discovery",
                    "load_balancer",
                    "api_gateway",
                    "monitoring",
                ],
                "minimum_components": 3,
            },
            {
                "template_type": "machine_learning",
                "expected_components": [
                    "data_pipeline",
                    "model_training",
                    "inference",
                    "monitoring",
                ],
                "minimum_components": 3,
            },
            {
                "template_type": "web_application",
                "expected_components": [
                    "database",
                    "cache",
                    "session_store",
                    "static_files",
                ],
                "minimum_components": 3,
            },
        ]

        for template in template_requests:
            with patch.object(config_assistant, "_call_ai_model") as mock_ai:
                mock_ai.return_value = self._generate_template_config(template)

                config = config_assistant.generate_template_configuration(
                    template["template_type"]
                )

                # Validate template structure
                component_count = self._count_template_components(
                    config, template["expected_components"]
                )

                assert (
                    component_count >= template["minimum_components"]
                ), f"Template missing components for {template['template_type']}"

    def test_configuration_error_handling(self, config_assistant):
        """Test configuration assistant error handling."""
        error_scenarios = [
            {
                "request": "",  # Empty request
                "expected_error_type": "InvalidInputError",
            },
            {
                "request": "Configure something impossible with infinite resources",
                "expected_error_type": "ValidationError",
            },
            {
                "request": "Set up database with conflicting requirements: MySQL and PostgreSQL",
                "expected_error_type": "ConflictError",
            },
        ]

        for scenario in error_scenarios:
            with patch.object(config_assistant, "_call_ai_model") as mock_ai:
                # Simulate AI model errors
                if scenario["request"] == "":
                    mock_ai.side_effect = ValueError("Empty request")
                else:
                    mock_ai.return_value = {"error": "Invalid configuration"}

                result = config_assistant.generate_configuration_safe(
                    scenario["request"]
                )

                # Should handle errors gracefully
                assert (
                    result.success is False
                ), f"Should detect error for scenario: {scenario['request']}"
                assert (
                    result.error_type is not None
                ), f"Should provide error type for scenario: {scenario['request']}"

    def test_configuration_optimization_suggestions(self, config_assistant):
        """Test AI configuration optimization suggestions."""
        base_configs = [
            {
                "database": {
                    "host": "localhost",
                    "port": 5432,
                    "max_connections": 100,
                    "ssl_mode": "disable",
                },
                "expected_optimizations": [
                    "ssl_enable",
                    "connection_pooling",
                    "performance_tuning",
                ],
            },
            {
                "api": {"timeout": 300, "retry_count": 1, "cache_enabled": False},
                "expected_optimizations": [
                    "timeout_reduction",
                    "retry_strategy",
                    "cache_enable",
                ],
            },
        ]

        for config_test in base_configs:
            base_config = {
                k: v for k, v in config_test.items() if k != "expected_optimizations"
            }

            with patch.object(config_assistant, "_call_ai_model") as mock_ai:
                mock_ai.return_value = self._generate_optimization_suggestions(
                    config_test
                )

                optimizations = config_assistant.suggest_optimizations(base_config)

                # Validate optimization suggestions
                assert len(optimizations) > 0, "Should provide optimization suggestions"

                suggestion_types = [opt["type"] for opt in optimizations]
                expected_opts = config_test["expected_optimizations"]

                matching_opts = sum(
                    1
                    for opt in expected_opts
                    if any(opt in s for s in suggestion_types)
                )
                assert (
                    matching_opts >= len(expected_opts) * 0.6
                ), "Should provide relevant optimization suggestions"

    def test_configuration_security_validation(self, config_validator):
        """Test security validation of configurations."""
        security_tests = [
            {
                "config": {
                    "database": {
                        "password": "password123",
                        "ssl_mode": "disable",
                        "public_access": True,
                    }
                },
                "expected_security_issues": [
                    "weak_password",
                    "ssl_disabled",
                    "public_access",
                ],
            },
            {
                "config": {
                    "api": {
                        "api_key": "test-key",
                        "rate_limiting": False,
                        "cors_origins": ["*"],
                    }
                },
                "expected_security_issues": ["no_rate_limiting", "open_cors"],
            },
        ]

        for test in security_tests:
            security_report = config_validator.validate_security(test["config"])

            assert (
                len(security_report.issues) > 0
            ), "Should detect security issues in insecure configuration"

            # Check for specific expected issues
            issue_types = [issue.type for issue in security_report.issues]
            for expected_issue in test["expected_security_issues"]:
                assert any(
                    expected_issue in issue_type for issue_type in issue_types
                ), f"Should detect {expected_issue} security issue"

    def test_configuration_performance_validation(self, config_validator):
        """Test performance validation of configurations."""
        performance_tests = [
            {
                "config": {
                    "database": {
                        "connection_pool_size": 1,
                        "query_timeout": 1,
                        "cache_enabled": False,
                    }
                },
                "expected_performance_issues": [
                    "small_pool",
                    "short_timeout",
                    "no_cache",
                ],
            },
            {
                "config": {
                    "processing": {
                        "batch_size": 1,
                        "parallel_workers": 1,
                        "memory_limit": "128MB",
                    }
                },
                "expected_performance_issues": [
                    "small_batch",
                    "single_worker",
                    "low_memory",
                ],
            },
        ]

        for test in performance_tests:
            performance_report = config_validator.validate_performance(test["config"])

            assert (
                len(performance_report.warnings) > 0
            ), "Should detect performance issues in suboptimal configuration"

    @pytest.mark.performance
    def test_configuration_generation_performance(self, config_assistant):
        """Test performance of configuration generation."""
        import time

        test_requests = [
            "Simple database configuration",
            "Complex microservices architecture with multiple databases, caches, and monitoring",
            "Enterprise-grade configuration with security, monitoring, backup, and high availability requirements",
        ]

        max_times = [2.0, 5.0, 10.0]  # seconds

        for request, max_time in zip(test_requests, max_times):
            start_time = time.time()

            with patch.object(config_assistant, "_call_ai_model") as mock_ai:
                mock_ai.return_value = {"test": "config"}
                config = config_assistant.generate_configuration(request)

            end_time = time.time()
            generation_time = end_time - start_time

            assert (
                generation_time < max_time
            ), f"Configuration generation took {generation_time:.2f}s, expected < {max_time}s"

    # Helper methods

    def _generate_mock_config(self, scenario: Dict) -> Dict:
        """Generate mock configuration based on scenario."""
        config = {}
        for field in scenario["expected_fields"]:
            if "port" in field:
                config[field] = 5432
            elif "weight" in field:
                config[field] = 0.7
            elif "threshold" in field:
                config[field] = 0.8
            elif "max" in field:
                config[field] = 100
            elif "ssl" in field:
                config[field] = "require"
            else:
                config[field] = f"test_{field}"
        return config

    def _generate_valid_config(self, scenario: Dict) -> Dict:
        """Generate valid configuration for testing."""
        config = {}
        rules = scenario.get("validation_rules", {})

        for field, rule in rules.items():
            if rule["type"] == "integer":
                config[field] = rule.get("minimum", 1)
            elif rule["type"] == "float":
                range_min, range_max = rule.get("range", [0.0, 1.0])
                config[field] = (range_min + range_max) / 2
            elif rule["type"] == "string":
                allowed_values = rule.get("allowed_values", ["test_value"])
                config[field] = allowed_values[0]

        return config

    def _generate_invalid_config(self, scenario: Dict) -> Dict:
        """Generate invalid configuration for testing."""
        config = {}
        rules = scenario.get("validation_rules", {})

        for field, rule in rules.items():
            if rule["type"] == "integer":
                config[field] = -1  # Invalid negative value
            elif rule["type"] == "float":
                config[field] = 999.0  # Out of range
            elif rule["type"] == "string":
                config[field] = "invalid_value"

        return config

    def _calculate_completeness(
        self, config: Dict, expected_fields: List[str]
    ) -> float:
        """Calculate configuration completeness score."""
        if not expected_fields:
            return 1.0

        found_fields = 0
        config_str = json.dumps(config).lower()

        for field in expected_fields:
            if field.lower() in config_str:
                found_fields += 1

        return found_fields / len(expected_fields)

    def _count_matching_features(self, config: Dict, features: List[str]) -> int:
        """Count matching features in configuration."""
        config_str = json.dumps(config).lower()
        return sum(1 for feature in features if feature.lower() in config_str)

    def _generate_contextual_config(self, test: Dict) -> Dict:
        """Generate contextual configuration based on test requirements."""
        config = {"settings": {}}

        for feature_list in [
            "expected_security_features",
            "expected_performance_features",
            "expected_monitoring_features",
        ]:
            if feature_list in test:
                for feature in test[feature_list]:
                    config["settings"][feature] = True

        return config

    def _generate_template_config(self, template: Dict) -> Dict:
        """Generate template configuration."""
        config = {"template": template["template_type"], "components": {}}

        for component in template["expected_components"]:
            config["components"][component] = {"enabled": True, "config": {}}

        return config

    def _count_template_components(
        self, config: Dict, expected_components: List[str]
    ) -> int:
        """Count template components in configuration."""
        if "components" not in config:
            return 0

        found_components = 0
        for component in expected_components:
            if component in config["components"]:
                found_components += 1

        return found_components

    def _generate_optimization_suggestions(self, config_test: Dict) -> Dict:
        """Generate optimization suggestions."""
        return {
            "optimizations": [
                {"type": opt, "description": f"Optimize {opt}", "priority": "medium"}
                for opt in config_test["expected_optimizations"]
            ]
        }
