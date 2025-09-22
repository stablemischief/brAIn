"""
Test suite for the AI Configuration Wizard.

Tests configuration validation, template application, SQL generation,
and complete wizard workflow.
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

from config.models import (
    SystemConfig, DatabaseConfig, OpenAIConfig,
    SupabaseConfig, SecurityConfig, CostManagementConfig,
    ProcessingConfig, ConfigurationTemplate, ValidationResult
)
from config.validators import ConfigurationValidator
from config.templates import ConfigurationTemplates
from config.sql_generator import SQLScriptGenerator
from config.wizard import ConfigurationWizard


@pytest.fixture
def sample_database_config():
    """Sample database configuration."""
    return DatabaseConfig(
        host="localhost",
        port=5432,
        database="test_db",
        username="test_user",
        password="test_password",
        schema="public",
        pool_size=10,
        max_overflow=5
    )


@pytest.fixture
def sample_openai_config():
    """Sample OpenAI configuration."""
    return OpenAIConfig(
        api_key="sk-test-key-123456789012345678901234567890",
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=4000,
        timeout=30,
        max_retries=3
    )


@pytest.fixture
def sample_supabase_config():
    """Sample Supabase configuration."""
    return SupabaseConfig(
        url="https://test.supabase.co",
        anon_key="test-anon-key-with-sufficient-length-for-validation",
        service_key=None,
        jwt_secret=None
    )


@pytest.fixture
def sample_security_config():
    """Sample security configuration."""
    return SecurityConfig(
        jwt_enabled=True,
        jwt_secret="test-jwt-secret-that-is-at-least-32-characters-long",
        jwt_algorithm="HS256",
        jwt_expiry_hours=24,
        cors_enabled=True,
        cors_origins=["http://localhost:3000"],
        rate_limit_enabled=True,
        rate_limit_requests=100,
        input_validation_strict=True,
        sql_injection_protection=True,
        xss_protection=True
    )


@pytest.fixture
def sample_cost_config():
    """Sample cost management configuration."""
    return CostManagementConfig(
        daily_budget=50.0,
        monthly_budget=1000.0,
        alert_threshold_percent=80,
        hard_limit_enabled=True
    )


@pytest.fixture
def sample_processing_config():
    """Sample processing configuration."""
    return ProcessingConfig(
        batch_size=10,
        parallel_workers=4,
        chunk_size=1000,
        chunk_overlap=200,
        quality_threshold=0.7,
        duplicate_threshold=0.95,
        max_file_size_mb=100
    )


@pytest.fixture
def complete_system_config(
    sample_database_config,
    sample_openai_config,
    sample_supabase_config,
    sample_security_config,
    sample_cost_config,
    sample_processing_config
):
    """Complete system configuration."""
    return SystemConfig(
        environment="development",
        database=sample_database_config,
        openai=sample_openai_config,
        anthropic=None,
        supabase=sample_supabase_config,
        langfuse=None,
        security=sample_security_config,
        cost_management=sample_cost_config,
        processing=sample_processing_config
    )


class TestConfigurationModels:
    """Test configuration model validation."""

    def test_database_config_validation(self, sample_database_config):
        """Test database configuration validation."""
        assert sample_database_config.host == "localhost"
        assert sample_database_config.port == 5432

        # Test connection string generation
        conn_str = sample_database_config.get_connection_string(hide_password=True)
        assert "***" in conn_str
        assert "localhost:5432" in conn_str

        conn_str_full = sample_database_config.get_connection_string(hide_password=False)
        assert "test_password" in conn_str_full

    def test_database_config_sql_injection_protection(self):
        """Test SQL identifier validation."""
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            DatabaseConfig(
                host="localhost",
                port=5432,
                database="test'; DROP TABLE users; --",
                username="test_user",
                password="test_password"
            )

    def test_openai_config_validation(self, sample_openai_config):
        """Test OpenAI configuration validation."""
        assert sample_openai_config.model == "gpt-4o-mini"
        assert sample_openai_config.temperature == 0.7

        # Test invalid API key
        with pytest.raises(ValueError, match="Invalid OpenAI API key"):
            OpenAIConfig(
                api_key="invalid-key",
                model="gpt-4o-mini"
            )

    def test_security_config_validation(self, sample_security_config):
        """Test security configuration validation."""
        assert sample_security_config.jwt_enabled
        assert len(sample_security_config.jwt_secret.get_secret_value()) >= 32

        # Test weak JWT secret
        with pytest.raises(ValueError, match="JWT secret must be at least 32 characters"):
            SecurityConfig(
                jwt_enabled=True,
                jwt_secret="short-secret",
                jwt_algorithm="HS256",
                jwt_expiry_hours=24,
                cors_enabled=True,
                cors_origins=["http://localhost:3000"],
                rate_limit_enabled=True,
                rate_limit_requests=100
            )

    def test_system_config_validation(self, complete_system_config):
        """Test complete system configuration validation."""
        assert complete_system_config.environment == "development"
        assert complete_system_config.database is not None
        assert complete_system_config.openai is not None

    def test_production_config_requirements(self, complete_system_config):
        """Test production environment requirements."""
        # Modify to production
        complete_system_config.environment = "production"

        # Should fail without proper security
        complete_system_config.security.jwt_enabled = False
        with pytest.raises(ValueError, match="JWT must be enabled in production"):
            complete_system_config.model_validate(complete_system_config.model_dump())


class TestConfigurationTemplates:
    """Test configuration templates."""

    def test_get_all_templates(self):
        """Test retrieving all templates."""
        templates = ConfigurationTemplates.get_all_templates()
        assert len(templates) == 4

        template_names = [t.name for t in templates]
        assert "Development Environment" in template_names
        assert "Production Secure" in template_names
        assert "Cost Optimized" in template_names
        assert "High Performance" in template_names

    def test_development_template(self):
        """Test development template."""
        template = ConfigurationTemplates.get_development_template()
        assert template.name == "Development Environment"
        assert template.config["environment"] == "development"
        assert template.config["security"]["jwt_enabled"] == False
        assert template.config["cost_management"]["hard_limit_enabled"] == False

    def test_production_template(self):
        """Test production template."""
        template = ConfigurationTemplates.get_production_secure_template()
        assert template.name == "Production Secure"
        assert template.config["environment"] == "production"
        assert template.config["security"]["jwt_enabled"] == True
        assert template.config["cost_management"]["hard_limit_enabled"] == True
        assert template.config["langfuse"]["enabled"] == True

    def test_apply_template_with_overrides(self):
        """Test applying template with overrides."""
        template = ConfigurationTemplates.get_development_template()
        overrides = {
            "database": {
                "host": "custom-host",
                "port": 5433
            }
        }

        config = ConfigurationTemplates.apply_template(template, overrides)
        assert config["database"]["host"] == "custom-host"
        assert config["database"]["port"] == 5433
        assert config["database"]["database"] == "brain_dev"  # Original value


class TestSQLScriptGenerator:
    """Test SQL script generation."""

    def test_generate_complete_setup(self, complete_system_config):
        """Test complete SQL setup script generation."""
        generator = SQLScriptGenerator(complete_system_config)
        script = generator.generate_complete_setup()

        assert "CREATE EXTENSION IF NOT EXISTS" in script
        assert "CREATE TABLE IF NOT EXISTS documents" in script
        assert "CREATE INDEX IF NOT EXISTS" in script
        assert "CREATE TRIGGER" in script

    def test_generate_rollback_script(self, complete_system_config):
        """Test rollback script generation."""
        generator = SQLScriptGenerator(complete_system_config)
        script = generator.generate_rollback_script()

        assert "DROP TABLE IF EXISTS" in script
        assert "DROP FUNCTION IF EXISTS" in script
        assert "DROP TRIGGER IF EXISTS" in script


class TestConfigurationValidator:
    """Test configuration validation."""

    @pytest.mark.asyncio
    async def test_validate_environment_variables(self):
        """Test environment variable validation."""
        validator = ConfigurationValidator()

        # Mock environment variables
        with patch.dict('os.environ', {
            'DATABASE_URL': 'postgresql://user:pass@localhost/db',
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_ANON_KEY': 'test-key',
            'JWT_SECRET': 'test-secret'
        }):
            results = validator.validate_environment_variables()
            assert results['DATABASE_URL'] == True
            assert results['SUPABASE_URL'] == True

    @pytest.mark.asyncio
    async def test_validate_database_connection(self, sample_database_config):
        """Test database validation."""
        validator = ConfigurationValidator()

        # Mock psycopg2
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchone.side_effect = [
                ('PostgreSQL 14.0',),  # Version query
                ('100',)  # max_connections
            ]
            mock_cursor.fetchall.return_value = [('pgvector',), ('uuid-ossp',)]
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            result = await validator.validate_database(sample_database_config)
            assert result == True

    @pytest.mark.asyncio
    async def test_validate_openai(self, sample_openai_config):
        """Test OpenAI validation."""
        validator = ConfigurationValidator()

        # Mock OpenAI client
        with patch('config.validators.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_models = MagicMock()
            mock_models.data = [MagicMock(id='gpt-4o-mini')]
            mock_client.models.list.return_value = mock_models

            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock()]
            mock_client.chat.completions.create.return_value = mock_completion

            mock_openai.return_value = mock_client

            result = await validator.validate_openai(sample_openai_config)
            assert result == True

    def test_validate_security_settings(self, sample_security_config):
        """Test security validation."""
        validator = ConfigurationValidator()

        result = validator.validate_security(sample_security_config, "development")
        assert result == True

        # Test production requirements
        sample_security_config.cors_origins = ["*"]
        result = validator.validate_security(sample_security_config, "production")
        assert result == False
        assert "CORS wildcard" in validator.validation_result.errors[0]


class TestConfigurationWizard:
    """Test configuration wizard functionality."""

    @pytest.fixture
    def wizard(self, tmp_path):
        """Create wizard instance with temp path."""
        config_path = tmp_path / "config.json"
        return ConfigurationWizard(config_path)

    @pytest.mark.asyncio
    async def test_save_and_load_configuration(self, wizard, complete_system_config):
        """Test saving and loading configuration."""
        # Save configuration
        await wizard.save_configuration(complete_system_config)
        assert wizard.config_path.exists()

        # Load configuration
        loaded_config = await wizard.load_configuration()
        assert loaded_config.environment == complete_system_config.environment
        assert loaded_config.database.host == complete_system_config.database.host

    @pytest.mark.asyncio
    async def test_generate_sql_scripts(self, wizard, complete_system_config, tmp_path):
        """Test SQL script generation."""
        # Mock Path to use tmp_path
        with patch('config.wizard.Path') as mock_path:
            mock_path.return_value = tmp_path / "sql" / "setup.sql"
            mock_path.return_value.parent.mkdir(parents=True, exist_ok=True)

            await wizard._generate_sql_scripts(complete_system_config)

    def test_get_minimal_config(self, wizard):
        """Test minimal configuration generation."""
        config = wizard._get_minimal_config()
        assert "environment" in config
        assert "database" in config
        assert "security" in config

    @pytest.mark.asyncio
    async def test_wizard_template_selection(self, wizard):
        """Test template selection in wizard."""
        # Mock input
        with patch('builtins.input', return_value='1'):
            config = await wizard._select_template()
            assert config["environment"] == "development"

    @pytest.mark.asyncio
    async def test_validation_display(self, wizard):
        """Test validation result display."""
        result = ValidationResult(
            valid=False,
            errors=["Test error"],
            warnings=["Test warning"],
            recommendations=["Test recommendation"]
        )

        # Capture output
        with patch('builtins.print') as mock_print:
            wizard._display_validation_results(result)

            # Check that errors, warnings, and recommendations were printed
            calls = [str(call) for call in mock_print.call_args_list]
            assert any("Test error" in call for call in calls)
            assert any("Test warning" in call for call in calls)
            assert any("Test recommendation" in call for call in calls)


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete wizard flow."""

    @pytest.mark.asyncio
    async def test_complete_wizard_flow(self, tmp_path):
        """Test complete wizard flow with mocked inputs."""
        config_path = tmp_path / "config.json"
        wizard = ConfigurationWizard(config_path)

        # Mock all user inputs
        inputs = [
            '1',  # Select development template
            '',   # Use default environment
            'n',  # Don't use DATABASE_URL
            '',   # Default host
            '',   # Default port
            '',   # Default database
            '',   # Default username
            'password',  # Password
            '',   # Default schema
            '',   # Default pool size
            '',   # Default overflow
            'y',  # Configure OpenAI
            'n',  # Don't use env var
            'sk-test-key-123456789012345678901234567890',  # API key
            '',   # Default model
            '',   # Default temperature
            '',   # Default max tokens
            '',   # Default timeout
            '',   # Default retries
            'n',  # Don't configure Anthropic
            'https://test.supabase.co',  # Supabase URL
            'test-anon-key-with-sufficient-length',  # Anon key
            'n',  # No service key
            'n',  # No JWT for dev
            'test-jwt-secret-that-is-at-least-32-characters',  # JWT secret
            '',   # Default algorithm
            '',   # Default expiry
            '',   # Default CORS
            'n',  # No rate limiting for dev
            '',   # Default daily budget
            '',   # Default monthly budget
            '',   # Default alert threshold
            'n',  # No hard limits for dev
            '',   # Default batch size
            '',   # Default workers
            '',   # Default chunk size
            '',   # Default overlap
            '',   # Default quality threshold
            '',   # Default duplicate threshold
            '',   # Default max file size
            'n',  # No Langfuse
            'y',  # Save configuration
            'n'   # No SQL scripts
        ]

        with patch('builtins.input', side_effect=inputs):
            with patch('getpass.getpass', side_effect=['password',
                'sk-test-key-123456789012345678901234567890',
                'test-anon-key-with-sufficient-length',
                'test-jwt-secret-that-is-at-least-32-characters']):
                # Mock validation to pass
                with patch.object(ConfigurationValidator, 'validate_complete_config') as mock_validate:
                    mock_validate.return_value = ValidationResult(
                        valid=True,
                        tested_components={'database': True, 'openai': True}
                    )

                    config = await wizard.start_wizard()

                    assert config is not None
                    assert config.environment == "development"
                    assert wizard.config_path.exists()