"""
Configuration validators for the AI Configuration Wizard.

This module provides comprehensive validation functions for testing
configuration values, connections, and system requirements.
"""

import os
import re
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import httpx
from openai import OpenAI, AuthenticationError as OpenAIAuthError
from supabase import create_client, Client

from app.config.models import (
    DatabaseConfig,
    OpenAIConfig,
    AnthropicConfig,
    SupabaseConfig,
    SecurityConfig,
    SystemConfig,
    ValidationResult,
)


class ConfigurationValidator:
    """Comprehensive configuration validation system."""

    def __init__(self):
        """Initialize the configuration validator."""
        self.validation_result = ValidationResult(valid=True)

    async def validate_complete_config(self, config: SystemConfig) -> ValidationResult:
        """
        Validate complete system configuration.

        Args:
            config: Complete system configuration

        Returns:
            ValidationResult with detailed findings
        """
        self.validation_result = ValidationResult(valid=True)

        # Test database connection
        db_valid = await self.validate_database(config.database)
        self.validation_result.tested_components["database"] = db_valid

        # Test LLM providers
        if config.openai:
            openai_valid = await self.validate_openai(config.openai)
            self.validation_result.tested_components["openai"] = openai_valid

        if config.anthropic:
            anthropic_valid = await self.validate_anthropic(config.anthropic)
            self.validation_result.tested_components["anthropic"] = anthropic_valid

        # Test Supabase
        supabase_valid = await self.validate_supabase(config.supabase)
        self.validation_result.tested_components["supabase"] = supabase_valid

        # Test Langfuse if configured
        if config.langfuse and config.langfuse.enabled:
            langfuse_valid = await self.validate_langfuse(config.langfuse)
            self.validation_result.tested_components["langfuse"] = langfuse_valid

        # Validate security settings
        security_valid = self.validate_security(config.security, config.environment)
        self.validation_result.tested_components["security"] = security_valid

        # Check for any failures
        if self.validation_result.errors:
            self.validation_result.valid = False

        # Add recommendations
        self._add_recommendations(config)

        return self.validation_result

    async def validate_database(self, db_config: DatabaseConfig) -> bool:
        """
        Validate database connection and configuration.

        Args:
            db_config: Database configuration

        Returns:
            True if valid, False otherwise
        """
        try:
            # Test connection
            conn = psycopg2.connect(
                host=db_config.host,
                port=db_config.port,
                database=db_config.database,
                user=db_config.username,
                password=db_config.password.get_secret_value(),
            )

            # Test basic query
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]

            # Check for required extensions
            cursor.execute(
                """
                SELECT extname FROM pg_extension
                WHERE extname IN ('pgvector', 'uuid-ossp');
            """
            )
            extensions = [row[0] for row in cursor.fetchall()]

            if "pgvector" not in extensions:
                self.validation_result.warnings.append(
                    "pgvector extension not installed - required for embeddings"
                )

            if "uuid-ossp" not in extensions:
                self.validation_result.warnings.append(
                    "uuid-ossp extension not installed - recommended for UUID generation"
                )

            # Check connection pool settings
            cursor.execute("SHOW max_connections;")
            max_conn = int(cursor.fetchone()[0])

            if db_config.pool_size + db_config.max_overflow > max_conn * 0.8:
                self.validation_result.warnings.append(
                    f"Pool size ({db_config.pool_size + db_config.max_overflow}) "
                    f"may be too high for max_connections ({max_conn})"
                )

            cursor.close()
            conn.close()

            return True

        except psycopg2.OperationalError as e:
            self.validation_result.errors.append(
                f"Database connection failed: {str(e)}"
            )
            return False
        except Exception as e:
            self.validation_result.errors.append(f"Database validation error: {str(e)}")
            return False

    async def validate_openai(self, openai_config: OpenAIConfig) -> bool:
        """
        Validate OpenAI API configuration.

        Args:
            openai_config: OpenAI configuration

        Returns:
            True if valid, False otherwise
        """
        try:
            client = OpenAI(api_key=openai_config.api_key.get_secret_value())

            # Test with a minimal API call
            response = client.models.list()
            available_models = [model.id for model in response.data]

            # Check if specified model is available
            if openai_config.model not in available_models:
                self.validation_result.warnings.append(
                    f"Model {openai_config.model} not in available models. "
                    f"Available: {', '.join(available_models[:5])}"
                )

            # Test a minimal completion
            test_response = client.chat.completions.create(
                model=openai_config.model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10,
            )

            if test_response.choices:
                return True
            else:
                self.validation_result.errors.append(
                    "OpenAI API test failed - no response"
                )
                return False

        except OpenAIAuthError:
            self.validation_result.errors.append("OpenAI API key authentication failed")
            return False
        except Exception as e:
            self.validation_result.errors.append(f"OpenAI validation error: {str(e)}")
            return False

    async def validate_anthropic(self, anthropic_config: AnthropicConfig) -> bool:
        """
        Validate Anthropic Claude API configuration.

        Args:
            anthropic_config: Anthropic configuration

        Returns:
            True if valid, False otherwise
        """
        try:
            # Test API key with minimal request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": anthropic_config.api_key.get_secret_value(),
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": anthropic_config.model,
                        "messages": [{"role": "user", "content": "Test"}],
                        "max_tokens": 10,
                    },
                    timeout=anthropic_config.timeout,
                )

                if response.status_code == 200:
                    return True
                elif response.status_code == 401:
                    self.validation_result.errors.append(
                        "Anthropic API key authentication failed"
                    )
                    return False
                else:
                    self.validation_result.errors.append(
                        f"Anthropic API test failed with status {response.status_code}"
                    )
                    return False

        except httpx.TimeoutException:
            self.validation_result.errors.append(
                "Anthropic API timeout - check network connection"
            )
            return False
        except Exception as e:
            self.validation_result.errors.append(
                f"Anthropic validation error: {str(e)}"
            )
            return False

    async def validate_supabase(self, supabase_config: SupabaseConfig) -> bool:
        """
        Validate Supabase configuration.

        Args:
            supabase_config: Supabase configuration

        Returns:
            True if valid, False otherwise
        """
        try:
            # Create Supabase client
            supabase: Client = create_client(
                str(supabase_config.url), supabase_config.anon_key.get_secret_value()
            )

            # Test basic connection by checking auth status
            # This is a lightweight test that doesn't require authentication
            response = supabase.auth.get_session()

            # If we have a service key, test it
            if supabase_config.service_key:
                service_client: Client = create_client(
                    str(supabase_config.url),
                    supabase_config.service_key.get_secret_value(),
                )
                # Service role key should allow listing users (admin operation)
                try:
                    # This is just a test - we're not actually using the result
                    service_response = service_client.auth.admin.list_users()
                except Exception:
                    self.validation_result.warnings.append(
                        "Service key validation failed - may not have admin permissions"
                    )

            return True

        except Exception as e:
            self.validation_result.errors.append(f"Supabase validation error: {str(e)}")
            return False

    async def validate_langfuse(self, langfuse_config) -> bool:
        """
        Validate Langfuse monitoring configuration.

        Args:
            langfuse_config: Langfuse configuration

        Returns:
            True if valid, False otherwise
        """
        if not langfuse_config.enabled:
            return True

        try:
            # Test Langfuse connection
            headers = {"Authorization": f"Bearer {langfuse_config.public_key}"}

            url = (
                str(langfuse_config.host)
                if langfuse_config.host
                else "https://cloud.langfuse.com"
            )

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{url}/api/public/health", headers=headers, timeout=10.0
                )

                if response.status_code == 200:
                    return True
                else:
                    self.validation_result.warnings.append(
                        f"Langfuse health check returned status {response.status_code}"
                    )
                    return True  # Non-critical, just a warning

        except Exception as e:
            self.validation_result.warnings.append(
                f"Langfuse validation warning: {str(e)}"
            )
            return True  # Non-critical component

    def validate_security(
        self, security_config: SecurityConfig, environment: str
    ) -> bool:
        """
        Validate security configuration.

        Args:
            security_config: Security configuration
            environment: Deployment environment

        Returns:
            True if valid, False otherwise
        """
        valid = True

        # Check JWT configuration
        if security_config.jwt_enabled:
            secret = security_config.jwt_secret.get_secret_value()
            if len(secret) < 32:
                self.validation_result.errors.append(
                    "JWT secret too short - minimum 32 characters required"
                )
                valid = False

            # Check for default or weak secrets
            weak_patterns = ["secret", "password", "123456", "default"]
            if any(pattern in secret.lower() for pattern in weak_patterns):
                self.validation_result.errors.append(
                    "JWT secret appears to be weak or default"
                )
                valid = False

        # Check CORS in production
        if environment == "production":
            if not security_config.cors_enabled:
                self.validation_result.errors.append(
                    "CORS must be configured in production"
                )
                valid = False

            if "*" in security_config.cors_origins:
                self.validation_result.errors.append(
                    "CORS wildcard (*) not allowed in production"
                )
                valid = False

            if any("localhost" in origin for origin in security_config.cors_origins):
                self.validation_result.warnings.append(
                    "localhost in CORS origins for production environment"
                )

        # Check rate limiting
        if environment == "production" and not security_config.rate_limit_enabled:
            self.validation_result.errors.append(
                "Rate limiting must be enabled in production"
            )
            valid = False

        return valid

    def validate_environment_variables(self) -> Dict[str, bool]:
        """
        Validate required environment variables.

        Returns:
            Dictionary of environment variable validation results
        """
        required_vars = {
            "DATABASE_URL": "Database connection string",
            "SUPABASE_URL": "Supabase project URL",
            "SUPABASE_ANON_KEY": "Supabase anonymous key",
            "JWT_SECRET": "JWT signing secret",
        }

        optional_vars = {
            "OPENAI_API_KEY": "OpenAI API key",
            "ANTHROPIC_API_KEY": "Anthropic API key",
            "LANGFUSE_PUBLIC_KEY": "Langfuse public key",
            "LANGFUSE_SECRET_KEY": "Langfuse secret key",
        }

        results = {}

        # Check required variables
        for var, description in required_vars.items():
            value = os.getenv(var)
            if not value:
                self.validation_result.errors.append(
                    f"Required environment variable {var} ({description}) is not set"
                )
                results[var] = False
            else:
                results[var] = True

        # Check optional variables
        for var, description in optional_vars.items():
            value = os.getenv(var)
            if not value:
                self.validation_result.warnings.append(
                    f"Optional environment variable {var} ({description}) is not set"
                )
                results[var] = False
            else:
                results[var] = True

        return results

    def _add_recommendations(self, config: SystemConfig):
        """
        Add configuration recommendations based on analysis.

        Args:
            config: System configuration
        """
        # Cost management recommendations
        if config.cost_management.daily_budget > 100:
            self.validation_result.recommendations.append(
                "Consider setting a lower daily budget for cost control"
            )

        if not config.cost_management.hard_limit_enabled:
            self.validation_result.recommendations.append(
                "Enable hard budget limits to prevent unexpected costs"
            )

        # Security recommendations
        if config.environment == "production":
            if config.security.jwt_expiry_hours > 24:
                self.validation_result.recommendations.append(
                    "Consider shorter JWT expiry for production (24 hours or less)"
                )

            if config.security.rate_limit_requests > 1000:
                self.validation_result.recommendations.append(
                    "Consider stricter rate limiting for production"
                )

        # Processing recommendations
        if config.processing.parallel_workers > 8:
            self.validation_result.recommendations.append(
                "High parallel worker count may cause resource contention"
            )

        if config.processing.chunk_size > 4000:
            self.validation_result.recommendations.append(
                "Large chunk sizes may exceed model context limits"
            )

        # Monitoring recommendations
        if config.environment == "production" and not config.langfuse:
            self.validation_result.recommendations.append(
                "Enable Langfuse monitoring for production observability"
            )


async def test_configuration(config: SystemConfig) -> ValidationResult:
    """
    Convenience function to test a complete configuration.

    Args:
        config: System configuration to test

    Returns:
        Validation result with detailed findings
    """
    validator = ConfigurationValidator()
    return await validator.validate_complete_config(config)
