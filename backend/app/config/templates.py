"""
Configuration templates for common deployment scenarios.

This module provides pre-configured templates for different use cases,
making it easy to get started with the AI Configuration Wizard.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from app.config.models import (
    ConfigurationTemplate,
    DatabaseConfig,
    OpenAIConfig,
    AnthropicConfig,
    SupabaseConfig,
    LangfuseConfig,
    SecurityConfig,
    CostManagementConfig,
    ProcessingConfig,
)


class ConfigurationTemplates:
    """Pre-configured templates for common scenarios."""

    @staticmethod
    def get_development_template() -> ConfigurationTemplate:
        """
        Get development environment template.

        Returns:
            Development configuration template
        """
        return ConfigurationTemplate(
            name="Development Environment",
            description="Local development setup with minimal security and monitoring",
            use_case="Local development and testing",
            config={
                "environment": "development",
                "database": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "brain_dev",
                    "username": "developer",
                    "password": "dev_password_change_me",
                    "schema": "public",
                    "pool_size": 5,
                    "max_overflow": 2,
                },
                "openai": {
                    "api_key": "sk-your-openai-key-here",
                    "model": "gpt-4o-mini",
                    "temperature": 0.7,
                    "max_tokens": 4000,
                    "timeout": 30,
                    "max_retries": 3,
                },
                "supabase": {
                    "url": "https://your-project.supabase.co",
                    "anon_key": "your-anon-key-here",
                    "service_key": None,
                    "jwt_secret": None,
                },
                "langfuse": {
                    "enabled": False,
                    "public_key": None,
                    "secret_key": None,
                    "host": None,
                },
                "security": {
                    "jwt_enabled": False,
                    "jwt_secret": "development-secret-change-in-production",
                    "jwt_algorithm": "HS256",
                    "jwt_expiry_hours": 24,
                    "cors_enabled": True,
                    "cors_origins": ["http://localhost:3000", "http://localhost:5173"],
                    "rate_limit_enabled": False,
                    "rate_limit_requests": 1000,
                    "input_validation_strict": False,
                    "sql_injection_protection": True,
                    "xss_protection": True,
                },
                "cost_management": {
                    "daily_budget": 10.0,
                    "monthly_budget": 100.0,
                    "alert_threshold_percent": 80,
                    "hard_limit_enabled": False,
                    "cost_per_1k_tokens": {
                        "gpt-4o-mini": 0.00015,
                        "gpt-4o": 0.005,
                        "claude-3-5-sonnet-20241022": 0.003,
                        "text-embedding-3-small": 0.00002,
                    },
                },
                "processing": {
                    "batch_size": 10,
                    "parallel_workers": 2,
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "quality_threshold": 0.6,
                    "duplicate_threshold": 0.95,
                    "max_file_size_mb": 50,
                    "supported_formats": ["pdf", "docx", "txt", "md", "html"],
                },
            },
            tags=["development", "local", "testing"],
            recommended_for=["Local development", "Testing", "Prototyping"],
            is_official=True,
        )

    @staticmethod
    def get_production_secure_template() -> ConfigurationTemplate:
        """
        Get production environment template with security focus.

        Returns:
            Production-ready secure configuration template
        """
        return ConfigurationTemplate(
            name="Production Secure",
            description="Production setup with comprehensive security and monitoring",
            use_case="Production deployment with enterprise security",
            config={
                "environment": "production",
                "database": {
                    "host": "${DATABASE_HOST}",
                    "port": 5432,
                    "database": "${DATABASE_NAME}",
                    "username": "${DATABASE_USER}",
                    "password": "${DATABASE_PASSWORD}",
                    "schema": "public",
                    "pool_size": 20,
                    "max_overflow": 10,
                },
                "openai": {
                    "api_key": "${OPENAI_API_KEY}",
                    "model": "gpt-4o",
                    "temperature": 0.7,
                    "max_tokens": 4000,
                    "timeout": 60,
                    "max_retries": 5,
                },
                "anthropic": {
                    "api_key": "${ANTHROPIC_API_KEY}",
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 4000,
                    "temperature": 0.7,
                    "timeout": 60,
                },
                "supabase": {
                    "url": "${SUPABASE_URL}",
                    "anon_key": "${SUPABASE_ANON_KEY}",
                    "service_key": "${SUPABASE_SERVICE_KEY}",
                    "jwt_secret": "${SUPABASE_JWT_SECRET}",
                },
                "langfuse": {
                    "enabled": True,
                    "public_key": "${LANGFUSE_PUBLIC_KEY}",
                    "secret_key": "${LANGFUSE_SECRET_KEY}",
                    "host": None,
                },
                "security": {
                    "jwt_enabled": True,
                    "jwt_secret": "${JWT_SECRET}",
                    "jwt_algorithm": "HS256",
                    "jwt_expiry_hours": 12,
                    "cors_enabled": True,
                    "cors_origins": ["${FRONTEND_URL}"],
                    "rate_limit_enabled": True,
                    "rate_limit_requests": 100,
                    "input_validation_strict": True,
                    "sql_injection_protection": True,
                    "xss_protection": True,
                },
                "cost_management": {
                    "daily_budget": 50.0,
                    "monthly_budget": 1000.0,
                    "alert_threshold_percent": 75,
                    "hard_limit_enabled": True,
                    "cost_per_1k_tokens": {
                        "gpt-4o-mini": 0.00015,
                        "gpt-4o": 0.005,
                        "claude-3-5-sonnet-20241022": 0.003,
                        "text-embedding-3-small": 0.00002,
                    },
                },
                "processing": {
                    "batch_size": 20,
                    "parallel_workers": 8,
                    "chunk_size": 1500,
                    "chunk_overlap": 300,
                    "quality_threshold": 0.8,
                    "duplicate_threshold": 0.95,
                    "max_file_size_mb": 100,
                    "supported_formats": [
                        "pdf",
                        "docx",
                        "txt",
                        "md",
                        "html",
                        "json",
                        "csv",
                    ],
                },
            },
            tags=["production", "secure", "enterprise"],
            recommended_for=[
                "Enterprise deployment",
                "High-security environments",
                "Compliance-focused",
            ],
            is_official=True,
        )

    @staticmethod
    def get_cost_optimized_template() -> ConfigurationTemplate:
        """
        Get cost-optimized configuration template.

        Returns:
            Cost-optimized configuration template
        """
        return ConfigurationTemplate(
            name="Cost Optimized",
            description="Configuration optimized for minimal operational costs",
            use_case="Budget-conscious deployments with cost controls",
            config={
                "environment": "production",
                "database": {
                    "host": "${DATABASE_HOST}",
                    "port": 5432,
                    "database": "${DATABASE_NAME}",
                    "username": "${DATABASE_USER}",
                    "password": "${DATABASE_PASSWORD}",
                    "schema": "public",
                    "pool_size": 10,
                    "max_overflow": 5,
                },
                "openai": {
                    "api_key": "${OPENAI_API_KEY}",
                    "model": "gpt-4o-mini",  # Cheaper model
                    "temperature": 0.5,  # Lower temperature for consistency
                    "max_tokens": 2000,  # Lower token limit
                    "timeout": 30,
                    "max_retries": 2,
                },
                "supabase": {
                    "url": "${SUPABASE_URL}",
                    "anon_key": "${SUPABASE_ANON_KEY}",
                    "service_key": None,  # Skip service key to save on API calls
                    "jwt_secret": None,
                },
                "langfuse": {
                    "enabled": False,  # Disable to save on monitoring costs
                    "public_key": None,
                    "secret_key": None,
                    "host": None,
                },
                "security": {
                    "jwt_enabled": True,
                    "jwt_secret": "${JWT_SECRET}",
                    "jwt_algorithm": "HS256",
                    "jwt_expiry_hours": 24,
                    "cors_enabled": True,
                    "cors_origins": ["${FRONTEND_URL}"],
                    "rate_limit_enabled": True,
                    "rate_limit_requests": 50,  # Stricter rate limiting
                    "input_validation_strict": True,
                    "sql_injection_protection": True,
                    "xss_protection": True,
                },
                "cost_management": {
                    "daily_budget": 5.0,  # Very low daily budget
                    "monthly_budget": 100.0,  # Tight monthly budget
                    "alert_threshold_percent": 50,  # Early alerts
                    "hard_limit_enabled": True,  # Strict enforcement
                    "cost_per_1k_tokens": {
                        "gpt-4o-mini": 0.00015,
                        "gpt-4o": 0.005,
                        "claude-3-5-sonnet-20241022": 0.003,
                        "text-embedding-3-small": 0.00002,
                    },
                },
                "processing": {
                    "batch_size": 5,  # Smaller batches
                    "parallel_workers": 2,  # Fewer workers
                    "chunk_size": 800,  # Smaller chunks
                    "chunk_overlap": 100,  # Less overlap
                    "quality_threshold": 0.7,
                    "duplicate_threshold": 0.98,  # Stricter duplicate detection
                    "max_file_size_mb": 25,  # Lower file size limit
                    "supported_formats": ["pdf", "docx", "txt", "md"],  # Fewer formats
                },
            },
            tags=["cost-optimized", "budget", "economical"],
            recommended_for=[
                "Startups",
                "Personal projects",
                "Budget-limited deployments",
            ],
            is_official=True,
        )

    @staticmethod
    def get_high_performance_template() -> ConfigurationTemplate:
        """
        Get high-performance configuration template.

        Returns:
            High-performance configuration template
        """
        return ConfigurationTemplate(
            name="High Performance",
            description="Configuration optimized for maximum throughput and performance",
            use_case="High-volume processing with performance optimization",
            config={
                "environment": "production",
                "database": {
                    "host": "${DATABASE_HOST}",
                    "port": 5432,
                    "database": "${DATABASE_NAME}",
                    "username": "${DATABASE_USER}",
                    "password": "${DATABASE_PASSWORD}",
                    "schema": "public",
                    "pool_size": 50,  # Large pool
                    "max_overflow": 20,  # High overflow
                },
                "openai": {
                    "api_key": "${OPENAI_API_KEY}",
                    "model": "gpt-4o",  # Best model
                    "temperature": 0.7,
                    "max_tokens": 8000,  # High token limit
                    "timeout": 120,  # Long timeout
                    "max_retries": 5,
                },
                "anthropic": {
                    "api_key": "${ANTHROPIC_API_KEY}",
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 8000,
                    "temperature": 0.7,
                    "timeout": 120,
                },
                "supabase": {
                    "url": "${SUPABASE_URL}",
                    "anon_key": "${SUPABASE_ANON_KEY}",
                    "service_key": "${SUPABASE_SERVICE_KEY}",
                    "jwt_secret": "${SUPABASE_JWT_SECRET}",
                },
                "langfuse": {
                    "enabled": True,
                    "public_key": "${LANGFUSE_PUBLIC_KEY}",
                    "secret_key": "${LANGFUSE_SECRET_KEY}",
                    "host": None,
                },
                "security": {
                    "jwt_enabled": True,
                    "jwt_secret": "${JWT_SECRET}",
                    "jwt_algorithm": "HS256",
                    "jwt_expiry_hours": 24,
                    "cors_enabled": True,
                    "cors_origins": ["${FRONTEND_URL}"],
                    "rate_limit_enabled": True,
                    "rate_limit_requests": 1000,  # High rate limit
                    "input_validation_strict": True,
                    "sql_injection_protection": True,
                    "xss_protection": True,
                },
                "cost_management": {
                    "daily_budget": 500.0,  # High budget
                    "monthly_budget": 10000.0,  # Very high monthly budget
                    "alert_threshold_percent": 90,  # Late alerts
                    "hard_limit_enabled": False,  # No hard limits
                    "cost_per_1k_tokens": {
                        "gpt-4o-mini": 0.00015,
                        "gpt-4o": 0.005,
                        "claude-3-5-sonnet-20241022": 0.003,
                        "text-embedding-3-small": 0.00002,
                    },
                },
                "processing": {
                    "batch_size": 50,  # Large batches
                    "parallel_workers": 16,  # Maximum workers
                    "chunk_size": 2000,  # Large chunks
                    "chunk_overlap": 400,  # More overlap for quality
                    "quality_threshold": 0.9,  # High quality requirement
                    "duplicate_threshold": 0.99,  # Precise duplicate detection
                    "max_file_size_mb": 500,  # Very large files
                    "supported_formats": [
                        "pdf",
                        "docx",
                        "txt",
                        "md",
                        "html",
                        "json",
                        "csv",
                        "xml",
                        "rtf",
                    ],
                },
            },
            tags=["high-performance", "enterprise", "scalable"],
            recommended_for=[
                "Enterprise",
                "High-volume processing",
                "Mission-critical applications",
            ],
            is_official=True,
        )

    @staticmethod
    def get_all_templates() -> List[ConfigurationTemplate]:
        """
        Get all available configuration templates.

        Returns:
            List of all configuration templates
        """
        return [
            ConfigurationTemplates.get_development_template(),
            ConfigurationTemplates.get_production_secure_template(),
            ConfigurationTemplates.get_cost_optimized_template(),
            ConfigurationTemplates.get_high_performance_template(),
        ]

    @staticmethod
    def get_template_by_name(name: str) -> Optional[ConfigurationTemplate]:
        """
        Get a template by name.

        Args:
            name: Template name

        Returns:
            Configuration template if found, None otherwise
        """
        templates = ConfigurationTemplates.get_all_templates()
        for template in templates:
            if template.name.lower() == name.lower():
                return template
        return None

    @staticmethod
    def apply_template(
        template: ConfigurationTemplate, overrides: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Apply a template with optional overrides.

        Args:
            template: Configuration template to apply
            overrides: Optional configuration overrides

        Returns:
            Complete configuration dictionary
        """
        config = template.config.copy()

        if overrides:
            # Deep merge overrides into config
            def deep_merge(base: dict, override: dict):
                for key, value in override.items():
                    if (
                        key in base
                        and isinstance(base[key], dict)
                        and isinstance(value, dict)
                    ):
                        deep_merge(base[key], value)
                    else:
                        base[key] = value

            deep_merge(config, overrides)

        return config
