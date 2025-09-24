"""
Configuration models and schemas for the AI Configuration Wizard.

This module defines Pydantic models for configuration management, validation,
and persistence with comprehensive error handling and security features.
"""

from typing import Optional, Dict, Any, List, Literal, Union
from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    HttpUrl,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings
from datetime import datetime
import re
from pathlib import Path


class DatabaseConfig(BaseModel):
    """Database configuration with validation."""

    host: str = Field(..., description="Database host address")
    port: int = Field(5432, ge=1, le=65535, description="Database port")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Database username")
    password: SecretStr = Field(..., description="Database password")
    schema: str = Field("public", description="Database schema")
    pool_size: int = Field(10, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(5, ge=0, le=50, description="Max overflow connections")

    @field_validator("database", "username", "schema")
    @classmethod
    def validate_sql_identifier(cls, v: str) -> str:
        """Validate SQL identifiers to prevent injection."""
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v):
            raise ValueError(f"Invalid SQL identifier: {v}")
        return v

    def get_connection_string(self, hide_password: bool = True) -> str:
        """Generate PostgreSQL connection string."""
        password = "***" if hide_password else self.password.get_secret_value()
        return f"postgresql://{self.username}:{password}@{self.host}:{self.port}/{self.database}"


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    api_key: SecretStr = Field(..., description="OpenAI API key")
    model: str = Field("gpt-4o-mini", description="Model to use")
    temperature: float = Field(0.7, ge=0, le=2, description="Temperature setting")
    max_tokens: int = Field(4000, ge=1, le=128000, description="Max tokens per request")
    timeout: int = Field(30, ge=1, le=300, description="Request timeout in seconds")
    max_retries: int = Field(3, ge=0, le=10, description="Max retry attempts")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: SecretStr) -> SecretStr:
        """Validate OpenAI API key format."""
        key = v.get_secret_value()
        if not key.startswith("sk-") or len(key) < 20:
            raise ValueError("Invalid OpenAI API key format")
        return v


class AnthropicConfig(BaseModel):
    """Anthropic Claude API configuration."""

    api_key: SecretStr = Field(..., description="Anthropic API key")
    model: str = Field("claude-3-5-sonnet-20241022", description="Model to use")
    max_tokens: int = Field(4000, ge=1, le=200000, description="Max tokens per request")
    temperature: float = Field(0.7, ge=0, le=1, description="Temperature setting")
    timeout: int = Field(60, ge=1, le=300, description="Request timeout in seconds")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: SecretStr) -> SecretStr:
        """Validate Anthropic API key format."""
        key = v.get_secret_value()
        if not key.startswith("sk-ant-") or len(key) < 30:
            raise ValueError("Invalid Anthropic API key format")
        return v


class SupabaseConfig(BaseModel):
    """Supabase configuration."""

    url: HttpUrl = Field(..., description="Supabase project URL")
    anon_key: SecretStr = Field(..., description="Supabase anon key")
    service_key: Optional[SecretStr] = Field(
        None, description="Supabase service role key"
    )
    jwt_secret: Optional[SecretStr] = Field(
        None, description="JWT secret for validation"
    )

    @field_validator("anon_key", "service_key")
    @classmethod
    def validate_supabase_key(cls, v: Optional[SecretStr]) -> Optional[SecretStr]:
        """Validate Supabase key format."""
        if v is None:
            return v
        key = v.get_secret_value()
        if len(key) < 30:
            raise ValueError("Invalid Supabase key format")
        return v


class LangfuseConfig(BaseModel):
    """Langfuse monitoring configuration."""

    enabled: bool = Field(True, description="Enable Langfuse monitoring")
    public_key: Optional[str] = Field(None, description="Langfuse public key")
    secret_key: Optional[SecretStr] = Field(None, description="Langfuse secret key")
    host: Optional[HttpUrl] = Field(None, description="Self-hosted Langfuse URL")

    @model_validator(mode="after")
    def validate_langfuse_config(self) -> "LangfuseConfig":
        """Validate Langfuse configuration completeness."""
        if self.enabled and not (self.public_key and self.secret_key):
            raise ValueError("Langfuse keys required when enabled")
        return self


class SecurityConfig(BaseModel):
    """Security configuration settings."""

    jwt_enabled: bool = Field(True, description="Enable JWT validation")
    jwt_secret: SecretStr = Field(..., description="JWT signing secret")
    jwt_algorithm: str = Field("HS256", description="JWT algorithm")
    jwt_expiry_hours: int = Field(24, ge=1, le=720, description="JWT expiry in hours")

    cors_enabled: bool = Field(True, description="Enable CORS")
    cors_origins: List[str] = Field(
        ["http://localhost:3000"], description="Allowed CORS origins"
    )

    rate_limit_enabled: bool = Field(True, description="Enable rate limiting")
    rate_limit_requests: int = Field(
        100, ge=1, le=10000, description="Requests per minute"
    )

    input_validation_strict: bool = Field(True, description="Strict input validation")
    sql_injection_protection: bool = Field(True, description="SQL injection protection")
    xss_protection: bool = Field(True, description="XSS protection")

    @field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret(cls, v: SecretStr) -> SecretStr:
        """Ensure JWT secret is strong enough."""
        secret = v.get_secret_value()
        if len(secret) < 32:
            raise ValueError("JWT secret must be at least 32 characters")
        return v

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, v: List[str]) -> List[str]:
        """Validate CORS origins format."""
        for origin in v:
            if not re.match(r"^https?://[a-zA-Z0-9.-]+(?::[0-9]+)?$", origin):
                raise ValueError(f"Invalid CORS origin format: {origin}")
        return v


class CostManagementConfig(BaseModel):
    """Cost management and budget configuration."""

    daily_budget: float = Field(50.0, ge=0, description="Daily spending limit in USD")
    monthly_budget: float = Field(
        1000.0, ge=0, description="Monthly spending limit in USD"
    )

    alert_threshold_percent: int = Field(
        80, ge=0, le=100, description="Alert at % of budget"
    )
    hard_limit_enabled: bool = Field(
        True, description="Stop processing at budget limit"
    )

    cost_per_1k_tokens: Dict[str, float] = Field(
        default_factory=lambda: {
            "gpt-4o-mini": 0.00015,
            "gpt-4o": 0.005,
            "claude-3-5-sonnet-20241022": 0.003,
            "text-embedding-3-small": 0.00002,
        },
        description="Cost per 1000 tokens by model",
    )


class ProcessingConfig(BaseModel):
    """Document processing configuration."""

    batch_size: int = Field(10, ge=1, le=100, description="Batch processing size")
    parallel_workers: int = Field(
        4, ge=1, le=16, description="Parallel processing workers"
    )

    chunk_size: int = Field(1000, ge=100, le=8000, description="Text chunk size")
    chunk_overlap: int = Field(200, ge=0, le=1000, description="Chunk overlap size")

    quality_threshold: float = Field(
        0.7, ge=0, le=1, description="Quality score threshold"
    )
    duplicate_threshold: float = Field(
        0.95, ge=0, le=1, description="Duplicate similarity threshold"
    )

    max_file_size_mb: int = Field(100, ge=1, le=1000, description="Max file size in MB")
    supported_formats: List[str] = Field(
        default_factory=lambda: ["pdf", "docx", "txt", "md", "html", "json", "csv"],
        description="Supported file formats",
    )


class SystemConfig(BaseModel):
    """Complete system configuration."""

    environment: Literal["development", "staging", "production"] = Field(
        "development", description="Deployment environment"
    )

    database: DatabaseConfig
    openai: Optional[OpenAIConfig] = None
    anthropic: Optional[AnthropicConfig] = None
    supabase: SupabaseConfig
    langfuse: Optional[LangfuseConfig] = None
    security: SecurityConfig
    cost_management: CostManagementConfig
    processing: ProcessingConfig

    config_version: str = Field("1.0.0", description="Configuration version")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    @model_validator(mode="after")
    def validate_llm_config(self) -> "SystemConfig":
        """Ensure at least one LLM provider is configured."""
        if not (self.openai or self.anthropic):
            raise ValueError(
                "At least one LLM provider (OpenAI or Anthropic) must be configured"
            )
        return self

    @model_validator(mode="after")
    def validate_production_config(self) -> "SystemConfig":
        """Validate production configuration requirements."""
        if self.environment == "production":
            # Ensure security is properly configured
            if not self.security.jwt_enabled:
                raise ValueError("JWT must be enabled in production")
            if "*" in self.security.cors_origins:
                raise ValueError("CORS wildcard not allowed in production")
            if not self.security.rate_limit_enabled:
                raise ValueError("Rate limiting must be enabled in production")

            # Ensure monitoring is configured
            if not self.langfuse or not self.langfuse.enabled:
                raise ValueError("Langfuse monitoring must be enabled in production")

            # Ensure cost controls are in place
            if not self.cost_management.hard_limit_enabled:
                raise ValueError("Cost hard limits must be enabled in production")

        return self


class ConfigurationTemplate(BaseModel):
    """Configuration template for common scenarios."""

    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    use_case: str = Field(..., description="Primary use case")

    config: Dict[str, Any] = Field(..., description="Template configuration")

    tags: List[str] = Field(default_factory=list, description="Template tags")
    recommended_for: List[str] = Field(
        default_factory=list, description="Recommended scenarios"
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_official: bool = Field(False, description="Official template flag")


class ConfigurationExport(BaseModel):
    """Configuration export format."""

    version: str = Field(..., description="Export version")
    exported_at: datetime = Field(default_factory=datetime.utcnow)

    config: SystemConfig

    metadata: Dict[str, Any] = Field(default_factory=dict)
    checksum: Optional[str] = None

    def calculate_checksum(self) -> str:
        """Calculate configuration checksum for integrity."""
        import hashlib
        import json

        config_json = json.dumps(self.config.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(config_json.encode()).hexdigest()


class ValidationResult(BaseModel):
    """Configuration validation result."""

    valid: bool = Field(..., description="Overall validation status")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")

    tested_components: Dict[str, bool] = Field(
        default_factory=dict, description="Component test results"
    )

    recommendations: List[str] = Field(
        default_factory=list, description="Configuration recommendations"
    )

    timestamp: datetime = Field(default_factory=datetime.utcnow)
