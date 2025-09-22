"""
brAIn v2.0 Configuration Settings
Pydantic-based settings management with environment variable support
"""

import os
import secrets
from functools import lru_cache
from typing import List, Optional, Literal, Any, Dict

from pydantic import Field, field_validator, model_validator, AnyUrl
from pydantic_core import Url
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    All settings can be configured via environment variables.
    Boolean values accept: true/false, 1/0, yes/no, on/off (case insensitive).
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ========================================
    # APPLICATION SETTINGS
    # ========================================
    
    app_name: str = Field(default="brAIn v2.0", description="Application name")
    version: str = Field(default="2.0.0", description="Application version")
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment"
    )
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes for production")
    
    # Security
    secret_key: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret key for session encryption"
    )
    session_max_age: int = Field(default=86400, description="Session max age in seconds")

    # JWT Authentication Settings
    jwt_secret: Optional[str] = Field(default=None, description="JWT signing secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    jwt_expiry_minutes: int = Field(default=1440, description="JWT token expiry in minutes (24 hours)")
    jwt_refresh_expiry_days: int = Field(default=30, description="JWT refresh token expiry in days")

    # Authentication Settings
    auth_enabled: bool = Field(default=True, description="Enable authentication middleware")
    allow_api_key_auth: bool = Field(default=True, description="Allow API key authentication")
    admin_api_key: Optional[str] = Field(default=None, description="Admin API key for system access")
    
    # CORS Settings
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )
    allowed_hosts: List[str] = Field(
        default=["localhost", "127.0.0.1", "0.0.0.0"],
        description="Allowed trusted hosts"
    )
    
    # SSL Configuration
    ssl_keyfile: Optional[str] = Field(default=None, description="SSL key file path")
    ssl_certfile: Optional[str] = Field(default=None, description="SSL certificate file path")
    
    # ========================================
    # DATABASE SETTINGS
    # ========================================
    
    database_url: Optional[AnyUrl] = Field(
        default=None,
        description="PostgreSQL database URL"
    )
    
    # Database connection pool settings
    db_pool_size: int = Field(default=5, description="Database connection pool size")
    db_max_overflow: int = Field(default=10, description="Database max overflow connections")
    db_pool_timeout: int = Field(default=30, description="Database connection timeout")
    db_pool_recycle: int = Field(default=3600, description="Database connection recycle time")
    
    # ========================================
    # SUPABASE SETTINGS
    # ========================================
    
    supabase_url: Optional[str] = Field(default=None, description="Supabase project URL")
    supabase_anon_key: Optional[str] = Field(default=None, description="Supabase anonymous key")
    supabase_service_key: Optional[str] = Field(default=None, description="Supabase service role key")
    supabase_jwt_secret: Optional[str] = Field(default=None, description="Supabase JWT verification secret")
    
    # ========================================
    # OPENAI SETTINGS
    # ========================================
    
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4-turbo-preview", description="Default OpenAI model")
    openai_embedding_model: str = Field(default="text-embedding-3-large", description="OpenAI embedding model")
    openai_max_tokens: int = Field(default=4000, description="Max tokens per OpenAI request")
    openai_temperature: float = Field(default=0.1, description="OpenAI temperature setting")
    
    # ========================================
    # GOOGLE DRIVE SETTINGS
    # ========================================
    
    google_client_id: Optional[str] = Field(default=None, description="Google OAuth client ID")
    google_client_secret: Optional[str] = Field(default=None, description="Google OAuth client secret")
    google_redirect_uri: str = Field(default="http://localhost:8000/auth/callback", description="Google OAuth redirect URI")
    google_scopes: List[str] = Field(
        default=["https://www.googleapis.com/auth/drive.readonly"],
        description="Google Drive API scopes"
    )
    
    # ========================================
    # LANGFUSE SETTINGS
    # ========================================
    
    langfuse_public_key: Optional[str] = Field(default=None, description="Langfuse public key")
    langfuse_secret_key: Optional[str] = Field(default=None, description="Langfuse secret key")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", description="Langfuse host URL")
    langfuse_enabled: bool = Field(default=True, description="Enable Langfuse monitoring")
    
    # ========================================
    # REDIS SETTINGS
    # ========================================
    
    redis_url: str = Field(default="redis://redis:6379", description="Redis connection URL")
    redis_max_connections: int = Field(default=10, description="Redis max connection pool size")
    redis_socket_timeout: int = Field(default=5, description="Redis socket timeout")
    
    # ========================================
    # PROCESSING SETTINGS
    # ========================================
    
    max_file_size_mb: int = Field(default=50, description="Maximum file size in MB")
    batch_size: int = Field(default=10, description="Default batch size for processing")
    max_concurrent_tasks: int = Field(default=5, description="Max concurrent processing tasks")
    
    # Document processing
    supported_file_types: List[str] = Field(
        default=["pdf", "docx", "xlsx", "pptx", "txt", "md", "html", "csv", "json"],
        description="Supported file types for processing"
    )
    
    # Vector embeddings
    embedding_dimensions: int = Field(default=3072, description="Embedding vector dimensions")
    similarity_threshold: float = Field(default=0.8, description="Vector similarity threshold")
    
    # ========================================
    # COST MANAGEMENT SETTINGS
    # ========================================
    
    default_monthly_budget: float = Field(default=100.0, description="Default monthly budget in USD")
    cost_alert_threshold: float = Field(default=0.8, description="Cost alert threshold (80% of budget)")
    token_cost_per_1k: Dict[str, float] = Field(
        default={
            "gpt-4-turbo-preview": 0.01,
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "text-embedding-3-large": 0.00013,
            "text-embedding-3-small": 0.00002,
        },
        description="Cost per 1K tokens by model"
    )
    
    # ========================================
    # MONITORING SETTINGS
    # ========================================
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: Optional[str] = Field(default=None, description="Log file path")
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    
    # ========================================
    # VALIDATORS
    # ========================================
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Ensure environment is valid."""
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v

    @field_validator("openai_temperature")
    @classmethod
    def validate_temperature(cls, v):
        """Ensure temperature is between 0 and 2."""
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v):
        """Ensure similarity threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        return v

    @field_validator("cost_alert_threshold")
    @classmethod
    def validate_cost_alert_threshold(cls, v):
        """Ensure cost alert threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Cost alert threshold must be between 0 and 1")
        return v

    @model_validator(mode='before')
    @classmethod
    def validate_production_requirements(cls, values):
        """Validate production-specific requirements."""
        if isinstance(values, dict):
            environment = values.get("environment")

            if environment == "production":
                required_fields = [
                    "database_url",
                    "supabase_url",
                    "supabase_service_key",
                    "openai_api_key",
                    "secret_key"
                ]

                for field in required_fields:
                    if not values.get(field):
                        raise ValueError(f"{field} is required in production environment")

        return values
    
    # ========================================
    # COMPUTED PROPERTIES
    # ========================================
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    @property
    def database_config(self) -> Dict[str, Any]:
        """Get database configuration dictionary."""
        return {
            "url": str(self.database_url) if self.database_url else None,
            "pool_size": self.db_pool_size,
            "max_overflow": self.db_max_overflow,
            "pool_timeout": self.db_pool_timeout,
            "pool_recycle": self.db_pool_recycle,
        }
    
    @property
    def openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration dictionary."""
        return {
            "api_key": self.openai_api_key,
            "model": self.openai_model,
            "embedding_model": self.openai_embedding_model,
            "max_tokens": self.openai_max_tokens,
            "temperature": self.openai_temperature,
        }
    
    @property
    def langfuse_config(self) -> Dict[str, Any]:
        """Get Langfuse configuration dictionary."""
        return {
            "public_key": self.langfuse_public_key,
            "secret_key": self.langfuse_secret_key,
            "host": self.langfuse_host,
            "enabled": self.langfuse_enabled,
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Export settings for easy importing
settings = get_settings()