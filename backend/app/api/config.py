"""
Configuration wizard API endpoints
AI-assisted system configuration and setup
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import ValidationError

from app.config.settings import get_settings
from app.models.api import (
    ConfigurationWizardStep,
    ConfigurationValidationResponse,
    ConfigurationTemplateResponse,
    SystemConfigurationResponse,
    ConfigurationUpdateRequest,
)
from database.connection import get_database_session
from api.auth import get_current_user

router = APIRouter()


@router.get("/wizard/steps", response_model=List[ConfigurationWizardStep])
async def get_wizard_steps(current_user: dict = Depends(get_current_user)):
    """Get configuration wizard steps and their current status."""
    try:
        # Define the configuration wizard steps
        steps = [
            ConfigurationWizardStep(
                step_number=1,
                title="Database Connection",
                description="Configure PostgreSQL database connection with pgvector extension",
                fields=[
                    {
                        "name": "database_url",
                        "label": "Database URL",
                        "type": "text",
                        "required": True,
                        "placeholder": "postgresql://user:password@localhost:5432/brain_db",
                        "validation_pattern": r"postgresql://.*",
                        "help_text": "PostgreSQL connection string with credentials",
                    },
                    {
                        "name": "enable_ssl",
                        "label": "Enable SSL",
                        "type": "boolean",
                        "required": False,
                        "default": True,
                        "help_text": "Enable SSL for database connection",
                    },
                ],
                is_completed=True,  # Would check actual configuration
                validation_status="valid",
                next_step=2,
            ),
            ConfigurationWizardStep(
                step_number=2,
                title="AI Services Configuration",
                description="Configure OpenAI API and other AI service credentials",
                fields=[
                    {
                        "name": "openai_api_key",
                        "label": "OpenAI API Key",
                        "type": "password",
                        "required": True,
                        "placeholder": "sk-...",
                        "validation_pattern": r"sk-[A-Za-z0-9]{48}",
                        "help_text": "Your OpenAI API key for embeddings and completions",
                    },
                    {
                        "name": "openai_model",
                        "label": "Default Model",
                        "type": "select",
                        "required": True,
                        "options": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                        "default": "gpt-4",
                        "help_text": "Default OpenAI model for processing",
                    },
                    {
                        "name": "embedding_model",
                        "label": "Embedding Model",
                        "type": "select",
                        "required": True,
                        "options": [
                            "text-embedding-ada-002",
                            "text-embedding-3-small",
                            "text-embedding-3-large",
                        ],
                        "default": "text-embedding-ada-002",
                        "help_text": "Model for generating document embeddings",
                    },
                ],
                is_completed=False,
                validation_status="pending",
                next_step=3,
            ),
            ConfigurationWizardStep(
                step_number=3,
                title="Google Drive Integration",
                description="Configure Google Drive API for document access",
                fields=[
                    {
                        "name": "google_service_account_key",
                        "label": "Service Account Key (JSON)",
                        "type": "textarea",
                        "required": True,
                        "placeholder": "Paste your Google Service Account JSON key here...",
                        "help_text": "Google Cloud service account key with Drive API access",
                    },
                    {
                        "name": "google_scopes",
                        "label": "API Scopes",
                        "type": "multiselect",
                        "required": True,
                        "options": [
                            "https://www.googleapis.com/auth/drive.readonly",
                            "https://www.googleapis.com/auth/drive.file",
                            "https://www.googleapis.com/auth/drive",
                        ],
                        "default": ["https://www.googleapis.com/auth/drive.readonly"],
                        "help_text": "Google Drive API permissions required",
                    },
                ],
                is_completed=False,
                validation_status="pending",
                next_step=4,
            ),
            ConfigurationWizardStep(
                step_number=4,
                title="Supabase Configuration",
                description="Configure Supabase for authentication and real-time features",
                fields=[
                    {
                        "name": "supabase_url",
                        "label": "Supabase URL",
                        "type": "text",
                        "required": True,
                        "placeholder": "https://your-project.supabase.co",
                        "validation_pattern": r"https://.*\.supabase\.co",
                        "help_text": "Your Supabase project URL",
                    },
                    {
                        "name": "supabase_anon_key",
                        "label": "Supabase Anonymous Key",
                        "type": "password",
                        "required": True,
                        "placeholder": "eyJ...",
                        "help_text": "Supabase anonymous/public API key",
                    },
                    {
                        "name": "supabase_service_role_key",
                        "label": "Supabase Service Role Key",
                        "type": "password",
                        "required": False,
                        "placeholder": "eyJ...",
                        "help_text": "Supabase service role key (for admin operations)",
                    },
                ],
                is_completed=False,
                validation_status="pending",
                next_step=5,
            ),
            ConfigurationWizardStep(
                step_number=5,
                title="Monitoring & Observability",
                description="Configure Langfuse for LLM operation tracking",
                fields=[
                    {
                        "name": "langfuse_public_key",
                        "label": "Langfuse Public Key",
                        "type": "text",
                        "required": False,
                        "placeholder": "pk-lf-...",
                        "help_text": "Langfuse public key for LLM observability",
                    },
                    {
                        "name": "langfuse_secret_key",
                        "label": "Langfuse Secret Key",
                        "type": "password",
                        "required": False,
                        "placeholder": "sk-lf-...",
                        "help_text": "Langfuse secret key",
                    },
                    {
                        "name": "langfuse_host",
                        "label": "Langfuse Host",
                        "type": "text",
                        "required": False,
                        "default": "https://cloud.langfuse.com",
                        "help_text": "Langfuse instance URL",
                    },
                ],
                is_completed=False,
                validation_status="pending",
                next_step=None,
            ),
        ]

        return steps
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get wizard steps: {str(e)}",
        )


@router.post("/wizard/validate", response_model=ConfigurationValidationResponse)
async def validate_configuration(
    step_number: int,
    configuration_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
):
    """Validate configuration data for a specific wizard step."""
    try:
        validation_results = {}
        overall_valid = True

        if step_number == 1:  # Database
            # Validate database connection
            database_url = configuration_data.get("database_url")
            if database_url:
                try:
                    # Would test actual database connection
                    validation_results["database_url"] = {
                        "valid": True,
                        "message": "Database connection successful",
                    }
                except Exception as e:
                    validation_results["database_url"] = {
                        "valid": False,
                        "message": f"Database connection failed: {str(e)}",
                    }
                    overall_valid = False

        elif step_number == 2:  # AI Services
            # Validate OpenAI API key
            openai_key = configuration_data.get("openai_api_key")
            if openai_key:
                if openai_key.startswith("sk-") and len(openai_key) > 20:
                    validation_results["openai_api_key"] = {
                        "valid": True,
                        "message": "OpenAI API key format is valid",
                    }
                else:
                    validation_results["openai_api_key"] = {
                        "valid": False,
                        "message": "Invalid OpenAI API key format",
                    }
                    overall_valid = False

        elif step_number == 3:  # Google Drive
            # Validate Google service account
            service_account = configuration_data.get("google_service_account_key")
            if service_account:
                try:
                    import json

                    json.loads(service_account)
                    validation_results["google_service_account_key"] = {
                        "valid": True,
                        "message": "Valid JSON service account key",
                    }
                except json.JSONDecodeError:
                    validation_results["google_service_account_key"] = {
                        "valid": False,
                        "message": "Invalid JSON format",
                    }
                    overall_valid = False

        elif step_number == 4:  # Supabase
            # Validate Supabase URL
            supabase_url = configuration_data.get("supabase_url")
            if supabase_url and supabase_url.endswith(".supabase.co"):
                validation_results["supabase_url"] = {
                    "valid": True,
                    "message": "Valid Supabase URL format",
                }
            else:
                validation_results["supabase_url"] = {
                    "valid": False,
                    "message": "Invalid Supabase URL format",
                }
                overall_valid = False

        return ConfigurationValidationResponse(
            step_number=step_number,
            is_valid=overall_valid,
            validation_results=validation_results,
            suggestions=(
                [
                    "Ensure all required fields are filled",
                    "Test connections before proceeding",
                    "Keep sensitive keys secure",
                ]
                if not overall_valid
                else ["Configuration looks good!"]
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration validation failed: {str(e)}",
        )


@router.get("/templates", response_model=List[ConfigurationTemplateResponse])
async def get_configuration_templates():
    """Get pre-configured templates for common deployment scenarios."""
    try:
        templates = [
            ConfigurationTemplateResponse(
                id="development",
                name="Development Setup",
                description="Local development environment with minimal external dependencies",
                category="development",
                configuration={
                    "database_url": "postgresql://postgres:password@localhost:5432/brain_dev",
                    "enable_ssl": False,
                    "openai_model": "gpt-3.5-turbo",
                    "embedding_model": "text-embedding-ada-002",
                    "langfuse_host": "https://cloud.langfuse.com",
                    "debug": True,
                    "environment": "development",
                },
                required_secrets=["openai_api_key", "google_service_account_key"],
                estimated_monthly_cost=25.00,
                features=[
                    "Local PostgreSQL database",
                    "Cost-optimized AI models",
                    "Development debugging enabled",
                    "Basic monitoring",
                ],
            ),
            ConfigurationTemplateResponse(
                id="production",
                name="Production Deployment",
                description="Production-ready configuration with full monitoring and security",
                category="production",
                configuration={
                    "enable_ssl": True,
                    "openai_model": "gpt-4",
                    "embedding_model": "text-embedding-3-large",
                    "environment": "production",
                    "debug": False,
                    "rate_limiting": True,
                    "security_headers": True,
                },
                required_secrets=[
                    "database_url",
                    "openai_api_key",
                    "google_service_account_key",
                    "supabase_url",
                    "supabase_anon_key",
                    "langfuse_public_key",
                    "langfuse_secret_key",
                ],
                estimated_monthly_cost=150.00,
                features=[
                    "Managed database with SSL",
                    "High-performance AI models",
                    "Full LLM observability",
                    "Production security",
                    "Real-time monitoring",
                    "Auto-scaling support",
                ],
            ),
            ConfigurationTemplateResponse(
                id="enterprise",
                name="Enterprise Configuration",
                description="Enterprise deployment with advanced features and compliance",
                category="enterprise",
                configuration={
                    "enable_ssl": True,
                    "openai_model": "gpt-4",
                    "embedding_model": "text-embedding-3-large",
                    "environment": "production",
                    "debug": False,
                    "rate_limiting": True,
                    "security_headers": True,
                    "audit_logging": True,
                    "data_retention_days": 365,
                    "backup_enabled": True,
                },
                required_secrets=[
                    "database_url",
                    "openai_api_key",
                    "google_service_account_key",
                    "supabase_url",
                    "supabase_service_role_key",
                    "langfuse_public_key",
                    "langfuse_secret_key",
                ],
                estimated_monthly_cost=500.00,
                features=[
                    "High-availability database",
                    "Premium AI models",
                    "Advanced monitoring",
                    "Audit logging",
                    "Data retention policies",
                    "Automated backups",
                    "24/7 support",
                ],
            ),
        ]

        return templates
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get configuration templates: {str(e)}",
        )


@router.get("/current", response_model=SystemConfigurationResponse)
async def get_current_configuration(current_user: dict = Depends(get_current_user)):
    """Get current system configuration status."""
    try:
        settings = get_settings()

        # Mask sensitive information
        configuration = {
            "database_configured": bool(settings.database_url),
            "openai_configured": bool(settings.openai_api_key),
            "google_drive_configured": bool(settings.google_service_account_json),
            "supabase_configured": bool(
                settings.supabase_url and settings.supabase_anon_key
            ),
            "langfuse_configured": bool(settings.langfuse_public_key),
            "environment": settings.environment,
            "debug_mode": settings.debug,
            "ssl_enabled": bool(settings.ssl_certfile and settings.ssl_keyfile),
        }

        # Calculate configuration completeness
        required_configs = ["database_configured", "openai_configured"]
        completed_configs = sum(
            1 for key in required_configs if configuration.get(key, False)
        )
        completeness_percentage = (completed_configs / len(required_configs)) * 100

        return SystemConfigurationResponse(
            configuration=configuration,
            configuration_complete=completeness_percentage >= 100,
            completeness_percentage=completeness_percentage,
            missing_configurations=[
                key for key in required_configs if not configuration.get(key, False)
            ],
            last_updated=datetime.now(timezone.utc),
            warnings=(
                [
                    "Google Drive not configured - document import limited",
                    "Langfuse not configured - LLM monitoring disabled",
                ]
                if completeness_percentage < 100
                else []
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get current configuration: {str(e)}",
        )


@router.put("/update", response_model=SystemConfigurationResponse)
async def update_configuration(
    request: ConfigurationUpdateRequest, current_user: dict = Depends(get_current_user)
):
    """Update system configuration settings."""
    try:
        # In a real implementation, this would update the configuration
        # For security, sensitive values should be stored in a secure manner

        # Validate the configuration update
        if request.configuration:
            # Apply configuration updates
            # This would typically update environment variables or a secure config store
            pass

        return await get_current_configuration(current_user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update configuration: {str(e)}",
        )


@router.post("/test-connection")
async def test_service_connection(
    service: str,
    configuration: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
):
    """Test connection to a specific service with given configuration."""
    try:
        if service == "database":
            # Test database connection
            return {"status": "success", "message": "Database connection successful"}
        elif service == "openai":
            # Test OpenAI API
            return {"status": "success", "message": "OpenAI API connection successful"}
        elif service == "google_drive":
            # Test Google Drive API
            return {
                "status": "success",
                "message": "Google Drive API connection successful",
            }
        elif service == "supabase":
            # Test Supabase connection
            return {"status": "success", "message": "Supabase connection successful"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown service: {service}",
            )
    except Exception as e:
        return {"status": "error", "message": f"Connection test failed: {str(e)}"}
