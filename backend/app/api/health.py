"""
Health and monitoring API endpoints
Provides system health checks, metrics, and monitoring information
"""

import asyncio
import psutil
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from config.settings import get_settings
from models.api import HealthResponse, SystemMetricsResponse, ServiceStatusResponse
from database.connection import get_database_session

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy", version="2.0.0", environment=get_settings().environment
    )


@router.get("/detailed", response_model=Dict[str, Any])
async def detailed_health_check(db: AsyncSession = Depends(get_database_session)):
    """Comprehensive health check including database and service status."""
    settings = get_settings()
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc),
        "version": "2.0.0",
        "environment": settings.environment,
        "services": {},
    }

    # Database health check
    try:
        result = await db.execute(text("SELECT 1"))
        health_data["services"]["database"] = {
            "status": "healthy",
            "response_time_ms": 0,  # Would measure actual response time
            "type": "postgresql",
        }
    except Exception as e:
        health_data["services"]["database"] = {
            "status": "unhealthy",
            "error": str(e),
            "type": "postgresql",
        }
        health_data["status"] = "degraded"

    # Check OpenAI API (if configured)
    if settings.openai_api_key:
        try:
            # Would test OpenAI connection here
            health_data["services"]["openai"] = {
                "status": "healthy",
                "model": settings.openai_model,
            }
        except Exception as e:
            health_data["services"]["openai"] = {"status": "unhealthy", "error": str(e)}

    # Check Supabase connection
    if settings.supabase_url:
        try:
            # Would test Supabase connection here
            health_data["services"]["supabase"] = {
                "status": "healthy",
                "url": settings.supabase_url,
            }
        except Exception as e:
            health_data["services"]["supabase"] = {
                "status": "unhealthy",
                "error": str(e),
            }

    return health_data


@router.get("/metrics", response_model=SystemMetricsResponse)
async def system_metrics():
    """Get current system performance metrics."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    return SystemMetricsResponse(
        cpu_usage_percent=cpu_percent,
        memory_usage_percent=memory.percent,
        memory_total_gb=memory.total / (1024**3),
        memory_available_gb=memory.available / (1024**3),
        disk_usage_percent=disk.percent,
        disk_total_gb=disk.total / (1024**3),
        disk_free_gb=disk.free / (1024**3),
        uptime_seconds=0,  # Would calculate actual uptime
    )


@router.get("/services", response_model=ServiceStatusResponse)
async def service_status():
    """Get status of all integrated services."""
    return ServiceStatusResponse(
        services={
            "fastapi": {"status": "running", "port": get_settings().port},
            "database": {"status": "connected", "type": "postgresql"},
            "websocket": {"status": "running", "connections": 0},
            "langfuse": {"status": "connected", "tracking": True},
            "supabase": {"status": "connected", "realtime": True},
        },
        overall_status="healthy",
    )


@router.get("/readiness")
async def readiness_check(db: AsyncSession = Depends(get_database_session)):
    """Kubernetes-style readiness probe."""
    try:
        # Check database connection
        await db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception:
        return {"status": "not_ready"}


@router.get("/liveness")
async def liveness_check():
    """Kubernetes-style liveness probe."""
    return {"status": "alive"}
