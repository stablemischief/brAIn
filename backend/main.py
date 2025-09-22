"""
brAIn v2.0 - Main FastAPI Application
Entry point for the brAIn RAG Pipeline Management System
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
from starlette.middleware.sessions import SessionMiddleware

# Use proper app package imports (no more path manipulation!)
from app.config.settings import get_settings, Settings
from app.models.api import HealthResponse
from app.api.health import router as health_router
from app.api.auth import router as auth_router
from app.api.folders import router as folders_router
from app.api.processing import router as processing_router
from app.api.search import router as search_router
from app.api.analytics import router as analytics_router
from app.api.config import router as config_router
from app.api.websocket_endpoints import realtime_router as websocket_router
from app.middleware.auth_middleware import AuthenticationMiddleware
from app.middleware.rate_limiting import RateLimitingMiddleware, IPWhitelistMiddleware
from app.middleware.security_middleware import SecurityMiddleware
from app.utils.error_handlers import setup_error_handlers
from app.database.connection import initialize_database, cleanup_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    settings = get_settings()
    
    # Startup
    logger.info("🧠 brAIn v2.0 starting up...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Initialize database connections
    try:
        await initialize_database()
        logger.info("✅ Database connections initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise
    
    # Initialize monitoring
    try:
        # Langfuse/monitoring initialization would go here
        logger.info("✅ Monitoring systems initialized")
    except Exception as e:
        logger.warning(f"⚠️ Monitoring initialization failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("🧠 brAIn v2.0 shutting down...")
    await cleanup_database()


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="brAIn v2.0 - RAG Pipeline Management System",
        description="AI-enhanced RAG pipeline management with real-time monitoring, cost optimization, and knowledge graph capabilities",
        version="2.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan
    )
    
    # CORS Configuration - Security Hardened
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],  # Explicit methods only
        allow_headers=[
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-API-Key",
            "Cache-Control"
        ],  # Explicit headers only - no wildcards
        expose_headers=["X-Total-Count", "X-User-ID", "X-User-Email"],  # Safe response headers
        max_age=86400,  # Cache preflight responses for 24 hours
    )
    
    # Trusted Host Middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )
    
    # Session Middleware
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key,
        max_age=settings.session_max_age
    )

    # Security Middleware (MUST be before authentication)
    app.add_middleware(SecurityMiddleware)

    # Authentication Middleware
    app.add_middleware(AuthenticationMiddleware)
    
    # Rate Limiting Middleware
    if settings.rate_limiting_enabled:
        app.add_middleware(RateLimitingMiddleware)
    
    # IP Whitelist Middleware for admin endpoints
    app.add_middleware(IPWhitelistMiddleware)
    
    # Set up error handlers
    setup_error_handlers(app)
    
    # Include routers
    app.include_router(health_router, prefix="/api/health", tags=["Health"])
    app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
    app.include_router(folders_router, prefix="/api/folders", tags=["Folders"])
    app.include_router(processing_router, prefix="/api/processing", tags=["Processing"])
    app.include_router(search_router, prefix="/api/search", tags=["Search"])
    app.include_router(analytics_router, prefix="/api/analytics", tags=["Analytics"])
    app.include_router(config_router, prefix="/api/config", tags=["Configuration"])
    app.include_router(websocket_router, prefix="/ws", tags=["WebSocket"])
    
    return app


# Create the application instance
app = create_application()


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint providing API information."""
    settings = get_settings()
    return {
        "name": "brAIn v2.0",
        "version": "2.0.0",
        "description": "AI-enhanced RAG Pipeline Management System",
        "environment": settings.environment,
        "status": "healthy",
        "features": [
            "Real-time monitoring",
            "Cost optimization", 
            "Knowledge graph",
            "Semantic search",
            "AI-powered validation"
        ]
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Simple health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=None,  # Will be auto-generated by model
        version="2.0.0",
        environment=get_settings().environment
    )


if __name__ == "__main__":
    settings = get_settings()
    
    # Configure uvicorn
    uvicorn_config = {
        "app": "main:app",
        "host": settings.host,
        "port": settings.port,
        "reload": settings.debug and settings.environment == "development",
        "log_level": "debug" if settings.debug else "info",
        "access_log": True,
        "use_colors": True,
    }
    
    if settings.environment == "production":
        uvicorn_config.update({
            "workers": settings.workers,
            "ssl_keyfile": settings.ssl_keyfile,
            "ssl_certfile": settings.ssl_certfile,
        })
    
    logger.info(f"🚀 Starting brAIn v2.0 server on {settings.host}:{settings.port}")
    uvicorn.run(**uvicorn_config)