"""
Database connection management for brAIn v2.0
Handles PostgreSQL connections with SQLAlchemy and async support
"""

import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy import text

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# Global database engine
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[sessionmaker] = None


def get_database_engine() -> AsyncEngine:
    """Get or create the database engine."""
    global _engine
    
    if _engine is None:
        settings = get_settings()
        
        if not settings.database_url:
            raise RuntimeError("DATABASE_URL is not configured")
        
        # Convert sync PostgreSQL URL to async if needed
        database_url = settings.database_url
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif not database_url.startswith("postgresql+asyncpg://"):
            raise RuntimeError("Database URL must be a PostgreSQL URL")
        
        # Engine configuration
        engine_kwargs = {
            "echo": settings.debug,  # Log SQL queries in debug mode
            "pool_pre_ping": True,   # Validate connections before use
            "pool_recycle": 3600,    # Recycle connections every hour
        }
        
        # Add SSL configuration if specified
        if settings.database_ssl_mode:
            engine_kwargs["connect_args"] = {
                "sslmode": settings.database_ssl_mode
            }
        
        # For testing, we might use SQLite
        if database_url.startswith("sqlite"):
            engine_kwargs.update({
                "poolclass": StaticPool,
                "connect_args": {"check_same_thread": False}
            })
        
        _engine = create_async_engine(database_url, **engine_kwargs)
        logger.info(f"Database engine created for: {database_url.split('@')[-1]}")  # Log without credentials
    
    return _engine


def get_session_factory() -> sessionmaker:
    """Get or create the session factory."""
    global _session_factory
    
    if _session_factory is None:
        engine = get_database_engine()
        _session_factory = sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False
        )
        logger.info("Database session factory created")
    
    return _session_factory


async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function that provides a database session.
    Use with FastAPI's Depends() for automatic session management.
    """
    session_factory = get_session_factory()
    
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions.
    Use for manual session management outside of FastAPI.
    """
    session_factory = get_session_factory()
    
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def test_database_connection() -> bool:
    """Test database connection and return success status."""
    try:
        async with get_db_session() as session:
            # Simple query to test connection
            result = await session.execute(text("SELECT 1"))
            result.scalar()
            logger.info("Database connection test successful")
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


async def close_database_engine():
    """Close the database engine and cleanup connections."""
    global _engine, _session_factory
    
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database engine closed")


async def ensure_database_extensions():
    """Ensure required PostgreSQL extensions are installed."""
    try:
        async with get_db_session() as session:
            # Check for pgvector extension
            result = await session.execute(
                text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            )
            has_vector = result.scalar()
            
            if not has_vector:
                logger.warning("pgvector extension not found. Vector operations may not work.")
                # In production, you might want to fail here or attempt to install
                # await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            else:
                logger.info("pgvector extension is available")
            
            # Check for other useful extensions
            extensions_to_check = ["uuid-ossp", "pg_trgm"]
            for ext in extensions_to_check:
                result = await session.execute(
                    text(f"SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = '{ext}')")
                )
                has_ext = result.scalar()
                
                if has_ext:
                    logger.info(f"Extension '{ext}' is available")
                else:
                    logger.info(f"Extension '{ext}' is not installed")
            
            await session.commit()
            
    except Exception as e:
        logger.error(f"Failed to check database extensions: {e}")


class DatabaseHealthCheck:
    """Database health check utility."""
    
    @staticmethod
    async def check_connection() -> dict:
        """Check database connection and return health status."""
        try:
            start_time = logger.time()
            
            async with get_db_session() as session:
                # Test basic connectivity
                await session.execute(text("SELECT 1"))
                
                # Check if we can write
                await session.execute(text("CREATE TEMP TABLE IF NOT EXISTS health_check (id INT)"))
                await session.execute(text("INSERT INTO health_check (id) VALUES (1)"))
                
                # Clean up
                await session.execute(text("DROP TABLE health_check"))
                
            response_time = (logger.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "message": "Database connection successful"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "Database connection failed"
            }
    
    @staticmethod
    async def check_extensions() -> dict:
        """Check required database extensions."""
        try:
            extensions = {}
            
            async with get_db_session() as session:
                required_extensions = ["vector", "uuid-ossp", "pg_trgm"]
                
                for ext in required_extensions:
                    result = await session.execute(
                        text(f"SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = '{ext}')")
                    )
                    extensions[ext] = result.scalar()
            
            all_required_present = extensions.get("vector", False)
            
            return {
                "status": "healthy" if all_required_present else "degraded",
                "extensions": extensions,
                "message": "Extension check complete"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "Extension check failed"
            }
    
    @staticmethod
    async def get_database_info() -> dict:
        """Get database version and configuration info."""
        try:
            async with get_db_session() as session:
                # Get PostgreSQL version
                version_result = await session.execute(text("SELECT version()"))
                version = version_result.scalar()
                
                # Get database name and size
                db_result = await session.execute(text("SELECT current_database()"))
                database_name = db_result.scalar()
                
                # Get connection count
                conn_result = await session.execute(
                    text("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()")
                )
                connection_count = conn_result.scalar()
                
                return {
                    "version": version,
                    "database_name": database_name,
                    "active_connections": connection_count,
                    "status": "healthy"
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "Database info retrieval failed"
            }


# Database initialization function
async def initialize_database():
    """Initialize database connection and perform startup checks."""
    try:
        logger.info("Initializing database connection...")
        
        # Test basic connection
        if not await test_database_connection():
            raise RuntimeError("Database connection test failed")
        
        # Check extensions
        await ensure_database_extensions()
        
        # Get database info
        db_info = await DatabaseHealthCheck.get_database_info()
        logger.info(f"Connected to database: {db_info.get('database_name', 'unknown')}")
        
        logger.info("Database initialization complete")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


# Cleanup function for application shutdown
async def cleanup_database():
    """Cleanup database connections on application shutdown."""
    try:
        await close_database_engine()
        logger.info("Database cleanup complete")
    except Exception as e:
        logger.error(f"Database cleanup error: {e}")