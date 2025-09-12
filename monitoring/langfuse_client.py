"""
brAIn v2.0 Langfuse Client Configuration
Centralized Langfuse client for LLM observability and cost tracking.
"""

import os
import logging
from typing import Optional, Dict, Any
from functools import lru_cache

from langfuse import Langfuse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LangfuseConfig(BaseModel):
    """Langfuse configuration with validation"""
    
    public_key: str = Field(
        description="Langfuse public API key"
    )
    
    secret_key: str = Field(
        description="Langfuse secret API key"
    )
    
    host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse host URL"
    )
    
    environment: str = Field(
        default="production",
        description="Environment name for trace tagging"
    )
    
    enabled: bool = Field(
        default=True,
        description="Whether Langfuse tracking is enabled"
    )
    
    flush_at: int = Field(
        default=15,
        description="Number of events to batch before flushing"
    )
    
    flush_interval: float = Field(
        default=0.5,
        description="Interval in seconds to flush events"
    )
    
    max_retries: int = Field(
        default=3,
        description="Maximum retries for failed API calls"
    )
    
    timeout: float = Field(
        default=10.0,
        description="Timeout for API calls in seconds"
    )
    
    debug: bool = Field(
        default=False,
        description="Enable debug logging"
    )


class LangfuseClientManager:
    """
    Manages Langfuse client lifecycle and provides centralized access.
    
    Features:
    - Singleton pattern for client instance
    - Configuration validation
    - Connection health checking
    - Graceful error handling
    """
    
    _instance: Optional['LangfuseClientManager'] = None
    _client: Optional[Langfuse] = None
    _config: Optional[LangfuseConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls, config: Optional[LangfuseConfig] = None) -> 'LangfuseClientManager':
        """
        Initialize the Langfuse client manager.
        
        Args:
            config: Optional configuration. If None, loads from environment.
            
        Returns:
            Initialized client manager
        """
        instance = cls()
        
        if config is None:
            config = cls._load_config_from_env()
        
        instance._config = config
        
        if config.enabled:
            try:
                instance._client = Langfuse(
                    public_key=config.public_key,
                    secret_key=config.secret_key,
                    host=config.host,
                    environment=config.environment,
                    flush_at=config.flush_at,
                    flush_interval=config.flush_interval,
                    max_retries=config.max_retries,
                    timeout=config.timeout,
                    debug=config.debug
                )
                logger.info("Langfuse client initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse client: {e}")
                instance._client = None
                
        else:
            logger.info("Langfuse tracking is disabled")
            
        return instance
    
    @classmethod
    def _load_config_from_env(cls) -> LangfuseConfig:
        """Load configuration from environment variables"""
        return LangfuseConfig(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            environment=os.getenv("LANGFUSE_ENVIRONMENT", "production"),
            enabled=os.getenv("LANGFUSE_ENABLED", "true").lower() == "true",
            flush_at=int(os.getenv("LANGFUSE_FLUSH_AT", "15")),
            flush_interval=float(os.getenv("LANGFUSE_FLUSH_INTERVAL", "0.5")),
            max_retries=int(os.getenv("LANGFUSE_MAX_RETRIES", "3")),
            timeout=float(os.getenv("LANGFUSE_TIMEOUT", "10.0")),
            debug=os.getenv("LANGFUSE_DEBUG", "false").lower() == "true"
        )
    
    @property
    def client(self) -> Optional[Langfuse]:
        """Get the Langfuse client instance"""
        return self._client
    
    @property
    def config(self) -> Optional[LangfuseConfig]:
        """Get the current configuration"""
        return self._config
    
    @property
    def is_enabled(self) -> bool:
        """Check if Langfuse tracking is enabled and client is available"""
        return self._client is not None and (self._config.enabled if self._config else False)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Langfuse connection.
        
        Returns:
            Health check result with status and details
        """
        if not self.is_enabled:
            return {
                "status": "disabled",
                "message": "Langfuse tracking is disabled",
                "client_available": False
            }
        
        try:
            # Create a test trace to verify connection
            trace = self._client.trace(
                name="health_check",
                metadata={"test": True}
            )
            trace.update(output="Health check successful")
            
            return {
                "status": "healthy",
                "message": "Langfuse client is working correctly",
                "client_available": True,
                "config": {
                    "host": self._config.host,
                    "environment": self._config.environment
                }
            }
            
        except Exception as e:
            logger.error(f"Langfuse health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": f"Langfuse health check failed: {str(e)}",
                "client_available": True,
                "error": str(e)
            }
    
    def flush(self) -> None:
        """Flush any pending events to Langfuse"""
        if self._client:
            try:
                self._client.flush()
            except Exception as e:
                logger.error(f"Failed to flush Langfuse events: {e}")
    
    def shutdown(self) -> None:
        """Shutdown the Langfuse client and flush remaining events"""
        if self._client:
            try:
                self._client.flush()
                logger.info("Langfuse client shut down successfully")
            except Exception as e:
                logger.error(f"Error during Langfuse client shutdown: {e}")
            finally:
                self._client = None


@lru_cache(maxsize=1)
def get_langfuse_client() -> Optional[Langfuse]:
    """
    Get the singleton Langfuse client instance.
    
    Returns:
        Langfuse client or None if disabled/unavailable
    """
    manager = LangfuseClientManager()
    return manager.client


def get_langfuse_manager() -> LangfuseClientManager:
    """
    Get the singleton Langfuse client manager.
    
    Returns:
        LangfuseClientManager instance
    """
    return LangfuseClientManager()


def is_langfuse_enabled() -> bool:
    """
    Check if Langfuse tracking is enabled and available.
    
    Returns:
        True if Langfuse is enabled and client is available
    """
    manager = LangfuseClientManager()
    return manager.is_enabled


# Environment variable validation
def validate_langfuse_environment() -> Dict[str, Any]:
    """
    Validate Langfuse environment configuration.
    
    Returns:
        Validation result with status and missing variables
    """
    required_vars = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
    optional_vars = ["LANGFUSE_HOST", "LANGFUSE_ENVIRONMENT", "LANGFUSE_ENABLED"]
    
    missing_required = []
    available_optional = {}
    
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            available_optional[var] = value
    
    is_valid = len(missing_required) == 0
    
    return {
        "valid": is_valid,
        "missing_required": missing_required,
        "optional_configured": available_optional,
        "message": "Langfuse environment is properly configured" if is_valid 
                  else f"Missing required environment variables: {', '.join(missing_required)}"
    }


# Initialization utilities
def initialize_langfuse(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: Optional[str] = None,
    environment: Optional[str] = None,
    **kwargs
) -> LangfuseClientManager:
    """
    Initialize Langfuse with custom configuration.
    
    Args:
        public_key: Langfuse public key (overrides env var)
        secret_key: Langfuse secret key (overrides env var) 
        host: Langfuse host URL (overrides env var)
        environment: Environment name (overrides env var)
        **kwargs: Additional configuration options
        
    Returns:
        Initialized LangfuseClientManager
    """
    config_dict = {}
    
    # Override with provided values
    if public_key:
        config_dict["public_key"] = public_key
    if secret_key:
        config_dict["secret_key"] = secret_key
    if host:
        config_dict["host"] = host
    if environment:
        config_dict["environment"] = environment
    
    # Add any additional configuration
    config_dict.update(kwargs)
    
    # Load defaults from environment for missing values
    env_config = LangfuseClientManager._load_config_from_env()
    
    # Create final configuration
    final_config = LangfuseConfig(
        public_key=config_dict.get("public_key", env_config.public_key),
        secret_key=config_dict.get("secret_key", env_config.secret_key),
        host=config_dict.get("host", env_config.host),
        environment=config_dict.get("environment", env_config.environment),
        enabled=config_dict.get("enabled", env_config.enabled),
        flush_at=config_dict.get("flush_at", env_config.flush_at),
        flush_interval=config_dict.get("flush_interval", env_config.flush_interval),
        max_retries=config_dict.get("max_retries", env_config.max_retries),
        timeout=config_dict.get("timeout", env_config.timeout),
        debug=config_dict.get("debug", env_config.debug)
    )
    
    return LangfuseClientManager.initialize(final_config)


# Context manager for automatic flushing
class LangfuseContext:
    """Context manager for automatic Langfuse event flushing"""
    
    def __init__(self, auto_flush: bool = True):
        self.auto_flush = auto_flush
        self.manager = get_langfuse_manager()
    
    def __enter__(self):
        return self.manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.auto_flush and self.manager.is_enabled:
            self.manager.flush()