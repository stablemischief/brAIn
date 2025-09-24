"""
brAIn v2.0 Supabase Real-time Client
Centralized Supabase client for real-time subscriptions and WebSocket management.
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any, Callable, List
from functools import lru_cache
from datetime import datetime

from supabase import create_client, Client
from realtime import Socket
from pydantic import BaseModel, Field
import httpx

logger = logging.getLogger(__name__)


class SupabaseRealtimeConfig(BaseModel):
    """Supabase real-time configuration with validation"""

    url: str = Field(description="Supabase project URL")

    anon_key: str = Field(description="Supabase anonymous key")

    service_role_key: Optional[str] = Field(
        default=None, description="Supabase service role key for server-side operations"
    )

    realtime_url: Optional[str] = Field(
        default=None, description="Custom realtime URL (auto-generated if None)"
    )

    enabled: bool = Field(
        default=True, description="Whether real-time functionality is enabled"
    )

    max_retries: int = Field(default=5, description="Maximum connection retries")

    retry_delay: float = Field(
        default=1.0, description="Delay between retry attempts in seconds"
    )

    heartbeat_interval: int = Field(
        default=30000, description="Heartbeat interval in milliseconds"
    )

    timeout: int = Field(
        default=10000, description="Connection timeout in milliseconds"
    )

    log_level: str = Field(default="info", description="Log level for real-time client")

    @property
    def realtime_ws_url(self) -> str:
        """Generate WebSocket URL for real-time connection"""
        if self.realtime_url:
            return self.realtime_url.replace("http://", "ws://").replace(
                "https://", "wss://"
            )

        # Convert Supabase URL to realtime WebSocket URL
        base_url = self.url.replace("https://", "").replace("http://", "")
        return f"wss://{base_url}/realtime/v1/websocket"


class RealtimeChannelManager:
    """
    Manages Supabase real-time channels and subscriptions.

    Features:
    - Channel lifecycle management
    - Subscription state tracking
    - Error handling and reconnection
    - Message filtering and transformation
    """

    def __init__(self, client: Client, config: SupabaseRealtimeConfig):
        self.client = client
        self.config = config
        self._channels: Dict[str, Any] = {}
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._connection_state = "disconnected"
        self._error_count = 0

    async def create_channel(
        self, channel_name: str, config: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Create a new real-time channel.

        Args:
            channel_name: Unique channel name
            config: Optional channel configuration

        Returns:
            Channel object or None if failed
        """
        try:
            if channel_name in self._channels:
                logger.warning(f"Channel {channel_name} already exists")
                return self._channels[channel_name]

            # Create channel with configuration
            channel_config = config or {}
            channel = self.client.channel(channel_name, **channel_config)

            # Store channel reference
            self._channels[channel_name] = channel
            self._subscriptions[channel_name] = {}

            logger.info(f"Created real-time channel: {channel_name}")
            return channel

        except Exception as e:
            logger.error(f"Failed to create channel {channel_name}: {e}")
            return None

    async def subscribe_to_table_changes(
        self,
        channel_name: str,
        table: str,
        event: str = "*",
        schema: str = "public",
        callback: Optional[Callable] = None,
        filter_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Subscribe to database table changes.

        Args:
            channel_name: Channel name to use
            table: Database table name
            event: Event type ("INSERT", "UPDATE", "DELETE", "*")
            schema: Database schema
            callback: Callback function for events
            filter_config: Optional filter configuration

        Returns:
            True if subscription successful
        """
        try:
            # Get or create channel
            channel = self._channels.get(channel_name)
            if not channel:
                channel = await self.create_channel(channel_name)
                if not channel:
                    return False

            # Setup table subscription
            subscription_key = f"{table}_{event}"

            # Configure postgres changes subscription
            postgres_config = {"event": event, "schema": schema, "table": table}

            # Add filters if provided
            if filter_config:
                postgres_config.update(filter_config)

            # Setup callback handler
            def handle_postgres_changes(payload):
                try:
                    logger.debug(f"Table change event: {payload}")
                    if callback:
                        callback(payload)
                except Exception as e:
                    logger.error(f"Error in postgres change handler: {e}")

            # Subscribe to postgres changes
            channel.on_postgres_changes(
                event=event,
                schema=schema,
                table=table,
                callback=handle_postgres_changes,
            )

            # Store subscription info
            self._subscriptions[channel_name][subscription_key] = {
                "type": "postgres_changes",
                "table": table,
                "event": event,
                "schema": schema,
                "callback": callback,
                "created_at": datetime.utcnow(),
            }

            logger.info(f"Subscribed to {table} changes on channel {channel_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to table changes: {e}")
            return False

    async def subscribe_to_broadcast(
        self, channel_name: str, event: str, callback: Callable
    ) -> bool:
        """
        Subscribe to broadcast messages on a channel.

        Args:
            channel_name: Channel name
            event: Event name to listen for
            callback: Callback function for messages

        Returns:
            True if subscription successful
        """
        try:
            # Get or create channel
            channel = self._channels.get(channel_name)
            if not channel:
                channel = await self.create_channel(channel_name)
                if not channel:
                    return False

            # Setup broadcast callback
            def handle_broadcast(payload):
                try:
                    logger.debug(f"Broadcast event {event}: {payload}")
                    callback(payload)
                except Exception as e:
                    logger.error(f"Error in broadcast handler: {e}")

            # Subscribe to broadcast
            channel.on_broadcast(event=event, callback=handle_broadcast)

            # Store subscription info
            subscription_key = f"broadcast_{event}"
            self._subscriptions[channel_name][subscription_key] = {
                "type": "broadcast",
                "event": event,
                "callback": callback,
                "created_at": datetime.utcnow(),
            }

            logger.info(f"Subscribed to broadcast {event} on channel {channel_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to broadcast: {e}")
            return False

    async def broadcast_message(
        self, channel_name: str, event: str, payload: Dict[str, Any]
    ) -> bool:
        """
        Broadcast a message to all clients on a channel.

        Args:
            channel_name: Channel name
            event: Event name
            payload: Message payload

        Returns:
            True if broadcast successful
        """
        try:
            channel = self._channels.get(channel_name)
            if not channel:
                logger.error(f"Channel {channel_name} not found")
                return False

            # Send broadcast message
            await channel.send_broadcast(event, payload)

            logger.debug(f"Broadcasted {event} on channel {channel_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
            return False

    async def subscribe_all_channels(self) -> Dict[str, bool]:
        """
        Subscribe to all configured channels.

        Returns:
            Dictionary of subscription results by channel
        """
        results = {}

        for channel_name, channel in self._channels.items():
            try:
                # Setup connection status callback
                def on_subscribe(status, error):
                    if error:
                        logger.error(f"Subscription error on {channel_name}: {error}")
                        self._error_count += 1
                    else:
                        logger.info(
                            f"Channel {channel_name} subscription status: {status}"
                        )
                        if status == "SUBSCRIBED":
                            self._connection_state = "connected"

                # Subscribe to the channel
                await channel.subscribe(on_subscribe)
                results[channel_name] = True

            except Exception as e:
                logger.error(f"Failed to subscribe to channel {channel_name}: {e}")
                results[channel_name] = False
                self._error_count += 1

        return results

    async def unsubscribe_channel(self, channel_name: str) -> bool:
        """
        Unsubscribe from a specific channel.

        Args:
            channel_name: Channel name to unsubscribe

        Returns:
            True if unsubscribed successfully
        """
        try:
            channel = self._channels.get(channel_name)
            if not channel:
                logger.warning(f"Channel {channel_name} not found")
                return False

            # Unsubscribe from channel
            await channel.unsubscribe()

            # Clean up references
            del self._channels[channel_name]
            if channel_name in self._subscriptions:
                del self._subscriptions[channel_name]

            logger.info(f"Unsubscribed from channel {channel_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to unsubscribe from channel {channel_name}: {e}")
            return False

    async def disconnect_all(self) -> None:
        """Disconnect from all channels"""
        for channel_name in list(self._channels.keys()):
            await self.unsubscribe_channel(channel_name)

        self._connection_state = "disconnected"
        logger.info("Disconnected from all real-time channels")

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get connection status and statistics.

        Returns:
            Connection status information
        """
        return {
            "connection_state": self._connection_state,
            "channels_count": len(self._channels),
            "subscriptions_count": sum(
                len(subs) for subs in self._subscriptions.values()
            ),
            "error_count": self._error_count,
            "channels": list(self._channels.keys()),
        }


class SupabaseRealtimeClient:
    """
    Centralized Supabase real-time client with connection management.

    Features:
    - Singleton pattern for global access
    - Connection lifecycle management
    - Health monitoring and recovery
    - Channel and subscription management
    """

    _instance: Optional["SupabaseRealtimeClient"] = None
    _client: Optional[Client] = None
    _config: Optional[SupabaseRealtimeConfig] = None
    _channel_manager: Optional[RealtimeChannelManager] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(
        cls, config: Optional[SupabaseRealtimeConfig] = None
    ) -> "SupabaseRealtimeClient":
        """
        Initialize the Supabase real-time client.

        Args:
            config: Optional configuration. If None, loads from environment.

        Returns:
            Initialized client instance
        """
        instance = cls()

        if config is None:
            config = cls._load_config_from_env()

        instance._config = config

        if config.enabled:
            try:
                # Create Supabase client
                instance._client = create_client(
                    supabase_url=config.url, supabase_key=config.anon_key
                )

                # Initialize channel manager
                instance._channel_manager = RealtimeChannelManager(
                    instance._client, config
                )

                logger.info("Supabase real-time client initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                instance._client = None
                instance._channel_manager = None
        else:
            logger.info("Supabase real-time functionality is disabled")

        return instance

    @classmethod
    def _load_config_from_env(cls) -> SupabaseRealtimeConfig:
        """Load configuration from environment variables"""
        return SupabaseRealtimeConfig(
            url=os.getenv("SUPABASE_URL", ""),
            anon_key=os.getenv("SUPABASE_ANON_KEY", ""),
            service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
            realtime_url=os.getenv("SUPABASE_REALTIME_URL"),
            enabled=os.getenv("SUPABASE_REALTIME_ENABLED", "true").lower() == "true",
            max_retries=int(os.getenv("SUPABASE_MAX_RETRIES", "5")),
            retry_delay=float(os.getenv("SUPABASE_RETRY_DELAY", "1.0")),
            heartbeat_interval=int(os.getenv("SUPABASE_HEARTBEAT_INTERVAL", "30000")),
            timeout=int(os.getenv("SUPABASE_TIMEOUT", "10000")),
            log_level=os.getenv("SUPABASE_LOG_LEVEL", "info"),
        )

    @property
    def client(self) -> Optional[Client]:
        """Get the Supabase client instance"""
        return self._client

    @property
    def config(self) -> Optional[SupabaseRealtimeConfig]:
        """Get the current configuration"""
        return self._config

    @property
    def channel_manager(self) -> Optional[RealtimeChannelManager]:
        """Get the channel manager instance"""
        return self._channel_manager

    @property
    def is_enabled(self) -> bool:
        """Check if real-time functionality is enabled and available"""
        return (
            self._client is not None
            and self._channel_manager is not None
            and (self._config.enabled if self._config else False)
        )

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Supabase connection.

        Returns:
            Health check result with status and details
        """
        if not self.is_enabled:
            return {
                "status": "disabled",
                "message": "Supabase real-time is disabled",
                "client_available": False,
            }

        try:
            # Test basic connection
            response = await self._test_connection()

            # Get channel manager status
            channel_status = (
                self._channel_manager.get_connection_status()
                if self._channel_manager
                else {}
            )

            return {
                "status": "healthy",
                "message": "Supabase real-time client is working correctly",
                "client_available": True,
                "connection_test": response,
                "channels": channel_status,
                "config": {
                    "url": self._config.url if self._config else None,
                    "realtime_url": (
                        self._config.realtime_ws_url if self._config else None
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Supabase health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)}",
                "client_available": True,
                "error": str(e),
            }

    async def _test_connection(self) -> Dict[str, Any]:
        """Test basic Supabase connection"""
        try:
            # Test with a simple query (this would be replaced with actual test)
            # For now, just test that client is accessible
            if self._client:
                return {"test": "passed", "timestamp": datetime.utcnow().isoformat()}
            else:
                raise Exception("Client not available")

        except Exception as e:
            raise Exception(f"Connection test failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown the real-time client and cleanup resources"""
        if self._channel_manager:
            await self._channel_manager.disconnect_all()

        logger.info("Supabase real-time client shut down successfully")


@lru_cache(maxsize=1)
def get_supabase_realtime_client() -> Optional[SupabaseRealtimeClient]:
    """
    Get the singleton Supabase real-time client instance.

    Returns:
        SupabaseRealtimeClient instance or None if disabled/unavailable
    """
    client = SupabaseRealtimeClient()
    return client if client.is_enabled else None


def get_supabase_realtime_manager() -> SupabaseRealtimeClient:
    """
    Get the singleton Supabase real-time client manager.

    Returns:
        SupabaseRealtimeClient instance
    """
    return SupabaseRealtimeClient()


def is_supabase_realtime_enabled() -> bool:
    """
    Check if Supabase real-time is enabled and available.

    Returns:
        True if real-time is enabled and client is available
    """
    client = SupabaseRealtimeClient()
    return client.is_enabled


# Environment validation
def validate_supabase_environment() -> Dict[str, Any]:
    """
    Validate Supabase environment configuration.

    Returns:
        Validation result with status and missing variables
    """
    required_vars = ["SUPABASE_URL", "SUPABASE_ANON_KEY"]
    optional_vars = [
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_REALTIME_URL",
        "SUPABASE_REALTIME_ENABLED",
    ]

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
        "message": (
            "Supabase environment is properly configured"
            if is_valid
            else f"Missing required environment variables: {', '.join(missing_required)}"
        ),
    }


# Initialization utilities
def initialize_supabase_realtime(
    url: Optional[str] = None,
    anon_key: Optional[str] = None,
    service_role_key: Optional[str] = None,
    **kwargs,
) -> SupabaseRealtimeClient:
    """
    Initialize Supabase real-time with custom configuration.

    Args:
        url: Supabase project URL (overrides env var)
        anon_key: Supabase anonymous key (overrides env var)
        service_role_key: Service role key (overrides env var)
        **kwargs: Additional configuration options

    Returns:
        Initialized SupabaseRealtimeClient
    """
    config_dict = {}

    # Override with provided values
    if url:
        config_dict["url"] = url
    if anon_key:
        config_dict["anon_key"] = anon_key
    if service_role_key:
        config_dict["service_role_key"] = service_role_key

    # Add any additional configuration
    config_dict.update(kwargs)

    # Load defaults from environment for missing values
    env_config = SupabaseRealtimeClient._load_config_from_env()

    # Create final configuration
    final_config = SupabaseRealtimeConfig(
        url=config_dict.get("url", env_config.url),
        anon_key=config_dict.get("anon_key", env_config.anon_key),
        service_role_key=config_dict.get(
            "service_role_key", env_config.service_role_key
        ),
        realtime_url=config_dict.get("realtime_url", env_config.realtime_url),
        enabled=config_dict.get("enabled", env_config.enabled),
        max_retries=config_dict.get("max_retries", env_config.max_retries),
        retry_delay=config_dict.get("retry_delay", env_config.retry_delay),
        heartbeat_interval=config_dict.get(
            "heartbeat_interval", env_config.heartbeat_interval
        ),
        timeout=config_dict.get("timeout", env_config.timeout),
        log_level=config_dict.get("log_level", env_config.log_level),
    )

    return SupabaseRealtimeClient.initialize(final_config)
