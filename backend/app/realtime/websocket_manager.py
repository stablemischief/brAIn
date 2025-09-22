"""
brAIn v2.0 WebSocket Connection Manager
Advanced WebSocket management for real-time dashboard updates with FastAPI integration.
"""

import json
import logging
import asyncio
from typing import Dict, Set, Optional, Any, List, Callable
from datetime import datetime, timezone, timedelta
from uuid import uuid4, UUID
from enum import Enum
from dataclasses import dataclass, asdict

from fastapi import WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConnectionState(str, Enum):
    """WebSocket connection states"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class MessageType(str, Enum):
    """WebSocket message types"""
    # System messages
    PING = "ping"
    PONG = "pong"
    ERROR = "error"
    ACK = "ack"
    
    # Subscription messages
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    SUBSCRIPTION_ACK = "subscription_ack"
    
    # Data messages
    PROCESSING_STATUS = "processing_status"
    COST_UPDATE = "cost_update"
    SYSTEM_HEALTH = "system_health"
    KNOWLEDGE_GRAPH_UPDATE = "knowledge_graph_update"
    USER_ACTIVITY = "user_activity"
    
    # Broadcast messages
    BROADCAST = "broadcast"
    NOTIFICATION = "notification"


@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    id: str
    type: MessageType
    data: Dict[str, Any]
    timestamp: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    channel: Optional[str] = None


@dataclass 
class ConnectionMetrics:
    """Connection performance metrics"""
    messages_sent: int = 0
    messages_received: int = 0
    last_ping: Optional[datetime] = None
    last_pong: Optional[datetime] = None
    average_latency: float = 0.0
    connection_quality: str = "good"  # good, fair, poor


class ConnectionInfo(BaseModel):
    """WebSocket connection information"""
    
    connection_id: str = Field(
        description="Unique connection identifier"
    )
    
    user_id: Optional[UUID] = Field(
        default=None,
        description="User ID associated with connection"
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for grouping connections"
    )
    
    connected_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Connection timestamp"
    )
    
    last_ping: Optional[datetime] = Field(
        default=None,
        description="Last ping received"
    )
    
    subscriptions: Set[str] = Field(
        default_factory=set,
        description="Active subscriptions"
    )
    
    state: ConnectionState = Field(
        default=ConnectionState.CONNECTING,
        description="Current connection state"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional connection metadata"
    )
    
    class Config:
        # Allow set type for subscriptions
        arbitrary_types_allowed = True


class WebSocketConnectionManager:
    """
    Advanced WebSocket connection manager with subscription handling.
    
    Features:
    - Connection lifecycle management
    - Subscription-based message routing
    - Rate limiting and message queuing
    - Health monitoring and recovery
    - Multi-user and session support
    """
    
    def __init__(self):
        # Connection storage
        self.connections: Dict[str, WebSocket] = {}
        self.connection_info: Dict[str, ConnectionInfo] = {}
        self.connection_metrics: Dict[str, ConnectionMetrics] = {}
        self.connection_states: Dict[str, ConnectionState] = {}
        
        # Subscription management
        self.subscriptions: Dict[str, Set[str]] = {}  # channel -> connection_ids
        self.user_connections: Dict[UUID, Set[str]] = {}  # user_id -> connection_ids
        self.session_connections: Dict[str, Set[str]] = {}  # session_id -> connection_ids
        
        # Message handling
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.message_queue: Dict[str, List[WebSocketMessage]] = {}
        
        # Lifecycle management
        self.cleanup_tasks: Dict[str, asyncio.Task] = {}
        self.ping_tasks: Dict[str, asyncio.Task] = {}
        self.reconnect_attempts: Dict[str, int] = {}
        
        # Configuration
        self.ping_interval = 30  # seconds
        self.ping_timeout = 10   # seconds
        self.max_reconnect_attempts = 3
        self.cleanup_delay = 300  # seconds before final cleanup
        
        # Statistics
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "reconnections": 0
        }
        
        # Setup default handlers
        self._setup_default_handlers()
        
        # Health monitoring
        self._health_check_task = None
        self._start_health_monitoring()
    
    def _setup_default_handlers(self):
        """Setup default message handlers"""
        self.message_handlers.update({
            MessageType.PING: self._handle_ping,
            MessageType.SUBSCRIBE: self._handle_subscribe,
            MessageType.UNSUBSCRIBE: self._handle_unsubscribe,
            MessageType.ERROR: self._handle_error
        })
    
    def _start_health_monitoring(self):
        """Start background health monitoring task"""
        if not self._health_check_task:
            self._health_check_task = asyncio.create_task(self._health_monitor_loop())
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._check_connection_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    async def _check_connection_health(self):
        """Check health of all connections"""
        current_time = datetime.utcnow()
        stale_connections = []
        
        for conn_id, info in self.connection_info.items():
            # Check for stale connections (no ping in 60 seconds)
            if info.last_ping:
                time_since_ping = (current_time - info.last_ping).seconds
                if time_since_ping > 60:
                    stale_connections.append(conn_id)
        
        # Clean up stale connections
        for conn_id in stale_connections:
            logger.warning(f"Removing stale connection: {conn_id}")
            await self.disconnect(conn_id, reason="Connection timeout")
    
    async def connect(
        self,
        websocket: WebSocket,
        user_id: Optional[UUID] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Accept a new WebSocket connection with full lifecycle management.
        
        Args:
            websocket: FastAPI WebSocket instance
            user_id: Optional user ID
            session_id: Optional session ID
            metadata: Optional connection metadata
            
        Returns:
            Connection ID
        """
        try:
            # Accept the WebSocket connection
            await websocket.accept()
            
            # Generate unique connection ID
            connection_id = str(uuid4())
            
            # Initialize connection state
            self.connection_states[connection_id] = ConnectionState.CONNECTING
            
            # Store connection
            self.connections[connection_id] = websocket
            
            # Create connection info
            info = ConnectionInfo(
                connection_id=connection_id,
                user_id=user_id,
                session_id=session_id,
                state=ConnectionState.CONNECTED,
                metadata=metadata or {}
            )
            self.connection_info[connection_id] = info
            
            # Initialize connection metrics
            self.connection_metrics[connection_id] = ConnectionMetrics()
            
            # Update connection state
            self.connection_states[connection_id] = ConnectionState.CONNECTED
            
            # Index by user and session
            if user_id:
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = set()
                self.user_connections[user_id].add(connection_id)
            
            if session_id:
                if session_id not in self.session_connections:
                    self.session_connections[session_id] = set()
                self.session_connections[session_id].add(connection_id)
            
            # Start lifecycle management tasks
            self._start_ping_task(connection_id)
            
            # Update statistics
            self.stats["total_connections"] += 1
            self.stats["active_connections"] += 1
            
            # Send connection acknowledgment
            await self.send_message(
                connection_id,
                MessageType.ACK,
                {
                    "connection_id": connection_id,
                    "status": "connected",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"WebSocket connected: {connection_id} (user: {user_id})")
            return connection_id
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            self.stats["errors"] += 1
            raise
    
    async def disconnect(
        self,
        connection_id: str,
        code: int = status.WS_1000_NORMAL_CLOSURE,
        reason: str = "Normal closure"
    ) -> bool:
        """
        Disconnect a WebSocket connection.
        
        Args:
            connection_id: Connection ID to disconnect
            code: WebSocket close code
            reason: Disconnect reason
            
        Returns:
            True if disconnected successfully
        """
        try:
            if connection_id not in self.connections:
                return False
            
            websocket = self.connections[connection_id]
            info = self.connection_info[connection_id]
            
            # Update state
            info.state = ConnectionState.DISCONNECTING
            self.connection_states[connection_id] = ConnectionState.DISCONNECTING
            
            # Clean up lifecycle tasks
            self._cleanup_connection_tasks(connection_id)
            
            # Close WebSocket connection
            try:
                await websocket.close(code=code, reason=reason)
            except:
                pass  # Connection might already be closed
            
            # Clean up subscriptions
            await self._cleanup_connection_subscriptions(connection_id)
            
            # Remove from indexes
            if info.user_id and info.user_id in self.user_connections:
                self.user_connections[info.user_id].discard(connection_id)
                if not self.user_connections[info.user_id]:
                    del self.user_connections[info.user_id]
            
            if info.session_id and info.session_id in self.session_connections:
                self.session_connections[info.session_id].discard(connection_id)
                if not self.session_connections[info.session_id]:
                    del self.session_connections[info.session_id]
            
            # Remove connection
            del self.connections[connection_id]
            del self.connection_info[connection_id]
            
            # Clean up metrics and state
            self.connection_metrics.pop(connection_id, None)
            self.connection_states.pop(connection_id, None)
            
            # Clean up message queue
            if connection_id in self.message_queue:
                del self.message_queue[connection_id]
            
            # Update statistics
            self.stats["active_connections"] -= 1
            
            logger.info(f"WebSocket disconnected: {connection_id} (reason: {reason})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disconnect WebSocket {connection_id}: {e}")
            return False
    
    async def _cleanup_connection_subscriptions(self, connection_id: str):
        """Clean up all subscriptions for a connection"""
        info = self.connection_info.get(connection_id)
        if not info:
            return
        
        for channel in info.subscriptions.copy():
            await self.unsubscribe(connection_id, channel)
    
    async def send_message(
        self,
        connection_id: str,
        message_type: MessageType,
        data: Dict[str, Any],
        channel: Optional[str] = None
    ) -> bool:
        """
        Send a message to a specific connection.
        
        Args:
            connection_id: Target connection ID
            message_type: Type of message
            data: Message data
            channel: Optional channel name
            
        Returns:
            True if sent successfully
        """
        try:
            if connection_id not in self.connections:
                return False
            
            websocket = self.connections[connection_id]
            info = self.connection_info[connection_id]
            
            # Create message
            message = WebSocketMessage(
                id=str(uuid4()),
                type=message_type,
                data=data,
                timestamp=datetime.utcnow().isoformat(),
                user_id=str(info.user_id) if info.user_id else None,
                session_id=info.session_id,
                channel=channel
            )
            
            # Send message
            await websocket.send_text(json.dumps(asdict(message)))
            
            # Update statistics
            self.stats["messages_sent"] += 1
            
            return True
            
        except WebSocketDisconnect:
            # Connection was closed, clean up
            await self.disconnect(connection_id, reason="Connection lost")
            return False
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            self.stats["errors"] += 1
            return False
    
    async def broadcast_to_channel(
        self,
        channel: str,
        message_type: MessageType,
        data: Dict[str, Any],
        exclude_connections: Optional[Set[str]] = None
    ) -> int:
        """
        Broadcast a message to all connections subscribed to a channel.
        
        Args:
            channel: Channel name
            message_type: Type of message
            data: Message data
            exclude_connections: Optional set of connection IDs to exclude
            
        Returns:
            Number of connections message was sent to
        """
        if channel not in self.subscriptions:
            return 0
        
        exclude_set = exclude_connections or set()
        sent_count = 0
        
        for connection_id in self.subscriptions[channel]:
            if connection_id not in exclude_set:
                success = await self.send_message(
                    connection_id, message_type, data, channel
                )
                if success:
                    sent_count += 1
        
        return sent_count
    
    async def broadcast_to_user(
        self,
        user_id: UUID,
        message_type: MessageType,
        data: Dict[str, Any],
        channel: Optional[str] = None
    ) -> int:
        """
        Broadcast a message to all connections for a specific user.
        
        Args:
            user_id: Target user ID
            message_type: Type of message
            data: Message data
            channel: Optional channel name
            
        Returns:
            Number of connections message was sent to
        """
        if user_id not in self.user_connections:
            return 0
        
        sent_count = 0
        for connection_id in self.user_connections[user_id]:
            success = await self.send_message(
                connection_id, message_type, data, channel
            )
            if success:
                sent_count += 1
        
        return sent_count
    
    async def broadcast_to_session(
        self,
        session_id: str,
        message_type: MessageType,
        data: Dict[str, Any],
        channel: Optional[str] = None
    ) -> int:
        """
        Broadcast a message to all connections in a session.
        
        Args:
            session_id: Target session ID
            message_type: Type of message
            data: Message data
            channel: Optional channel name
            
        Returns:
            Number of connections message was sent to
        """
        if session_id not in self.session_connections:
            return 0
        
        sent_count = 0
        for connection_id in self.session_connections[session_id]:
            success = await self.send_message(
                connection_id, message_type, data, channel
            )
            if success:
                sent_count += 1
        
        return sent_count
    
    async def subscribe(self, connection_id: str, channel: str) -> bool:
        """
        Subscribe a connection to a channel.
        
        Args:
            connection_id: Connection ID
            channel: Channel name to subscribe to
            
        Returns:
            True if subscribed successfully
        """
        try:
            if connection_id not in self.connections:
                return False
            
            info = self.connection_info[connection_id]
            
            # Add to channel subscriptions
            if channel not in self.subscriptions:
                self.subscriptions[channel] = set()
            self.subscriptions[channel].add(connection_id)
            
            # Add to connection subscriptions
            info.subscriptions.add(channel)
            
            # Send subscription acknowledgment
            await self.send_message(
                connection_id,
                MessageType.SUBSCRIPTION_ACK,
                {
                    "channel": channel,
                    "status": "subscribed",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.debug(f"Connection {connection_id} subscribed to {channel}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe {connection_id} to {channel}: {e}")
            return False
    
    async def unsubscribe(self, connection_id: str, channel: str) -> bool:
        """
        Unsubscribe a connection from a channel.
        
        Args:
            connection_id: Connection ID
            channel: Channel name to unsubscribe from
            
        Returns:
            True if unsubscribed successfully
        """
        try:
            if connection_id not in self.connections:
                return False
            
            info = self.connection_info[connection_id]
            
            # Remove from channel subscriptions
            if channel in self.subscriptions:
                self.subscriptions[channel].discard(connection_id)
                if not self.subscriptions[channel]:
                    del self.subscriptions[channel]
            
            # Remove from connection subscriptions
            info.subscriptions.discard(channel)
            
            logger.debug(f"Connection {connection_id} unsubscribed from {channel}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe {connection_id} from {channel}: {e}")
            return False
    
    async def handle_message(
        self,
        connection_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """
        Handle incoming WebSocket message.
        
        Args:
            connection_id: Source connection ID
            message: Message data
            
        Returns:
            True if handled successfully
        """
        try:
            message_type = MessageType(message.get("type"))
            data = message.get("data", {})
            
            # Update statistics
            self.stats["messages_received"] += 1
            
            # Update last activity
            if connection_id in self.connection_info:
                self.connection_info[connection_id].last_ping = datetime.utcnow()
            
            # Call appropriate handler
            handler = self.message_handlers.get(message_type)
            if handler:
                return await handler(connection_id, data)
            else:
                logger.warning(f"No handler for message type: {message_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to handle message from {connection_id}: {e}")
            await self.send_message(
                connection_id,
                MessageType.ERROR,
                {"error": str(e), "original_message": message}
            )
            return False
    
    async def _handle_ping(self, connection_id: str, data: Dict[str, Any]) -> bool:
        """Handle ping message"""
        await self.send_message(connection_id, MessageType.PONG, {"timestamp": datetime.utcnow().isoformat()})
        return True
    
    async def _handle_subscribe(self, connection_id: str, data: Dict[str, Any]) -> bool:
        """Handle subscription request"""
        channel = data.get("channel")
        if not channel:
            return False
        
        return await self.subscribe(connection_id, channel)
    
    async def _handle_unsubscribe(self, connection_id: str, data: Dict[str, Any]) -> bool:
        """Handle unsubscription request"""
        channel = data.get("channel")
        if not channel:
            return False
        
        return await self.unsubscribe(connection_id, channel)
    
    async def _handle_error(self, connection_id: str, data: Dict[str, Any]) -> bool:
        """Handle error message"""
        logger.error(f"Client error from {connection_id}: {data}")
        return True
    
    def register_handler(self, message_type: MessageType, handler: Callable) -> None:
        """
        Register a custom message handler.
        
        Args:
            message_type: Message type to handle
            handler: Handler function (async)
        """
        self.message_handlers[message_type] = handler
    
    def get_connection_info(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get connection information"""
        return self.connection_info.get(connection_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics.
        
        Returns:
            Dictionary with connection statistics
        """
        return {
            **self.stats,
            "channels": len(self.subscriptions),
            "subscriptions_total": sum(len(subs) for subs in self.subscriptions.values()),
            "users_connected": len(self.user_connections),
            "sessions_active": len(self.session_connections)
        }
    
    def get_channel_info(self, channel: str) -> Dict[str, Any]:
        """Get information about a specific channel"""
        if channel not in self.subscriptions:
            return {"exists": False}
        
        connection_ids = self.subscriptions[channel]
        
        return {
            "exists": True,
            "subscriber_count": len(connection_ids),
            "connections": list(connection_ids)
        }
    
    def _start_ping_task(self, connection_id: str) -> None:
        """Start ping task for connection lifecycle management."""
        task = asyncio.create_task(self._ping_connection(connection_id))
        self.ping_tasks[connection_id] = task
    
    async def _ping_connection(self, connection_id: str) -> None:
        """Periodically ping connection to maintain health."""
        try:
            while connection_id in self.connections:
                await asyncio.sleep(self.ping_interval)
                
                if connection_id not in self.connections:
                    break
                
                # Send ping
                try:
                    ping_time = datetime.now(timezone.utc)
                    await self.send_message(
                        connection_id,
                        MessageType.PING,
                        {"timestamp": ping_time.isoformat()}
                    )
                    
                    # Update metrics
                    if connection_id in self.connection_metrics:
                        self.connection_metrics[connection_id].last_ping = ping_time
                    
                except Exception as e:
                    logger.warning(f"Failed to ping connection {connection_id}: {e}")
                    # Connection may be dead, trigger cleanup
                    await self._handle_connection_failure(connection_id)
                    break
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Ping task error for {connection_id}: {e}")
    
    async def _handle_connection_failure(self, connection_id: str) -> None:
        """Handle connection failure with reconnection support."""
        try:
            if connection_id not in self.connection_states:
                return
            
            # Update state
            self.connection_states[connection_id] = ConnectionState.ERROR
            
            # Check reconnection attempts
            attempts = self.reconnect_attempts.get(connection_id, 0)
            
            if attempts < self.max_reconnect_attempts:
                # Schedule reconnection cleanup
                self.reconnect_attempts[connection_id] = attempts + 1
                task = asyncio.create_task(self._schedule_cleanup(connection_id))
                self.cleanup_tasks[connection_id] = task
                
                logger.info(f"Connection {connection_id} failed, attempt {attempts + 1}/{self.max_reconnect_attempts}")
            else:
                # Max attempts reached, force disconnect
                logger.warning(f"Connection {connection_id} exceeded max reconnection attempts")
                await self.disconnect(connection_id, reason="Max reconnection attempts exceeded")
        
        except Exception as e:
            logger.error(f"Error handling connection failure for {connection_id}: {e}")
    
    async def _schedule_cleanup(self, connection_id: str) -> None:
        """Schedule delayed cleanup for failed connections."""
        try:
            await asyncio.sleep(self.cleanup_delay)
            
            # If connection still exists and in error state, clean up
            if (connection_id in self.connection_states and 
                self.connection_states[connection_id] == ConnectionState.ERROR):
                
                logger.info(f"Performing delayed cleanup for {connection_id}")
                await self.disconnect(connection_id, reason="Cleanup timeout")
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Cleanup task error for {connection_id}: {e}")
    
    def _cleanup_connection_tasks(self, connection_id: str) -> None:
        """Clean up tasks associated with a connection."""
        # Cancel ping task
        if connection_id in self.ping_tasks:
            task = self.ping_tasks.pop(connection_id)
            if not task.done():
                task.cancel()
        
        # Cancel cleanup task
        if connection_id in self.cleanup_tasks:
            task = self.cleanup_tasks.pop(connection_id)
            if not task.done():
                task.cancel()
        
        # Clear reconnection attempts
        self.reconnect_attempts.pop(connection_id, None)
    
    async def _handle_pong_message(self, connection_id: str, data: Dict[str, Any]) -> None:
        """Handle pong message to update connection health."""
        try:
            if connection_id in self.connection_metrics:
                pong_time = datetime.now(timezone.utc)
                self.connection_metrics[connection_id].last_pong = pong_time
                
                # Calculate latency if ping timestamp available
                if "timestamp" in data:
                    ping_timestamp = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
                    latency = (pong_time - ping_timestamp).total_seconds() * 1000  # ms
                    
                    metrics = self.connection_metrics[connection_id]
                    if metrics.average_latency == 0:
                        metrics.average_latency = latency
                    else:
                        metrics.average_latency = (metrics.average_latency + latency) / 2
                    
                    # Update connection quality based on latency
                    if latency < 100:
                        metrics.connection_quality = "good"
                    elif latency < 300:
                        metrics.connection_quality = "fair"
                    else:
                        metrics.connection_quality = "poor"
        
        except Exception as e:
            logger.error(f"Error handling pong from {connection_id}: {e}")
    
    def get_connection_health(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get health information for a specific connection."""
        if connection_id not in self.connection_metrics:
            return None
        
        metrics = self.connection_metrics[connection_id]
        state = self.connection_states.get(connection_id, ConnectionState.DISCONNECTED)
        
        return {
            "connection_id": connection_id,
            "state": state,
            "metrics": {
                "messages_sent": metrics.messages_sent,
                "messages_received": metrics.messages_received,
                "last_ping": metrics.last_ping.isoformat() if metrics.last_ping else None,
                "last_pong": metrics.last_pong.isoformat() if metrics.last_pong else None,
                "average_latency": metrics.average_latency,
                "connection_quality": metrics.connection_quality
            },
            "reconnect_attempts": self.reconnect_attempts.get(connection_id, 0)
        }
    
    async def shutdown(self) -> None:
        """Shutdown the WebSocket manager and disconnect all clients"""
        # Cancel health monitoring
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all lifecycle tasks
        for task in self.ping_tasks.values():
            if not task.done():
                task.cancel()
        
        for task in self.cleanup_tasks.values():
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        all_tasks = list(self.ping_tasks.values()) + list(self.cleanup_tasks.values())
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Disconnect all connections
        connection_ids = list(self.connections.keys())
        for connection_id in connection_ids:
            await self.disconnect(connection_id, reason="Server shutdown")
        
        logger.info("WebSocket manager shut down successfully")


# Global WebSocket manager instance
_websocket_manager = WebSocketConnectionManager()


def get_websocket_manager() -> WebSocketConnectionManager:
    """Get the global WebSocket manager instance"""
    return _websocket_manager