"""
Message broadcasting system for real-time communication.
Handles message routing, filtering, and delivery across WebSocket connections.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any, Union, Callable
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass, asdict

from pydantic import BaseModel, Field
import redis.asyncio as redis

from .websocket_manager import WebSocketConnectionManager, ConnectionInfo
from .supabase_client import SupabaseRealtimeClient


logger = logging.getLogger(__name__)


class MessagePriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MessageScope(str, Enum):
    BROADCAST = "broadcast"  # All connected users
    USER = "user"           # Specific user
    SESSION = "session"     # Specific session
    ROLE = "role"          # Users with specific role
    CHANNEL = "channel"    # Specific channel subscribers


class BroadcastMessage(BaseModel):
    """Standard message format for broadcasting."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str
    channel: str
    payload: Dict[str, Any]
    scope: MessageScope = MessageScope.BROADCAST
    target_id: Optional[str] = None  # user_id, session_id, role, or channel
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: Optional[int] = None  # Time to live
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class DeliveryStats:
    """Statistics for message delivery."""
    total_messages: int = 0
    delivered_messages: int = 0
    failed_messages: int = 0
    queued_messages: int = 0
    average_delivery_time: float = 0.0


class MessageFilter:
    """Base class for message filtering."""
    
    def should_deliver(self, message: BroadcastMessage, connection_info: ConnectionInfo) -> bool:
        """Determine if message should be delivered to connection."""
        return True


class UserScopeFilter(MessageFilter):
    """Filter messages by user scope."""
    
    def should_deliver(self, message: BroadcastMessage, connection_info: ConnectionInfo) -> bool:
        if message.scope == MessageScope.USER:
            return str(connection_info.user_id) == message.target_id if connection_info.user_id else False
        elif message.scope == MessageScope.SESSION:
            return connection_info.session_id == message.target_id
        elif message.scope == MessageScope.ROLE:
            return message.target_id in (connection_info.metadata.get("roles", []) if connection_info.metadata else [])
        elif message.scope == MessageScope.CHANNEL:
            return message.target_id in connection_info.subscriptions
        return message.scope == MessageScope.BROADCAST


class PriorityQueue:
    """Priority-based message queue."""
    
    def __init__(self):
        self._queues = {priority: asyncio.Queue() for priority in MessagePriority}
        self._lock = asyncio.Lock()
    
    async def put(self, message: BroadcastMessage) -> None:
        """Add message to appropriate priority queue."""
        await self._queues[message.priority].put(message)
    
    async def get(self) -> BroadcastMessage:
        """Get next message by priority."""
        # Check queues in priority order
        priorities = [MessagePriority.URGENT, MessagePriority.HIGH, MessagePriority.NORMAL, MessagePriority.LOW]
        
        while True:
            for priority in priorities:
                try:
                    message = self._queues[priority].get_nowait()
                    return message
                except asyncio.QueueEmpty:
                    continue
            
            # If all queues empty, wait for any message
            tasks = [asyncio.create_task(queue.get()) for queue in self._queues.values()]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            # Return the first completed task result
            for task in done:
                return await task
    
    def qsize(self) -> Dict[MessagePriority, int]:
        """Get queue sizes by priority."""
        return {priority: queue.qsize() for priority, queue in self._queues.items()}


class MessageBroadcaster:
    """Advanced message broadcasting system with Redis persistence and filtering."""
    
    def __init__(
        self,
        websocket_manager: WebSocketConnectionManager,
        supabase_client: SupabaseRealtimeClient,
        redis_url: Optional[str] = None,
        max_queue_size: int = 10000,
        delivery_timeout: float = 30.0
    ):
        self.websocket_manager = websocket_manager
        self.supabase_client = supabase_client
        self.redis_client: Optional[redis.Redis] = None
        self.redis_url = redis_url
        
        self.message_queue = PriorityQueue()
        self.max_queue_size = max_queue_size
        self.delivery_timeout = delivery_timeout
        
        self.filters: List[MessageFilter] = [UserScopeFilter()]
        self.stats = DeliveryStats()
        
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._redis_subscriber_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> bool:
        """Initialize broadcaster with Redis connection."""
        try:
            if self.redis_url:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                await self.redis_client.ping()
                logger.info("Connected to Redis for message persistence")
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize message broadcaster: {e}")
            return False
    
    async def start(self) -> None:
        """Start the message broadcasting worker."""
        if self._running:
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._message_worker())
        
        if self.redis_client:
            self._redis_subscriber_task = asyncio.create_task(self._redis_subscriber())
        
        logger.info("Message broadcaster started")
    
    async def stop(self) -> None:
        """Stop the message broadcasting worker."""
        self._running = False
        
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        if self._redis_subscriber_task:
            self._redis_subscriber_task.cancel()
            try:
                await self._redis_subscriber_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Message broadcaster stopped")
    
    async def broadcast_message(
        self,
        message_type: str,
        channel: str,
        payload: Dict[str, Any],
        scope: MessageScope = MessageScope.BROADCAST,
        target_id: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl_seconds: Optional[int] = None
    ) -> str:
        """Broadcast a message to connected clients."""
        message = BroadcastMessage(
            type=message_type,
            channel=channel,
            payload=payload,
            scope=scope,
            target_id=target_id,
            priority=priority,
            ttl_seconds=ttl_seconds
        )
        
        await self._queue_message(message)
        return message.id
    
    async def broadcast_to_user(
        self,
        user_id: UUID,
        message_type: str,
        channel: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> str:
        """Broadcast message to specific user."""
        return await self.broadcast_message(
            message_type=message_type,
            channel=channel,
            payload=payload,
            scope=MessageScope.USER,
            target_id=str(user_id),
            priority=priority
        )
    
    async def broadcast_to_session(
        self,
        session_id: str,
        message_type: str,
        channel: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> str:
        """Broadcast message to specific session."""
        return await self.broadcast_message(
            message_type=message_type,
            channel=channel,
            payload=payload,
            scope=MessageScope.SESSION,
            target_id=session_id,
            priority=priority
        )
    
    async def broadcast_to_channel_subscribers(
        self,
        channel: str,
        message_type: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> str:
        """Broadcast message to channel subscribers."""
        return await self.broadcast_message(
            message_type=message_type,
            channel=channel,
            payload=payload,
            scope=MessageScope.CHANNEL,
            target_id=channel,
            priority=priority
        )
    
    async def _queue_message(self, message: BroadcastMessage) -> None:
        """Add message to processing queue."""
        # Check TTL
        if message.ttl_seconds:
            age = (datetime.now(timezone.utc) - message.timestamp).total_seconds()
            if age > message.ttl_seconds:
                logger.debug(f"Message {message.id} expired, not queuing")
                return
        
        # Add to local queue
        await self.message_queue.put(message)
        
        # Persist to Redis if available
        if self.redis_client:
            try:
                await self.redis_client.lpush(
                    "broadcast_messages",
                    json.dumps(message.dict(), default=str)
                )
                # Set TTL on Redis list
                if message.ttl_seconds:
                    await self.redis_client.expire("broadcast_messages", message.ttl_seconds)
            except Exception as e:
                logger.error(f"Failed to persist message to Redis: {e}")
        
        self.stats.queued_messages += 1
    
    async def _message_worker(self) -> None:
        """Background worker for processing messages."""
        logger.info("Message worker started")
        
        while self._running:
            try:
                # Get next message
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                
                # Process message
                await self._process_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in message worker: {e}")
                await asyncio.sleep(0.1)
        
        logger.info("Message worker stopped")
    
    async def _process_message(self, message: BroadcastMessage) -> None:
        """Process and deliver a message."""
        start_time = datetime.now()
        delivered_count = 0
        failed_count = 0
        
        try:
            # Check TTL again
            if message.ttl_seconds:
                age = (datetime.now(timezone.utc) - message.timestamp).total_seconds()
                if age > message.ttl_seconds:
                    logger.debug(f"Message {message.id} expired during processing")
                    return
            
            # Get all connections
            connections = self.websocket_manager.get_all_connections()
            
            # Filter connections based on message scope
            target_connections = []
            for conn_id, conn_info in connections.items():
                if self._should_deliver_message(message, conn_info):
                    target_connections.append((conn_id, conn_info))
            
            # Prepare message data
            message_data = {
                "id": message.id,
                "type": message.type,
                "channel": message.channel,
                "payload": message.payload,
                "timestamp": message.timestamp.isoformat(),
                "priority": message.priority
            }
            
            # Deliver to target connections
            if target_connections:
                tasks = []
                for conn_id, conn_info in target_connections:
                    task = asyncio.create_task(
                        self._deliver_to_connection(conn_id, message_data)
                    )
                    tasks.append(task)
                
                # Wait for all deliveries with timeout
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=self.delivery_timeout
                    )
                    
                    for result in results:
                        if isinstance(result, Exception):
                            failed_count += 1
                        else:
                            delivered_count += 1
                
                except asyncio.TimeoutError:
                    logger.warning(f"Message {message.id} delivery timed out")
                    failed_count = len(tasks)
            
            # Update statistics
            self.stats.total_messages += 1
            self.stats.delivered_messages += delivered_count
            self.stats.failed_messages += failed_count
            
            delivery_time = (datetime.now() - start_time).total_seconds()
            self.stats.average_delivery_time = (
                (self.stats.average_delivery_time * (self.stats.total_messages - 1) + delivery_time) /
                self.stats.total_messages
            )
            
            # Log delivery results
            if delivered_count > 0 or failed_count > 0:
                logger.info(
                    f"Message {message.id} delivered to {delivered_count} connections, "
                    f"{failed_count} failures in {delivery_time:.3f}s"
                )
        
        except Exception as e:
            logger.error(f"Failed to process message {message.id}: {e}")
            self.stats.failed_messages += 1
            
            # Retry logic
            if message.retry_count < message.max_retries:
                message.retry_count += 1
                await asyncio.sleep(min(2 ** message.retry_count, 30))  # Exponential backoff
                await self._queue_message(message)
    
    async def _deliver_to_connection(self, connection_id: str, message_data: Dict[str, Any]) -> None:
        """Deliver message to specific connection."""
        try:
            await self.websocket_manager.send_to_connection(
                connection_id, 
                json.dumps(message_data)
            )
        except Exception as e:
            logger.warning(f"Failed to deliver message to connection {connection_id}: {e}")
            raise
    
    def _should_deliver_message(self, message: BroadcastMessage, connection_info: ConnectionInfo) -> bool:
        """Check if message should be delivered to connection."""
        for filter_obj in self.filters:
            if not filter_obj.should_deliver(message, connection_info):
                return False
        return True
    
    async def _redis_subscriber(self) -> None:
        """Subscribe to Redis for distributed message broadcasting."""
        if not self.redis_client:
            return
        
        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe("broadcast_messages")
            
            logger.info("Redis subscriber started")
            
            async for message in pubsub.listen():
                if not self._running:
                    break
                
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        broadcast_message = BroadcastMessage(**data)
                        await self.message_queue.put(broadcast_message)
                    except Exception as e:
                        logger.error(f"Failed to process Redis message: {e}")
        
        except Exception as e:
            logger.error(f"Redis subscriber error: {e}")
        finally:
            if self.redis_client:
                await pubsub.unsubscribe("broadcast_messages")
    
    def add_filter(self, filter_obj: MessageFilter) -> None:
        """Add message filter."""
        self.filters.append(filter_obj)
    
    def remove_filter(self, filter_obj: MessageFilter) -> None:
        """Remove message filter."""
        if filter_obj in self.filters:
            self.filters.remove(filter_obj)
    
    def get_stats(self) -> DeliveryStats:
        """Get current delivery statistics."""
        return self.stats
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        queue_sizes = self.message_queue.qsize()
        return {
            "queue_sizes": queue_sizes,
            "total_queued": sum(queue_sizes.values()),
            "max_queue_size": self.max_queue_size,
            "stats": asdict(self.stats)
        }


# Global broadcaster instance
_broadcaster: Optional[MessageBroadcaster] = None


def get_message_broadcaster() -> Optional[MessageBroadcaster]:
    """Get global message broadcaster instance."""
    return _broadcaster


def initialize_message_broadcaster(
    websocket_manager: WebSocketConnectionManager,
    supabase_client: SupabaseRealtimeClient,
    redis_url: Optional[str] = None
) -> MessageBroadcaster:
    """Initialize global message broadcaster."""
    global _broadcaster
    _broadcaster = MessageBroadcaster(
        websocket_manager=websocket_manager,
        supabase_client=supabase_client,
        redis_url=redis_url
    )
    return _broadcaster