"""
brAIn v2.0 Real-time Subscription Handlers
Specialized handlers for different real-time channels and event types.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from uuid import UUID
from enum import Enum

from .websocket_manager import WebSocketConnectionManager, MessageType, get_websocket_manager
from .supabase_client import get_supabase_realtime_client
from ..models.api import ProcessingStatusMessage, CostUpdateMessage

logger = logging.getLogger(__name__)


class ChannelType(str, Enum):
    """Real-time channel types"""
    PROCESSING_STATUS = "processing_status"
    SYSTEM_HEALTH = "system_health"
    COST_MONITORING = "cost_monitoring"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    USER_ACTIVITY = "user_activity"


class SubscriptionHandler:
    """
    Base class for real-time subscription handlers.
    
    Each handler manages subscriptions for a specific type of real-time data.
    """
    
    def __init__(
        self,
        channel_type: ChannelType,
        websocket_manager: WebSocketConnectionManager
    ):
        self.channel_type = channel_type
        self.websocket_manager = websocket_manager
        self.supabase_client = get_supabase_realtime_client()
        self.active_subscriptions: Dict[str, Any] = {}
        self.message_filters: Dict[str, Callable] = {}
        
    async def setup_supabase_subscriptions(self) -> bool:
        """
        Setup Supabase real-time subscriptions for this handler.
        
        Returns:
            True if setup successful
        """
        raise NotImplementedError("Subclasses must implement setup_supabase_subscriptions")
    
    async def handle_supabase_event(self, payload: Dict[str, Any]) -> None:
        """
        Handle events from Supabase real-time.
        
        Args:
            payload: Event payload from Supabase
        """
        raise NotImplementedError("Subclasses must implement handle_supabase_event")
    
    async def broadcast_to_subscribers(
        self,
        message_type: MessageType,
        data: Dict[str, Any],
        user_filter: Optional[Callable[[UUID], bool]] = None
    ) -> int:
        """
        Broadcast message to all subscribers of this channel.
        
        Args:
            message_type: WebSocket message type
            data: Message data
            user_filter: Optional function to filter users
            
        Returns:
            Number of connections message was sent to
        """
        channel_name = f"{self.channel_type.value}_channel"
        
        # Apply user filtering if specified
        if user_filter:
            sent_count = 0
            for user_id, connection_ids in self.websocket_manager.user_connections.items():
                if user_filter(user_id):
                    sent_count += await self.websocket_manager.broadcast_to_user(
                        user_id, message_type, data, channel_name
                    )
            return sent_count
        else:
            return await self.websocket_manager.broadcast_to_channel(
                channel_name, message_type, data
            )
    
    def add_message_filter(self, filter_name: str, filter_func: Callable) -> None:
        """Add a message filter function"""
        self.message_filters[filter_name] = filter_func
    
    def remove_message_filter(self, filter_name: str) -> None:
        """Remove a message filter function"""
        if filter_name in self.message_filters:
            del self.message_filters[filter_name]


class ProcessingStatusHandler(SubscriptionHandler):
    """
    Handler for document processing status updates.
    
    Monitors:
    - Document processing progress
    - Queue status changes
    - Processing completions and failures
    - Batch operation status
    """
    
    def __init__(self, websocket_manager: WebSocketConnectionManager):
        super().__init__(ChannelType.PROCESSING_STATUS, websocket_manager)
    
    async def setup_supabase_subscriptions(self) -> bool:
        """Setup processing status subscriptions"""
        try:
            if not self.supabase_client or not self.supabase_client.channel_manager:
                logger.warning("Supabase client not available for processing status")
                return False
            
            channel_manager = self.supabase_client.channel_manager
            
            # Subscribe to processing_queue table changes
            success1 = await channel_manager.subscribe_to_table_changes(
                channel_name="processing_status",
                table="processing_queue",
                event="*",
                callback=self._handle_processing_queue_change
            )
            
            # Subscribe to documents table processing status changes
            success2 = await channel_manager.subscribe_to_table_changes(
                channel_name="processing_status",
                table="documents",
                event="UPDATE",
                callback=self._handle_document_processing_change,
                filter_config={"filter": "processing_status=neq.pending"}
            )
            
            # Subscribe to batch operation broadcasts
            success3 = await channel_manager.subscribe_to_broadcast(
                channel_name="processing_status",
                event="batch_operation_update",
                callback=self._handle_batch_operation_update
            )
            
            logger.info("Processing status subscriptions setup completed")
            return success1 and success2 and success3
            
        except Exception as e:
            logger.error(f"Failed to setup processing status subscriptions: {e}")
            return False
    
    async def _handle_processing_queue_change(self, payload: Dict[str, Any]) -> None:
        """Handle processing queue table changes"""
        try:
            event_type = payload.get("eventType")
            record = payload.get("new", payload.get("old", {}))
            
            if not record:
                return
            
            # Extract relevant data
            task_data = {
                "task_id": record.get("id"),
                "document_id": record.get("document_id"),
                "user_id": record.get("user_id"),
                "task_type": record.get("task_type"),
                "status": record.get("status"),
                "progress_percentage": record.get("progress_percentage"),
                "error_message": record.get("error_message"),
                "updated_at": record.get("updated_at")
            }
            
            # Create WebSocket message
            message_data = {
                "event_type": event_type,
                "task": task_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Broadcast to subscribers (filtered by user)
            user_id = record.get("user_id")
            if user_id:
                await self.websocket_manager.broadcast_to_user(
                    UUID(user_id),
                    MessageType.PROCESSING_STATUS,
                    message_data
                )
            
        except Exception as e:
            logger.error(f"Error handling processing queue change: {e}")
    
    async def _handle_document_processing_change(self, payload: Dict[str, Any]) -> None:
        """Handle document processing status changes"""
        try:
            record = payload.get("new", {})
            
            if not record:
                return
            
            # Extract document processing data
            document_data = {
                "document_id": record.get("id"),
                "user_id": record.get("user_id"),
                "title": record.get("title"),
                "processing_status": record.get("processing_status"),
                "extraction_quality": record.get("extraction_quality"),
                "processing_cost": record.get("processing_cost"),
                "updated_at": record.get("updated_at")
            }
            
            # Create WebSocket message
            message_data = {
                "event_type": "document_processed",
                "document": document_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Broadcast to user
            user_id = record.get("user_id")
            if user_id:
                await self.websocket_manager.broadcast_to_user(
                    UUID(user_id),
                    MessageType.PROCESSING_STATUS,
                    message_data
                )
            
        except Exception as e:
            logger.error(f"Error handling document processing change: {e}")
    
    async def _handle_batch_operation_update(self, payload: Dict[str, Any]) -> None:
        """Handle batch operation updates"""
        try:
            # Extract batch operation data
            batch_data = payload.get("payload", {})
            
            message_data = {
                "event_type": "batch_operation_update",
                "batch": batch_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Broadcast to relevant users
            user_id = batch_data.get("user_id")
            if user_id:
                await self.websocket_manager.broadcast_to_user(
                    UUID(user_id),
                    MessageType.PROCESSING_STATUS,
                    message_data
                )
            
        except Exception as e:
            logger.error(f"Error handling batch operation update: {e}")
    
    async def handle_supabase_event(self, payload: Dict[str, Any]) -> None:
        """Handle general Supabase events for processing status"""
        # This method can be used for additional event handling if needed
        pass
    
    async def notify_processing_started(
        self,
        user_id: UUID,
        document_id: str,
        task_type: str
    ) -> None:
        """Notify that processing has started"""
        message_data = {
            "event_type": "processing_started",
            "document_id": document_id,
            "task_type": task_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.websocket_manager.broadcast_to_user(
            user_id, MessageType.PROCESSING_STATUS, message_data
        )
    
    async def notify_processing_completed(
        self,
        user_id: UUID,
        document_id: str,
        result: Dict[str, Any]
    ) -> None:
        """Notify that processing has completed"""
        message_data = {
            "event_type": "processing_completed",
            "document_id": document_id,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.websocket_manager.broadcast_to_user(
            user_id, MessageType.PROCESSING_STATUS, message_data
        )


class CostMonitoringHandler(SubscriptionHandler):
    """
    Handler for cost monitoring and budget alerts.
    
    Monitors:
    - Real-time cost accumulation
    - Budget threshold alerts
    - Token usage updates
    - Spending projections
    """
    
    def __init__(self, websocket_manager: WebSocketConnectionManager):
        super().__init__(ChannelType.COST_MONITORING, websocket_manager)
    
    async def setup_supabase_subscriptions(self) -> bool:
        """Setup cost monitoring subscriptions"""
        try:
            if not self.supabase_client or not self.supabase_client.channel_manager:
                logger.warning("Supabase client not available for cost monitoring")
                return False
            
            channel_manager = self.supabase_client.channel_manager
            
            # Subscribe to LLM usage table for real-time cost updates
            success1 = await channel_manager.subscribe_to_table_changes(
                channel_name="cost_monitoring",
                table="llm_usage",
                event="INSERT",
                callback=self._handle_llm_usage_insert
            )
            
            # Subscribe to daily cost summary updates
            success2 = await channel_manager.subscribe_to_table_changes(
                channel_name="cost_monitoring",
                table="daily_cost_summary",
                event="*",
                callback=self._handle_daily_cost_update
            )
            
            # Subscribe to budget alert broadcasts
            success3 = await channel_manager.subscribe_to_broadcast(
                channel_name="cost_monitoring",
                event="budget_alert",
                callback=self._handle_budget_alert
            )
            
            logger.info("Cost monitoring subscriptions setup completed")
            return success1 and success2 and success3
            
        except Exception as e:
            logger.error(f"Failed to setup cost monitoring subscriptions: {e}")
            return False
    
    async def _handle_llm_usage_insert(self, payload: Dict[str, Any]) -> None:
        """Handle new LLM usage records"""
        try:
            record = payload.get("new", {})
            
            if not record:
                return
            
            # Extract cost data
            cost_data = {
                "operation_id": record.get("id"),
                "user_id": record.get("user_id"),
                "operation_type": record.get("operation_type"),
                "model_name": record.get("model_name"),
                "cost": record.get("cost", 0),
                "total_tokens": record.get("total_tokens", 0),
                "timestamp": record.get("created_at")
            }
            
            # Create WebSocket message
            message_data = {
                "event_type": "cost_update",
                "operation": cost_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Broadcast to user
            user_id = record.get("user_id")
            if user_id:
                await self.websocket_manager.broadcast_to_user(
                    UUID(user_id),
                    MessageType.COST_UPDATE,
                    message_data
                )
            
        except Exception as e:
            logger.error(f"Error handling LLM usage insert: {e}")
    
    async def _handle_daily_cost_update(self, payload: Dict[str, Any]) -> None:
        """Handle daily cost summary updates"""
        try:
            record = payload.get("new", {})
            
            if not record:
                return
            
            # Extract daily summary data
            summary_data = {
                "user_id": record.get("user_id"),
                "date": record.get("date"),
                "total_cost": record.get("total_cost", 0),
                "operation_costs": record.get("operation_costs", {}),
                "total_tokens": record.get("total_tokens", 0),
                "budget_limit": record.get("budget_limit", 0),
                "is_over_budget": record.get("total_cost", 0) > record.get("budget_limit", 0)
            }
            
            # Create WebSocket message
            message_data = {
                "event_type": "daily_summary_update",
                "summary": summary_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Broadcast to user
            user_id = record.get("user_id")
            if user_id:
                await self.websocket_manager.broadcast_to_user(
                    UUID(user_id),
                    MessageType.COST_UPDATE,
                    message_data
                )
            
        except Exception as e:
            logger.error(f"Error handling daily cost update: {e}")
    
    async def _handle_budget_alert(self, payload: Dict[str, Any]) -> None:
        """Handle budget alert broadcasts"""
        try:
            alert_data = payload.get("payload", {})
            
            message_data = {
                "event_type": "budget_alert",
                "alert": alert_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Broadcast to specific user
            user_id = alert_data.get("user_id")
            if user_id:
                await self.websocket_manager.broadcast_to_user(
                    UUID(user_id),
                    MessageType.COST_UPDATE,
                    message_data
                )
            
        except Exception as e:
            logger.error(f"Error handling budget alert: {e}")
    
    async def handle_supabase_event(self, payload: Dict[str, Any]) -> None:
        """Handle general Supabase events for cost monitoring"""
        pass
    
    async def notify_budget_threshold(
        self,
        user_id: UUID,
        threshold_percentage: float,
        current_spending: float,
        budget_limit: float
    ) -> None:
        """Notify user of budget threshold crossing"""
        message_data = {
            "event_type": "budget_threshold",
            "threshold_percentage": threshold_percentage,
            "current_spending": current_spending,
            "budget_limit": budget_limit,
            "percentage_used": current_spending / budget_limit if budget_limit > 0 else 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.websocket_manager.broadcast_to_user(
            user_id, MessageType.COST_UPDATE, message_data
        )


class SystemHealthHandler(SubscriptionHandler):
    """
    Handler for system health monitoring.
    
    Monitors:
    - Service status changes
    - Performance metrics
    - Error alerts
    - System maintenance notifications
    """
    
    def __init__(self, websocket_manager: WebSocketConnectionManager):
        super().__init__(ChannelType.SYSTEM_HEALTH, websocket_manager)
    
    async def setup_supabase_subscriptions(self) -> bool:
        """Setup system health subscriptions"""
        try:
            if not self.supabase_client or not self.supabase_client.channel_manager:
                logger.warning("Supabase client not available for system health")
                return False
            
            channel_manager = self.supabase_client.channel_manager
            
            # Subscribe to system health table
            success1 = await channel_manager.subscribe_to_table_changes(
                channel_name="system_health",
                table="system_health",
                event="*",
                callback=self._handle_health_update
            )
            
            # Subscribe to system alerts broadcasts
            success2 = await channel_manager.subscribe_to_broadcast(
                channel_name="system_health",
                event="system_alert",
                callback=self._handle_system_alert
            )
            
            logger.info("System health subscriptions setup completed")
            return success1 and success2
            
        except Exception as e:
            logger.error(f"Failed to setup system health subscriptions: {e}")
            return False
    
    async def _handle_health_update(self, payload: Dict[str, Any]) -> None:
        """Handle system health updates"""
        try:
            record = payload.get("new", {})
            
            if not record:
                return
            
            # Extract health data
            health_data = {
                "service_name": record.get("service_name"),
                "status": record.get("status"),
                "response_time_ms": record.get("response_time_ms"),
                "cpu_usage_percent": record.get("cpu_usage_percent"),
                "memory_usage_percent": record.get("memory_usage_percent"),
                "error_rate_percent": record.get("error_rate_percent"),
                "timestamp": record.get("created_at")
            }
            
            # Create WebSocket message
            message_data = {
                "event_type": "health_update",
                "service": health_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Broadcast to all connected users (system-wide info)
            await self.broadcast_to_subscribers(
                MessageType.SYSTEM_HEALTH,
                message_data
            )
            
        except Exception as e:
            logger.error(f"Error handling health update: {e}")
    
    async def _handle_system_alert(self, payload: Dict[str, Any]) -> None:
        """Handle system alert broadcasts"""
        try:
            alert_data = payload.get("payload", {})
            
            message_data = {
                "event_type": "system_alert",
                "alert": alert_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Broadcast to all users
            await self.broadcast_to_subscribers(
                MessageType.SYSTEM_HEALTH,
                message_data
            )
            
        except Exception as e:
            logger.error(f"Error handling system alert: {e}")
    
    async def handle_supabase_event(self, payload: Dict[str, Any]) -> None:
        """Handle general Supabase events for system health"""
        pass
    
    async def notify_service_status_change(
        self,
        service_name: str,
        old_status: str,
        new_status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Notify of service status change"""
        message_data = {
            "event_type": "service_status_change",
            "service_name": service_name,
            "old_status": old_status,
            "new_status": new_status,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.broadcast_to_subscribers(
            MessageType.SYSTEM_HEALTH,
            message_data
        )


class RealtimeSubscriptionManager:
    """
    Central manager for all real-time subscription handlers.
    
    Coordinates multiple subscription handlers and provides unified interface.
    """
    
    def __init__(self):
        self.websocket_manager = get_websocket_manager()
        self.handlers: Dict[ChannelType, SubscriptionHandler] = {}
        self.initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize all subscription handlers.
        
        Returns:
            True if all handlers initialized successfully
        """
        try:
            # Create handlers
            self.handlers[ChannelType.PROCESSING_STATUS] = ProcessingStatusHandler(
                self.websocket_manager
            )
            self.handlers[ChannelType.COST_MONITORING] = CostMonitoringHandler(
                self.websocket_manager
            )
            self.handlers[ChannelType.SYSTEM_HEALTH] = SystemHealthHandler(
                self.websocket_manager
            )
            
            # Setup Supabase subscriptions for each handler
            setup_results = {}
            for channel_type, handler in self.handlers.items():
                setup_results[channel_type.value] = await handler.setup_supabase_subscriptions()
            
            # Check if all subscriptions were setup successfully
            all_success = all(setup_results.values())
            
            if all_success:
                self.initialized = True
                logger.info("All real-time subscription handlers initialized successfully")
            else:
                failed_handlers = [k for k, v in setup_results.items() if not v]
                logger.warning(f"Some handlers failed to initialize: {failed_handlers}")
            
            return all_success
            
        except Exception as e:
            logger.error(f"Failed to initialize subscription manager: {e}")
            return False
    
    def get_handler(self, channel_type: ChannelType) -> Optional[SubscriptionHandler]:
        """Get handler for specific channel type"""
        return self.handlers.get(channel_type)
    
    async def shutdown(self) -> None:
        """Shutdown all handlers and cleanup"""
        for handler in self.handlers.values():
            if hasattr(handler, 'shutdown'):
                await handler.shutdown()
        
        self.handlers.clear()
        self.initialized = False
        
        logger.info("Subscription manager shut down successfully")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all subscription handlers"""
        return {
            "initialized": self.initialized,
            "handlers": list(self.handlers.keys()),
            "websocket_stats": self.websocket_manager.get_stats()
        }


# Global subscription manager instance
_subscription_manager = RealtimeSubscriptionManager()


def get_subscription_manager() -> RealtimeSubscriptionManager:
    """Get the global subscription manager instance"""
    return _subscription_manager