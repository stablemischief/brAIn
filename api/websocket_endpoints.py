"""
Real-time WebSocket API endpoints for brAIn v2.0.
Handles WebSocket connections, subscriptions, and real-time data streaming.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from realtime.websocket_manager import (
    WebSocketConnectionManager, 
    get_websocket_manager,
    MessageType,
    ConnectionState
)
from realtime.message_broadcaster import (
    MessageBroadcaster,
    get_message_broadcaster,
    MessageScope,
    MessagePriority
)
from realtime.supabase_client import SupabaseRealtimeClient
from realtime.subscription_handlers import (
    ProcessingStatusHandler,
    CostMonitoringHandler,
    SystemHealthHandler
)

logger = logging.getLogger(__name__)

# Router for real-time endpoints
realtime_router = APIRouter(prefix="/api/realtime", tags=["realtime"])


class WebSocketConnectionRequest(BaseModel):
    """Request model for WebSocket connection info."""
    user_id: Optional[UUID] = Field(None, description="User ID for the connection")
    session_id: Optional[str] = Field(None, description="Session ID for grouping connections")
    channels: List[str] = Field(default_factory=list, description="Initial channels to subscribe to")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional connection metadata")


class SubscriptionRequest(BaseModel):
    """Request model for channel subscription."""
    channel: str = Field(description="Channel name to subscribe to")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional subscription filters")


class BroadcastRequest(BaseModel):
    """Request model for broadcasting messages."""
    message_type: str = Field(description="Type of message")
    channel: str = Field(description="Target channel")
    payload: Dict[str, Any] = Field(description="Message payload")
    scope: MessageScope = Field(default=MessageScope.BROADCAST, description="Message scope")
    target_id: Optional[str] = Field(None, description="Target ID for scoped messages")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL, description="Message priority")
    ttl_seconds: Optional[int] = Field(None, description="Time to live in seconds")


# WebSocket endpoint for real-time connections
@realtime_router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: Optional[str] = Query(None, description="User ID"),
    session_id: Optional[str] = Query(None, description="Session ID"),
    channels: Optional[str] = Query(None, description="Comma-separated list of channels to subscribe to")
):
    """
    Main WebSocket endpoint for real-time communication.
    
    Query Parameters:
        user_id: Optional user ID for the connection
        session_id: Optional session ID for grouping connections
        channels: Comma-separated list of initial channels to subscribe to
    """
    websocket_manager = get_websocket_manager()
    connection_id = None
    
    try:
        # Parse user_id
        parsed_user_id = None
        if user_id:
            try:
                parsed_user_id = UUID(user_id)
            except ValueError:
                await websocket.close(code=4000, reason="Invalid user_id format")
                return
        
        # Parse channels
        channel_list = []
        if channels:
            channel_list = [ch.strip() for ch in channels.split(",") if ch.strip()]
        
        # Connect to WebSocket manager
        connection_id = await websocket_manager.connect(
            websocket=websocket,
            user_id=parsed_user_id,
            session_id=session_id,
            metadata={"initial_channels": channel_list}
        )
        
        # Subscribe to initial channels
        for channel in channel_list:
            await websocket_manager.subscribe(connection_id, channel)
        
        logger.info(f"WebSocket connection established: {connection_id}")
        
        # Message handling loop
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle the message
                await websocket_manager.handle_message(connection_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {connection_id}")
                break
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON received from {connection_id}: {e}")
                await websocket_manager.send_message(
                    connection_id,
                    MessageType.ERROR,
                    {"error": "Invalid JSON format", "details": str(e)}
                )
            except Exception as e:
                logger.error(f"Error handling message from {connection_id}: {e}")
                await websocket_manager.send_message(
                    connection_id,
                    MessageType.ERROR,
                    {"error": "Message handling failed", "details": str(e)}
                )
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    
    finally:
        # Clean up connection
        if connection_id:
            await websocket_manager.disconnect(connection_id, reason="Connection closed")


@realtime_router.get("/connections")
async def get_connections(
    manager: WebSocketConnectionManager = Depends(get_websocket_manager)
) -> Dict[str, Any]:
    """Get information about active WebSocket connections."""
    try:
        connections = manager.get_all_connections()
        
        # Format connection info
        connection_data = []
        for conn_id, conn_info in connections.items():
            health = manager.get_connection_health(conn_id)
            connection_data.append({
                "connection_id": conn_id,
                "user_id": str(conn_info.user_id) if conn_info.user_id else None,
                "session_id": conn_info.session_id,
                "connected_at": conn_info.connected_at.isoformat(),
                "subscriptions": list(conn_info.subscriptions),
                "state": conn_info.state,
                "health": health
            })
        
        stats = manager.get_stats()
        
        return {
            "total_connections": len(connection_data),
            "connections": connection_data,
            "stats": stats
        }
    
    except Exception as e:
        logger.error(f"Error getting connections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@realtime_router.get("/connections/{connection_id}")
async def get_connection_info(
    connection_id: str,
    manager: WebSocketConnectionManager = Depends(get_websocket_manager)
) -> Dict[str, Any]:
    """Get detailed information about a specific connection."""
    try:
        connections = manager.get_all_connections()
        
        if connection_id not in connections:
            raise HTTPException(status_code=404, detail="Connection not found")
        
        conn_info = connections[connection_id]
        health = manager.get_connection_health(connection_id)
        
        return {
            "connection_id": connection_id,
            "user_id": str(conn_info.user_id) if conn_info.user_id else None,
            "session_id": conn_info.session_id,
            "connected_at": conn_info.connected_at.isoformat(),
            "subscriptions": list(conn_info.subscriptions),
            "state": conn_info.state,
            "metadata": conn_info.metadata,
            "health": health
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting connection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@realtime_router.post("/connections/{connection_id}/subscribe")
async def subscribe_to_channel(
    connection_id: str,
    subscription: SubscriptionRequest,
    manager: WebSocketConnectionManager = Depends(get_websocket_manager)
) -> Dict[str, Any]:
    """Subscribe a connection to a channel."""
    try:
        success = await manager.subscribe(connection_id, subscription.channel)
        
        if not success:
            raise HTTPException(
                status_code=400, 
                detail="Failed to subscribe to channel"
            )
        
        return {
            "success": True,
            "connection_id": connection_id,
            "channel": subscription.channel,
            "message": "Successfully subscribed to channel"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error subscribing to channel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@realtime_router.delete("/connections/{connection_id}/subscribe/{channel}")
async def unsubscribe_from_channel(
    connection_id: str,
    channel: str,
    manager: WebSocketConnectionManager = Depends(get_websocket_manager)
) -> Dict[str, Any]:
    """Unsubscribe a connection from a channel."""
    try:
        success = await manager.unsubscribe(connection_id, channel)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to unsubscribe from channel"
            )
        
        return {
            "success": True,
            "connection_id": connection_id,
            "channel": channel,
            "message": "Successfully unsubscribed from channel"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unsubscribing from channel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@realtime_router.delete("/connections/{connection_id}")
async def disconnect_connection(
    connection_id: str,
    reason: Optional[str] = Query("Forced disconnect", description="Disconnect reason"),
    manager: WebSocketConnectionManager = Depends(get_websocket_manager)
) -> Dict[str, Any]:
    """Force disconnect a WebSocket connection."""
    try:
        success = await manager.disconnect(connection_id, reason=reason)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Connection not found or already disconnected"
            )
        
        return {
            "success": True,
            "connection_id": connection_id,
            "reason": reason,
            "message": "Connection disconnected successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disconnecting connection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@realtime_router.post("/broadcast")
async def broadcast_message(
    broadcast_req: BroadcastRequest,
    broadcaster: Optional[MessageBroadcaster] = Depends(get_message_broadcaster)
) -> Dict[str, Any]:
    """Broadcast a message to connected clients."""
    try:
        if not broadcaster:
            raise HTTPException(
                status_code=503,
                detail="Message broadcaster not available"
            )
        
        message_id = await broadcaster.broadcast_message(
            message_type=broadcast_req.message_type,
            channel=broadcast_req.channel,
            payload=broadcast_req.payload,
            scope=broadcast_req.scope,
            target_id=broadcast_req.target_id,
            priority=broadcast_req.priority,
            ttl_seconds=broadcast_req.ttl_seconds
        )
        
        return {
            "success": True,
            "message_id": message_id,
            "message": "Message queued for broadcasting"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error broadcasting message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@realtime_router.post("/broadcast/user/{user_id}")
async def broadcast_to_user(
    user_id: UUID,
    message_type: str,
    channel: str,
    payload: Dict[str, Any],
    priority: MessagePriority = MessagePriority.NORMAL,
    broadcaster: Optional[MessageBroadcaster] = Depends(get_message_broadcaster)
) -> Dict[str, Any]:
    """Broadcast a message to a specific user."""
    try:
        if not broadcaster:
            raise HTTPException(
                status_code=503,
                detail="Message broadcaster not available"
            )
        
        message_id = await broadcaster.broadcast_to_user(
            user_id=user_id,
            message_type=message_type,
            channel=channel,
            payload=payload,
            priority=priority
        )
        
        return {
            "success": True,
            "message_id": message_id,
            "user_id": str(user_id),
            "message": "Message queued for user"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error broadcasting to user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@realtime_router.post("/broadcast/session/{session_id}")
async def broadcast_to_session(
    session_id: str,
    message_type: str,
    channel: str,
    payload: Dict[str, Any],
    priority: MessagePriority = MessagePriority.NORMAL,
    broadcaster: Optional[MessageBroadcaster] = Depends(get_message_broadcaster)
) -> Dict[str, Any]:
    """Broadcast a message to a specific session."""
    try:
        if not broadcaster:
            raise HTTPException(
                status_code=503,
                detail="Message broadcaster not available"
            )
        
        message_id = await broadcaster.broadcast_to_session(
            session_id=session_id,
            message_type=message_type,
            channel=channel,
            payload=payload,
            priority=priority
        )
        
        return {
            "success": True,
            "message_id": message_id,
            "session_id": session_id,
            "message": "Message queued for session"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error broadcasting to session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@realtime_router.post("/broadcast/channel/{channel}")
async def broadcast_to_channel(
    channel: str,
    message_type: str,
    payload: Dict[str, Any],
    priority: MessagePriority = MessagePriority.NORMAL,
    broadcaster: Optional[MessageBroadcaster] = Depends(get_message_broadcaster)
) -> Dict[str, Any]:
    """Broadcast a message to all subscribers of a channel."""
    try:
        if not broadcaster:
            raise HTTPException(
                status_code=503,
                detail="Message broadcaster not available"
            )
        
        message_id = await broadcaster.broadcast_to_channel_subscribers(
            channel=channel,
            message_type=message_type,
            payload=payload,
            priority=priority
        )
        
        return {
            "success": True,
            "message_id": message_id,
            "channel": channel,
            "message": "Message queued for channel subscribers"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error broadcasting to channel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@realtime_router.get("/channels")
async def get_channels(
    manager: WebSocketConnectionManager = Depends(get_websocket_manager)
) -> Dict[str, Any]:
    """Get information about active channels."""
    try:
        stats = manager.get_stats()
        
        # Get channel information
        channels = {}
        for channel_name in manager.subscriptions.keys():
            channel_info = manager.get_channel_info(channel_name)
            channels[channel_name] = channel_info
        
        return {
            "total_channels": len(channels),
            "channels": channels,
            "stats": stats
        }
    
    except Exception as e:
        logger.error(f"Error getting channels: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@realtime_router.get("/channels/{channel}")
async def get_channel_info(
    channel: str,
    manager: WebSocketConnectionManager = Depends(get_websocket_manager)
) -> Dict[str, Any]:
    """Get detailed information about a specific channel."""
    try:
        channel_info = manager.get_channel_info(channel)
        
        if not channel_info["exists"]:
            raise HTTPException(status_code=404, detail="Channel not found")
        
        return {
            "channel": channel,
            **channel_info
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting channel info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@realtime_router.get("/stats")
async def get_realtime_stats(
    manager: WebSocketConnectionManager = Depends(get_websocket_manager),
    broadcaster: Optional[MessageBroadcaster] = Depends(get_message_broadcaster)
) -> Dict[str, Any]:
    """Get comprehensive real-time system statistics."""
    try:
        websocket_stats = manager.get_stats()
        
        broadcaster_stats = {}
        if broadcaster:
            broadcaster_stats = broadcaster.get_queue_status()
        
        return {
            "websocket": websocket_stats,
            "broadcaster": broadcaster_stats,
            "system": {
                "status": "healthy",
                "components": {
                    "websocket_manager": True,
                    "message_broadcaster": broadcaster is not None,
                    "supabase_realtime": True  # TODO: Add health check
                }
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@realtime_router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for the real-time system."""
    try:
        manager = get_websocket_manager()
        broadcaster = get_message_broadcaster()
        
        # Basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",  # TODO: Use actual timestamp
            "components": {
                "websocket_manager": manager is not None,
                "message_broadcaster": broadcaster is not None,
                "active_connections": len(manager.connections) if manager else 0
            }
        }
        
        # Determine overall health
        all_healthy = all(health_status["components"].values())
        health_status["status"] = "healthy" if all_healthy else "degraded"
        
        status_code = 200 if all_healthy else 503
        return JSONResponse(content=health_status, status_code=status_code)
    
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": "2024-01-01T00:00:00Z"
            },
            status_code=503
        )


# Export the router
__all__ = ["realtime_router"]