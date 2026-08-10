# app/services/websocket/service.py
import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from fastapi import WebSocket, WebSocketDisconnect

from app.logger import get_logger
from app.services.websocket.models import (
    WSChannel,
    WSConnection,
    WSMessage,
    WSMessageType,
)

logger = get_logger(__name__)


class WebSocketService:
    """WebSocket service for real-time communication.
    
    Consumes:
    - logger: structured logging
    - auth: user authentication for connections
    
    Provides:
    - Channel-based pub/sub
    - Connection management
    - Real-time event broadcasting
    - Heartbeat/health checks
    """
    
    def __init__(self):
        self._connections: Dict[str, WSConnection] = {}  # connection_id -> WSConnection
        self._channels: Dict[WSChannel, Set[str]] = {  # channel -> set of connection_ids
            ch: set() for ch in WSChannel
        }
        self._heartbeat_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start heartbeat and cleanup tasks."""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("WebSocket service started")
    
    async def stop(self):
        """Stop background tasks."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        # Close all connections
        for conn in self._connections.values():
            try:
                await conn.websocket.close()
            except Exception:
                pass
        self._connections.clear()
        for ch in self._channels:
            self._channels[ch].clear()
        logger.info("WebSocket service stopped")
    
    async def connect(
        self,
        websocket: WebSocket,
        channels: Optional[List[WSChannel]] = None,
        user_id: Optional[str] = None,
    ) -> WSConnection:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        connection_id = uuid4()
        subscribed_channels = channels or [WSChannel.SYSTEM]
        
        connection = WSConnection(
            id=connection_id,
            websocket=websocket,
            channels=subscribed_channels,
            user_id=user_id,
        )
        
        self._connections[str(connection_id)] = connection
        
        # Subscribe to channels
        for channel in subscribed_channels:
            if channel in self._channels:
                self._channels[channel].add(str(connection_id))
        
        logger.info(
            f"WebSocket connected: {connection_id} "
            f"(user={user_id}, channels={[ch.value for ch in subscribed_channels]})"
        )
        
        # Send welcome message
        await self.send_to_connection(connection, WSMessage(
            type=WSMessageType.NOTIFICATION,
            channel=WSChannel.SYSTEM,
            data={
                "message": "Connected to AgriMind AI Platform",
                "connection_id": str(connection_id),
                "channels": [ch.value for ch in subscribed_channels],
            },
        ))
        
        return connection
    
    async def disconnect(self, connection_id: str):
        """Handle connection disconnect."""
        connection = self._connections.pop(connection_id, None)
        if connection:
            # Unsubscribe from all channels
            for channel in connection.channels:
                if channel in self._channels:
                    self._channels[channel].discard(connection_id)
            
            logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def subscribe(self, connection_id: str, channels: List[WSChannel]):
        """Subscribe a connection to additional channels."""
        connection = self._connections.get(connection_id)
        if connection:
            for channel in channels:
                if channel not in connection.channels:
                    connection.channels.append(channel)
                if channel in self._channels:
                    self._channels[channel].add(connection_id)
            
            logger.debug(f"Connection {connection_id} subscribed to: {[ch.value for ch in channels]}")
    
    async def unsubscribe(self, connection_id: str, channels: List[WSChannel]):
        """Unsubscribe from channels."""
        connection = self._connections.get(connection_id)
        if connection:
            for channel in channels:
                if channel in connection.channels:
                    connection.channels.remove(channel)
                if channel in self._channels:
                    self._channels[channel].discard(connection_id)
    
    async def broadcast(self, message: WSMessage):
        """Broadcast a message to all subscribers of a channel."""
        channel = message.channel
        connection_ids = self._channels.get(channel, set())
        
        disconnected = []
        
        for conn_id in connection_ids:
            connection = self._connections.get(conn_id)
            if connection:
                try:
                    await self._send_json(connection.websocket, {
                        "type": message.type.value,
                        "channel": message.channel.value,
                        "data": message.data,
                        "message_id": str(message.message_id),
                        "timestamp": message.timestamp.isoformat(),
                        "sender": message.sender,
                    })
                    connection.last_activity = datetime.now(timezone.utc)
                except Exception:
                    disconnected.append(conn_id)
            else:
                disconnected.append(conn_id)
        
        # Clean up disconnected
        for conn_id in disconnected:
            self._channels[channel].discard(conn_id)
        
        if disconnected:
            logger.debug(f"Cleaned up {len(disconnected)} stale connections from {channel.value}")
    
    async def send_to_connection(self, connection: WSConnection, message: WSMessage):
        """Send a message to a specific connection."""
        try:
            await self._send_json(connection.websocket, {
                "type": message.type.value,
                "channel": message.channel.value,
                "data": message.data,
                "message_id": str(message.message_id),
                "timestamp": message.timestamp.isoformat(),
                "sender": message.sender,
            })
            connection.last_activity = datetime.now(timezone.utc)
        except Exception as e:
            logger.warning(f"Failed to send to connection {connection.id}: {e}")
    
    async def send_to_user(self, user_id: str, message: WSMessage):
        """Send a message to all connections for a specific user."""
        for conn in self._connections.values():
            if conn.user_id == user_id:
                await self.send_to_connection(conn, message)
    
    async def _send_json(self, websocket: WebSocket, data: dict):
        """Send JSON data over WebSocket."""
        await websocket.send_text(json.dumps(data, default=str))
    
    async def _heartbeat_loop(self, interval: int = 30):
        """Send periodic heartbeat to all connections."""
        while True:
            try:
                await asyncio.sleep(interval)
                
                heartbeat = WSMessage(
                    type=WSMessageType.HEALTH_CHECK,
                    channel=WSChannel.SYSTEM,
                    data={"timestamp": datetime.now(timezone.utc).isoformat()},
                )
                
                # Broadcast to system channel
                await self.broadcast(heartbeat)
                
                # Clean up stale connections
                stale_timeout = 120  # seconds
                now = datetime.now(timezone.utc)
                stale_connections = [
                    conn_id for conn_id, conn in self._connections.items()
                    if (now - conn.last_activity).total_seconds() > stale_timeout
                ]
                
                for conn_id in stale_connections:
                    await self.disconnect(conn_id)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket service statistics."""
        return {
            "active_connections": len(self._connections),
            "channels": {
                ch.value: len(conns)
                for ch, conns in self._channels.items()
            },
            "total_channels": len(self._channels),
        }