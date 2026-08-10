# app/services/websocket/router.py
from typing import List, Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, Query

from app.logger import get_logger
from app.services.websocket.models import WSChannel, WSMessage, WSMessageType
from app.services.websocket.service import WebSocketService

logger = get_logger(__name__)
router = APIRouter(prefix="/ws", tags=["WebSocket"])

_ws_service = WebSocketService()


def get_ws_service() -> WebSocketService:
    return _ws_service


@router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    channels: Optional[str] = Query(None),
    token: Optional[str] = Query(None),
    service: WebSocketService = Depends(get_ws_service),
):
    """Main WebSocket connection endpoint.
    
    Query params:
    - channels: comma-separated list of channels to subscribe to
    - token: JWT token for authentication
    """
    # Parse channels
    subscribed_channels = [WSChannel.SYSTEM]
    if channels:
        try:
            subscribed_channels = [
                WSChannel(ch.strip())
                for ch in channels.split(",")
                if ch.strip() in [c.value for c in WSChannel]
            ]
        except ValueError:
            await websocket.close(code=4000, reason="Invalid channel")
            return
    
    # Authenticate (optional for system channel)
    user_id = None
    if token:
        try:
            from app.services.auth.service import AuthService
            claims = await AuthService.verify_token(token)
            user_id = claims.get("sub")
        except Exception:
            await websocket.close(code=4001, reason="Invalid token")
            return
    
    # Connect
    connection = await service.connect(websocket, subscribed_channels, user_id)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            
            # Handle client messages
            msg_type = data.get("type")
            
            if msg_type == "subscribe":
                new_channels = data.get("channels", [])
                parsed = []
                for ch in new_channels:
                    try:
                        parsed.append(WSChannel(ch))
                    except ValueError:
                        pass
                if parsed:
                    await service.subscribe(str(connection.id), parsed)
            
            elif msg_type == "unsubscribe":
                remove_channels = data.get("channels", [])
                parsed = []
                for ch in remove_channels:
                    try:
                        parsed.append(WSChannel(ch))
                    except ValueError:
                        pass
                if parsed:
                    await service.unsubscribe(str(connection.id), parsed)
            
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": data.get("timestamp")})
            
            else:
                # Echo back for unknown types
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": f"Unknown message type: {msg_type}"}
                })
                
    except WebSocketDisconnect:
        await service.disconnect(str(connection.id))
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await service.disconnect(str(connection.id))


@router.get("/stats", response_model=dict)
async def get_ws_stats(
    service: WebSocketService = Depends(get_ws_service),
):
    """Get WebSocket service statistics."""
    return service.get_stats()