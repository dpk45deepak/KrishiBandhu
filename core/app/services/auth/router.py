# app/services/auth/router.py
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import OAuth2PasswordRequestForm

from app.logger import get_logger
from app.services.api.dependencies import get_current_user, require_permission
from app.services.auth.models import (
    APIKeyCreate,
    APIKeyResponse,
    Permission,
    RefreshTokenRequest,
    TokenResponse,
    UserCreate,
)
from app.services.auth.service import AuthService

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=dict, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """Register a new user account."""
    try:
        user = await AuthService.create_user(user_data)
        return {"message": "User created", "username": user.username, "id": str(user.id)}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.post("/token", response_model=TokenResponse)
async def login_password(form_data: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 password flow - obtain JWT access token."""
    token = await AuthService.authenticate_oauth2(
        username=form_data.username,
        password=form_data.password,
    )
    
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    logger.info(f"Token issued for user: {form_data.username}")
    return token


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """Refresh an expired access token using a refresh token."""
    token = await AuthService.refresh_access_token(request.refresh_token)
    
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    
    return token


@router.get("/me", response_model=dict)
async def get_current_user_info(
    current_user: dict = Depends(get_current_user),
):
    """Get the currently authenticated user's information."""
    return current_user


@router.post("/api-keys", response_model=dict)
async def create_api_key(
    key_data: APIKeyCreate,
    current_user: dict = Depends(require_permission(Permission.ADMIN.value)),
):
    """Create a new API key (admin only)."""
    key = await AuthService.create_api_key(
        user_id=current_user["sub"],
        key_data=key_data,
    )
    
    return {
        "id": str(key.id),
        "name": key.name,
        "key": key._raw_key,  # Only time the raw key is returned
        "key_prefix": key.key_prefix,
        "role": key.role.value,
        "permissions": [p.value for p in key.permissions],
        "created_at": key.created_at.isoformat(),
        "expires_at": key.expires_at.isoformat() if key.expires_at else None,
        "message": "Store this key securely - it will not be shown again",
    }


@router.get("/api-keys", response_model=list[dict])
async def list_api_keys(
    current_user: dict = Depends(require_permission(Permission.ADMIN.value)),
):
    """List API keys (admin only, raw keys never returned)."""
    keys = []
    for key_record in AuthService._api_keys.values():
        if key_record.is_active:
            keys.append({
                "id": str(key_record.id),
                "name": key_record.name,
                "key_prefix": key_record.key_prefix,
                "role": key_record.role.value,
                "permissions": [p.value for p in key_record.permissions],
                "created_at": key_record.created_at.isoformat(),
                "expires_at": key_record.expires_at.isoformat() if key_record.expires_at else None,
            })
    return keys


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: dict = Depends(require_permission(Permission.ADMIN.value)),
):
    """Revoke an API key (admin only)."""
    success = await AuthService.revoke_api_key(key_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")
    return {"message": "API key revoked"}


@router.get("/users", response_model=list[dict])
async def list_users(
    current_user: dict = Depends(require_permission(Permission.ADMIN.value)),
):
    """List all users (admin only)."""
    return await AuthService.list_users()