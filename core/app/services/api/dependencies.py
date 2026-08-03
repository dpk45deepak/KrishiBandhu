# app/services/api/dependencies.py
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer

from app.config import settings
from app.services.health.service import HealthService

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

# Singleton instances
_health_service: Optional[HealthService] = None


def get_health_service() -> HealthService:
    """Dependency injection for HealthService singleton."""
    global _health_service
    if _health_service is None:
        _health_service = HealthService()
    return _health_service


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> dict:
    """Extract and validate current user from JWT token."""
    if credentials is None:
        # Check for API key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return await _validate_api_key_user(api_key)
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Will implement JWT validation in auth service
    from app.services.auth.service import AuthService
    return await AuthService.verify_token(credentials.credentials)


async def verify_api_key(
    x_api_key: Optional[str] = Header(None),
) -> Optional[str]:
    """Validate API key from header."""
    if x_api_key is None:
        return None
    # Will implement in auth service
    from app.services.auth.service import AuthService
    return await AuthService.verify_api_key(x_api_key)


async def _validate_api_key_user(api_key: str) -> dict:
    """Convert valid API key to user context."""
    from app.services.auth.service import AuthService
    user = await AuthService.get_user_from_api_key(api_key)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user


def require_permission(permission: str):
    """Dependency factory for RBAC permission checks."""
    async def _check_permission(
        current_user: dict = Depends(get_current_user),
    ) -> dict:
        permissions = current_user.get("permissions", [])
        if permission not in permissions and "admin" not in permissions:
            raise HTTPException(status_code=403, detail=f"Missing permission: {permission}")
        return current_user
    return _check_permission


async def get_db_session():
    """Get database session - placeholder for actual DB integration."""
    # Will integrate with existing data module's session management
    from app.data import get_session
    async with get_session() as session:
        yield session