# app/services/auth/service.py
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import jwt
from passlib.context import CryptContext

from app.config import settings
from app.logger import get_logger
from app.services.auth.models import (
    APIKeyCreate,
    APIKeyInDB,
    APIKeyResponse,
    Permission,
    ROLE_PERMISSIONS,
    Role,
    TokenResponse,
    UserCreate,
    UserCredentials,
    UserInDB,
)
from app.utils.decorators import timed

logger = get_logger(__name__)


class AuthService:
    """Authentication and authorization service.
    
    Consumes: config (settings), logger, utils (timed decorator)
    Provides: JWT, OAuth2, API Key, RBAC for all other services.
    """
    
    # Password hashing context - uses bcrypt from existing config
    _pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    # User store (in-memory for now; replace with DB integration)
    _users: dict[str, UserInDB] = {}
    _api_keys: dict[str, APIKeyInDB] = {}
    
    @classmethod
    def hash_password(cls, password: str) -> str:
        """Hash a password using configured scheme."""
        return cls._pwd_context.hash(password)
    
    @classmethod
    def verify_password(cls, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return cls._pwd_context.verify(plain_password, hashed_password)
    
    @classmethod
    @timed
    async def create_user(cls, user_data: UserCreate) -> UserInDB:
        """Create a new user. Consumes config for default settings."""
        import uuid
        
        if user_data.username in cls._users:
            raise ValueError(f"User '{user_data.username}' already exists")
        
        permissions = ROLE_PERMISSIONS[user_data.role]
        
        user = UserInDB(
            id=uuid.uuid4(),
            username=user_data.username,
            email=user_data.email,
            full_name=user_data.full_name,
            role=user_data.role,
            permissions=permissions,
            hashed_password=cls.hash_password(user_data.password),
        )
        
        cls._users[user_data.username] = user
        logger.info(f"User created: {user.username} with role {user.role.value}")
        return user
    
    @classmethod
    @timed
    async def authenticate(cls, credentials: UserCredentials) -> Optional[TokenResponse]:
        """Authenticate user and return JWT tokens."""
        user = cls._users.get(credentials.username)
        
        if user is None:
            logger.warning(f"Authentication failed: unknown user '{credentials.username}'")
            return None
        
        if not user.is_active:
            logger.warning(f"Authentication failed: inactive user '{credentials.username}'")
            return None
        
        if not cls.verify_password(credentials.password, user.hashed_password):
            logger.warning(f"Authentication failed: invalid password for '{credentials.username}'")
            return None
        
        logger.info(f"User authenticated: {user.username}")
        return cls._create_token_response(user)
    
    @classmethod
    @timed
    async def authenticate_oauth2(
        cls, username: str, password: str
    ) -> Optional[TokenResponse]:
        """OAuth2 password flow authentication."""
        credentials = UserCredentials(username=username, password=password)
        return await cls.authenticate(credentials)
    
    @classmethod
    def _create_token_response(cls, user: UserInDB) -> TokenResponse:
        """Create JWT access and refresh tokens."""
        access_token = cls._create_access_token(user)
        refresh_token = cls._create_refresh_token(user)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )
    
    @classmethod
    def _create_access_token(cls, user: UserInDB) -> str:
        """Create a JWT access token."""
        claims = user.to_claims()
        claims.update({
            "type": "access",
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(
                minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
            ),
        })
        
        return jwt.encode(
            claims,
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )
    
    @classmethod
    def _create_refresh_token(cls, user: UserInDB) -> str:
        """Create a JWT refresh token."""
        claims = {
            "sub": str(user.id),
            "type": "refresh",
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(
                days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
            ),
        }
        
        return jwt.encode(
            claims,
            settings.JWT_REFRESH_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )
    
    @classmethod
    @timed
    async def verify_token(cls, token: str) -> dict:
        """Verify a JWT access token and return claims."""
        try:
            claims = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
            )
            
            if claims.get("type") != "access":
                raise jwt.InvalidTokenError("Invalid token type")
            
            logger.debug(f"Token verified for user: {claims.get('username')}")
            return claims
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            raise
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            raise
    
    @classmethod
    @timed
    async def refresh_access_token(cls, refresh_token: str) -> Optional[TokenResponse]:
        """Create a new access token from a valid refresh token."""
        try:
            claims = jwt.decode(
                refresh_token,
                settings.JWT_REFRESH_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
            )
            
            if claims.get("type") != "refresh":
                raise jwt.InvalidTokenError("Invalid token type")
            
            user_id = claims.get("sub")
            user = cls._find_user_by_id(user_id)
            
            if user is None or not user.is_active:
                return None
            
            logger.info(f"Token refreshed for user: {user.username}")
            return cls._create_token_response(user)
            
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid refresh token: {e}")
            return None
    
    @classmethod
    @timed
    async def create_api_key(cls, user_id: str, key_data: APIKeyCreate) -> APIKeyResponse:
        """Create an API key for service-to-service authentication."""
        import uuid
        from datetime import timedelta
        
        raw_key = f"agm_{secrets.token_hex(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_prefix = raw_key[:11]  # "agm_" + first 8 hex chars
        
        permissions = key_data.permissions or ROLE_PERMISSIONS[key_data.role]
        
        api_key = APIKeyInDB(
            id=uuid.uuid4(),
            name=key_data.name,
            key_prefix=key_prefix,
            key_hash=key_hash,
            role=key_data.role,
            permissions=permissions,
            created_at=datetime.now(timezone.utc),
            created_by=uuid.UUID(user_id),
            expires_at=(
                datetime.now(timezone.utc) + timedelta(days=key_data.expires_in_days)
                if key_data.expires_in_days
                else None
            ),
        )
        
        cls._api_keys[key_hash] = api_key
        logger.info(f"API key created: {key_data.name} ({key_prefix}...)")
        
        # Return the raw key only once
        response = APIKeyResponse(
            id=api_key.id,
            name=api_key.name,
            key_prefix=key_prefix,
            role=api_key.role,
            permissions=api_key.permissions,
            created_at=api_key.created_at,
            expires_at=api_key.expires_at,
        )
        # Attach raw key (only time it's visible)
        response._raw_key = raw_key
        return response
    
    @classmethod
    @timed
    async def verify_api_key(cls, api_key: str) -> Optional[str]:
        """Verify an API key and return user context."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_record = cls._api_keys.get(key_hash)
        
        if key_record is None:
            return None
        
        if not key_record.is_active:
            logger.warning(f"Inactive API key used: {key_record.key_prefix}")
            return None
        
        if key_record.expires_at and key_record.expires_at < datetime.now(timezone.utc):
            logger.warning(f"Expired API key used: {key_record.key_prefix}")
            return None
        
        logger.debug(f"API key verified: {key_record.key_prefix}")
        return key_record.created_by  # Return the user ID who owns the key
    
    @classmethod
    async def get_user_from_api_key(cls, api_key: str) -> Optional[dict]:
        """Resolve API key to full user context."""
        user_id = await cls.verify_api_key(api_key)
        if user_id is None:
            return None
        
        user = cls._find_user_by_id(str(user_id))
        if user is None:
            return None
        
        # Also include the API key's specific permissions
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_record = cls._api_keys[key_hash]
        
        claims = user.to_claims()
        claims["permissions"] = [p.value for p in key_record.permissions]
        return claims
    
    @classmethod
    def _find_user_by_id(cls, user_id: str) -> Optional[UserInDB]:
        """Find a user by their ID."""
        for user in cls._users.values():
            if str(user.id) == user_id:
                return user
        return None
    
    @classmethod
    async def get_user(cls, username: str) -> Optional[dict]:
        """Get public user information."""
        user = cls._users.get(username)
        if user is None:
            return None
        return user.to_claims()
    
    @classmethod
    async def list_users(cls) -> list[dict]:
        """List all users (admin only)."""
        return [user.to_claims() for user in cls._users.values()]
    
    @classmethod
    async def revoke_api_key(cls, key_id: str) -> bool:
        """Revoke an API key."""
        for key_hash, key_record in cls._api_keys.items():
            if str(key_record.id) == key_id:
                key_record.is_active = False
                logger.info(f"API key revoked: {key_record.key_prefix}")
                return True
        return False