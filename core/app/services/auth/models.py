# app/services/auth/models.py
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4


class Permission(str, Enum):
    # Dataset permissions
    DATASET_READ = "dataset:read"
    DATASET_WRITE = "dataset:write"
    DATASET_DELETE = "dataset:delete"
    
    # Pipeline permissions
    PIPELINE_READ = "pipeline:read"
    PIPELINE_WRITE = "pipeline:write"
    PIPELINE_EXECUTE = "pipeline:execute"
    
    # Feature store permissions
    FEATURE_READ = "feature:read"
    FEATURE_WRITE = "feature:write"
    
    # ML permissions
    MODEL_TRAIN = "model:train"
    MODEL_READ = "model:read"
    MODEL_DEPLOY = "model:deploy"
    
    # Prediction permissions
    PREDICT_READ = "predict:read"
    PREDICT_EXECUTE = "predict:execute"
    
    # Report permissions
    REPORT_READ = "report:read"
    REPORT_CREATE = "report:create"
    
    # Admin
    ADMIN = "admin"


class Role(str, Enum):
    ADMIN = "admin"
    DATA_SCIENTIST = "data_scientist"
    ANALYST = "analyst"
    VIEWER = "viewer"
    SERVICE = "service"


# Role-to-permission mappings
ROLE_PERMISSIONS: dict[Role, List[Permission]] = {
    Role.ADMIN: [p for p in Permission],
    Role.DATA_SCIENTIST: [
        Permission.DATASET_READ, Permission.DATASET_WRITE,
        Permission.PIPELINE_READ, Permission.PIPELINE_WRITE, Permission.PIPELINE_EXECUTE,
        Permission.FEATURE_READ, Permission.FEATURE_WRITE,
        Permission.MODEL_TRAIN, Permission.MODEL_READ, Permission.MODEL_DEPLOY,
        Permission.PREDICT_READ, Permission.PREDICT_EXECUTE,
        Permission.REPORT_READ, Permission.REPORT_CREATE,
    ],
    Role.ANALYST: [
        Permission.DATASET_READ,
        Permission.PIPELINE_READ, Permission.PIPELINE_EXECUTE,
        Permission.FEATURE_READ,
        Permission.MODEL_READ,
        Permission.PREDICT_READ, Permission.PREDICT_EXECUTE,
        Permission.REPORT_READ, Permission.REPORT_CREATE,
    ],
    Role.VIEWER: [
        Permission.DATASET_READ,
        Permission.PIPELINE_READ,
        Permission.FEATURE_READ,
        Permission.MODEL_READ,
        Permission.PREDICT_READ,
        Permission.REPORT_READ,
    ],
    Role.SERVICE: [
        Permission.DATASET_READ, Permission.DATASET_WRITE,
        Permission.PIPELINE_EXECUTE,
        Permission.FEATURE_READ, Permission.FEATURE_WRITE,
        Permission.MODEL_READ,
        Permission.PREDICT_EXECUTE,
    ],
}


@dataclass
class UserCreate:
    username: str
    email: str
    password: str
    full_name: str
    role: Role = Role.VIEWER


@dataclass
class UserCredentials:
    username: str
    password: str


@dataclass
class UserInDB:
    id: UUID
    username: str
    email: str
    full_name: str
    role: Role
    permissions: List[Permission]
    hashed_password: str
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_claims(self) -> dict:
        """Convert to JWT claims (excludes sensitive data)."""
        return {
            "sub": str(self.id),
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role.value,
            "permissions": [p.value for p in self.permissions],
        }


@dataclass
class TokenResponse:
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    refresh_token: Optional[str] = None
    
    
@dataclass
class RefreshTokenRequest:
    refresh_token: str


@dataclass
class APIKeyCreate:
    name: str
    role: Role = Role.SERVICE
    permissions: Optional[List[Permission]] = None
    expires_in_days: Optional[int] = None


@dataclass
class APIKeyResponse:
    id: UUID
    name: str
    key_prefix: str
    role: Role
    permissions: List[Permission]
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
    
    
@dataclass
class APIKeyInDB(APIKeyResponse):
    key_hash: str
    created_by: UUID