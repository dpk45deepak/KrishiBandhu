# Add to existing config module - JWT settings
# app/config/jwt_config.py (or add to existing settings)
from pydantic_settings import BaseSettings
from typing import List


class JWTSettings(BaseSettings):
    """JWT configuration - consumed by AuthService."""
    
    JWT_SECRET_KEY: str = "dev-secret-key-change-in-production"
    JWT_REFRESH_SECRET_KEY: str = "dev-refresh-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    class Config:
        env_file = ".env"
        env_prefix = "AGRIMIND_"