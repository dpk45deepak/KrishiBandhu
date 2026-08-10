# app/config/__init__.py - Update to include JWT settings
from app.config.settings import Settings
from app.config.jwt_config import JWTSettings


class AppSettings(Settings, JWTSettings):
    """Unified application settings."""
    pass


settings = AppSettings()