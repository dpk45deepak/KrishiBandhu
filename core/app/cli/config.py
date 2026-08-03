# app/cli/config.py
"""
CLI configuration helpers - never hardcodes paths.
Consumes existing config module.
"""
from pathlib import Path
from typing import Optional

from app.config import settings


def get_config_path() -> Path:
    """Get CLI config directory."""
    path = Path(settings.CLI_CONFIG_PATH)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_token(token: str):
    """Save auth token to config."""
    config_file = get_config_path() / "auth.json"
    config_file.write_text(f'{{"token": "{token}"}}')


def load_token() -> Optional[str]:
    """Load saved auth token."""
    config_file = get_config_path() / "auth.json"
    if config_file.exists():
        import json
        return json.loads(config_file.read_text()).get("token")
    return None