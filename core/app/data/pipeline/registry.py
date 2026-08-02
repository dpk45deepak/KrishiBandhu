"""Stage registry for dynamic stage loading and management."""

from typing import Dict, Type, Optional, Callable, Any
from functools import wraps
from .models import StageType
from .exceptions import StageNotFoundError


class StageRegistry:
    """
    Registry for pipeline stages.
    Supports dynamic registration and retrieval of stages.
    """
    
    _instance = None
    _stages: Dict[StageType, Callable] = {}
    _stage_configs: Dict[StageType, Dict[str, Any]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(
        self,
        stage_type: StageType,
        config: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """
        Decorator to register a stage function.
        
        Usage:
            @registry.register(StageType.SCAN)
            def scan_stage(context, config):
                ...
        """
        def decorator(func: Callable) -> Callable:
            self._stages[stage_type] = func
            if config:
                self._stage_configs[stage_type] = config
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def get_stage(self, stage_type: StageType) -> Callable:
        """Get a registered stage function."""
        if stage_type not in self._stages:
            raise StageNotFoundError(f"Stage {stage_type} not registered")
        return self._stages[stage_type]
    
    def get_config(self, stage_type: StageType) -> Dict[str, Any]:
        """Get configuration for a stage."""
        return self._stage_configs.get(stage_type, {})
    
    def list_stages(self) -> Dict[StageType, Callable]:
        """List all registered stages."""
        return self._stages.copy()
    
    def is_registered(self, stage_type: StageType) -> bool:
        """Check if a stage is registered."""
        return stage_type in self._stages
    
    def clear(self) -> None:
        """Clear all registered stages."""
        self._stages.clear()
        self._stage_configs.clear()