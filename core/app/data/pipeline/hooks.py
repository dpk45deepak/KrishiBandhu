"""Hook system for pipeline lifecycle events."""

from typing import Dict, List, Callable, Optional, Any
from enum import Enum
from dataclasses import dataclass
from .models import PipelineEvent, StageType


class HookType(str, Enum):
    """Types of pipeline hooks."""
    PRE_EXECUTION = "pre_execution"
    POST_EXECUTION = "post_execution"
    PRE_STAGE = "pre_stage"
    POST_STAGE = "post_stage"
    ON_ERROR = "on_error"
    ON_RETRY = "on_retry"
    ON_CHECKPOINT = "on_checkpoint"


@dataclass
class Hook:
    """A hook with its priority."""
    callback: Callable
    priority: int = 0
    stage_type: Optional[StageType] = None


class HookManager:
    """
    Manages pipeline hooks for lifecycle events.
    Supports priority-based hook execution.
    """
    
    _hooks: Dict[HookType, List[Hook]] = {hook_type: [] for hook_type in HookType}
    
    def register(
        self,
        hook_type: HookType,
        priority: int = 0,
        stage_type: Optional[StageType] = None
    ) -> Callable:
        """
        Decorator to register a hook.
        
        Usage:
            @hook_manager.register(HookType.PRE_STAGE, stage_type=StageType.SCAN)
            def pre_scan_hook(event):
                ...
        """
        def decorator(func: Callable) -> Callable:
            hook = Hook(callback=func, priority=priority, stage_type=stage_type)
            self._hooks[hook_type].append(hook)
            # Sort by priority (higher priority first)
            self._hooks[hook_type].sort(key=lambda h: h.priority, reverse=True)
            return func
        return decorator
    
    def trigger(
        self,
        hook_type: HookType,
        event: PipelineEvent,
        stage_type: Optional[StageType] = None
    ) -> None:
        """
        Trigger all hooks of a specific type.
        Filters hooks by stage_type if provided.
        """
        hooks = self._hooks.get(hook_type, [])
        
        for hook in hooks:
            # Skip if hook is stage-specific and doesn't match
            if hook.stage_type and stage_type and hook.stage_type != stage_type:
                continue
            
            try:
                hook.callback(event)
            except Exception as e:
                # Log but don't fail pipeline for hook errors
                import logging
                logging.warning(f"Hook {hook.callback.__name__} failed: {e}")
    
    def clear(self) -> None:
        """Clear all registered hooks."""
        for hook_type in HookType:
            self._hooks[hook_type].clear()