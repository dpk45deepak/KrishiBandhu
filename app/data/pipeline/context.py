"""Thread-safe execution context for pipeline stages."""

import threading
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineContext:
    """
    Thread-safe shared context for pipeline execution.
    Stages can read/write data through this context.
    """
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _data: Dict[str, Any] = field(default_factory=dict)
    _metadata: Dict[str, Any] = field(default_factory=dict)
    _artifacts_dir: Optional[Path] = None
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the context."""
        with self._lock:
            self._data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context."""
        with self._lock:
            return self._data.get(key, default)
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update multiple values at once."""
        with self._lock:
            self._data.update(data)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        with self._lock:
            self._metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        with self._lock:
            return self._metadata.get(key, default)
    
    @property
    def artifacts_dir(self) -> Optional[Path]:
        """Get the artifacts directory."""
        return self._artifacts_dir
    
    @artifacts_dir.setter
    def artifacts_dir(self, path: Path) -> None:
        """Set the artifacts directory."""
        path.mkdir(parents=True, exist_ok=True)
        self._artifacts_dir = path
    
    def clear(self) -> None:
        """Clear all context data."""
        with self._lock:
            self._data.clear()
            self._metadata.clear()
    
    def snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of the current context."""
        with self._lock:
            return {
                "data": self._data.copy(),
                "metadata": self._metadata.copy(),
                "artifacts_dir": str(self._artifacts_dir) if self._artifacts_dir else None
            }
    
    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore context from a snapshot."""
        with self._lock:
            self._data = snapshot.get("data", {}).copy()
            self._metadata = snapshot.get("metadata", {}).copy()
            if snapshot.get("artifacts_dir"):
                self._artifacts_dir = Path(snapshot["artifacts_dir"])