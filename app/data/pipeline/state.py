"""Pipeline state management with checkpointing support."""

import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
from .models import PipelineState, StageType, StageMetadata, StageStatus, PipelineConfig
from .exceptions import CheckpointError


class StateManager:
    """
    Manages pipeline state with checkpoint persistence.
    Supports saving/loading state for recovery.
    """
    
    def __init__(self, checkpoint_dir: Path = Path(".pipeline_checkpoints")):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._state: Optional[PipelineState] = None
        self._checkpoint_id: Optional[str] = None
    
    def initialize(
        self,
        config: PipelineConfig,
        pipeline_id: Optional[str] = None
    ) -> PipelineState:
        """Initialize a new pipeline state."""
        if pipeline_id is None:
            pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        stages = {
            stage.name: StageMetadata(stage_type=stage.name)
            for stage in config.stages
            if stage.enabled
        }
        
        self._state = PipelineState(
            pipeline_id=pipeline_id,
            config=config,
            stages=stages
        )
        self._checkpoint_id = None
        
        return self._state
    
    def load_checkpoint(self, checkpoint_path: Optional[Path] = None) -> PipelineState:
        """Load pipeline state from a checkpoint."""
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
            if not checkpoints:
                raise CheckpointError("No checkpoint found")
            checkpoint_path = checkpoints[-1]
        
        try:
            with open(checkpoint_path, "rb") as f:
                self._state = pickle.load(f)
            self._checkpoint_id = checkpoint_path.stem
            return self._state
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}")
    
    def save_checkpoint(self) -> Path:
        """Save current state as a checkpoint."""
        if self._state is None:
            raise CheckpointError("No state to checkpoint")
        
        checkpoint_id = f"checkpoint_{self._state.pipeline_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        
        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(self._state, f)
            self._checkpoint_id = checkpoint_id
            self._state.checkpoint = checkpoint_path
            return checkpoint_path
        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}")
    
    def update_stage_metadata(
        self,
        stage_type: StageType,
        **kwargs
    ) -> None:
        """Update metadata for a specific stage."""
        if self._state is None:
            return
        
        if stage_type not in self._state.stages:
            self._state.stages[stage_type] = StageMetadata(stage_type=stage_type)
        
        metadata = self._state.stages[stage_type]
        for key, value in kwargs.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
    
    def get_stage_metadata(self, stage_type: StageType) -> Optional[StageMetadata]:
        """Get metadata for a specific stage."""
        if self._state is None:
            return None
        return self._state.stages.get(stage_type)
    
    def update_status(self, status: StageStatus) -> None:
        """Update overall pipeline status."""
        if self._state:
            self._state.status = status
            if status == StageStatus.COMPLETED:
                self._state.end_time = datetime.now()
    
    def get_completed_stages(self) -> set:
        """Get set of completed stage types."""
        if self._state is None:
            return set()
        
        return {
            stage_type
            for stage_type, metadata in self._state.stages.items()
            if metadata.status == StageStatus.COMPLETED
        }
    
    def get_failed_stages(self) -> set:
        """Get set of failed stage types."""
        if self._state is None:
            return set()
        
        return {
            stage_type
            for stage_type, metadata in self._state.stages.items()
            if metadata.status == StageStatus.FAILED
        }
    
    def get_pending_stages(self) -> set:
        """Get set of pending stage types."""
        if self._state is None:
            return set()
        
        return {
            stage_type
            for stage_type, metadata in self._state.stages.items()
            if metadata.status in (StageStatus.PENDING, StageStatus.RETRYING)
        }
    
    @property
    def state(self) -> Optional[PipelineState]:
        """Get current state."""
        return self._state
    
    @property
    def checkpoint_id(self) -> Optional[str]:
        """Get current checkpoint ID."""
        return self._checkpoint_id