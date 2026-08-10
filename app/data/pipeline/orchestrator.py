"""Main pipeline orchestrator coordinating all components."""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import yaml
from .models import (
    PipelineConfig, StageConfig, StageType, StageStatus,
    PipelineEvent, ExecutionMode, StageMetadata
)
from .exceptions import (
    PipelineConfigurationError, StageExecutionError,
    ParallelExecutionError, CheckpointError
)
from .context import PipelineContext
from .registry import StageRegistry
from .hooks import HookManager, HookType
from .state import StateManager
from .monitor import PipelineMonitor
from .executor import StageExecutor
from .scheduler import PipelineScheduler
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn


class PipelineOrchestrator:
    """
    Main orchestrator for the data pipeline.
    Coordinates all stages and handles the complete execution lifecycle.
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        config_dict: Optional[Dict[str, Any]] = None
    ):
        self.config = self._load_config(config_path, config_dict)
        self.context = PipelineContext()
        self.registry = StageRegistry()
        self.hooks = HookManager()
        self.state = StateManager()
        self.monitor = PipelineMonitor()
        self.scheduler = PipelineScheduler(self.config.execution_mode)
        self.executor = StageExecutor(
            self.registry,
            self.hooks,
            self.state,
            self.monitor
        )
        self.console = Console()
        
        # Set context artifacts directory
        self.context.artifacts_dir = self.config.artifact_dir
        
        # Initialize state
        self.state.initialize(self.config)
        
        # Build scheduler graph
        self.scheduler.build_graph(self.config.stages)
    
    def _load_config(
        self,
        config_path: Optional[Path],
        config_dict: Optional[Dict[str, Any]]
    ) -> PipelineConfig:
        """Load pipeline configuration from file or dict."""
        if config_path:
            with open(config_path) as f:
                if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
        elif config_dict:
            config_data = config_dict
        else:
            raise PipelineConfigurationError("No configuration provided")
        
        # Validate and create config
        return PipelineConfig(**config_data)
    
    def register_stages(self, stage_registry: Dict[StageType, callable]) -> None:
        """Register stage implementations."""
        for stage_type, func in stage_registry.items():
            self.registry.register(stage_type)(func)
    
    def run(
        self,
        resume: bool = False,
        checkpoint_path: Optional[Path] = None,
        dry_run: bool = False
    ) -> bool:
        """
        Run the pipeline.
        
        Args:
            resume: Resume from last checkpoint
            checkpoint_path: Specific checkpoint to resume from
            dry_run: Validate pipeline without executing
            
        Returns:
            True if pipeline completed successfully, False otherwise
        """
        try:
            # Initialize or resume
            if resume:
                self._resume_pipeline(checkpoint_path)
            else:
                self._initialize_pipeline()
            
            # Dry run
            if dry_run:
                return self._dry_run()
            
            # Get execution order
            execution_order = self.scheduler.get_execution_order()
            
            # Log execution plan
            self._log_execution_plan(execution_order)
            
            # Execute pipeline
            success = self._execute_pipeline(execution_order)
            
            # Generate reports
            self._generate_reports()
            
            return success
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self._handle_critical_error(e)
            return False
        finally:
            self._shutdown()
    
    def _initialize_pipeline(self) -> None:
        """Initialize pipeline state and emit events."""
        self.state.update_status(StageStatus.RUNNING)
        
        event = PipelineEvent(
            event_type="pipeline_start",
            pipeline_id=self.state.state.pipeline_id
        )
        self.monitor.record_event(event)
        self.hooks.trigger(HookType.PRE_EXECUTION, event)
        
        logger.info(f"Starting pipeline: {self.config.name} (ID: {self.state.state.pipeline_id})")
    
    def _resume_pipeline(self, checkpoint_path: Optional[Path] = None) -> None:
        """Resume pipeline from checkpoint."""
        try:
            self.state.load_checkpoint(checkpoint_path)
            logger.info(f"Resuming pipeline from checkpoint: {self.state.checkpoint_id}")
            
            # Restore context from checkpoint
            if self.state.state.shared_context:
                self.context.restore(self.state.state.shared_context)
            
            # Emit resume event
            event = PipelineEvent(
                event_type="pipeline_resume",
                pipeline_id=self.state.state.pipeline_id,
                data={"checkpoint_id": self.state.checkpoint_id}
            )
            self.monitor.record_event(event)
            
        except CheckpointError as e:
            logger.error(f"Failed to resume: {e}")
            raise
    
    def _dry_run(self) -> bool:
        """Validate pipeline without executing."""
        logger.info("Performing dry run...")
        
        # Validate all stages are registered
        missing_stages = []
        for stage in self.config.stages:
            if stage.enabled and not self.registry.is_registered(stage.name):
                missing_stages.append(stage.name)
        
        if missing_stages:
            logger.error(f"Missing stage implementations: {missing_stages}")
            return False
        
        # Validate dependencies
        try:
            order = self.scheduler.get_execution_order()
            logger.info(f"Execution order validated: {len(order)} stages")
            
            # Print execution plan
            self._log_execution_plan(order)
            
            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def _execute_pipeline(self, execution_order: List[List[StageType]]) -> bool:
        """
        Execute pipeline stages according to the execution order.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task(
                "[cyan]Executing pipeline...",
                total=sum(len(level) for level in execution_order)
            )
            
            for level_idx, level in enumerate(execution_order):
                if self.mode == ExecutionMode.PARALLEL and len(level) > 1:
                    # Execute stages in parallel
                    stage_configs = {
                        stage_type: self._get_stage_config(stage_type)
                        for stage_type in level
                    }
                    
                    results = self.executor.execute_parallel(
                        stage_configs,
                        self.context
                    )
                    
                    # Check results
                    for stage_type, result in results.items():
                        if not result.success:
                            logger.error(f"Stage {stage_type} failed")
                            self._handle_stage_failure(stage_type, result)
                            self.state.update_status(StageStatus.FAILED)
                            
                            # Emit pipeline failure
                            event = PipelineEvent(
                                event_type="pipeline_failed",
                                pipeline_id=self.state.state.pipeline_id,
                                data={"failed_stage": stage_type}
                            )
                            self.monitor.record_event(event)
                            self.hooks.trigger(HookType.ON_ERROR, event)
                            
                            return False
                        
                        # Update progress
                        progress.advance(task, 1)
                    
                    # Save checkpoint after parallel execution
                    if self.config.enable_checkpointing:
                        self.state.save_checkpoint()
                    
                else:
                    # Execute stages sequentially
                    for stage_type in level:
                        # Check if stage should be executed based on condition
                        stage_config = self._get_stage_config(stage_type)
                        
                        if not self._should_execute_stage(stage_type, stage_config):
                            self.state.update_stage_metadata(
                                stage_type,
                                status=StageStatus.SKIPPED
                            )
                            progress.advance(task, 1)
                            continue
                        
                        # Execute stage
                        result = self.executor.execute_stage(
                            stage_type,
                            stage_config,
                            self.context
                        )
                        
                        # Check result
                        if not result.success:
                            logger.error(f"Stage {stage_type} failed: {result.error}")
                            self._handle_stage_failure(stage_type, result)
                            self.state.update_status(StageStatus.FAILED)
                            
                            # Emit pipeline failure
                            event = PipelineEvent(
                                event_type="pipeline_failed",
                                pipeline_id=self.state.state.pipeline_id,
                                data={"failed_stage": stage_type}
                            )
                            self.monitor.record_event(event)
                            self.hooks.trigger(HookType.ON_ERROR, event)
                            
                            return False
                        
                        # Save output to context for later stages
                        if result.output:
                            self.context.set(f"{stage_type.value}_output", result.output)
                        
                        # Update progress
                        progress.advance(task, 1)
                        
                        # Save checkpoint
                        if self.config.enable_checkpointing:
                            self.state.save_checkpoint()
            
            # Pipeline completed successfully
            self.state.update_status(StageStatus.COMPLETED)
            
            # Emit completion event
            event = PipelineEvent(
                event_type="pipeline_complete",
                pipeline_id=self.state.state.pipeline_id,
                data={"duration": self._get_pipeline_duration()}
            )
            self.monitor.record_event(event)
            self.hooks.trigger(HookType.POST_EXECUTION, event)
            
            logger.success("Pipeline completed successfully!")
            return True
    
    def _get_stage_config(self, stage_type: StageType) -> StageConfig:
        """Get configuration for a specific stage."""
        for stage in self.config.stages:
            if stage.name == stage_type:
                return stage
        raise ValueError(f"Stage {stage_type} not found in configuration")
    
    def _should_execute_stage(self, stage_type: StageType, config: StageConfig) -> bool:
        """Determine if a stage should be executed based on conditions."""
        if not config.enabled:
            return False
        
        # Check if already completed (for resume)
        metadata = self.state.get_stage_metadata(stage_type)
        if metadata and metadata.status == StageStatus.COMPLETED:
            return False
        
        # Check condition
        if config.condition:
            # Evaluate condition expression
            # For now, just check if it's a simple key in context
            condition = config.condition.strip()
            if condition.startswith("context."):
                key = condition[8:]  # Remove 'context.'
                if not self.context.get(key):
                    return False
        
        return True
    
    def _handle_stage_failure(self, stage_type: StageType, result) -> None:
        """Handle a failed stage."""
        logger.error(f"Stage {stage_type} failed: {result.error}")
        
        # Update metadata
        self.state.update_stage_metadata(
            stage_type,
            status=StageStatus.FAILED,
            error=str(result.error)
        )
    
    def _log_execution_plan(self, execution_order: List[List[StageType]]) -> None:
        """Log the execution plan."""
        table = Table(title="Pipeline Execution Plan")
        table.add_column("Level", style="cyan")
        table.add_column("Stages", style="green")
        
        for idx, level in enumerate(execution_order):
            table.add_row(
                str(idx + 1),
                ", ".join(stage.value for stage in level)
            )
        
        self.console.print(table)
        
        # Log details
        for level in execution_order:
            for stage_type in level:
                deps = self.scheduler.get_dependencies(stage_type)
                if deps:
                    logger.debug(f"{stage_type} depends on: {', '.join(d.value for d in deps)}")
    
    def _generate_reports(self) -> None:
        """Generate pipeline reports."""
        # Metrics report
        metrics_report = self.monitor.generate_report()
        
        # Save report
        report_path = self.monitor.save_report()
        logger.info(f"Metrics report saved to: {report_path}")
        
        # Save state summary
        summary = self._generate_summary()
        summary_path = self.config.artifact_dir / "pipeline_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Pipeline summary saved to: {summary_path}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate pipeline summary."""
        state = self.state.state
        if not state:
            return {}
        
        return {
            "pipeline_id": state.pipeline_id,
            "name": self.config.name,
            "version": self.config.version,
            "status": state.status,
            "start_time": state.start_time.isoformat() if state.start_time else None,
            "end_time": state.end_time.isoformat() if state.end_time else None,
            "duration": self._get_pipeline_duration(),
            "stages": {
                stage_type.value: {
                    "status": metadata.status,
                    "duration": metadata.duration,
                    "retry_count": metadata.retry_count,
                    "error": metadata.error
                }
                for stage_type, metadata in state.stages.items()
            },
            "metrics": self.monitor.get_pipeline_metrics()
        }
    
    def _get_pipeline_duration(self) -> Optional[float]:
        """Get total pipeline duration."""
        state = self.state.state
        if not state or not state.start_time:
            return None
        
        end_time = state.end_time or datetime.now()
        return (end_time - state.start_time).total_seconds()
    
    def _handle_critical_error(self, error: Exception) -> None:
        """Handle a critical pipeline error."""
        self.state.update_status(StageStatus.FAILED)
        
        event = PipelineEvent(
            event_type="pipeline_failed",
            pipeline_id=self.state.state.pipeline_id,
            data={"error": str(error)}
        )
        self.monitor.record_event(event)
        self.hooks.trigger(HookType.ON_ERROR, event)
    
    def _shutdown(self) -> None:
        """Shutdown the pipeline orchestrator."""
        self.executor.shutdown()
        logger.info("Pipeline orchestrator shutdown complete")
    
    @property
    def mode(self) -> ExecutionMode:
        """Get execution mode."""
        return self.config.execution_mode
    
    @mode.setter
    def mode(self, value: ExecutionMode) -> None:
        """Set execution mode."""
        self.config.execution_mode = value
        self.scheduler = PipelineScheduler(value)
        self.scheduler.build_graph(self.config.stages)