# app/services/pipeline/engine.py
import asyncio
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from app.logger import get_logger
from app.services.pipeline.models import (
    PipelineRun,
    PipelineStage,
    RunLog,
    StageStatus,
    StageType,
)
from app.utils.decorators import timed

logger = get_logger(__name__)


class PipelineEngine:
    """DAG-based pipeline execution engine.
    
    Consumes: logger, utils
    Coordinates: datasets service, feature_store service, ml service
    """
    
    # Registry of stage executors - maps StageType to execution function
    _executors: Dict[StageType, Callable] = {}
    
    @classmethod
    def register_executor(cls, stage_type: StageType, executor: Callable):
        """Register a stage executor function."""
        cls._executors[stage_type] = executor
        logger.debug(f"Registered executor for stage: {stage_type.value}")
    
    @classmethod
    @timed
    async def execute_pipeline(
        cls,
        pipeline_run: PipelineRun,
        context: Dict[str, Any],
        stage_callbacks: Optional[Dict[str, Callable]] = None,
    ) -> PipelineRun:
        """Execute all stages in dependency order."""
        pipeline_run.status = StageStatus.RUNNING
        pipeline_run.started_at = datetime.now(timezone.utc)
        
        cls._log(pipeline_run, "info", None, f"Pipeline execution started: {pipeline_run.pipeline_name}")
        
        try:
            # Build execution graph
            completed: Dict[str, PipelineStage] = {}
            failed: Dict[str, PipelineStage] = {}
            remaining = list(pipeline_run.stages)
            
            while remaining:
                # Find stages whose dependencies are all completed
                ready_stages = []
                still_waiting = []
                
                for stage in remaining:
                    deps_met = all(
                        dep in completed and completed[dep].status == StageStatus.COMPLETED
                        for dep in stage.config.depends_on
                    )
                    # Also check if any dependency failed and on_failure is "stop"
                    dep_failed = any(
                        dep in failed
                        for dep in stage.config.depends_on
                    )
                    
                    if dep_failed and stage.config.on_failure == "stop":
                        stage.status = StageStatus.SKIPPED
                        cls._log(
                            pipeline_run, "warning", stage.name,
                            f"Skipped due to failed dependency"
                        )
                        continue
                    
                    if deps_met:
                        ready_stages.append(stage)
                    else:
                        still_waiting.append(stage)
                
                if not ready_stages and still_waiting:
                    # Circular dependency or all remaining depend on failed stages
                    for stage in still_waiting:
                        stage.status = StageStatus.SKIPPED
                    break
                
                # Execute ready stages concurrently
                tasks = []
                for stage in ready_stages:
                    task = cls._execute_stage(stage, context, pipeline_run)
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for stage, result in zip(ready_stages, results):
                    if isinstance(result, Exception):
                        stage.status = StageStatus.FAILED
                        stage.error_message = str(result)
                        failed[stage.name] = stage
                        cls._log(
                            pipeline_run, "error", stage.name,
                            f"Stage failed: {result}"
                        )
                        
                        # Check if we should stop the entire pipeline
                        if stage.config.on_failure == "stop":
                            pipeline_run.status = StageStatus.FAILED
                            cls._log(
                                pipeline_run, "error", None,
                                f"Pipeline failed at stage: {stage.name}"
                            )
                            # Skip remaining stages
                            for s in still_waiting:
                                s.status = StageStatus.SKIPPED
                            remaining = []
                            break
                    else:
                        completed[stage.name] = stage
                        # Update context with stage outputs
                        if result:
                            context.update(result)
                
                remaining = still_waiting
            
            # Determine final status
            if pipeline_run.status != StageStatus.FAILED:
                if all(s.status == StageStatus.COMPLETED for s in pipeline_run.stages):
                    pipeline_run.status = StageStatus.COMPLETED
                elif any(s.status == StageStatus.FAILED for s in pipeline_run.stages):
                    pipeline_run.status = StageStatus.FAILED
                else:
                    pipeline_run.status = StageStatus.COMPLETED  # Some skipped is ok
            
        except Exception as e:
            pipeline_run.status = StageStatus.FAILED
            cls._log(pipeline_run, "error", None, f"Pipeline execution error: {e}")
            logger.exception(f"Pipeline execution failed: {pipeline_run.pipeline_name}")
        
        finally:
            pipeline_run.completed_at = datetime.now(timezone.utc)
            if pipeline_run.started_at:
                pipeline_run.duration_seconds = (
                    pipeline_run.completed_at - pipeline_run.started_at
                ).total_seconds()
            
            cls._log(
                pipeline_run, "info", None,
                f"Pipeline execution finished: {pipeline_run.status.value} "
                f"({pipeline_run.duration_seconds:.1f}s)"
            )
        
        return pipeline_run
    
    @classmethod
    async def _execute_stage(
        cls,
        stage: PipelineStage,
        context: Dict[str, Any],
        pipeline_run: PipelineRun,
    ) -> Optional[Dict[str, Any]]:
        """Execute a single pipeline stage."""
        stage.status = StageStatus.RUNNING
        stage.started_at = datetime.now(timezone.utc)
        
        cls._log(pipeline_run, "info", stage.name, f"Stage started: {stage.stage_type.value}")
        
        executor = cls._executors.get(stage.stage_type)
        if executor is None:
            raise ValueError(f"No executor registered for stage type: {stage.stage_type.value}")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                executor(stage, context, pipeline_run),
                timeout=stage.config.timeout_seconds,
            )
            
            stage.status = StageStatus.COMPLETED
            stage.completed_at = datetime.now(timezone.utc)
            stage.duration_seconds = (
                stage.completed_at - stage.started_at
            ).total_seconds()
            
            cls._log(
                pipeline_run, "info", stage.name,
                f"Stage completed in {stage.duration_seconds:.1f}s"
            )
            
            return result
            
        except asyncio.TimeoutError:
            stage.status = StageStatus.FAILED
            stage.error_message = f"Timeout after {stage.config.timeout_seconds}s"
            raise
            
        except Exception as e:
            stage.status = StageStatus.FAILED
            stage.error_message = str(e)
            stage.completed_at = datetime.now(timezone.utc)
            raise
    
    @classmethod
    def _log(
        cls,
        run: PipelineRun,
        level: str,
        stage_name: Optional[str],
        message: str,
        metadata: Optional[Dict] = None,
    ):
        """Add a log entry to the run."""
        log_entry = RunLog(
            timestamp=datetime.now(timezone.utc),
            level=level,
            stage_name=stage_name,
            message=message,
            metadata=metadata or {},
        )
        run.logs.append(log_entry)
        
        # Also log to the system logger
        log_fn = getattr(logger, level, logger.info)
        log_fn(f"[Pipeline:{run.pipeline_name}] [{stage_name or 'system'}] {message}")