"""Pipeline stage executor with retry logic and error handling."""

import time
from typing import Optional, Callable, Dict, Any
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from .models import (
    StageType, StageStatus, StageMetadata, PipelineEvent,
    StageResult, StageConfig
)
from .exceptions import StageExecutionError, RetryExhaustedError
from .context import PipelineContext
from .registry import StageRegistry
from .hooks import HookManager, HookType
from .state import StateManager
from .monitor import PipelineMonitor
from loguru import logger


class StageExecutor:
    """
    Executes pipeline stages with retry logic.
    Handles stage lifecycle and error recovery.
    """
    
    def __init__(
        self,
        registry: StageRegistry,
        hook_manager: HookManager,
        state_manager: StateManager,
        monitor: PipelineMonitor
    ):
        self.registry = registry
        self.hooks = hook_manager
        self.state = state_manager
        self.monitor = monitor
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def execute_stage(
        self,
        stage_type: StageType,
        config: StageConfig,
        context: PipelineContext,
        retry_count: int = 0
    ) -> StageResult:
        """
        Execute a single stage with retry logic.
        """
        # Check if stage is enabled
        if not config.enabled:
            logger.info(f"Stage {stage_type} is disabled, skipping")
            return StageResult(
                success=True,
                stage_type=stage_type,
                metadata=StageMetadata(
                    stage_type=stage_type,
                    status=StageStatus.SKIPPED
                )
            )
        
        # Update state
        self.state.update_stage_metadata(
            stage_type,
            status=StageStatus.RUNNING,
            start_time=datetime.now(),
            retry_count=retry_count
        )
        
        # Emit pre-stage event
        event = PipelineEvent(
            event_type="stage_start",
            pipeline_id=self.state.state.pipeline_id,
            stage_type=stage_type,
            data={"config": config.model_dump()}
        )
        self.monitor.record_event(event)
        self.hooks.trigger(HookType.PRE_STAGE, event, stage_type)
        
        try:
            # Get stage function
            stage_func = self.registry.get_stage(stage_type)
            
            # Execute with timeout
            result = self._execute_with_timeout(
                stage_func,
                context,
                config.config,
                config.timeout
            )
            
            # Update metadata
            end_time = datetime.now()
            duration = (end_time - event.timestamp).total_seconds()
            
            self.state.update_stage_metadata(
                stage_type,
                status=StageStatus.COMPLETED,
                end_time=end_time,
                duration=duration,
                artifacts=result.get("artifacts", {}),
                metrics=result.get("metrics", {}),
                output_path=result.get("output_path")
            )
            
            # Emit success event
            complete_event = PipelineEvent(
                event_type="stage_complete",
                pipeline_id=self.state.state.pipeline_id,
                stage_type=stage_type,
                data={"duration": duration, "result": result}
            )
            self.monitor.record_event(complete_event)
            self.hooks.trigger(HookType.POST_STAGE, complete_event, stage_type)
            
            return StageResult(
                success=True,
                stage_type=stage_type,
                metadata=self.state.get_stage_metadata(stage_type),
                output=result
            )
            
        except Exception as e:
            # Handle failure
            return self._handle_stage_failure(
                stage_type,
                config,
                e,
                retry_count,
                context
            )
    
    def _execute_with_timeout(
        self,
        func: Callable,
        context: PipelineContext,
        config: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        """Execute function with timeout."""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(func, context, config)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise StageExecutionError(f"Stage execution timed out after {timeout}s")
    
    def _handle_stage_failure(
        self,
        stage_type: StageType,
        config: StageConfig,
        error: Exception,
        retry_count: int,
        context: PipelineContext
    ) -> StageResult:
        """Handle stage failure with retry logic."""
        
        # Check if we should retry
        if retry_count < config.retry_count:
            # Emit retry event
            retry_event = PipelineEvent(
                event_type="stage_retry",
                pipeline_id=self.state.state.pipeline_id,
                stage_type=stage_type,
                data={"attempt": retry_count + 1, "error": str(error)}
            )
            self.monitor.record_event(retry_event)
            self.hooks.trigger(HookType.ON_RETRY, retry_event, stage_type)
            
            logger.warning(
                f"Stage {stage_type} failed, retrying ({retry_count + 1}/{config.retry_count}): {error}"
            )
            
            # Wait before retry
            time.sleep(config.retry_delay)
            
            # Retry
            return self.execute_stage(
                stage_type,
                config,
                context,
                retry_count + 1
            )
        
        # No more retries, mark as failed
        end_time = datetime.now()
        
        self.state.update_stage_metadata(
            stage_type,
            status=StageStatus.FAILED,
            end_time=end_time,
            error=str(error)
        )
        
        # Emit failure event
        fail_event = PipelineEvent(
            event_type="stage_failed",
            pipeline_id=self.state.state.pipeline_id,
            stage_type=stage_type,
            data={"error": str(error), "retry_count": retry_count}
        )
        self.monitor.record_event(fail_event)
        self.hooks.trigger(HookType.ON_ERROR, fail_event, stage_type)
        
        return StageResult(
            success=False,
            stage_type=stage_type,
            metadata=self.state.get_stage_metadata(stage_type),
            error=error
        )
    
    def execute_parallel(
        self,
        stages: Dict[StageType, StageConfig],
        context: PipelineContext
    ) -> Dict[StageType, StageResult]:
        """Execute multiple stages in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(stages)) as executor:
            future_to_stage = {
                executor.submit(
                    self.execute_stage,
                    stage_type,
                    config,
                    context,
                    0
                ): stage_type
                for stage_type, config in stages.items()
            }
            
            for future in as_completed(future_to_stage):
                stage_type = future_to_stage[future]
                try:
                    results[stage_type] = future.result()
                except Exception as e:
                    results[stage_type] = StageResult(
                        success=False,
                        stage_type=stage_type,
                        metadata=StageMetadata(
                            stage_type=stage_type,
                            status=StageStatus.FAILED,
                            error=str(e)
                        ),
                        error=e
                    )
        
        return results
    
    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown(wait=True)