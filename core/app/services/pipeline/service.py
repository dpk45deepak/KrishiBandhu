# app/services/pipeline/service.py
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from app.logger import get_logger
from app.services.pipeline.engine import PipelineEngine
from app.services.pipeline.models import (
    PipelineConfig,
    PipelineCreate,
    PipelineResponse,
    PipelineRun,
    PipelineStage,
    PipelineStatus,
    StageConfig,
    StageStatus,
)
from app.utils.decorators import timed

logger = get_logger(__name__)


class PipelineService:
    """Pipeline management service.
    
    Consumes: logger, utils
    Coordinates: datasets service, feature_store service, ml service
    """
    
    def __init__(self):
        self._pipelines: Dict[str, PipelineResponse] = {}
        self._runs: Dict[str, List[PipelineRun]] = {}
        self._active_runs: Dict[str, asyncio.Task] = {}
    
    @timed
    async def create_pipeline(
        self, pipeline_data: PipelineCreate, user_id: str
    ) -> PipelineResponse:
        """Create a new pipeline definition."""
        pipeline_id = uuid4()
        
        # Validate stages
        stage_names = set()
        for stage in pipeline_data.config.stages:
            if stage.stage_type.value in stage_names:
                raise ValueError(f"Duplicate stage name: {stage.stage_type.value}")
            stage_names.add(stage.stage_type.value)
        
        # Validate dependencies reference existing stages
        for stage in pipeline_data.config.stages:
            for dep in stage.depends_on:
                if dep not in stage_names:
                    raise ValueError(f"Stage '{stage.stage_type.value}' depends on unknown stage: {dep}")
        
        pipeline = PipelineResponse(
            id=pipeline_id,
            name=pipeline_data.name,
            description=pipeline_data.description,
            status=PipelineStatus.DRAFT,
            config=pipeline_data.config,
            dataset_id=pipeline_data.dataset_id,
            created_by=user_id,
        )
        
        self._pipelines[str(pipeline_id)] = pipeline
        self._runs[str(pipeline_id)] = []
        
        logger.info(f"Pipeline created: {pipeline.name} with {len(pipeline_data.config.stages)} stages")
        return pipeline
    
    @timed
    async def run_pipeline(
        self,
        pipeline_id: str,
        params: Optional[Dict[str, Any]] = None,
        triggered_by: Optional[str] = None,
        trigger_type: str = "manual",
    ) -> PipelineRun:
        """Execute a pipeline."""
        pipeline = self._get_pipeline(pipeline_id)
        
        if pipeline.status == PipelineStatus.ARCHIVED:
            raise ValueError("Cannot run archived pipeline")
        
        # Build run stages from config
        stages = []
        for stage_config in pipeline.config.stages:
            stage = PipelineStage(
                id=uuid4(),
                name=stage_config.stage_type.value,
                stage_type=stage_config.stage_type,
                config=stage_config,
            )
            stages.append(stage)
        
        run = PipelineRun(
            id=uuid4(),
            pipeline_id=pipeline.id,
            pipeline_name=pipeline.name,
            stages=stages,
            triggered_by=triggered_by,
            trigger_type=trigger_type,
            input_params=params or {},
        )
        
        # Build execution context
        context = {
            "pipeline_id": str(pipeline.id),
            "run_id": str(run.id),
            "dataset_id": pipeline.dataset_id,
            "params": params or {},
            "artifacts": {},
        }
        
        self._runs[pipeline_id].append(run)
        
        # Execute pipeline
        logger.info(f"Pipeline run started: {pipeline.name} (run={run.id})")
        run = await PipelineEngine.execute_pipeline(run, context)
        
        # Update pipeline stats
        pipeline.latest_run = run
        pipeline.run_count += 1
        if pipeline.run_count > 0:
            completed = sum(
                1 for r in self._runs[pipeline_id]
                if r.status == StageStatus.COMPLETED
            )
            pipeline.success_rate = completed / pipeline.run_count * 100
        
        pipeline.updated_at = datetime.now(timezone.utc)
        
        logger.info(
            f"Pipeline run finished: {pipeline.name} - {run.status.value} "
            f"({run.duration_seconds:.1f}s)"
        )
        
        return run
    
    @timed
    async def run_pipeline_async(
        self,
        pipeline_id: str,
        params: Optional[Dict[str, Any]] = None,
        triggered_by: Optional[str] = None,
    ) -> PipelineRun:
        """Start pipeline execution asynchronously and return immediately."""
        pipeline = self._get_pipeline(pipeline_id)
        
        # Build run with PENDING status
        stages = []
        for stage_config in pipeline.config.stages:
            stage = PipelineStage(
                id=uuid4(),
                name=stage_config.stage_type.value,
                stage_type=stage_config.stage_type,
                config=stage_config,
            )
            stages.append(stage)
        
        run = PipelineRun(
            id=uuid4(),
            pipeline_id=pipeline.id,
            pipeline_name=pipeline.name,
            stages=stages,
            triggered_by=triggered_by,
            trigger_type="api",
            input_params=params or {},
        )
        
        self._runs[str(pipeline_id)].append(run)
        
        # Start background execution
        context = {
            "pipeline_id": str(pipeline.id),
            "run_id": str(run.id),
            "dataset_id": pipeline.dataset_id,
            "params": params or {},
            "artifacts": {},
        }
        
        task = asyncio.create_task(
            PipelineEngine.execute_pipeline(run, context)
        )
        self._active_runs[str(run.id)] = task
        
        # Add callback to update pipeline on completion
        def on_complete(t: asyncio.Task):
            completed_run = t.result()
            pipeline.latest_run = completed_run
            pipeline.run_count += 1
            pipeline.updated_at = datetime.now(timezone.utc)
            self._active_runs.pop(str(run.id), None)
            logger.info(f"Async pipeline completed: {pipeline.name} - {completed_run.status.value}")
        
        task.add_done_callback(on_complete)
        
        logger.info(f"Pipeline started async: {pipeline.name} (run={run.id})")
        return run
    
    async def get_pipeline(self, pipeline_id: str) -> Optional[PipelineResponse]:
        """Get pipeline by ID."""
        return self._pipelines.get(pipeline_id)
    
    async def list_pipelines(
        self, status: Optional[PipelineStatus] = None
    ) -> List[PipelineResponse]:
        """List all pipelines."""
        pipelines = list(self._pipelines.values())
        if status:
            pipelines = [p for p in pipelines if p.status == status]
        return pipelines
    
    async def get_run(self, pipeline_id: str, run_id: str) -> Optional[PipelineRun]:
        """Get a specific pipeline run."""
        runs = self._runs.get(pipeline_id, [])
        for run in runs:
            if str(run.id) == run_id:
                return run
        return None
    
    async def list_runs(
        self, pipeline_id: str, limit: int = 20
    ) -> List[PipelineRun]:
        """List runs for a pipeline."""
        runs = self._runs.get(pipeline_id, [])
        return sorted(runs, key=lambda r: r.started_at or datetime.min, reverse=True)[:limit]
    
    async def cancel_run(self, pipeline_id: str, run_id: str) -> bool:
        """Cancel a running pipeline."""
        task = self._active_runs.get(run_id)
        if task and not task.done():
            task.cancel()
            logger.info(f"Pipeline run cancelled: {run_id}")
            return True
        return False
    
    async def update_status(self, pipeline_id: str, status: PipelineStatus) -> PipelineResponse:
        """Update pipeline status."""
        pipeline = self._get_pipeline(pipeline_id)
        pipeline.status = status
        pipeline.updated_at = datetime.now(timezone.utc)
        logger.info(f"Pipeline status updated: {pipeline.name} -> {status.value}")
        return pipeline
    
    async def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a pipeline."""
        if pipeline_id not in self._pipelines:
            return False
        
        # Cancel any active runs
        for run_id in list(self._active_runs.keys()):
            if run_id in self._runs.get(pipeline_id, []):
                await self.cancel_run(pipeline_id, run_id)
        
        pipeline = self._pipelines.pop(pipeline_id)
        self._runs.pop(pipeline_id, None)
        logger.info(f"Pipeline deleted: {pipeline.name}")
        return True
    
    def _get_pipeline(self, pipeline_id: str) -> PipelineResponse:
        """Get pipeline or raise."""
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline is None:
            raise ValueError(f"Pipeline not found: {pipeline_id}")
        return pipeline