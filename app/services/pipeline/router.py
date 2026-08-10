# app/services/pipeline/router.py
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.logger import get_logger
from app.services.api.dependencies import get_current_user, require_permission
from app.services.auth.models import Permission
from app.services.pipeline.models import (
    PipelineCreate,
    PipelineResponse,
    PipelineRun,
    PipelineStatus,
)
from app.services.pipeline.service import PipelineService

logger = get_logger(__name__)
router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

_pipeline_service = PipelineService()


def get_pipeline_service() -> PipelineService:
    return _pipeline_service


@router.post("", response_model=PipelineResponse, status_code=status.HTTP_201_CREATED)
async def create_pipeline(
    pipeline: PipelineCreate,
    current_user: dict = Depends(require_permission(Permission.PIPELINE_WRITE.value)),
    service: PipelineService = Depends(get_pipeline_service),
):
    """Create a new pipeline definition."""
    try:
        return await service.create_pipeline(pipeline, user_id=current_user["sub"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=List[PipelineResponse])
async def list_pipelines(
    status: Optional[PipelineStatus] = Query(None),
    current_user: dict = Depends(require_permission(Permission.PIPELINE_READ.value)),
    service: PipelineService = Depends(get_pipeline_service),
):
    """List all pipelines."""
    return await service.list_pipelines(status=status)


@router.get("/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(
    pipeline_id: str,
    current_user: dict = Depends(require_permission(Permission.PIPELINE_READ.value)),
    service: PipelineService = Depends(get_pipeline_service),
):
    """Get pipeline details."""
    pipeline = await service.get_pipeline(pipeline_id)
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return pipeline


@router.post("/{pipeline_id}/run", response_model=PipelineRun)
async def run_pipeline(
    pipeline_id: str,
    params: Optional[Dict[str, Any]] = None,
    current_user: dict = Depends(require_permission(Permission.PIPELINE_EXECUTE.value)),
    service: PipelineService = Depends(get_pipeline_service),
):
    """Execute a pipeline (synchronous)."""
    try:
        return await service.run_pipeline(
            pipeline_id,
            params=params,
            triggered_by=current_user["username"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{pipeline_id}/run-async", response_model=PipelineRun)
async def run_pipeline_async(
    pipeline_id: str,
    params: Optional[Dict[str, Any]] = None,
    current_user: dict = Depends(require_permission(Permission.PIPELINE_EXECUTE.value)),
    service: PipelineService = Depends(get_pipeline_service),
):
    """Start pipeline execution asynchronously."""
    try:
        return await service.run_pipeline_async(
            pipeline_id,
            params=params,
            triggered_by=current_user["username"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{pipeline_id}/runs", response_model=List[PipelineRun])
async def list_runs(
    pipeline_id: str,
    limit: int = Query(20, le=100),
    current_user: dict = Depends(require_permission(Permission.PIPELINE_READ.value)),
    service: PipelineService = Depends(get_pipeline_service),
):
    """List runs for a pipeline."""
    return await service.list_runs(pipeline_id, limit=limit)


@router.get("/{pipeline_id}/runs/{run_id}", response_model=PipelineRun)
async def get_run(
    pipeline_id: str,
    run_id: str,
    current_user: dict = Depends(require_permission(Permission.PIPELINE_READ.value)),
    service: PipelineService = Depends(get_pipeline_service),
):
    """Get a specific pipeline run."""
    run = await service.get_run(pipeline_id, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post("/{pipeline_id}/runs/{run_id}/cancel")
async def cancel_run(
    pipeline_id: str,
    run_id: str,
    current_user: dict = Depends(require_permission(Permission.PIPELINE_EXECUTE.value)),
    service: PipelineService = Depends(get_pipeline_service),
):
    """Cancel a running pipeline."""
    cancelled = await service.cancel_run(pipeline_id, run_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail="Run not found or already completed")
    return {"message": "Pipeline run cancelled"}


@router.patch("/{pipeline_id}/status", response_model=PipelineResponse)
async def update_status(
    pipeline_id: str,
    status: PipelineStatus,
    current_user: dict = Depends(require_permission(Permission.PIPELINE_WRITE.value)),
    service: PipelineService = Depends(get_pipeline_service),
):
    """Update pipeline status."""
    try:
        return await service.update_status(pipeline_id, status)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{pipeline_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_pipeline(
    pipeline_id: str,
    current_user: dict = Depends(require_permission(Permission.PIPELINE_WRITE.value)),
    service: PipelineService = Depends(get_pipeline_service),
):
    """Delete a pipeline."""
    deleted = await service.delete_pipeline(pipeline_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Pipeline not found")