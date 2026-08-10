# app/services/reports/router.py
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.logger import get_logger
from app.services.api.dependencies import get_current_user, require_permission
from app.services.auth.models import Permission
from app.services.reports.models import ReportRequest, ReportType, ExportFormat
from app.services.reports.service import ReportService

logger = get_logger(__name__)
router = APIRouter(prefix="/reports", tags=["Reports"])

_report_service = ReportService()


def get_report_service() -> ReportService:
    return _report_service


@router.post("/generate", response_model=dict)
async def generate_report(
    request: ReportRequest,
    current_user: dict = Depends(require_permission(Permission.REPORT_CREATE.value)),
    service: ReportService = Depends(get_report_service),
):
    """Generate a new report."""
    try:
        report = await service.generate_report(request, user_id=current_user["sub"])
        return {
            "id": str(report.id),
            "title": report.title,
            "report_type": report.report_type.value,
            "sections_count": len(report.sections),
            "summary": report.summary,
            "exports": report.exports,
            "generated_at": report.generated_at.isoformat(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=List[dict])
async def list_reports(
    report_type: Optional[ReportType] = Query(None),
    current_user: dict = Depends(require_permission(Permission.REPORT_READ.value)),
    service: ReportService = Depends(get_report_service),
):
    """List generated reports."""
    reports = await service.list_reports(report_type=report_type)
    return [
        {
            "id": str(r.id),
            "title": r.title,
            "report_type": r.report_type.value,
            "summary": r.summary,
            "exports": r.exports,
            "generated_at": r.generated_at.isoformat(),
        }
        for r in reports
    ]


@router.get("/{report_id}", response_model=dict)
async def get_report(
    report_id: str,
    format: Optional[str] = Query("json"),
    current_user: dict = Depends(require_permission(Permission.REPORT_READ.value)),
    service: ReportService = Depends(get_report_service),
):
    """Get a specific report."""
    report = await service.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Check if requested format is available
    if format in report.exports:
        from fastapi.responses import FileResponse
        return FileResponse(report.exports[format])
    
    return {
        "id": str(report.id),
        "title": report.title,
        "report_type": report.report_type.value,
        "summary": report.summary,
        "sections": [
            {
                "title": s.title,
                "description": s.description,
                "metrics": s.metrics,
                "insights": s.insights,
                "recommendations": s.recommendations,
            }
            for s in report.sections
        ],
        "exports": report.exports,
        "generated_at": report.generated_at.isoformat(),
    }