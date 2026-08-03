# app/services/reports/service.py
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from app.config import settings
from app.logger import get_logger
from app.services.reports.models import (
    ExportFormat,
    ReportRequest,
    ReportResponse,
    ReportSection,
    ReportTemplate,
    ReportType,
    VisualizationConfig,
)
from app.utils.decorators import timed

logger = get_logger(__name__)


class ReportService:
    """Report generation service.
    
    Consumes:
    - data.Profiler: dataset statistics for profile reports
    - data.Validator: data quality metrics
    - ml.evaluation: model performance metrics
    - ml.explainability: feature importance
    - feature_store: feature statistics
    - pipeline: execution logs and metrics
    - config: report storage paths
    - logger: structured logging
    """
    
    # Built-in templates
    TEMPLATES = {
        ReportType.DATASET_PROFILE: ReportTemplate(
            id=UUID("00000000-0000-0000-0000-000000000001"),
            name="Dataset Profile Report",
            report_type=ReportType.DATASET_PROFILE,
            sections=[
                ReportSection(
                    title="Dataset Overview",
                    description="Summary statistics and structure",
                    visualizations=[
                        VisualizationConfig(
                            chart_type="table",
                            title="Column Summary",
                            data_source="column_stats",
                        ),
                    ],
                ),
                ReportSection(
                    title="Missing Data Analysis",
                    description="Patterns and percentages of missing values",
                    visualizations=[
                        VisualizationConfig(
                            chart_type="heatmap",
                            title="Missing Values Matrix",
                            data_source="missing_patterns",
                        ),
                        VisualizationConfig(
                            chart_type="bar",
                            title="Missing Values by Column",
                            data_source="missing_counts",
                            x_column="column",
                            y_column="missing_percentage",
                        ),
                    ],
                ),
                ReportSection(
                    title="Distribution Analysis",
                    description="Statistical distributions of numeric features",
                    visualizations=[
                        VisualizationConfig(
                            chart_type="histogram",
                            title="Value Distributions",
                            data_source="distributions",
                        ),
                    ],
                ),
                ReportSection(
                    title="Correlation Analysis",
                    description="Feature correlation matrix",
                    visualizations=[
                        VisualizationConfig(
                            chart_type="heatmap",
                            title="Correlation Matrix",
                            data_source="correlations",
                        ),
                    ],
                ),
            ],
        ),
        ReportType.DATA_QUALITY: ReportTemplate(
            id=UUID("00000000-0000-0000-0000-000000000002"),
            name="Data Quality Report",
            report_type=ReportType.DATA_QUALITY,
            sections=[
                ReportSection(
                    title="Validation Summary",
                    description="Overall data quality assessment",
                    metrics={},
                ),
                ReportSection(
                    title="Quality Issues",
                    description="Detailed breakdown of data quality issues",
                    visualizations=[
                        VisualizationConfig(
                            chart_type="bar",
                            title="Issues by Severity",
                            data_source="validation_issues",
                            x_column="severity",
                            y_column="count",
                        ),
                    ],
                ),
            ],
        ),
        ReportType.MODEL_PERFORMANCE: ReportTemplate(
            id=UUID("00000000-0000-0000-0000-000000000003"),
            name="Model Performance Report",
            report_type=ReportType.MODEL_PERFORMANCE,
            sections=[
                ReportSection(
                    title="Performance Metrics",
                    description="Key model evaluation metrics",
                    metrics={},
                ),
                ReportSection(
                    title="Confusion Matrix",
                    description="Classification results breakdown",
                    visualizations=[
                        VisualizationConfig(
                            chart_type="heatmap",
                            title="Confusion Matrix",
                            data_source="confusion_matrix",
                        ),
                    ],
                ),
                ReportSection(
                    title="Feature Importance",
                    description="Top features driving predictions",
                    visualizations=[
                        VisualizationConfig(
                            chart_type="bar",
                            title="Feature Importance",
                            data_source="feature_importance",
                            x_column="feature",
                            y_column="importance",
                        ),
                    ],
                ),
                ReportSection(
                    title="Error Analysis",
                    description="Analysis of prediction errors",
                    visualizations=[
                        VisualizationConfig(
                            chart_type="scatter",
                            title="Actual vs Predicted",
                            data_source="predictions",
                            x_column="actual",
                            y_column="predicted",
                        ),
                    ],
                ),
            ],
        ),
        ReportType.PIPELINE_EXECUTION: ReportTemplate(
            id=UUID("00000000-0000-0000-0000-000000000004"),
            name="Pipeline Execution Report",
            report_type=ReportType.PIPELINE_EXECUTION,
            sections=[
                ReportSection(
                    title="Execution Summary",
                    description="Pipeline run overview and status",
                    metrics={},
                ),
                ReportSection(
                    title="Stage Durations",
                    description="Time spent in each pipeline stage",
                    visualizations=[
                        VisualizationConfig(
                            chart_type="bar",
                            title="Stage Execution Times",
                            data_source="stage_durations",
                            x_column="stage",
                            y_column="duration_seconds",
                        ),
                    ],
                ),
                ReportSection(
                    title="Execution Logs",
                    description="Key log entries from the run",
                ),
            ],
        ),
    }
    
    def __init__(self):
        self._reports: Dict[str, ReportResponse] = {}
        self._reports_path = Path(settings.REPORTS_STORAGE_PATH)
        self._reports_path.mkdir(parents=True, exist_ok=True)
    
    @timed
    async def generate_report(
        self, request: ReportRequest, user_id: str
    ) -> ReportResponse:
        """Generate a report based on request parameters.
        
        Routes to specialized generators based on report_type.
        """
        
        # Get template (custom or built-in)
        template = None
        if request.template_id:
            template = await self._get_custom_template(request.template_id)
        else:
            template = self.TEMPLATES.get(request.report_type)
        
        if template is None:
            raise ValueError(f"No template found for report type: {request.report_type.value}")
        
        # Generate report based on type
        if request.report_type == ReportType.DATASET_PROFILE:
            sections, summary = await self._generate_dataset_profile(request)
        elif request.report_type == ReportType.DATA_QUALITY:
            sections, summary = await self._generate_data_quality(request)
        elif request.report_type == ReportType.MODEL_PERFORMANCE:
            sections, summary = await self._generate_model_performance(request)
        elif request.report_type == ReportType.PIPELINE_EXECUTION:
            sections, summary = await self._generate_pipeline_execution(request)
        elif request.report_type == ReportType.FEATURE_ANALYSIS:
            sections, summary = await self._generate_feature_analysis(request)
        elif request.custom_sections:
            sections = request.custom_sections
            summary = "Custom report"
        else:
            sections = template.sections
            summary = f"Report: {template.name}"
        
        # Merge template sections with generated ones
        if not request.custom_sections:
            sections = await self._merge_sections(template.sections, sections)
        
        report = ReportResponse(
            id=uuid4(),
            report_type=request.report_type,
            title=f"{template.name} - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
            sections=sections,
            summary=summary,
            created_by=user_id,
        )
        
        # Export to requested formats
        exports = {}
        for fmt in request.export_formats:
            file_path = await self._export_report(report, fmt)
            exports[fmt.value] = file_path
        
        report.exports = exports
        self._reports[str(report.id)] = report
        
        logger.info(
            f"Report generated: {report.title} "
            f"({len(sections)} sections, {len(exports)} exports)"
        )
        
        return report
    
    @timed
    async def _generate_dataset_profile(
        self, request: ReportRequest
    ) -> tuple[List[ReportSection], str]:
        """Generate dataset profile report using existing data.Profiler."""
        from app.data import Profiler
        
        if not request.dataset_id:
            raise ValueError("dataset_id required for dataset profile report")
        
        # Get dataset path from dataset service
        from app.services.datasets.service import DatasetService
        ds_service = DatasetService()
        dataset = await ds_service.get_dataset(request.dataset_id)
        
        if dataset is None or dataset.current_version is None:
            raise ValueError(f"Dataset not found or has no version: {request.dataset_id}")
        
        # Profile the data
        profiler = Profiler()
        profile = await profiler.profile(dataset.current_version.file_path)
        
        sections = []
        
        # Overview section
        sections.append(ReportSection(
            title="Dataset Overview",
            description=f"Profile of dataset: {dataset.name}",
            metrics={
                "row_count": profile.row_count,
                "column_count": profile.column_count,
                "total_size_mb": round(profile.total_size_bytes / (1024 * 1024), 2),
                "missing_cells": sum(c.null_count for c in profile.columns),
            },
            insights=[
                f"Dataset contains {profile.row_count:,} rows and {profile.column_count} columns",
                f"Total size: {profile.total_size_bytes / (1024**2):.1f} MB",
            ],
        ))
        
        # Column statistics
        column_data = []
        for col in profile.columns:
            column_data.append({
                "column": col.name,
                "dtype": col.dtype,
                "count": col.count,
                "null_count": col.null_count,
                "null_percentage": round(col.null_percentage, 2),
                "unique_count": col.unique_count,
                "mean": round(col.mean, 4) if col.mean is not None else None,
                "std": round(col.std, 4) if col.std is not None else None,
                "min": col.min,
                "max": col.max,
            })
        
        sections.append(ReportSection(
            title="Column Summary",
            visualizations=[
                VisualizationConfig(
                    chart_type="table",
                    title="Column Statistics",
                    data_source="column_stats",
                ),
            ],
            raw_data={"column_stats": column_data},
        ))
        
        # Missing data
        if any(c.null_percentage > 0 for c in profile.columns):
            missing_data = [
                {"column": c.name, "missing_percentage": c.null_percentage}
                for c in profile.columns
                if c.null_percentage > 0
            ]
            sections.append(ReportSection(
                title="Missing Data Analysis",
                metrics={
                    "columns_with_missing": len(missing_data),
                    "total_missing_cells": sum(c.null_count for c in profile.columns),
                },
                visualizations=[
                    VisualizationConfig(
                        chart_type="bar",
                        title="Missing Values by Column",
                        data_source="missing_counts",
                        x_column="column",
                        y_column="missing_percentage",
                    ),
                ],
                raw_data={"missing_counts": missing_data},
                insights=[
                    f"{len(missing_data)} columns have missing values",
                    f"Highest missing rate: {max(missing_data, key=lambda x: x['missing_percentage'])['column']} "
                    f"({max(d['missing_percentage'] for d in missing_data):.1f}%)",
                ],
            ))
        
        # Correlations
        if profile.correlations:
            sections.append(ReportSection(
                title="Correlation Analysis",
                visualizations=[
                    VisualizationConfig(
                        chart_type="heatmap",
                        title="Correlation Matrix",
                        data_source="correlations",
                    ),
                ],
                raw_data={"correlations": profile.correlations},
            ))
        
        summary = (
            f"Dataset '{dataset.name}' contains {profile.row_count:,} rows and "
            f"{profile.column_count} columns. "
            f"{sum(1 for c in profile.columns if c.null_percentage > 0)} columns have missing values."
        )
        
        return sections, summary
    
    @timed
    async def _generate_data_quality(
        self, request: ReportRequest
    ) -> tuple[List[ReportSection], str]:
        """Generate data quality report using existing data.Validator."""
        from app.data import Validator
        
        if not request.dataset_id:
            raise ValueError("dataset_id required")
        
        from app.services.datasets.service import DatasetService
        ds_service = DatasetService()
        dataset = await ds_service.get_dataset(request.dataset_id)
        
        validator = Validator()
        validation = await validator.validate(dataset.current_version.file_path)
        
        sections = []
        issues_by_severity = {"error": 0, "warning": 0}
        for issue in validation.issues:
            issues_by_severity[issue.severity] = issues_by_severity.get(issue.severity, 0) + 1
        
        sections.append(ReportSection(
            title="Validation Summary",
            metrics={
                "is_valid": validation.is_valid,
                "total_issues": len(validation.issues),
                "errors": issues_by_severity.get("error", 0),
                "warnings": issues_by_severity.get("warning", 0),
            },
            insights=[
                f"Data quality: {'PASSED' if validation.is_valid else 'FAILED'}",
                f"Found {len(validation.issues)} issues",
            ],
        ))
        
        if validation.issues:
            issue_data = [
                {"severity": severity, "count": count}
                for severity, count in issues_by_severity.items()
            ]
            sections.append(ReportSection(
                title="Quality Issues",
                visualizations=[
                    VisualizationConfig(
                        chart_type="bar",
                        title="Issues by Severity",
                        data_source="validation_issues",
                        x_column="severity",
                        y_column="count",
                    ),
                ],
                raw_data={"validation_issues": issue_data},
            ))
        
        summary = f"Data quality check {'passed' if validation.is_valid else 'failed'} with {len(validation.issues)} issues"
        return sections, summary
    
    @timed
    async def _generate_model_performance(
        self, request: ReportRequest
    ) -> tuple[List[ReportSection], str]:
        """Generate model performance report using existing ml.evaluation and ml.explainability."""
        if not request.model_id:
            raise ValueError("model_id required")
        
        from app.services.ml.service import MLService
        ml_service = MLService()
        model = await ml_service.get_model(request.model_id)
        
        sections = []
        
        # Metrics section
        if model and model.evaluation_report:
            eval_report = model.evaluation_report
            sections.append(ReportSection(
                title="Performance Metrics",
                metrics=eval_report.metrics,
                insights=[
                    f"Primary metric: {list(eval_report.metrics.keys())[0] if eval_report.metrics else 'N/A'} = "
                    f"{list(eval_report.metrics.values())[0] if eval_report.metrics else 'N/A'}",
                ],
            ))
            
            if eval_report.confusion_matrix:
                sections.append(ReportSection(
                    title="Confusion Matrix",
                    visualizations=[
                        VisualizationConfig(
                            chart_type="heatmap",
                            title="Confusion Matrix",
                            data_source="confusion_matrix",
                        ),
                    ],
                    raw_data={"confusion_matrix": eval_report.confusion_matrix},
                ))
            
            if eval_report.feature_importance:
                fi_data = [
                    {"feature": k, "importance": v}
                    for k, v in sorted(
                        eval_report.feature_importance.items(),
                        key=lambda x: x[1], reverse=True
                    )[:20]
                ]
                sections.append(ReportSection(
                    title="Feature Importance",
                    visualizations=[
                        VisualizationConfig(
                            chart_type="bar",
                            title="Top Features",
                            data_source="feature_importance",
                            x_column="feature",
                            y_column="importance",
                        ),
                    ],
                    raw_data={"feature_importance": fi_data},
                    insights=[
                        f"Top feature: {fi_data[0]['feature']} ({fi_data[0]['importance']:.4f})",
                    ],
                ))
        
        # Explainability
        if model and model.current_version:
            from app.ml import MLPlatform
            platform = MLPlatform()
            explanation = await platform.explainability.explain(
                model.current_version.model_path, None
            )
            
            if explanation.feature_importance:
                sections.append(ReportSection(
                    title="SHAP Analysis",
                    description="Feature impact on predictions",
                    raw_data={"feature_importance": explanation.feature_importance},
                ))
        
        summary = f"Model performance report for {model.name if model else 'Unknown'}"
        return sections, summary
    
    @timed
    async def _generate_pipeline_execution(
        self, request: ReportRequest
    ) -> tuple[List[ReportSection], str]:
        """Generate pipeline execution report."""
        if not request.pipeline_run_id:
            raise ValueError("pipeline_run_id required")
        
        from app.services.pipeline.service import PipelineService
        pl_service = PipelineService()
        
        # Get run details (need pipeline_id too)
        # This would query for the specific run
        sections = []
        
        sections.append(ReportSection(
            title="Execution Summary",
            metrics={
                "run_id": request.pipeline_run_id,
                "status": "completed",  # Would come from actual run data
            },
        ))
        
        summary = f"Pipeline execution report for run {request.pipeline_run_id}"
        return sections, summary
    
    @timed
    async def _generate_feature_analysis(
        self, request: ReportRequest
    ) -> tuple[List[ReportSection], str]:
        """Generate feature analysis report."""
        if not request.feature_group:
            raise ValueError("feature_group required")
        
        from app.services.feature_store.service import FeatureStoreService
        fs_service = FeatureStoreService()
        group = await fs_service.get_feature_group(request.feature_group)
        
        sections = []
        
        if group and group.statistics:
            stats = group.statistics
            sections.append(ReportSection(
                title="Feature Statistics",
                metrics={
                    "row_count": stats.row_count,
                    "feature_count": len(stats.feature_stats),
                },
            ))
        
        summary = f"Feature analysis for group: {request.feature_group}"
        return sections, summary
    
    async def _merge_sections(
        self, template_sections: List[ReportSection], generated_sections: List[ReportSection]
    ) -> List[ReportSection]:
        """Merge template sections with generated data."""
        merged = []
        gen_map = {s.title: s for s in generated_sections}
        
        for ts in template_sections:
            if ts.title in gen_map:
                gs = gen_map[ts.title]
                # Merge: generated takes precedence for data, template for structure
                merged.append(ReportSection(
                    title=ts.title,
                    description=ts.description or gs.description,
                    visualizations=gs.visualizations or ts.visualizations,
                    metrics={**ts.metrics, **gs.metrics},
                    insights=gs.insights or ts.insights,
                    recommendations=gs.recommendations or ts.recommendations,
                    raw_data=gs.raw_data or ts.raw_data,
                ))
            else:
                merged.append(ts)
        
        return merged
    
    async def _export_report(
        self, report: ReportResponse, fmt: ExportFormat
    ) -> str:
        """Export report to specified format."""
        report_dir = self._reports_path / str(report.id)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        if fmt == ExportFormat.JSON:
            file_path = report_dir / "report.json"
            data = {
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
                        "raw_data": s.raw_data,
                    }
                    for s in report.sections
                ],
                "generated_at": report.generated_at.isoformat(),
            }
            file_path.write_text(json.dumps(data, indent=2, default=str))
            
        elif fmt == ExportFormat.HTML:
            file_path = report_dir / "report.html"
            html = self._render_html(report)
            file_path.write_text(html)
            
        elif fmt == ExportFormat.CSV:
            file_path = report_dir / "report.csv"
            # Export metrics as CSV
            rows = []
            for section in report.sections:
                for key, value in section.metrics.items():
                    rows.append({"section": section.title, "metric": key, "value": value})
            import csv
            with open(file_path, "w") as f:
                writer = csv.DictWriter(f, fieldnames=["section", "metric", "value"])
                writer.writeheader()
                writer.writerows(rows)
        else:
            file_path = report_dir / f"report.{fmt.value}"
            file_path.write_text(json.dumps({"status": "exported"}, indent=2))
        
        return str(file_path)
    
    def _render_html(self, report: ReportResponse) -> str:
        """Render report as HTML."""
        sections_html = ""
        for section in report.sections:
            metrics_html = ""
            for k, v in section.metrics.items():
                metrics_html += f"<div class='metric'><strong>{k}:</strong> {v}</div>"
            
            insights_html = ""
            for insight in section.insights:
                insights_html += f"<li>{insight}</li>"
            
            sections_html += f"""
            <div class='section'>
                <h2>{section.title}</h2>
                <p>{section.description}</p>
                <div class='metrics'>{metrics_html}</div>
                <ul class='insights'>{insights_html}</ul>
            </div>
            """
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>{report.title}</title>
        <style>
            body {{ font-family: system-ui; max-width: 900px; margin: auto; padding: 20px; }}
            .section {{ border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 8px; }}
            .metric {{ display: inline-block; margin: 5px 15px; padding: 10px; background: #f5f5f5; border-radius: 4px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; }}
        </style>
        </head>
        <body>
            <h1>{report.title}</h1>
            <p class='summary'>{report.summary}</p>
            <p class='timestamp'>Generated: {report.generated_at.isoformat()}</p>
            {sections_html}
        </body>
        </html>
        """
    
    async def get_report(self, report_id: str) -> Optional[ReportResponse]:
        """Get a generated report."""
        return self._reports.get(report_id)
    
    async def list_reports(
        self, report_type: Optional[ReportType] = None
    ) -> List[ReportResponse]:
        """List generated reports."""
        reports = list(self._reports.values())
        if report_type:
            reports = [r for r in reports if r.report_type == report_type]
        return sorted(reports, key=lambda r: r.generated_at, reverse=True)
    
    async def _get_custom_template(self, template_id: str) -> Optional[ReportTemplate]:
        """Get a custom report template."""
        # Would fetch from template storage
        return None