# app/services/reports/models.py
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class ReportType(str, Enum):
    DATASET_PROFILE = "dataset_profile"
    DATA_QUALITY = "data_quality"
    PIPELINE_EXECUTION = "pipeline_execution"
    MODEL_PERFORMANCE = "model_performance"
    FEATURE_ANALYSIS = "feature_analysis"
    PREDICTION_SUMMARY = "prediction_summary"
    DRIFT_DETECTION = "drift_detection"
    CUSTOM = "custom"


class ExportFormat(str, Enum):
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    DASHBOARD = "dashboard"


@dataclass
class VisualizationConfig:
    """Configuration for a visualization in a report."""
    chart_type: str  # "bar", "line", "scatter", "heatmap", "histogram", "box", "table"
    title: str
    data_source: str  # Reference to data in context
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportSection:
    """A section within a report."""
    title: str
    description: str = ""
    visualizations: List[VisualizationConfig] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class ReportTemplate:
    """Pre-defined report template."""
    id: UUID
    name: str
    report_type: ReportType
    sections: List[ReportSection]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ReportRequest:
    """Request to generate a report."""
    report_type: ReportType
    template_id: Optional[str] = None
    dataset_id: Optional[str] = None
    model_id: Optional[str] = None
    pipeline_run_id: Optional[str] = None
    feature_group: Optional[str] = None
    date_range: Optional[Dict[str, str]] = None
    custom_sections: Optional[List[ReportSection]] = None
    export_formats: List[ExportFormat] = field(default_factory=lambda: [ExportFormat.HTML])
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportResponse:
    """Generated report response."""
    id: UUID
    report_type: ReportType
    title: str
    sections: List[ReportSection]
    summary: str
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exports: Dict[str, str] = field(default_factory=dict)  # format -> file_path
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_by: Optional[str] = None