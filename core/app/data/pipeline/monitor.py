"""Pipeline monitoring and metrics collection."""

from typing import Dict, List, Optional, Any
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from .models import PipelineEvent, StageType, StageStatus


class PipelineMonitor:
    """
    Monitors pipeline execution and collects metrics.
    Provides performance tracking and reporting.
    """
    
    def __init__(self, metrics_dir: Path = Path("reports/pipeline/metrics")):
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self._events: List[PipelineEvent] = []
        self._stage_metrics: Dict[StageType, Dict[str, Any]] = defaultdict(dict)
        self._pipeline_metrics: Dict[str, Any] = {}
    
    def record_event(self, event: PipelineEvent) -> None:
        """Record a pipeline event."""
        self._events.append(event)
        
        # Update metrics
        if event.stage_type:
            self._update_stage_metrics(event)
        else:
            self._update_pipeline_metrics(event)
    
    def _update_stage_metrics(self, event: PipelineEvent) -> None:
        """Update stage-specific metrics."""
        stage_type = event.stage_type
        metrics = self._stage_metrics[stage_type]
        
        if event.event_type == "stage_start":
            metrics["start_time"] = event.timestamp
            metrics["attempts"] = metrics.get("attempts", 0) + 1
        elif event.event_type == "stage_complete":
            if "start_time" in metrics:
                metrics["duration"] = (event.timestamp - metrics["start_time"]).total_seconds()
            metrics["status"] = "success"
            metrics["end_time"] = event.timestamp
        elif event.event_type == "stage_failed":
            if "start_time" in metrics:
                metrics["duration"] = (event.timestamp - metrics["start_time"]).total_seconds()
            metrics["status"] = "failed"
            metrics["end_time"] = event.timestamp
            metrics["error"] = event.data.get("error")
    
    def _update_pipeline_metrics(self, event: PipelineEvent) -> None:
        """Update pipeline-level metrics."""
        if event.event_type == "pipeline_start":
            self._pipeline_metrics["start_time"] = event.timestamp
        elif event.event_type == "pipeline_complete":
            self._pipeline_metrics["end_time"] = event.timestamp
            if "start_time" in self._pipeline_metrics:
                self._pipeline_metrics["total_duration"] = (
                    event.timestamp - self._pipeline_metrics["start_time"]
                ).total_seconds()
            self._pipeline_metrics["status"] = "success"
        elif event.event_type == "pipeline_failed":
            self._pipeline_metrics["end_time"] = event.timestamp
            if "start_time" in self._pipeline_metrics:
                self._pipeline_metrics["total_duration"] = (
                    event.timestamp - self._pipeline_metrics["start_time"]
                ).total_seconds()
            self._pipeline_metrics["status"] = "failed"
            self._pipeline_metrics["error"] = event.data.get("error")
    
    def get_stage_metrics(self, stage_type: StageType) -> Dict[str, Any]:
        """Get metrics for a specific stage."""
        return self._stage_metrics.get(stage_type, {})
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get overall pipeline metrics."""
        metrics = self._pipeline_metrics.copy()
        metrics["total_stages"] = len(self._stage_metrics)
        metrics["successful_stages"] = sum(
            1 for m in self._stage_metrics.values()
            if m.get("status") == "success"
        )
        metrics["failed_stages"] = sum(
            1 for m in self._stage_metrics.values()
            if m.get("status") == "failed"
        )
        metrics["events_count"] = len(self._events)
        return metrics
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive metrics report."""
        return {
            "pipeline": self.get_pipeline_metrics(),
            "stages": dict(self._stage_metrics),
            "events": [
                {
                    "type": e.event_type,
                    "stage": e.stage_type.value if e.stage_type else None,
                    "timestamp": e.timestamp.isoformat(),
                    "data": e.data
                }
                for e in self._events
            ]
        }
    
    def save_report(self, filename: Optional[str] = None) -> Path:
        """Save metrics report to file."""
        if filename is None:
            filename = f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.generate_report()
        report_path = self.metrics_dir / filename
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        return report_path
    
    def clear(self) -> None:
        """Clear all metrics and events."""
        self._events.clear()
        self._stage_metrics.clear()
        self._pipeline_metrics.clear()