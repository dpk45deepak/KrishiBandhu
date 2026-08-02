"""Report generation for cleaning operations."""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from jinja2 import Template
from loguru import logger

from .models import CleaningMetadata
from .exceptions import ReportGenerationError


class CleaningReportGenerator:
    """Generates cleaning reports in multiple formats."""
    
    def __init__(self):
        self.report_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load report templates."""
        # HTML Template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cleaning Report - {{ metadata.dataset_name }}</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    color: #333;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }
                .summary-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }
                .summary-card {
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                }
                .summary-card h3 {
                    margin: 0 0 5px 0;
                    color: #6c757d;
                    font-size: 14px;
                }
                .summary-card .value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #333;
                }
                .section {
                    background: white;
                    border: 1px solid #e9ecef;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                }
                .section h2 {
                    margin-top: 0;
                    color: #495057;
                    border-bottom: 2px solid #e9ecef;
                    padding-bottom: 10px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }
                th, td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #e9ecef;
                }
                th {
                    background-color: #f8f9fa;
                    font-weight: 600;
                }
                tr:hover {
                    background-color: #f8f9fa;
                }
                .status-success {
                    color: #28a745;
                }
                .status-warning {
                    color: #ffc107;
                }
                .status-danger {
                    color: #dc3545;
                }
                .timestamp {
                    color: #6c757d;
                    font-size: 14px;
                }
                .operations-list {
                    list-style: none;
                    padding: 0;
                }
                .operations-list li {
                    padding: 10px;
                    margin: 5px 0;
                    background: #f8f9fa;
                    border-radius: 5px;
                }
                .badge {
                    display: inline-block;
                    padding: 3px 8px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: 600;
                }
                .badge-success {
                    background: #d4edda;
                    color: #155724;
                }
                .badge-warning {
                    background: #fff3cd;
                    color: #856404;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Data Cleaning Report</h1>
                <p><strong>Dataset:</strong> {{ metadata.dataset_name }}</p>
                <p><strong>Source:</strong> {{ metadata.source_path }}</p>
                <p class="timestamp">Generated: {{ metadata.end_time.strftime('%Y-%m-%d %H:%M:%S') }}</p>
            </div>
            
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Rows</h3>
                    <div class="value">{{ metadata.rows_before }} → {{ metadata.rows_after }}</div>
                </div>
                <div class="summary-card">
                    <h3>Columns</h3>
                    <div class="value">{{ metadata.columns_before|length }} → {{ metadata.columns_after|length }}</div>
                </div>
                <div class="summary-card">
                    <h3>Duplicates Removed</h3>
                    <div class="value">{{ metadata.duplicates_removed }}</div>
                </div>
                <div class="summary-card">
                    <h3>Missing Values Fixed</h3>
                    <div class="value">{{ metadata.missing_values_fixed.values()|sum }}</div>
                </div>
                <div class="summary-card">
                    <h3>Outliers Handled</h3>
                    <div class="value">{{ metadata.outliers_handled.values()|sum }}</div>
                </div>
                <div class="summary-card">
                    <h3>Execution Time</h3>
                    <div class="value">{{ "%.2f"|format(metadata.execution_time_seconds) }}s</div>
                </div>
            </div>
            
            {% if metadata.missing_values_fixed %}
            <div class="section">
                <h2>📊 Missing Values Fixed</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Missing Values Fixed</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for col, count in metadata.missing_values_fixed.items() %}
                        <tr>
                            <td><strong>{{ col }}</strong></td>
                            <td>{{ count }}</td>
                            <td><span class="badge badge-success">Fixed</span></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            {% endif %}
            
            {% if metadata.outliers_handled %}
            <div class="section">
                <h2>🎯 Outliers Handled</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Outliers Fixed</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for col, count in metadata.outliers_handled.items() %}
                        <tr>
                            <td><strong>{{ col }}</strong></td>
                            <td>{{ count }}</td>
                            <td><span class="badge badge-success">Handled</span></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            {% endif %}
            
            {% if metadata.datatype_changes %}
            <div class="section">
                <h2>🔄 Data Type Changes</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Old Type</th>
                            <th>New Type</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for col, changes in metadata.datatype_changes.items() %}
                        <tr>
                            <td><strong>{{ col }}</strong></td>
                            <td>{{ changes.old }}</td>
                            <td>{{ changes.new }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            {% endif %}
            
            {% if metadata.operations %}
            <div class="section">
                <h2>📝 Cleaning Operations</h2>
                <ul class="operations-list">
                    {% for op in metadata.operations %}
                    <li>
                        <strong>{{ op.operation }}</strong>
                        {% if op.column %}on <strong>{{ op.column }}</strong>{% endif %}
                        - {{ op.changes_made }} changes
                        <span class="badge badge-success">Applied</span>
                    </li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
            
            {% if metadata.warnings %}
            <div class="section">
                <h2>⚠️ Warnings</h2>
                <ul>
                    {% for warning in metadata.warnings %}
                    <li class="status-warning">⚠️ {{ warning }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
            
            <div style="text-align: center; color: #6c757d; margin-top: 40px; font-size: 14px;">
                <p>AgriMind AI - Data Cleaning Pipeline</p>
                <p>Report generated automatically</p>
            </div>
        </body>
        </html>
        """
        
        # Markdown Template
        markdown_template = """
        # Data Cleaning Report: {{ metadata.dataset_name }}
        
        **Generated:** {{ metadata.end_time.strftime('%Y-%m-%d %H:%M:%S') }}
        
        ## 📊 Summary
        
        | Metric | Value |
        |--------|-------|
        | Rows | {{ metadata.rows_before }} → {{ metadata.rows_after }} |
        | Columns | {{ metadata.columns_before|length }} → {{ metadata.columns_after|length }} |
        | Duplicates Removed | {{ metadata.duplicates_removed }} |
        | Missing Values Fixed | {{ metadata.missing_values_fixed.values()|sum }} |
        | Outliers Handled | {{ metadata.outliers_handled.values()|sum }} |
        | Execution Time | {{ "%.2f"|format(metadata.execution_time_seconds) }} seconds |
        
        ## 📋 Column Changes
        
        ### Before Cleaning
        {{ metadata.columns_before|join(', ') }}
        
        ### After Cleaning  
        {{ metadata.columns_after|join(', ') }}
        
        {% if metadata.missing_values_fixed %}
        ### Missing Values Fixed
        
        | Column | Count |
        |--------|-------|
        {% for col, count in metadata.missing_values_fixed.items() %}
        | {{ col }} | {{ count }} |
        {% endfor %}
        {% endif %}
        
        {% if metadata.outliers_handled %}
        ### Outliers Handled
        
        | Column | Count |
        |--------|-------|
        {% for col, count in metadata.outliers_handled.items() %}
        | {{ col }} | {{ count }} |
        {% endfor %}
        {% endif %}
        
        {% if metadata.datatype_changes %}
        ### Data Type Changes
        
        | Column | Old Type | New Type |
        |--------|----------|----------|
        {% for col, changes in metadata.datatype_changes.items() %}
        | {{ col }} | {{ changes.old }} | {{ changes.new }} |
        {% endfor %}
        {% endif %}
        
        {% if metadata.operations %}
        ### Operations Performed
        
        {% for op in metadata.operations %}
        - **{{ op.operation }}** {% if op.column %}({{ op.column }}){% endif %}: {{ op.changes_made }} changes
        {% endfor %}
        {% endif %}
        
        {% if metadata.warnings %}
        ## ⚠️ Warnings
        
        {% for warning in metadata.warnings %}
        - {{ warning }}
        {% endfor %}
        {% endif %}
        """
        
        return {
            "html": html_template,
            "markdown": markdown_template,
        }
    
    def generate_report(self, metadata: CleaningMetadata) -> str:
        """Generate a cleaning report."""
        try:
            # Use Markdown as default
            template = Template(self.report_templates["markdown"])
            report = template.render(metadata=metadata)
            return report
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise ReportGenerationError(f"Report generation failed: {e}")
    
    def generate_html_report(self, metadata: CleaningMetadata) -> str:
        """Generate an HTML cleaning report."""
        try:
            template = Template(self.report_templates["html"])
            report = template.render(metadata=metadata)
            return report
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            raise ReportGenerationError(f"HTML report generation failed: {e}")
    
    def generate_json_report(self, metadata: CleaningMetadata) -> str:
        """Generate a JSON cleaning report."""
        try:
            # Convert metadata to JSON serializable format
            data = self._metadata_to_dict(metadata)
            return json.dumps(data, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to generate JSON report: {e}")
            raise ReportGenerationError(f"JSON report generation failed: {e}")
    
    def save_report(self, 
                   metadata: CleaningMetadata,
                   output_dir: Optional[Union[str, Path]] = None,
                   formats: List[str] = ["md", "html", "json"]) -> Dict[str, Path]:
        """Save reports in multiple formats."""
        if output_dir is None:
            output_dir = Path("reports/cleaning")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        # Generate base filename
        base_name = metadata.dataset_name.replace(" ", "_").lower()
        timestamp = metadata.end_time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"{base_name}_{timestamp}"
        
        if "md" in formats:
            report_md = self.generate_report(metadata)
            path = output_dir / f"{base_filename}_cleaning_report.md"
            path.write_text(report_md)
            saved_paths["md"] = path
        
        if "html" in formats:
            report_html = self.generate_html_report(metadata)
            path = output_dir / f"{base_filename}_cleaning_report.html"
            path.write_text(report_html)
            saved_paths["html"] = path
        
        if "json" in formats:
            report_json = self.generate_json_report(metadata)
            path = output_dir / f"{base_filename}_cleaning_report.json"
            path.write_text(report_json)
            saved_paths["json"] = path
        
        logger.info(f"Saved reports to {output_dir}")
        return saved_paths
    
    def _metadata_to_dict(self, metadata: CleaningMetadata) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "dataset_name": metadata.dataset_name,
            "source_path": metadata.source_path,
            "output_path": metadata.output_path,
            "rows_before": metadata.rows_before,
            "rows_after": metadata.rows_after,
            "columns_before": metadata.columns_before,
            "columns_after": metadata.columns_after,
            "operations": [
                {
                    "operation": op.operation,
                    "column": op.column,
                    "changes_made": op.changes_made,
                    "execution_time_ms": op.execution_time_ms,
                }
                for op in metadata.operations
            ],
            "missing_values_fixed": metadata.missing_values_fixed,
            "duplicates_removed": metadata.duplicates_removed,
            "outliers_handled": metadata.outliers_handled,
            "datatype_changes": metadata.datatype_changes,
            "start_time": metadata.start_time.isoformat(),
            "end_time": metadata.end_time.isoformat(),
            "execution_time_seconds": metadata.execution_time_seconds,
            "validation_status": metadata.validation_status,
            "warnings": metadata.warnings,
            "errors": metadata.errors,
        }