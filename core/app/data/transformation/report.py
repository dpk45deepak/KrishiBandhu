"""Report generation for standardization processes."""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import logging
from datetime import datetime

from .models import StandardizationReport

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates detailed reports for standardization."""
    
    def __init__(self, report_dir: str = "reports/standardization"):
        """
        Initialize report generator.
        
        Args:
            report_dir: Directory to store reports
        """
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, report: StandardizationReport, 
                       output_path: Optional[Path] = None) -> Path:
        """
        Generate a detailed report from standardization results.
        
        Args:
            report: StandardizationReport object
            output_path: Optional output path
            
        Returns:
            Path to generated report file
        """
        if not output_path:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"standardization_report_{timestamp}"
            output_path = self.report_dir / f"{filename}.txt"
        
        # Generate HTML report
        html_path = output_path.with_suffix('.html')
        self._generate_html_report(report, html_path)
        
        # Generate JSON report
        json_path = output_path.with_suffix('.json')
        self._generate_json_report(report, json_path)
        
        # Generate text summary
        self._generate_text_summary(report, output_path)
        
        logger.info(f"Reports generated: {html_path}, {json_path}, {output_path}")
        return output_path
    
    def _generate_html_report(self, report: StandardizationReport, output_path: Path):
        """Generate HTML report."""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Standardization Report - {report.schema_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
                .header {{ border-bottom: 2px solid #4CAF50; padding-bottom: 10px; margin-bottom: 20px; }}
                .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
                .card {{ background: #f9f9f9; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; }}
                .card-title {{ font-weight: bold; color: #666; font-size: 12px; text-transform: uppercase; }}
                .card-value {{ font-size: 24px; font-weight: bold; }}
                .section {{ margin-top: 30px; }}
                .section-title {{ color: #333; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .badge {{ display: inline-block; padding: 2px 8px; background: #4CAF50; color: white; border-radius: 12px; font-size: 12px; }}
                .error {{ color: #f44336; }}
                .warning {{ color: #ff9800; }}
                .success {{ color: #4CAF50; }}
                .progress {{ height: 20px; background: #f0f0f0; border-radius: 10px; overflow: hidden; margin: 5px 0; }}
                .progress-bar {{ height: 100%; background: #4CAF50; border-radius: 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Standardization Report</h1>
                    <p><strong>Schema:</strong> {report.schema_name} v{report.schema_version}</p>
                    <p><strong>Source:</strong> {report.source_file}</p>
                    <p><strong>Generated:</strong> {report.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="summary">
                    <div class="card">
                        <div class="card-title">Total Rows</div>
                        <div class="card-value">{report.total_rows:,}</div>
                    </div>
                    <div class="card">
                        <div class="card-title">Total Columns</div>
                        <div class="card-value">{report.total_columns}</div>
                    </div>
                    <div class="card">
                        <div class="card-title">Processing Time</div>
                        <div class="card-value">{report.processing_time:.2f}s</div>
                    </div>
                    <div class="card">
                        <div class="card-title">Transformations</div>
                        <div class="card-value">{len(report.transformations_applied)}</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2 class="section-title">Column Mappings</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Original Column</th>
                                <th>Standardized Column</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(f'<tr><td>{orig}</td><td>{std}</td></tr>' for orig, std in report.column_mappings.items())}
                        </tbody>
                    </table>
                </div>
                
                <div class="section">
                    <h2 class="section-title">Transformations Applied</h2>
                    <ul>
                        {''.join(f'<li>{t}</li>' for t in report.transformations_applied)}
                    </ul>
                </div>
                
                <div class="section">
                    <h2 class="section-title">Errors Encountered</h2>
                    {''.join(f'<div class="error">⚠️ {e}</div>' for e in report.errors_encountered) or '<div class="success">✅ No errors encountered</div>'}
                </div>
                
                <div class="section">
                    <h2 class="section-title">Unit Conversions</h2>
                    {''.join(f'<div><strong>{col}:</strong> {info.get("target_unit", "unknown")}</div>' for col, info in report.unit_conversions.items()) or '<div>No unit conversions performed</div>'}
                </div>
                
                <div class="section">
                    <h2 class="section-title">Category Mappings</h2>
                    {''.join(f'<div><strong>{col}:</strong> {info.get("unique_values", 0)} unique values ({info.get("category_type", "unknown")})</div>' for col, info in report.category_mappings.items()) or '<div>No category mappings performed</div>'}
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_template)
    
    def _generate_json_report(self, report: StandardizationReport, output_path: Path):
        """Generate JSON report."""
        with open(output_path, 'w') as f:
            json.dump(report.dict(), f, indent=2, default=str)
    
    def _generate_text_summary(self, report: StandardizationReport, output_path: Path):
        """Generate text summary."""
        lines = [
            "=" * 80,
            f"STANDARDIZATION REPORT - {report.schema_name} v{report.schema_version}",
            "=" * 80,
            "",
            f"Source File: {report.source_file}",
            f"Output File: {report.output_file}",
            f"Generated: {report.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Processing Time: {report.processing_time:.2f}s",
            "",
            "=" * 40,
            "SUMMARY",
            "=" * 40,
            f"Total Rows: {report.total_rows:,}",
            f"Total Columns: {report.total_columns}",
            f"Transformations Applied: {len(report.transformations_applied)}",
            f"Errors Encountered: {len(report.errors_encountered)}",
            "",
            "=" * 40,
            "COLUMN MAPPINGS",
            "=" * 40,
        ]
        
        for original, standardized in report.column_mappings.items():
            lines.append(f"  {original} -> {standardized}")
        
        if report.transformations_applied:
            lines.append("")
            lines.append("=" * 40)
            lines.append("TRANSFORMATIONS")
            lines.append("=" * 40)
            for transform in report.transformations_applied:
                lines.append(f"  ✓ {transform}")
        
        if report.errors_encountered:
            lines.append("")
            lines.append("=" * 40)
            lines.append("ERRORS")
            lines.append("=" * 40)
            for error in report.errors_encountered:
                lines.append(f"  ⚠️ {error}")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))