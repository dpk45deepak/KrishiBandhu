# app/data/eda/report.py
"""
Report generation engine.
Creates HTML, Markdown, and JSON reports from EDA results.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import polars as pl
from loguru import logger
from app.data.eda.models import EDAReport
from app.data.eda.visualizer import VisualizationEngine


class ReportGenerator:
    """Report generation engine"""
    
    def __init__(self, report: EDAReport, output_dir: Path):
        self.report = report
        self.output_dir = output_dir
        self.logger = logger.bind(module="report_generator")
        
        # Setup Jinja2
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True
        )
        
        # Visualization engine for creating figures
        self.viz_engine = VisualizationEngine(report.config.visualization if hasattr(report, 'config') else None)
        
    def generate_all(self) -> Dict[str, Path]:
        """
        Generate all report formats.
        
        Returns:
            Dictionary mapping format to file path
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Generate HTML report
        html_path = self.output_dir / "report.html"
        self._generate_html(html_path)
        results['html'] = html_path
        
        # Generate Markdown report
        md_path = self.output_dir / "report.md"
        self._generate_markdown(md_path)
        results['markdown'] = md_path
        
        # Generate JSON report
        json_path = self.output_dir / "report.json"
        self._generate_json(json_path)
        results['json'] = json_path
        
        self.logger.info(f"All reports generated in {self.output_dir}")
        return results
    
    def _generate_html(self, output_path: Path) -> None:
        """Generate HTML report."""
        template = self.env.get_template("report.html.j2")
        
        # Prepare data
        context = self._prepare_template_context()
        
        # Render template
        html_content = template.render(**context)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        self.logger.info(f"HTML report saved: {output_path}")
    
    def _generate_markdown(self, output_path: Path) -> None:
        """Generate Markdown report."""
        template = self.env.get_template("report.md.j2")
        
        # Prepare data
        context = self._prepare_template_context()
        
        # Render template
        md_content = template.render(**context)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        self.logger.info(f"Markdown report saved: {output_path}")
    
    def _generate_json(self, output_path: Path) -> None:
        """Generate JSON report."""
        # Convert report to dict
        report_dict = self.report.dict()
        
        # Handle datetime serialization
        report_dict['generation_timestamp'] = report_dict['generation_timestamp'].isoformat()
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, default=str)
            
        self.logger.info(f"JSON report saved: {output_path}")
    
    def _prepare_template_context(self) -> Dict[str, Any]:
        """Prepare context for template rendering."""
        return {
            'report': self.report,
            'dataset_name': self.report.dataset_name,
            'timestamp': self.report.generation_timestamp,
            'overview': self.report.dataset_overview,
            'feature_summaries': self.report.feature_summaries,
            'quality_scores': self.report.quality_scores,
            'correlation': self.report.correlation_matrix,
            'pca_results': self.report.pca_results,
            'clustering_results': self.report.clustering_results,
            'missingness_patterns': self.report.missingness_patterns,
            'outliers': self.report.outliers_detected,
            'ml_readiness': self.report.ml_readiness,
            'recommendations': self.report.recommendations,
            'executive_summary': self.report.executive_summary
        }