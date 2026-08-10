"""
Report generation for model evaluation.
"""

from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from ..common.models import BaseMLModel
from ..common.utils import ensure_directory, save_json, get_timestamp
from .evaluator import ModelEvaluator


class ReportGenerator:
    """
    Generator for comprehensive model evaluation reports.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        ensure_directory(self.output_dir)
        
        self.report_data = {}
        logger.info(f"ReportGenerator initialized at {self.output_dir}")
    
    def generate_report(
        self,
        model: BaseMLModel,
        evaluation_results: Dict[str, Any],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        model_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a complete evaluation report.
        
        Args:
            model: Trained model
            evaluation_results: Results from ModelEvaluator
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            **kwargs: Additional metadata
            
        Returns:
            Report dictionary
        """
        try:
            model_name = model_name or model.name
            
            report = {
                'model_name': model_name,
                'model_type': 'classification' if hasattr(model, 'get_n_classes') else 'regression',
                'timestamp': get_timestamp(),
                'n_samples': len(y_test),
                'n_features': X_test.shape[1] if hasattr(X_test, 'shape') else None
            }
            
            # Add evaluation metrics
            report['metrics'] = evaluation_results.get('metrics', {})
            
            # Add model info
            report['model_info'] = {
                'hyperparameters': model.get_params(),
                'is_fitted': model.is_fitted()
            }
            
            if hasattr(model, 'get_n_classes'):
                report['model_info']['n_classes'] = model.get_n_classes()
                report['model_info']['classes'] = model.get_classes().tolist()
            
            # Add training metadata if available
            if model._metadata:
                report['model_info']['metadata'] = model._metadata.model_dump()
            
            # Add additional info
            report['additional_info'] = kwargs
            
            # Store report data
            self.report_data = report
            
            # Save report
            self._save_report(report)
            
            # Generate HTML report
            self._generate_html_report(report)
            
            logger.info(f"Report generated for {model_name}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate report: {str(e)}")
            raise
    
    def _save_report(self, report: Dict[str, Any]) -> None:
        """Save report in multiple formats."""
        # Save as JSON
        json_path = self.output_dir / 'report.json'
        save_json(report, json_path)
        
        # Save as CSV (metrics only)
        if report.get('metrics'):
            metrics_df = pd.DataFrame([report['metrics']])
            metrics_path = self.output_dir / 'metrics.csv'
            metrics_df.to_csv(metrics_path, index=False)
        
        # Save as Markdown
        self._generate_markdown_report(report)
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> None:
        """Generate Markdown report."""
        md_path = self.output_dir / 'report.md'
        
        lines = []
        lines.append(f"# Model Evaluation Report")
        lines.append(f"")
        lines.append(f"**Model:** {report['model_name']}")
        lines.append(f"**Type:** {report['model_type']}")
        lines.append(f"**Generated:** {report['timestamp']}")
        lines.append(f"**Samples:** {report['n_samples']}")
        lines.append(f"**Features:** {report.get('n_features', 'N/A')}")
        lines.append(f"")
        lines.append(f"## Metrics")
        lines.append(f"")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        
        metrics = report.get('metrics', {})
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                lines.append(f"| {metric} | {value:.4f} |")
            else:
                lines.append(f"| {metric} | {value} |")
        
        lines.append(f"")
        lines.append(f"## Model Information")
        lines.append(f"")
        lines.append(f"```json")
        lines.append(json.dumps(report.get('model_info', {}), indent=2))
        lines.append(f"```")
        
        with open(md_path, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Markdown report saved to {md_path}")
    
    def _generate_html_report(self, report: Dict[str, Any]) -> None:
        """Generate HTML report."""
        html_path = self.output_dir / 'report.html'
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                .header {{ background: #34495e; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
                .card {{ background: #ecf0f1; padding: 15px; border-radius: 5px; }}
                .card h3 {{ margin-top: 0; color: #2c3e50; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #34495e; color: white; }}
                .metrics {{ background: #f9f9f9; padding: 20px; border-radius: 5px; }}
                .highlight {{ background: #3498db; color: white; padding: 2px 8px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 Model Evaluation Report</h1>
                    <p><strong>Model:</strong> {report['model_name']}</p>
                    <p><strong>Type:</strong> {report['model_type']}</p>
                    <p><strong>Generated:</strong> {report['timestamp']}</p>
                </div>
                
                <div class="grid">
                    <div class="card">
                        <h3>📊 Dataset</h3>
                        <p><strong>Samples:</strong> {report['n_samples']}</p>
                        <p><strong>Features:</strong> {report.get('n_features', 'N/A')}</p>
                    </div>
                    <div class="card">
                        <h3>📈 Performance</h3>
        """
        
        # Add best metric
        metrics = report.get('metrics', {})
        if metrics:
            best_metric = max([(v, k) for k, v in metrics.items() if isinstance(v, (int, float))])
            html += f"""
                        <p><strong>Best Metric:</strong> <span class="highlight">{best_metric[1]} = {best_metric[0]:.4f}</span></p>
                    </div>
                </div>
                
                <h2>📋 Metrics</h2>
                <div class="metrics">
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
            """
            
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    html += f"""
                        <tr>
                            <td>{metric}</td>
                            <td>{value:.4f}</td>
                        </tr>
                    """
                else:
                    html += f"""
                        <tr>
                            <td>{metric}</td>
                            <td>{value}</td>
                        </tr>
                    """
        
        html += """
                    </table>
                </div>
                
                <h2>🔧 Model Information</h2>
                <pre style="background: #f5f5f5; padding: 15px; border-radius: 5px; overflow: auto;">
        """
        html += json.dumps(report.get('model_info', {}), indent=2)
        
        html += """
                </pre>
            </div>
        </body>
        </html>
        """
        
        with open(html_path, 'w') as f:
            f.write(html)
        
        logger.info(f"HTML report saved to {html_path}")
    
    def generate_comparison_report(
        self,
        results: Dict[str, Dict[str, Any]],
        output_name: str = 'comparison'
    ) -> None:
        """
        Generate a comparison report for multiple models.
        
        Args:
            results: Dictionary mapping model names to evaluation results
            output_name: Name of the output file
        """
        try:
            comparison_dir = self.output_dir / 'comparisons'
            ensure_directory(comparison_dir)
            
            # Create comparison table
            metrics = {}
            model_names = list(results.keys())
            
            for model_name, eval_results in results.items():
                for metric, value in eval_results.get('metrics', {}).items():
                    if metric not in metrics:
                        metrics[metric] = {}
                    metrics[metric][model_name] = value
            
            # Save as CSV
            comparison_df = pd.DataFrame(metrics).T
            comparison_path = comparison_dir / f'{output_name}.csv'
            comparison_df.to_csv(comparison_path)
            
            # Generate comparison JSON
            comparison_json = {
                'models': model_names,
                'metrics': metrics,
                'timestamp': get_timestamp()
            }
            json_path = comparison_dir / f'{output_name}.json'
            save_json(comparison_json, json_path)
            
            # Generate comparison markdown
            md_path = comparison_dir / f'{output_name}.md'
            lines = [f"# Model Comparison Report", ""]
            lines.append(f"**Generated:** {get_timestamp()}")
            lines.append("")
            lines.append("## Metrics Comparison")
            lines.append("")
            lines.append("| Metric | " + " | ".join(model_names) + " |")
            lines.append("|--------|" + "|".join(["--------"] * len(model_names)) + "|")
            
            for metric, values in metrics.items():
                row = [str(metric)]
                for model in model_names:
                    val = values.get(model, 'N/A')
                    if isinstance(val, (int, float)):
                        row.append(f"{val:.4f}")
                    else:
                        row.append(str(val))
                lines.append("| " + " | ".join(row) + " |")
            
            with open(md_path, 'w') as f:
                f.write('\n'.join(lines))
            
            logger.info(f"Comparison report saved to {comparison_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate comparison report: {str(e)}")
            raise
    
    def get_report(self) -> Dict[str, Any]:
        """Get the generated report."""
        return self.report_data