"""Report generation for validation results.

The :class:`ValidationReportGenerator` renders a :class:`ValidationReport`
into HTML (with Plotly charts), JSON, Markdown, and a CSV of failed rows.
All outputs are written to a configurable directory (default:
``reports/validation``).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from loguru import logger

from app.data.validation.models import ValidationReport
from app.utils.decorators import timer
from app.utils.path_utils import ensure_dir, resolve_path


class ValidationReportGenerator:
    """Generates HTML, JSON, Markdown, and CSV reports for validation runs.

    Attributes:
        output_dir: Directory where report artifacts are stored.
    """

    # Report file names (overwritten on each run for a dataset)
    REPORT_HTML: str = "validation_report.html"
    REPORT_JSON: str = "validation_report.json"
    REPORT_MD: str = "validation_report.md"
    REPORT_CSV: str = "failed_rows.csv"

    def __init__(self, output_dir: str | Path = "reports/validation") -> None:
        """Initialize the report generator.

        Args:
            output_dir: Directory to save report artifacts.
        """
        self.output_dir = ensure_dir(output_dir)
        logger.debug(f"ValidationReportGenerator initialized, output: {self.output_dir}")

    @timer
    def generate_all(
        self, report: ValidationReport, failed_df: pd.DataFrame | None = None
    ) -> dict[str, str]:
        """Generate every report artifact for a validation report.

        Args:
            report: The validation report to render.
            failed_df: Optional DataFrame containing only failed rows.

        Returns:
            Mapping of format name to output file path.
        """
        paths: dict[str, str] = {}
        paths["html"] = self.generate_html(report)
        paths["json"] = self.generate_json(report)
        paths["markdown"] = self.generate_markdown(report)
        paths["csv"] = self.generate_failed_rows_csv(report, failed_df)
        logger.info(f"Validation reports generated for '{report.summary.dataset_name}': {paths}")
        return paths

    def generate_html(self, report: ValidationReport) -> str:
        """Render the validation report to a self-contained HTML file.

        Args:
            report: The validation report.

        Returns:
            Path to the generated HTML file.
        """
        file_path = resolve_path(self.output_dir / self.REPORT_HTML)
        figures = self._build_figures(report)
        html = self._render_html(report, figures)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"HTML validation report saved: {file_path}")
        return str(file_path)

    def generate_json(self, report: ValidationReport) -> str:
        """Serialize the validation report to a JSON file.

        Args:
            report: The validation report.

        Returns:
            Path to the generated JSON file.
        """
        file_path = resolve_path(self.output_dir / self.REPORT_JSON)
        data = report.model_dump(mode="json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"JSON validation report saved: {file_path}")
        return str(file_path)

    def generate_markdown(self, report: ValidationReport) -> str:
        """Render the validation report to a Markdown file.

        Args:
            report: The validation report.

        Returns:
            Path to the generated Markdown file.
        """
        file_path = resolve_path(self.output_dir / self.REPORT_MD)
        content = self._render_markdown(report)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Markdown validation report saved: {file_path}")
        return str(file_path)

    def generate_failed_rows_csv(
        self,
        report: ValidationReport,
        failed_df: pd.DataFrame | None = None,
    ) -> str:
        """Write failed rows to a CSV file.

        Args:
            report: The validation report (used to infer failed indices).
            failed_df: Optional DataFrame of failed rows.

        Returns:
            Path to the generated CSV file ('' if no failures).
        """
        file_path = resolve_path(self.output_dir / self.REPORT_CSV)
        if failed_df is None or failed_df.empty:
            logger.info("No failed rows to export; skipping CSV")
            return ""

        failed_df.to_csv(file_path, index=False)
        logger.info(f"Failed rows CSV saved: {file_path} ({len(failed_df)} rows)")
        return str(file_path)

    # ------------------------------------------------------------------
    # Chart builders
    # ------------------------------------------------------------------

    def _build_figures(self, report: ValidationReport) -> list[go.Figure]:
        """Build Plotly figures for the HTML report."""
        figures = [self._build_severity_chart(report)]
        rule_chart = self._build_rule_chart(report)
        if rule_chart:
            figures.append(rule_chart)
        return figures

    def _build_severity_chart(self, report: ValidationReport) -> go.Figure:
        """Build a bar chart of severity counts."""
        counts = report.summary.severity_counts
        colors = {
            "critical": "#F44336",
            "error": "#FF9800",
            "warning": "#FFC107",
            "info": "#2196F3",
        }
        labels = [k for k, v in counts.items() if v > 0]
        values = [counts[k] for k in labels]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=[colors.get(k, "#9E9E9E") for k in labels],
                    text=values,
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title="Validation Errors by Severity",
            xaxis_title="Severity",
            yaxis_title="Error Count",
            template="plotly_white",
            height=360,
        )
        return fig

    def _build_rule_chart(self, report: ValidationReport) -> go.Figure | None:
        """Build a bar chart of passed vs failed rules."""
        stats = report.rule_statistics
        if not stats:
            return None
        names = [s.rule_name for s in stats]
        passed = [1 if s.passed else 0 for s in stats]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=names,
                    y=passed,
                    marker_color=["#4CAF50" if s.passed else "#F44336" for s in stats],
                    text=["PASS" if s.passed else "FAIL" for s in stats],
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title="Rule Execution Results",
            xaxis_title="Rule",
            yaxis_title="Passed (1) / Failed (0)",
            template="plotly_white",
            height=400,
            xaxis_tickangle=-45,
        )
        return fig

    # ------------------------------------------------------------------
    # Renderers
    # ------------------------------------------------------------------

    def _render_html(self, report: ValidationReport, figures: list[go.Figure]) -> str:
        """Render the complete self-contained HTML report."""
        figure_divs = "\n".join(
            fig.to_html(full_html=False, include_plotlyjs=False) for fig in figures
        )
        s = report.summary
        score_class = (
            "excellent"
            if s.validation_score >= 0.9
            else (
                "good"
                if s.validation_score >= 0.75
                else "fair" if s.validation_score >= 0.5 else "poor"
            )
        )

        failure_list = "".join(f"<li>{reason}</li>\n" for reason in report.failure_reasons[:100])
        failure_html = (
            f"<ol>{failure_list}</ol>"
            if report.failure_reasons
            else "<p>No validation failures detected.</p>"
        )

        rule_rows = "".join(f"""<tr>
                <td>{stat.rule_name}</td>
                <td>{stat.rule_type}</td>
                <td>{'<span class="badge badge-true">PASS</span>' if stat.passed else '<span class="badge badge-false">FAIL</span>'}</td>
                <td>{stat.errors}</td>
                <td>{stat.failed_rows}</td>
            </tr>\n""" for stat in report.rule_statistics)

        col_rows = "".join(f"""<tr>
                <td>{stat.column}</td>
                <td>{stat.rules_checked}</td>
                <td>{stat.rules_failed}</td>
                <td>{stat.errors}</td>
                <td>{stat.failed_rows}</td>
            </tr>\n""" for stat in report.column_statistics)

        severity_badges = " ".join(
            f'<span class="severity-badge sev-{sev}">{sev}: {count}</span>'
            for sev, count in s.severity_counts.items()
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validation Report - {s.dataset_name}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #212121; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #2E7D32, #4CAF50); color: white; padding: 30px; border-radius: 12px; margin-bottom: 24px; }}
        .header h1 {{ font-size: 28px; margin-bottom: 8px; }}
        .header .subtitle {{ font-size: 14px; opacity: 0.9; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-card .label {{ font-size: 12px; text-transform: uppercase; color: #666; }}
        .stat-card .value {{ font-size: 22px; font-weight: bold; color: #2E7D32; }}
        .stat-card .value.danger {{ color: #F44336; }}
        .stat-card .value.warning {{ color: #FF9800; }}
        .section {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 24px; }}
        .section h2 {{ font-size: 20px; color: #2E7D32; margin-bottom: 16px; border-bottom: 2px solid #4CAF50; padding-bottom: 8px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
        th {{ background: #f0f0f0; font-weight: 600; }}
        tr:hover {{ background: #f9f9f9; }}
        .badge {{ padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }}
        .badge-true {{ background: #E8F5E9; color: #2E7D32; }}
        .badge-false {{ background: #FFEBEE; color: #F44336; }}
        .severity-badge {{ padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; margin-right: 6px; }}
        .sev-critical {{ background: #FFEBEE; color: #F44336; }}
        .sev-error {{ background: #FFF3E0; color: #FF9800; }}
        .sev-warning {{ background: #FFF8E1; color: #F9A825; }}
        .sev-info {{ background: #E3F2FD; color: #1976D2; }}
        .score {{ font-size: 48px; font-weight: bold; text-align: center; padding: 16px; }}
        .score.excellent {{ color: #2E7D32; }}
        .score.good {{ color: #4CAF50; }}
        .score.fair {{ color: #FF9800; }}
        .score.poor {{ color: #F44336; }}
        .footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 40px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>✅ Data Validation Report</h1>
        <div class="subtitle">
            <strong>{s.dataset_name}</strong> &middot;
            Generated by AgriMind AI &middot;
            {s.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="label">Validation Score</div>
            <div class="score {score_class}">{s.validation_score:.3f}</div>
        </div>
        <div class="stat-card">
            <div class="label">Rows Validated</div>
            <div class="value">{s.total_rows:,}</div>
        </div>
        <div class="stat-card">
            <div class="label">Rows Passed</div>
            <div class="value">{s.rows_passed:,}</div>
        </div>
        <div class="stat-card">
            <div class="label">Rows Failed</div>
            <div class="value {'danger' if s.rows_failed else ''}">{s.rows_failed:,}</div>
        </div>
        <div class="stat-card">
            <div class="label">Rules Checked</div>
            <div class="value">{s.rules_checked}</div>
        </div>
        <div class="stat-card">
            <div class="label">Rules Passed / Failed</div>
            <div class="value">{s.rules_passed} / {s.rules_failed}</div>
        </div>
        <div class="stat-card">
            <div class="label">Total Errors</div>
            <div class="value {'danger' if s.total_errors else ''}">{s.total_errors}</div>
        </div>
        <div class="stat-card">
            <div class="label">Severity</div>
            <div>{severity_badges}</div>
        </div>
    </div>

    <div class="section">
        <h2>📈 Visualizations</h2>
        {figure_divs}
    </div>

    <div class="section">
        <h2>📋 Rule Statistics</h2>
        <table>
            <thead>
                <tr><th>Rule</th><th>Type</th><th>Status</th><th>Errors</th><th>Failed Rows</th></tr>
            </thead>
            <tbody>
                {rule_rows}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>🔍 Column Statistics</h2>
        <table>
            <thead>
                <tr><th>Column</th><th>Rules Checked</th><th>Rules Failed</th><th>Errors</th><th>Failed Rows</th></tr>
            </thead>
            <tbody>
                {col_rows}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>🚨 Failure Reasons</h2>
        {failure_html}
    </div>

    <div class="footer">
        <p>Generated by AgriMind AI &mdash; Agricultural Intelligence Platform</p>
        <p>SQL-like status: {'✅ PASSED' if s.passed else '❌ FAILED'}</p>
    </div>
</body>
</html>"""
        return html

    def _render_markdown(self, report: ValidationReport) -> str:
        """Render the validation report to Markdown text."""
        s = report.summary
        lines = [
            f"# Validation Report: {s.dataset_name}",
            "",
            f"- **Generated**: {s.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Validation Score**: {s.validation_score:.4f}",
            f"- **Status**: {'✅ PASSED' if s.passed else '❌ FAILED'}",
            f"- **Rows Validated**: {s.total_rows:,}",
            f"- **Rows Passed**: {s.rows_passed:,}",
            f"- **Rows Failed**: {s.rows_failed:,}",
            f"- **Rules Checked**: {s.rules_checked}",
            f"- **Rules Passed**: {s.rules_passed}",
            f"- **Rules Failed**: {s.rules_failed}",
            f"- **Total Errors**: {s.total_errors}",
            "",
            "## Severity Counts",
            "",
            "| Severity | Count |",
            "|----------|-------|",
        ]
        for sev, count in s.severity_counts.items():
            lines.append(f"| {sev} | {count} |")

        lines.extend(
            [
                "",
                "## Rule Statistics",
                "",
                "| Rule | Type | Status | Errors | Failed Rows |",
                "|------|------|--------|--------|-------------|",
            ]
        )
        for stat in report.rule_statistics:
            status = "PASS" if stat.passed else "FAIL"
            lines.append(
                f"| {stat.rule_name} | {stat.rule_type} | {status} | {stat.errors} | {stat.failed_rows} |"
            )

        lines.extend(
            [
                "",
                "## Column Statistics",
                "",
                "| Column | Rules Checked | Rules Failed | Errors | Failed Rows |",
                "|--------|---------------|--------------|--------|-------------|",
            ]
        )
        for stat in report.column_statistics:
            lines.append(
                f"| {stat.column} | {stat.rules_checked} | {stat.rules_failed} | "
                f"{stat.errors} | {stat.failed_rows} |"
            )

        lines.extend(["", "## Failure Reasons", ""])
        if report.failure_reasons:
            for reason in report.failure_reasons:
                lines.append(f"- {reason}")
        else:
            lines.append("None.")

        return "\n".join(lines)

    def build_failed_rows(
        self,
        df: pd.DataFrame,
        report: ValidationReport,
    ) -> pd.DataFrame:
        """Extract the subset of rows that failed at least one rule.

        Args:
            df: The original validated DataFrame.
            report: Validation report containing failure row indices.

        Returns:
            DataFrame containing only rows that failed validation.
        """
        failed_indices: set[int] = set()
        for result in report.results:
            if not result.passed:
                for error in result.errors:
                    failed_indices.update(error.row_indices)

        if not failed_indices or df.empty:
            return pd.DataFrame(columns=df.columns)

        failed_mask = df.index.isin(sorted(failed_indices))
        return df.loc[failed_mask].copy()

    # ------------------------------------------------------------------
    # JSON serialization helper
    # ------------------------------------------------------------------

    @staticmethod
    def _serializable(value: Any) -> Any:
        """Ensure a value is JSON serializable."""
        if isinstance(value, datetime):
            return value.isoformat()
        return value
