"""Report generation for data profiling results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from loguru import logger

from app.constants.constants import COLORS
from app.data.profiling.profiler import ProfilingResult
from app.utils.decorators import timer
from app.utils.path_utils import ensure_dir, resolve_path


class ReportGenerator:
    """Generates profiling reports in HTML, JSON, and Markdown formats.

    HTML reports include interactive Plotly visualizations.
    JSON reports contain the full structured profiling data.
    Markdown reports provide a human-readable summary.
    """

    def __init__(self, output_dir: str | Path = "reports/profiling") -> None:
        """Initialize the report generator.

        Args:
            output_dir: Directory to save reports.
        """
        self.output_dir = ensure_dir(output_dir)
        logger.debug(f"ReportGenerator initialized, output: {self.output_dir}")

    @timer
    def generate_all(self, result: ProfilingResult) -> dict[str, str]:
        """Generate all report formats for a profiling result.

        Args:
            result: The profiling result to report on.

        Returns:
            Dict mapping format names to file paths.
        """
        paths = {}
        paths["html"] = self._save_html(result)
        paths["json"] = self._save_json(result)
        paths["markdown"] = self._save_markdown(result)
        logger.info(f"Reports generated for {result.filename}: {paths}")
        return paths

    def _save_html(self, result: ProfilingResult) -> str:
        """Generate and save an HTML report with interactive visualizations."""
        safe_name = result.filename.rsplit(".", 1)[0]
        file_path = self.output_dir / f"{safe_name}_profile_report.html"
        file_path = resolve_path(file_path)

        figures = self._build_figures(result)
        html_content = self._render_html(result, figures)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML report saved: {file_path}")
        return str(file_path)

    def _build_figures(self, result: ProfilingResult) -> list[go.Figure]:
        """Build Plotly figures from profiling results."""
        figures: list[go.Figure] = []

        fig_missing = self._build_missing_chart(result)
        if fig_missing:
            figures.append(fig_missing)

        fig_dtypes = self._build_dtype_chart(result)
        if fig_dtypes:
            figures.append(fig_dtypes)

        fig_corr = self._build_correlation_heatmap(result)
        if fig_corr:
            figures.append(fig_corr)

        fig_outliers = self._build_outlier_chart(result)
        if fig_outliers:
            figures.append(fig_outliers)

        return figures

    def _build_missing_chart(self, result: ProfilingResult) -> go.Figure | None:
        """Build a bar chart of missing values per column."""
        cols_with_missing = [c for c in result.columns if c.missing_count > 0]
        if not cols_with_missing:
            fig = go.Figure()
            fig.add_annotation(text="No missing values found", showarrow=False)
            fig.update_layout(title="Missing Values")
            return fig

        fig = go.Figure(
            data=[
                go.Bar(
                    x=[c.name for c in cols_with_missing],
                    y=[c.missing_count for c in cols_with_missing],
                    marker_color=COLORS["missing"],
                    text=[f"{c.missing_ratio:.1%}" for c in cols_with_missing],
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title="Missing Values by Column",
            xaxis_title="Column",
            yaxis_title="Missing Count",
            template="plotly_white",
            height=400,
        )
        return fig

    def _build_dtype_chart(self, result: ProfilingResult) -> go.Figure | None:
        """Build a pie chart showing data type distribution."""
        dtype_counts: dict[str, int] = {}
        for col in result.columns:
            base_type = col.dtype.split("[")[0]
            base_type = base_type.split("(")[0]
            dtype_counts[base_type] = dtype_counts.get(base_type, 0) + 1

        if not dtype_counts:
            return None

        color_list = [
            COLORS["primary"],
            COLORS["secondary"],
            COLORS["info"],
            COLORS["warning"],
            COLORS["danger"],
        ]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(dtype_counts.keys()),
                    values=list(dtype_counts.values()),
                    hole=0.3,
                    marker=dict(colors=color_list[: len(dtype_counts)]),
                )
            ]
        )
        fig.update_layout(
            title="Data Type Distribution",
            template="plotly_white",
            height=400,
        )
        return fig

    def _build_correlation_heatmap(self, result: ProfilingResult) -> go.Figure | None:
        """Build a correlation heatmap for numeric columns."""
        if len(result.numeric_columns) < 2 or not result.correlation_matrix:
            return None

        cols = result.numeric_columns
        corr_data = []
        for col in cols:
            row = []
            for other in cols:
                val = result.correlation_matrix.get(col, {}).get(other, 0)
                row.append(val)
            corr_data.append(row)

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_data,
                x=cols,
                y=cols,
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr_data],
                texttemplate="%{text}",
                textfont={"size": 10},
            )
        )
        fig.update_layout(
            title="Correlation Matrix (Numeric Columns)",
            template="plotly_white",
            height=max(400, len(cols) * 40),
            width=max(500, len(cols) * 50),
        )
        return fig

    def _build_outlier_chart(self, result: ProfilingResult) -> go.Figure | None:
        """Build a bar chart of outlier counts per numeric column."""
        outlier_data = []
        for col in result.columns:
            if col.numeric_summary and col.numeric_summary.outlier_count > 0:
                outlier_data.append(
                    {
                        "column": col.name,
                        "outliers": col.numeric_summary.outlier_count,
                        "ratio": col.numeric_summary.outlier_ratio,
                    }
                )

        if not outlier_data:
            return None

        fig = go.Figure(
            data=[
                go.Bar(
                    x=[d["column"] for d in outlier_data],
                    y=[d["outliers"] for d in outlier_data],
                    marker_color=COLORS["outlier"],
                    text=[f"{d['ratio']:.1%}" for d in outlier_data],
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title="Outliers by Column (IQR Method)",
            xaxis_title="Column",
            yaxis_title="Outlier Count",
            template="plotly_white",
            height=400,
        )
        return fig

    def _render_html(self, result: ProfilingResult, figures: list[go.Figure]) -> str:
        """Render the complete HTML report."""
        figure_divs = ""
        for fig in figures:
            figure_divs += fig.to_html(full_html=False, include_plotlyjs=False)
            figure_divs += "\n"

        def _format_optional(value: Any, fmt: str = "") -> str:
            if value is None:
                return "n/a"
            if fmt:
                return format(value, fmt)
            return str(value)

        # Build column info rows
        column_rows = ""
        for col in result.columns:
            extra_info = ""
            if col.numeric_summary:
                extra_info = (
                    f"Mean: {_format_optional(col.numeric_summary.mean, '.4f')} | "
                    f"Std: {_format_optional(col.numeric_summary.std, '.4f')} | "
                    f"Outliers: {_format_optional(col.numeric_summary.outlier_count)}"
                )
            elif col.categorical_summary:
                extra_info = (
                    f"Unique: {_format_optional(col.categorical_summary.unique_count)} | "
                    f"High Card: {_format_optional(col.categorical_summary.is_high_cardinality)}"
                )

            column_rows += f"""<tr>
                <td>{col.name}</td>
                <td>{col.dtype}</td>
                <td>{col.missing_count}</td>
                <td>{col.missing_ratio:.2%}</td>
                <td>{col.unique_count}</td>
                <td>{col.is_constant}</td>
                <td>{col.is_numeric}</td>
                <td>{col.is_categorical}</td>
                <td>{col.is_target_candidate}</td>
                <td>{extra_info}</td>
            </tr>\n"""

        # High correlations rows
        corr_rows = ""
        for col1, col2, val in result.high_correlations:
            corr_rows += f"<li>{col1} ↔ {col2}: {val:.4f}</li>\n"

        # Include plotlyjs once
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Profile Report - {result.filename}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #212121; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #2E7D32, #4CAF50); color: white; padding: 30px; border-radius: 12px; margin-bottom: 24px; }}
        .header h1 {{ font-size: 28px; margin-bottom: 8px; }}
        .header .subtitle {{ font-size: 14px; opacity: 0.9; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-card .label {{ font-size: 12px; text-transform: uppercase; color: #666; }}
        .stat-card .value {{ font-size: 24px; font-weight: bold; color: #2E7D32; }}
        .stat-card .value.warning {{ color: #FF9800; }}
        .stat-card .value.danger {{ color: #F44336; }}
        .section {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 24px; }}
        .section h2 {{ font-size: 20px; color: #2E7D32; margin-bottom: 16px; border-bottom: 2px solid #4CAF50; padding-bottom: 8px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
        th {{ background: #f0f0f0; font-weight: 600; position: sticky; top: 0; }}
        tr:hover {{ background: #f9f9f9; }}
        .badge {{ padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }}
        .badge-true {{ background: #E8F5E9; color: #2E7D32; }}
        .badge-false {{ background: #FFEBEE; color: #F44336; }}
        .plot-container {{ margin: 16px 0; }}
        .quality-score {{ font-size: 48px; font-weight: bold; text-align: center; padding: 20px; }}
        .quality-excellent {{ color: #2E7D32; }}
        .quality-good {{ color: #4CAF50; }}
        .quality-fair {{ color: #FF9800; }}
        .quality-poor {{ color: #F44336; }}
        .footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 40px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Data Profile Report</h1>
        <div class="subtitle">
            <strong>{result.filename}</strong> &middot;
            Generated by AgriMind AI &middot;
            {result.suggested_ml_task.upper()} task suggested
        </div>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="label">Rows</div>
            <div class="value">{result.row_count:,}</div>
        </div>
        <div class="stat-card">
            <div class="label">Columns</div>
            <div class="value">{result.column_count}</div>
        </div>
        <div class="stat-card">
            <div class="label">Missing Values</div>
            <div class="value {'danger' if result.total_missing_ratio > 0.1 else 'warning' if result.total_missing_ratio > 0 else ''}">{result.total_missing:,} ({result.total_missing_ratio:.2%})</div>
        </div>
        <div class="stat-card">
            <div class="label">Duplicate Rows</div>
            <div class="value">{result.duplicate_rows:,} ({result.duplicate_ratio:.2%})</div>
        </div>
        <div class="stat-card">
            <div class="label">Memory Usage</div>
            <div class="value">{result.memory_usage_mb:.2f} MB</div>
        </div>
        <div class="stat-card">
            <div class="label">Data Quality Score</div>
            <div class="quality-score {'quality-excellent' if result.quality_score >= 0.8 else 'quality-good' if result.quality_score >= 0.6 else 'quality-fair' if result.quality_score >= 0.4 else 'quality-poor'}">{result.quality_score:.2f}</div>
        </div>
    </div>

    <div class="section">
        <h2>📋 Column Profile</h2>
        <div style="overflow-x: auto; max-height: 500px; overflow-y: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Dtype</th>
                        <th>Missing</th>
                        <th>Missing %</th>
                        <th>Unique</th>
                        <th>Constant</th>
                        <th>Numeric</th>
                        <th>Categorical</th>
                        <th>Target?</th>
                        <th>Extra</th>
                    </tr>
                </thead>
                <tbody>
                    {column_rows}
                </tbody>
            </table>
        </div>
    </div>

    <div class="section">
        <h2>📈 Visualizations</h2>
        {figure_divs}
    </div>

    <div class="section">
        <h2>🔗 High Correlations (|r| > 0.7)</h2>
        {f'<ol>{corr_rows}</ol>' if corr_rows else '<p>No high correlations found.</p>'}
    </div>

    <div class="section">
        <h2>🎯 ML Task Suggestion</h2>
        <p style="font-size: 18px;">
            <strong>{result.suggested_ml_task.upper()}</strong>
            &mdash; inferred from {len(result.target_candidates)} target candidate(s): {', '.join(result.target_candidates) if result.target_candidates else 'none identified'}.
        </p>
        <p>
            Numeric features: {len(result.numeric_columns)} | Categorical features: {len(result.categorical_columns)} |
            Constant columns: {len(result.constant_columns)} | High cardinality: {len(result.high_cardinality_columns)}
        </p>
    </div>

    <div class="footer">
        <p>Generated by AgriMind AI &mdash; Agricultural Intelligence Platform</p>
        <p>Report saved at: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>"""
        return html

    def _save_json(self, result: ProfilingResult) -> str:
        """Generate and save a JSON report with full profiling data."""
        safe_name = result.filename.rsplit(".", 1)[0]
        file_path = self.output_dir / f"{safe_name}_profile_report.json"
        file_path = resolve_path(file_path)

        data = self._result_to_dict(result)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"JSON report saved: {file_path}")
        return str(file_path)

    def _save_markdown(self, result: ProfilingResult) -> str:
        """Generate and save a Markdown summary report."""
        safe_name = result.filename.rsplit(".", 1)[0]
        file_path = self.output_dir / f"{safe_name}_profile_summary.md"
        file_path = resolve_path(file_path)

        md = self._render_markdown(result)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md)

        logger.info(f"Markdown report saved: {file_path}")
        return str(file_path)

    def _result_to_dict(self, result: ProfilingResult) -> dict[str, Any]:
        """Convert a ProfilingResult to a JSON-serializable dict."""
        columns_data = []
        for col in result.columns:
            col_dict: dict[str, Any] = {
                "name": col.name,
                "dtype": col.dtype,
                "missing_count": col.missing_count,
                "missing_ratio": col.missing_ratio,
                "unique_count": col.unique_count,
                "unique_ratio": col.unique_ratio,
                "is_constant": col.is_constant,
                "is_numeric": col.is_numeric,
                "is_categorical": col.is_categorical,
                "is_target_candidate": col.is_target_candidate,
            }
            if col.numeric_summary:
                ns = col.numeric_summary
                col_dict["numeric_summary"] = {
                    "mean": ns.mean,
                    "median": ns.median,
                    "std": ns.std,
                    "min": ns.min,
                    "max": ns.max,
                    "q1": ns.q1,
                    "q3": ns.q3,
                    "iqr": ns.iqr,
                    "skewness": ns.skewness,
                    "kurtosis": ns.kurtosis,
                    "outlier_count": ns.outlier_count,
                    "outlier_ratio": ns.outlier_ratio,
                    "zero_count": ns.zero_count,
                    "negative_count": ns.negative_count,
                }
            if col.categorical_summary:
                cs = col.categorical_summary
                col_dict["categorical_summary"] = {
                    "unique_count": cs.unique_count,
                    "top_values": cs.top_values,
                    "cardinality_ratio": cs.cardinality_ratio,
                    "is_high_cardinality": cs.is_high_cardinality,
                }
            columns_data.append(col_dict)

        return {
            "filename": result.filename,
            "file_path": result.file_path,
            "shape": list(result.shape),
            "row_count": result.row_count,
            "column_count": result.column_count,
            "total_missing": result.total_missing,
            "total_missing_ratio": result.total_missing_ratio,
            "duplicate_rows": result.duplicate_rows,
            "duplicate_ratio": result.duplicate_ratio,
            "memory_usage_mb": result.memory_usage_mb,
            "numeric_columns": result.numeric_columns,
            "categorical_columns": result.categorical_columns,
            "constant_columns": result.constant_columns,
            "high_cardinality_columns": result.high_cardinality_columns,
            "target_candidates": result.target_candidates,
            "suggested_ml_task": result.suggested_ml_task,
            "quality_score": result.quality_score,
            "high_correlations": [
                {"col1": c1, "col2": c2, "correlation": v}
                for c1, c2, v in result.high_correlations
            ],
            "columns": columns_data,
        }

    def _render_markdown(self, result: ProfilingResult) -> str:
        """Render a Markdown summary report."""
        def _format_optional(value: Any, fmt: str = "") -> str:
            if value is None:
                return "n/a"
            if fmt:
                return format(value, fmt)
            return str(value)

        lines = [
            f"# Data Profile Summary: {result.filename}",
            "",
            f"- **Rows**: {result.row_count:,}",
            f"- **Columns**: {result.column_count}",
            f"- **Total Missing**: {result.total_missing:,} ({result.total_missing_ratio:.2%})",
            f"- **Duplicate Rows**: {result.duplicate_rows:,} ({result.duplicate_ratio:.2%})",
            f"- **Memory**: {result.memory_usage_mb:.2f} MB",
            f"- **Quality Score**: {result.quality_score:.4f}",
            f"- **Suggested ML Task**: {result.suggested_ml_task.upper()}",
            "",
            "## Column Details",
            "",
            "| Column | Dtype | Missing | Unique | Constant | Numeric | Target? |",
            "|--------|-------|---------|--------|----------|---------|--------|",
        ]
        for col in result.columns:
            lines.append(
                f"| {col.name} | {col.dtype} | {col.missing_ratio:.2%} | "
                f"{col.unique_count} | {col.is_constant} | {col.is_numeric} | "
                f"{col.is_target_candidate} |"
            )

        lines.extend(["", "## Numeric Columns", ""])
        for col in result.columns:
            if col.numeric_summary:
                ns = col.numeric_summary
                lines.append(
                    f"- **{col.name}**: mean={_format_optional(ns.mean, '.4f')}, "
                    f"std={_format_optional(ns.std, '.4f')}, min={_format_optional(ns.min)}, "
                    f"max={_format_optional(ns.max)}, skew={_format_optional(ns.skewness, '.4f')}, "
                    f"kurt={_format_optional(ns.kurtosis, '.4f')}, "
                    f"outliers={_format_optional(ns.outlier_count)} ({_format_optional(ns.outlier_ratio, '.2%')})"
                )
            else:
                lines.append(f"- **{col.name}**: no numeric summary available")

        lines.extend(["", "## High Correlations", ""])
        if result.high_correlations:
            for col1, col2, val in result.high_correlations:
                lines.append(f"- {col1} ↔ {col2}: |r| = {val:.4f}")
        else:
            lines.append("None found.")

        lines.extend(["", "## Target Candidates", ""])
        if result.target_candidates:
            for col in result.target_candidates:
                lines.append(f"- {col}")
        else:
            lines.append("No clear target candidates identified.")

        return "\n".join(lines)
