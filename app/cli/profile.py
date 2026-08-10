"""Profile CLI command implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from app.config.config import load_config as load_project_config
from app.data.profiling.profiler import DataProfiler
from app.data.profiling.report_generator import ReportGenerator
from app.utils.dataset_scanner import DatasetScanner

console = Console()


def _display_profile_summary(result) -> None:
    """Display a profiling summary in the console."""
    quality = result.quality_score
    quality_color = "green" if quality >= 0.8 else "yellow" if quality >= 0.6 else "red"

    info = (
        f"Rows: {result.row_count:,} | "
        f"Cols: {result.column_count} | "
        f"Missing: {result.total_missing:,} ({result.total_missing_ratio:.2%}) | "
        f"Duplicates: {result.duplicate_rows:,} ({result.duplicate_ratio:.2%}) | "
        f"Numeric: {len(result.numeric_columns)} | "
        f"Categorical: {len(result.categorical_columns)} | "
        f"Quality: [{quality_color}]{quality:.4f}[/{quality_color}] | "
        f"ML Task: [bold]{result.suggested_ml_task.upper()}[/bold]"
    )
    console.print(Panel(info, title=f"📊 {result.filename}", border_style="green"))


def _display_report_paths(paths: dict[str, str]) -> None:
    """Display the paths of generated reports."""
    for fmt, path in paths.items():
        icon = {"html": "🌐", "json": "📋", "markdown": "📝"}.get(fmt, "📄")
        console.print(f"  {icon} {fmt.upper()}: {path}")


def register_profile_command(app: typer.Typer) -> None:
    """Register the profile subcommand on the main CLI app."""

    @app.command()
    def profile(
        file_path: Optional[str] = typer.Argument(None, help="Path to a specific file to profile (default: all files in data/raw/)"),
        output_dir: str = typer.Option("reports/profiling", "--output", "-o", help="Report output directory"),
    ) -> None:
        """Profile datasets and generate comprehensive reports."""
        load_project_config()
        profiler = DataProfiler(engine="pandas")
        report_gen = ReportGenerator(output_dir=output_dir)

        if file_path:
            target_path = Path(file_path)
            if not target_path.exists():
                logger.error(f"File not found: {target_path}")
                console.print(f"[red]Error:[/red] File not found: {target_path}")
                raise typer.Exit(code=1)

            console.print(f"\n[bold]Profiling:[/bold] {target_path.name}")
            result = profiler.profile(target_path)
            paths = report_gen.generate_all(result)
            _display_profile_summary(result)
            _display_report_paths(paths)
        else:
            scanner = DatasetScanner()
            datasets = scanner.scan(recursive=False)

            if not datasets:
                console.print(
                    "[yellow]No datasets found in data/raw/.[/yellow]\n"
                    "Use 'python main.py scan' to discover datasets, or "
                    "specify a file with 'python main.py profile <file>'."
                )
                raise typer.Exit(code=0)

            for ds in datasets:
                console.print(f"\n[bold]Profiling:[/bold] {ds.filename}")
                result = profiler.profile(ds.file_path)
                paths = report_gen.generate_all(result)
                _display_profile_summary(result)
                _display_report_paths(paths)

        console.print("[green]✓[/green] Profiling complete.")
