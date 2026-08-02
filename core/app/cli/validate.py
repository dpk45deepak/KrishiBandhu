"""Validation CLI command implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from app.config.config import load_config as load_project_config
from app.data.validation.report import ValidationReportGenerator
from app.data.validation.validator import ValidationEngine, ValidationEngineConfig
from app.utils.dataset_scanner import DatasetScanner

console = Console()


def _load_dataframe(file_path: str | Path):
    """Load a DataFrame for failed-row extraction using the project loader."""
    from app.data.ingestion.loader import DataLoader

    loader = DataLoader(engine="pandas")
    return loader.load(file_path)


def _display_validation_summary(report) -> None:
    """Display a validation summary in the console."""
    s = report.summary
    status = "✅ PASSED" if s.passed else "❌ FAILED"
    status_color = "green" if s.passed else "red"
    score_color = "green" if s.validation_score >= 0.9 else "yellow" if s.validation_score >= 0.75 else "red"

    info = (
        f"Status: [{status_color}]{status}[/{status_color}] | "
        f"Score: [{score_color}]{s.validation_score:.4f}[/{score_color}] | "
        f"Rows: {s.total_rows:,} (passed {s.rows_passed:,} / failed {s.rows_failed:,}) | "
        f"Rules: {s.rules_checked} (passed {s.rules_passed} / failed {s.rules_failed}) | "
        f"Errors: {s.total_errors}"
    )
    console.print(Panel(info, title=f"🔍 {s.dataset_name}", border_style="green" if s.passed else "red"))


def _display_report_paths(paths: dict[str, str]) -> None:
    """Display the paths of generated reports."""
    for fmt, path in paths.items():
        icon = {"html": "🌐", "json": "📋", "markdown": "📝"}.get(fmt, "📄")
        console.print(f"  {icon} {fmt.upper()}: {path}")


def register_validate_command(app: typer.Typer) -> None:
    """Register the validate subcommand on the main CLI app."""

    @app.command()
    def validate(
        file_path: Optional[str] = typer.Argument(None, help="Path to a dataset to validate (default: all files in data/raw/)"),
        schema: Optional[str] = typer.Option(None, "--schema", "-s", help="Path to a validation schema YAML file"),
        strict: bool = typer.Option(False, "--strict", help="Enable strict mode (raise on failure)"),
        fail_fast: bool = typer.Option(False, "--fail-fast", help="Stop at the first failing rule"),
        output_dir: str = typer.Option("reports/validation", "--output", "-o", help="Report output directory"),
    ) -> None:
        """Validate datasets against a schema and generate validation reports."""
        config = load_project_config()
        engine_config = ValidationEngineConfig(
            strict_mode=strict or config.validation.strict_mode,
            fail_fast=fail_fast or config.validation.fail_fast,
            max_missing_percentage=config.validation.max_missing_percentage,
            duplicate_threshold=config.validation.duplicate_threshold,
            report_generation=config.validation.report_generation,
        )

        schema_path = schema or config.validation.default_schema
        engine = ValidationEngine(schema=schema_path, config=engine_config)
        report_gen = ValidationReportGenerator(output_dir=output_dir)

        if file_path:
            target_path = Path(file_path)
            if not target_path.exists():
                logger.error(f"File not found: {target_path}")
                console.print(f"[red]Error:[/red] File not found: {target_path}")
                raise typer.Exit(code=1)

            console.print(f"\n[bold]Validating:[/bold] {target_path.name}")
            report = engine.load_and_validate(target_path)
            failed_df = report_gen.build_failed_rows(_load_dataframe(target_path), report)
            paths = report_gen.generate_all(report, failed_df)
            _display_validation_summary(report)
            _display_report_paths(paths)
        else:
            scanner = DatasetScanner()
            datasets = scanner.scan(recursive=False)

            if not datasets:
                console.print(
                    "[yellow]No datasets found in data/raw/.[/yellow]\n"
                    "Use 'python main.py scan' to discover datasets, or "
                    "specify a file with 'python main.py validate <file>'."
                )
                raise typer.Exit(code=0)

            for ds in datasets:
                console.print(f"\n[bold]Validating:[/bold] {ds.filename}")
                report = engine.load_and_validate(ds.file_path)
                failed_df = report_gen.build_failed_rows(_load_dataframe(ds.file_path), report)
                paths = report_gen.generate_all(report, failed_df)
                _display_validation_summary(report)
                _display_report_paths(paths)

        console.print("[green]✓[/green] Validation complete.")
