"""AgriMind AI - Agricultural Intelligence Platform CLI.

Usage:
    python main.py scan
    python main.py profile
    python main.py report
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from app.config.config import load_config
from app.constants.constants import get_project_root
from app.data.profiling.profiler import DataProfiler
from app.data.profiling.report_generator import ReportGenerator
from app.data.validation.report import ValidationReportGenerator
from app.data.validation.validator import ValidationEngine, ValidationEngineConfig
from app.logger.logger import setup_logger
from app.utils.dataset_scanner import DatasetScanner

# Create Typer app
app = typer.Typer(
    name="agrimind",
    help="AgriMind AI - Agricultural Intelligence Platform",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

console = Console()


@app.callback()
def main_callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """AgriMind AI CLI - Dataset scanning, profiling, and reporting."""
    level = "DEBUG" if verbose else "INFO"
    config = load_config()
    setup_logger(level=level, colored=True)
    if verbose:
        logger.info("Verbose logging enabled")


@app.command()
def scan(
    data_dir: Optional[str] = typer.Argument(
        None, help="Directory to scan (default: data/raw)"
    ),
    recursive: bool = typer.Option(True, "--recursive", "-r", help="Scan recursively"),
) -> None:
    """Scan a directory for supported datasets and display a summary."""
    config = load_config()
    scanner = DatasetScanner(data_dir=data_dir)
    results = scanner.scan(recursive=recursive)

    if not results:
        console.print(
            Panel(
                "[yellow]No supported datasets found.[/yellow]\n\n"
                "Supported formats: .csv, .xls, .xlsx, .parquet\n\n"
                "Place your data files in data/raw/ or specify a custom path.",
                title="Scan Complete",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=0)

    scanner.print_summary_table()

    console.print(
        Panel(
            f"Total datasets: {len(results)}\n"
            f"Total rows: {sum(r.rows or 0 for r in results):,}\n"
            f"Total estimated memory: {sum(r.memory_usage_mb for r in results):.2f} MB",
            title="Summary",
            border_style="green",
        )
    )


@app.command()
def profile(
    file_path: Optional[str] = typer.Argument(
        None, help="Path to a specific file to profile (default: all files in data/raw/)"
    ),
    output_dir: str = typer.Option(
        "reports/profiling", "--output", "-o", help="Report output directory"
    ),
) -> None:
    """Profile datasets and generate comprehensive reports."""
    config = load_config()
    profiler = DataProfiler(engine="pandas")
    report_gen = ReportGenerator(output_dir=output_dir)

    if file_path:
        # Profile a single file
        target_path = Path(file_path)
        if not target_path.exists():
            logger.error(f"File not found: {target_path}")
            console.print(f"[red]Error:[/red] File not found: {target_path}")
            raise typer.Exit(code=1)

        console.print(f"\n[bold]Profiling:[/bold] {target_path.name}")
        result = profiler.profile(target_path)
        paths = report_gen.generate_all(result)

        # Display summary
        _display_profile_summary(result)
        _display_report_paths(paths)
    else:
        # Profile all datasets in data/raw/
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


@app.command()
def report(
    report_dir: str = typer.Argument(
        "reports/profiling", help="Directory containing profiling reports"
    ),
) -> None:
    """Show an overview of available profiling reports."""
    report_path = Path(report_dir)

    if not report_path.exists():
        console.print(f"[red]Error:[/red] Report directory not found: {report_path}")
        raise typer.Exit(code=1)

    html_files = sorted(report_path.glob("*_profile_report.html"))
    json_files = sorted(report_path.glob("*_profile_report.json"))
    md_files = sorted(report_path.glob("*_profile_summary.md"))

    if not html_files and not json_files and not md_files:
        console.print(
            f"[yellow]No profiling reports found in {report_dir}.[/yellow]\n"
            "Run 'python main.py profile' first to generate reports."
        )
        raise typer.Exit(code=0)

    console.print(Panel(f"[bold]Reports in:[/bold] {report_path.resolve()}", title="Report Overview"))

    if html_files:
        console.print("\n[bold]HTML Reports:[/bold]")
        for f in html_files:
            console.print(f"  📊 {f.name}")

    if json_files:
        console.print("\n[bold]JSON Reports:[/bold]")
        for f in json_files:
            console.print(f"  📋 {f.name}")

    if md_files:
        console.print("\n[bold]Markdown Summaries:[/bold]")
        for f in md_files:
            console.print(f"  📝 {f.name}")

    console.print(f"\nTotal: {len(html_files)} HTML, {len(json_files)} JSON, {len(md_files)} Markdown")


@app.command()
def validate(
    file_path: Optional[str] = typer.Argument(
        None, help="Path to a dataset to validate (default: all files in data/raw/)"
    ),
    schema: Optional[str] = typer.Option(
        None, "--schema", "-s", help="Path to a validation schema YAML file"
    ),
    strict: bool = typer.Option(False, "--strict", help="Enable strict mode (raise on failure)"),
    fail_fast: bool = typer.Option(False, "--fail-fast", help="Stop at the first failing rule"),
    output_dir: str = typer.Option(
        "reports/validation", "--output", "-o", help="Report output directory"
    ),
) -> None:
    """Validate datasets against a schema and generate validation reports."""
    config = load_config()
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
        failed_df = report_gen.build_failed_rows(
            _load_dataframe(target_path), report
        )
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
            failed_df = report_gen.build_failed_rows(
                _load_dataframe(ds.file_path), report
            )
            paths = report_gen.generate_all(report, failed_df)
            _display_validation_summary(report)
            _display_report_paths(paths)

    console.print("[green]✓[/green] Validation complete.")


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


def main() -> None:
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
