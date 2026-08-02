"""Cleaning CLI command implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from app.data.cleaning import CleaningConfig, CleaningPipeline

console = Console()


def load_cleaning_config(config_path: Optional[str]) -> CleaningConfig:
    """Load cleaning configuration from YAML with fallback to defaults."""
    if not config_path:
        default_config_path = Path("configs/cleaning.yaml")
        if default_config_path.exists():
            config_path = str(default_config_path)
        else:
            logger.warning("No cleaning configuration file found. Using default cleaning configuration.")
            return CleaningConfig()

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config_dict = yaml.safe_load(handle) or {}
        return CleaningConfig(**config_dict)
    except Exception as exc:
        logger.error(f"Failed to load cleaning configuration: {exc}")
        raise typer.Exit(code=1) from exc


def register_clean_command(app: typer.Typer) -> None:
    """Register the clean subcommand on the main CLI app."""

    @app.command()
    def clean(
        dataset: Optional[str] = typer.Argument(None, help="Path to dataset file or directory"),
        strategy: Optional[str] = typer.Option(None, "--strategy", "-s", help="Path to cleaning strategy YAML file"),
        save_interim: bool = typer.Option(False, "--save-interim", help="Save cleaned datasets to interim directory"),
        parallel: bool = typer.Option(False, "--parallel", help="Process multiple datasets in parallel"),
        workers: int = typer.Option(4, "--workers", help="Number of parallel workers"),
        output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory for cleaned datasets"),
        report: Optional[str] = typer.Option(None, "--report", "-r", help="Generate cleaning report and save to the specified directory"),
    ) -> None:
        """Clean dataset(s) using the configured cleaning pipeline."""
        cleaning_config = load_cleaning_config(strategy)
        pipeline = CleaningPipeline(config=cleaning_config, max_workers=workers, parallel=parallel)

        if not dataset:
            console.print("[yellow]Please specify a dataset to clean.[/yellow]")
            raise typer.Exit(code=1)

        dataset_path = Path(dataset)
        if not dataset_path.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            console.print(f"[red]Error:[/red] Dataset path does not exist: {dataset_path}")
            raise typer.Exit(code=1)

        if dataset_path.is_dir():
            logger.info(f"Cleaning all datasets in directory: {dataset_path}")
            results = pipeline.clean_directory(dataset_path, recursive=True, save_interim=save_interim)
            if report:
                pipeline.generate_pipeline_report(report)
            if results:
                summary = pipeline.get_summary()
                console.print(
                    Panel(
                        f"Total datasets: {summary['total_datasets']}\n"
                        f"Successful: {summary['successful']}\n"
                        f"Failed: {summary['failed']}\n"
                        f"Rows before: {summary['total_rows_before']:,} → after: {summary['total_rows_after']:,}",
                        title="Cleaning Summary",
                        border_style="green",
                    )
                )
            else:
                console.print("[yellow]No datasets were cleaned.[/yellow]")
            return

        logger.info(f"Cleaning dataset: {dataset_path}")
        cleaned_data = pipeline.clean_single(dataset_path, save_interim=save_interim, output_path=output)
        if report:
            pipeline.generate_pipeline_report(report)

        metadata = pipeline.results[-1] if pipeline.results else None
        if metadata:
            console.print(
                Panel(
                    f"Dataset: {metadata.dataset_name}\n"
                    f"Rows: {metadata.rows_before:,} → {metadata.rows_after:,}\n"
                    f"Columns: {len(metadata.columns_before)} → {len(metadata.columns_after)}\n"
                    f"Duplicates removed: {metadata.duplicates_removed:,}\n"
                    f"Execution time: {metadata.execution_time_seconds:.2f} seconds",
                    title="Cleaning Summary",
                    border_style="green",
                )
            )
        else:
            console.print(f"[green]Cleaned[/green] {len(cleaned_data)} rows")
