"""Scan CLI command implementation."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from app.config.config import load_config as load_project_config
from app.utils.dataset_scanner import DatasetScanner

console = Console()


def register_scan_command(app: typer.Typer) -> None:
    """Register the scan subcommand on the main CLI app."""

    @app.command()
    def scan(
        data_dir: Optional[str] = typer.Argument(None, help="Directory to scan (default: data/raw)"),
        recursive: bool = typer.Option(True, "--recursive", "-r", help="Scan recursively"),
    ) -> None:
        """Scan a directory for supported datasets and display a summary."""
        load_project_config()
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
