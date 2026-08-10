"""Report CLI command implementation."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()


def register_report_command(app: typer.Typer) -> None:
    """Register the report subcommand on the main CLI app."""

    @app.command()
    def report(report_dir: str = typer.Argument("reports/profiling", help="Directory containing profiling reports")) -> None:
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
