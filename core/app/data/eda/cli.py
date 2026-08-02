# app/data/eda/cli.py
"""
AgriMind AI - CLI Interface
"""
import typer
from typing import Optional, List
from pathlib import Path
import polars as pl
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint
from loguru import logger
import sys
from datetime import datetime

from app.data.eda.analyzer import EDAAnalyzer
from app.data.eda.models import EDAAnalysisConfig
from app.data.eda.report import ReportGenerator
from app.data.eda.dashboard import DashboardGenerator

app = typer.Typer(
    name="AgriMind",
    help="Agricultural Intelligence Platform",
    add_completion=False
)
console = Console()


@app.command()
def eda(
    dataset: Optional[Path] = typer.Argument(
        None,
        help="Path to dataset file (CSV, Parquet, etc.)"
    ),
    dashboard: bool = typer.Option(
        False,
        "--dashboard",
        help="Generate interactive dashboard"
    ),
    html: bool = typer.Option(
        False,
        "--html",
        help="Generate HTML report"
    ),
    figures: bool = typer.Option(
        False,
        "--figures",
        help="Generate standalone figures"
    ),
    output_dir: Path = typer.Option(
        Path("./reports/eda"),
        "--output-dir",
        "-o",
        help="Output directory for reports"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file path"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """
    Run automated EDA on agricultural dataset.
    """
    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"
    )
    
    # Display header
    console.print(Panel.fit(
        "[bold green]🌾 AgriMind AI - Automated EDA Platform[/bold green]",
        border_style="green"
    ))
    
    # Check if dataset provided
    if dataset is None:
        console.print("[red]Error: Please provide a dataset file path[/red]")
        raise typer.Exit(code=1)
    
    if not dataset.exists():
        console.print(f"[red]Error: Dataset file not found: {dataset}[/red]")
        raise typer.Exit(code=1)
    
    # Load configuration
    config = EDAAnalysisConfig()
    if config_file and config_file.exists():
        config = EDAAnalysisConfig.parse_file(config_file)
        console.print(f"[green]✓ Loaded configuration from {config_file}[/green]")
    
    # Load dataset
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Loading dataset...", total=None)
        
        try:
            # Determine file format
            ext = dataset.suffix.lower()
            if ext == '.csv':
                df = pl.read_csv(dataset, infer_schema_length=10000)
            elif ext == '.parquet':
                df = pl.read_parquet(dataset)
            elif ext in ['.xlsx', '.xls']:
                df = pl.read_excel(dataset)
            else:
                console.print(f"[red]Unsupported file format: {ext}[/red]")
                raise typer.Exit(code=1)
                
            progress.update(task, completed=True)
        except Exception as e:
            console.print(f"[red]Failed to load dataset: {e}[/red]")
            raise typer.Exit(code=1)
    
    # Display dataset info
    console.print("\n[bold]Dataset Info[/bold]")
    info_table = Table()
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="white")
    
    info_table.add_row("Rows", f"{df.height:,}")
    info_table.add_row("Columns", f"{df.width:,}")
    info_table.add_row("Memory", f"{df.estimated_size('mb'):.2f} MB")
    
    console.print(info_table)
    
    # Run EDA
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Running EDA analysis...", total=None)
        
        try:
            analyzer = EDAAnalyzer(df, config, dataset.stem)
            report = analyzer.analyze()
            progress.update(task, completed=True)
        except Exception as e:
            console.print(f"[red]EDA analysis failed: {e}[/red]")
            logger.exception(e)
            raise typer.Exit(code=1)
    
    # Generate reports
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Generate reports
        task = progress.add_task("[cyan]Generating reports...", total=None)
        
        try:
            report_gen = ReportGenerator(report, output_dir)
            report_paths = report_gen.generate_all()
            progress.update(task, completed=True)
        except Exception as e:
            console.print(f"[red]Report generation failed: {e}[/red]")
            logger.exception(e)
            raise typer.Exit(code=1)
    
    # Generate dashboard if requested
    if dashboard:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Generating dashboard...", total=None)
            
            try:
                dashboard_gen = DashboardGenerator(report, output_dir)
                dashboard_path = dashboard_gen.generate_dashboard()
                progress.update(task, completed=True)
            except Exception as e:
                console.print(f"[red]Dashboard generation failed: {e}[/red]")
                logger.exception(e)
                raise typer.Exit(code=1)
    
    # Summary
    console.print("\n[bold green]✅ EDA completed successfully![/bold green]")
    console.print(f"\n[bold]Reports generated in:[/bold] {output_dir}/")
    
    if html:
        console.print(f"  • HTML Report: {output_dir / 'report.html'}")
    if dashboard:
        console.print(f"  • Dashboard: {output_dir / 'dashboard.html'}")
    console.print(f"  • Markdown: {output_dir / 'report.md'}")
    console.print(f"  • JSON: {output_dir / 'report.json'}")
    
    # Display quick stats
    console.print("\n[bold]Quick Stats[/bold]")
    stats_table = Table()
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")
    
    stats_table.add_row(
        "Data Quality Score",
        f"{np.mean([s.overall_score for s in report.quality_scores.values()]):.2f}/1.0"
    )
    stats_table.add_row(
        "Features with Missing Data",
        f"{len([p for p in report.missingness_patterns if p.missing_count > 0])}"
    )
    stats_table.add_row(
        "ML Readiness",
        "Ready" if report.ml_readiness.is_regression_ready or report.ml_readiness.is_classification_ready else "Needs Work"
    )
    stats_table.add_row(
        "Recommendations",
        f"{len(report.recommendations)}"
    )
    
    console.print(stats_table)


@app.command()
def version():
    """Display version information."""
    console.print("[bold green]AgriMind AI v1.0.0[/bold green]")
    console.print("Sprint 6 - Automated EDA Platform")


if __name__ == "__main__":
    app()