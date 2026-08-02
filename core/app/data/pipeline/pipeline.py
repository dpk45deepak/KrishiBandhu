"""Main pipeline module with CLI interface."""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from .orchestrator import PipelineOrchestrator
from .models import StageType
from loguru import logger

app = typer.Typer()
console = Console()


@app.command()
def run(
    dataset: Optional[Path] = typer.Argument(None, help="Path to dataset file"),
    config: Path = typer.Option("configs/pipeline.yaml", "--config", "-c", help="Pipeline configuration file"),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume from last checkpoint"),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", help="Specific checkpoint to resume from"),
    stage: Optional[StageType] = typer.Option(None, "--stage", "-s", help="Run specific stage only"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate pipeline without executing"),
    parallel: bool = typer.Option(False, "--parallel", "-p", help="Enable parallel execution"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """
    Run the AgriMind AI data pipeline.
    """
    if verbose:
        logger.remove()
        logger.add(lambda msg: print(msg), level="DEBUG")
    
    # Load configuration
    config_path = Path(config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise typer.Exit(1)
    
    # Override dataset if provided
    if dataset:
        # This would update the config to use the specified dataset
        pass
    
    # Override execution mode
    if parallel:
        # This would update the config to use parallel execution
        pass
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(config_path=config_path)
    
    # Register stages (would be done elsewhere in actual implementation)
    # orchestrator.register_stages(stage_registry)
    
    # Run pipeline
    success = orchestrator.run(
        resume=resume,
        checkpoint_path=checkpoint,
        dry_run=dry_run
    )
    
    if not success:
        raise typer.Exit(1)


@app.command()
def status(checkpoint_dir: Path = typer.Option(".pipeline_checkpoints", help="Checkpoint directory")):
    """
    Show pipeline status and available checkpoints.
    """
    table = Table(title="Pipeline Checkpoints")
    table.add_column("Checkpoint", style="cyan")
    table.add_column("Created", style="green")
    table.add_column("Size", style="yellow")
    
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.exists():
        for checkpoint in sorted(checkpoint_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True):
            size = checkpoint.stat().st_size / 1024  # KB
            created = checkpoint.stat().st_ctime
            table.add_row(
                checkpoint.stem,
                str(created),
                f"{size:.2f} KB"
            )
    
    console.print(table)


@app.command()
def clean(checkpoint_dir: Path = typer.Option(".pipeline_checkpoints", help="Checkpoint directory")):
    """
    Clean pipeline checkpoints and artifacts.
    """
    import shutil
    
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        logger.info(f"Cleaned checkpoint directory: {checkpoint_dir}")
    
    reports_dir = Path("reports/pipeline")
    if reports_dir.exists():
        shutil.rmtree(reports_dir)
        logger.info(f"Cleaned reports directory: {reports_dir}")


@app.command()
def list_stages():
    """
    List all available pipeline stages.
    """
    table = Table(title="Pipeline Stages")
    table.add_column("Stage", style="cyan")
    table.add_column("Description", style="green")
    
    stage_info = {
        StageType.SCAN: "Dataset scanning and discovery",
        StageType.PROFILE: "Data profiling and statistics",
        StageType.VALIDATE: "Data quality validation",
        StageType.CLEAN: "Automated data cleaning",
        StageType.STANDARDIZE: "Dataset standardization",
        StageType.FEATURE_ENGINEERING: "Feature extraction and transformation",
        StageType.FEATURE_STORE: "Feature storage and management",
        StageType.EDA: "Exploratory data analysis",
        StageType.SAVE: "Save artifacts and results"
    }
    
    for stage_type, description in stage_info.items():
        table.add_row(stage_type.value, description)
    
    console.print(table)


if __name__ == "__main__":
    app()