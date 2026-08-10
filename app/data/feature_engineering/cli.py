# app/data/feature_engineering/cli.py
import typer
import pandas as pd
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger

from app.data.feature_engineering.feature_pipeline import FeaturePipeline
from app.data.feature_engineering.feature_registry import FeatureRegistry
from app.data.feature_engineering.models import FeatureDefinition, FeatureType
from app.data.feature_engineering.exceptions import FeatureEngineeringError

console = Console()
app = typer.Typer()


@app.command()
def generate(
    dataset_path: Path = typer.Argument(..., help="Path to input dataset"),
    config_path: Path = typer.Option(
        "configs/features.yaml", 
        "--config", "-c",
        help="Path to feature configuration file"
    ),
    feature_store_path: Path = typer.Option(
        "data/feature_store",
        "--store", "-s",
        help="Path to feature store"
    ),
    version: str = typer.Option(
        "1.0.0",
        "--version", "-v",
        help="Feature version"
    ),
    owner: str = typer.Option(
        "system",
        "--owner", "-o",
        help="Feature owner"
    )
):
    """
    Generate features from a dataset.
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading dataset...", total=None)
            
            # Load dataset
            df = pd.read_csv(dataset_path)
            progress.update(task, description="Initializing feature pipeline...")
            
            # Initialize pipeline
            pipeline = FeaturePipeline(
                config_path=config_path,
                feature_store_path=feature_store_path
            )
            
            progress.update(task, description="Generating features...")
            
            # Run pipeline
            result_df = pipeline.run_pipeline(
                df,
                dataset_path.stem,
                owner=owner,
                version=version
            )
            
            progress.update(task, description="Done!", completed=True)
        
        # Show results
        console.print("\n[bold green]✓ Feature generation completed![/bold green]")
        console.print(f"Generated {len(result_df.columns)} features from {len(df)} samples")
        
        # Show feature list
        table = Table(title="Generated Features")
        table.add_column("Feature", style="cyan")
        table.add_column("Type", style="green")
        
        for col in result_df.columns[:10]:
            table.add_row(col, str(result_df[col].dtype))
        
        if len(result_df.columns) > 10:
            table.add_row("...", f"and {len(result_df.columns) - 10} more")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def registry(
    action: str = typer.Argument(..., help="Registry action: list, get, search, deactivate"),
    feature_name: Optional[str] = typer.Argument(None, help="Feature name (for get, deactivate)"),
    feature_store_path: Path = typer.Option(
        "data/feature_store",
        "--store", "-s",
        help="Path to feature store"
    ),
    feature_type: Optional[str] = typer.Option(
        None,
        "--type", "-t",
        help="Filter by feature type"
    ),
    query: Optional[str] = typer.Option(
        None,
        "--query", "-q",
        help="Search query"
    )
):
    """
    Manage feature registry.
    """
    try:
        registry = FeatureRegistry(feature_store_path / 'registry')
        
        if action == 'list':
            features = registry.list_features(feature_type)
            
            if features:
                table = Table(title="Feature Registry")
                table.add_column("Name", style="cyan")
                table.add_column("Type", style="green")
                table.add_column("Version", style="yellow")
                table.add_column("Owner", style="blue")
                table.add_column("Created", style="magenta")
                
                for feature in features:
                    table.add_row(
                        feature['name'],
                        feature['type'],
                        feature['latest_version'],
                        feature['owner'],
                        feature['created_date'][:10]
                    )
                
                console.print(table)
                console.print(f"\nTotal: {len(features)} features")
            else:
                console.print("No features found in registry")
        
        elif action == 'get':
            if not feature_name:
                console.print("[bold red]Error: Feature name required for get action[/bold red]")
                raise typer.Exit(1)
            
            metadata = registry.get_feature(feature_name)
            console.print(f"\n[bold]Feature: {metadata.feature_name}[/bold]")
            console.print(f"Description: {metadata.description}")
            console.print(f"Type: {metadata.feature_type}")
            console.print(f"Version: {metadata.version}")
            console.print(f"Owner: {metadata.owner}")
            console.print(f"Created: {metadata.created_date}")
            console.print(f"Source Columns: {', '.join(metadata.source_columns)}")
            console.print(f"Dependencies: {', '.join(metadata.dependencies)}")
            console.print(f"Tags: {', '.join(metadata.tags)}")
            console.print(f"Formula: {metadata.formula}")
        
        elif action == 'search':
            if not query:
                console.print("[bold red]Error: Query required for search action[/bold red]")
                raise typer.Exit(1)
            
            results = registry.search_features(query)
            
            if results:
                table = Table(title="Search Results")
                table.add_column("Name", style="cyan")
                table.add_column("Description", style="green")
                table.add_column("Type", style="yellow")
                table.add_column("Tags", style="blue")
                
                for result in results:
                    table.add_row(
                        result['name'],
                        result['description'][:50] + '...' if len(result['description']) > 50 else result['description'],
                        result['type'],
                        ', '.join(result['tags'])
                    )
                
                console.print(table)
                console.print(f"\nFound {len(results)} features")
            else:
                console.print("No features found matching the query")
        
        elif action == 'deactivate':
            if not feature_name:
                console.print("[bold red]Error: Feature name required for deactivate action[/bold red]")
                raise typer.Exit(1)
            
            registry.deactivate_feature(feature_name)
            console.print(f"[bold green]✓ Feature {feature_name} deactivated[/bold green]")
        
        else:
            console.print(f"[bold red]Error: Unknown action '{action}'[/bold red]")
            console.print("Available actions: list, get, search, deactivate")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def report(
    dataset_path: Path = typer.Argument(..., help="Path to feature dataset"),
    dataset_name: Optional[str] = typer.Option(
        None,
        "--name", "-n",
        help="Dataset name (defaults to filename)"
    ),
    feature_store_path: Path = typer.Option(
        "data/feature_store",
        "--store", "-s",
        help="Path to feature store"
    ),
    version: str = typer.Option(
        "1.0.0",
        "--version", "-v",
        help="Feature version"
    )
):
    """
    Generate feature reports for a dataset.
    """
    try:
        from app.data.feature_engineering.report import FeatureReport
        
        # Load dataset
        df = pd.read_parquet(dataset_path)
        
        # Initialize report generator
        report = FeatureReport(feature_store_path / 'reports' / 'features')
        
        # Determine dataset name
        if dataset_name is None:
            dataset_name = dataset_path.stem
        
        # Generate reports
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating reports...", total=None)
            
            report.generate_feature_summary(df, dataset_name, version)
            report.generate_correlation_report(df, dataset_name, version)
            report.generate_distribution_report(df, dataset_name, version)
            report.generate_feature_metadata_report(None, dataset_name, version)
            
            progress.update(task, description="Done!", completed=True)
        
        console.print("[bold green]✓ Reports generated successfully![/bold green]")
        console.print(f"Reports saved to: {feature_store_path}/reports/features/{dataset_name}")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def serve(
    feature_store_path: Path = typer.Option(
        "data/feature_store",
        "--store", "-s",
        help="Path to feature store"
    ),
    port: int = typer.Option(
        8000,
        "--port", "-p",
        help="Port to serve on"
    )
):
    """
    Serve feature metadata as a web service.
    """
    try:
        from fastapi import FastAPI, HTTPException
        import uvicorn
        
        # Initialize registry
        registry = FeatureRegistry(feature_store_path / 'registry')
        
        # Create FastAPI app
        app = FastAPI(title="Feature Registry API")
        
        @app.get("/features")
        def list_features(feature_type: Optional[str] = None):
            return registry.list_features(feature_type)
        
        @app.get("/features/{feature_name}")
        def get_feature(feature_name: str, version: Optional[str] = None):
            try:
                return registry.get_feature(feature_name, version)
            except FeatureEngineeringError:
                raise HTTPException(status_code=404, detail="Feature not found")
        
        @app.get("/features/search")
        def search_features(query: str):
            return registry.search_features(query)
        
        console.print(f"[bold green]✓ Feature registry API running on http://localhost:{port}[/bold green]")
        uvicorn.run(app, host="0.0.0.0", port=port)
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


def main():
    app()


if __name__ == "__main__":
    main()