# app/cli/main.py
"""
Main CLI entry point - Typer application.

Every command delegates to APIClient.
No direct business logic in CLI.
"""
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import asyncio

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax

from app.cli.client import APIClient
from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

app = typer.Typer(
    name="agrimind",
    help="AgriMind AI Platform CLI",
    add_completion=True,
)

console = Console()

# Global client (initialized with login)
_client: Optional[APIClient] = None


def get_client() -> APIClient:
    """Get or raise if not authenticated."""
    if _client is None:
        console.print("[red]Not authenticated. Run 'agrimind login' first.[/red]")
        raise typer.Exit(1)
    return _client


# ============ Auth Commands ============

@app.command()
def login(
    username: str = typer.Option(..., prompt=True),
    password: str = typer.Option(..., prompt=True, hide_input=True),
    api_url: str = typer.Option(None, help="API base URL"),
):
    """Authenticate with the AgriMind platform."""
    global _client
    
    _client = APIClient(base_url=api_url)
    
    async def _login():
        try:
            result = await _client.login(username, password)
            console.print(f"[green]✓[/green] Authenticated as [bold]{username}[/bold]")
            console.print(f"  Token expires in: {result.get('expires_in', 'N/A')}s")
        except Exception as e:
            console.print(f"[red]✗ Login failed: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_login())


@app.command()
def logout():
    """Clear authentication."""
    global _client
    _client = None
    console.print("[green]✓[/green] Logged out")


# ============ Health Commands ============

@app.command()
def health():
    """Check platform health."""
    client = get_client()
    
    async def _health():
        result = await client.health_check()
        
        status_color = {
            "healthy": "green",
            "degraded": "yellow",
            "unhealthy": "red",
        }
        
        color = status_color.get(result["status"], "white")
        console.print(f"Status: [bold {color}]{result['status'].upper()}[/bold {color}]")
        console.print(f"Version: {result['version']}")
        console.print(f"Uptime: {result['uptime_seconds']:.0f}s")
        
        table = Table(title="Component Health")
        table.add_column("Component")
        table.add_column("Status")
        table.add_column("Latency")
        
        for name, comp in result.get("components", {}).items():
            c_color = status_color.get(comp["status"], "white")
            table.add_row(
                name,
                f"[{c_color}]{comp['status']}[/{c_color}]",
                f"{comp.get('latency_ms', 0):.1f}ms",
            )
        
        console.print(table)
    
    asyncio.run(_health())


# ============ Dataset Commands ============

dataset_app = typer.Typer(help="Dataset management commands")
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("list")
def list_datasets(
    status: Optional[str] = typer.Option(None, help="Filter by status"),
):
    """List all datasets."""
    client = get_client()
    
    async def _list():
        datasets = await client.list_datasets(status=status)
        
        if not datasets:
            console.print("[yellow]No datasets found[/yellow]")
            return
        
        table = Table(title="Datasets")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Status")
        table.add_column("Format")
        table.add_column("Created")
        
        for ds in datasets:
            table.add_row(
                ds["id"][:8] + "...",
                ds["name"],
                ds["status"],
                ds.get("format", "N/A"),
                ds.get("created_at", "N/A")[:10],
            )
        
        console.print(table)
    
    asyncio.run(_list())


@dataset_app.command("create")
def create_dataset(
    name: str = typer.Argument(..., help="Dataset name"),
    description: str = typer.Option("", help="Dataset description"),
    format: str = typer.Option("csv", help="Dataset format"),
):
    """Create a new dataset."""
    client = get_client()
    
    async def _create():
        result = await client.create_dataset(name, description, format=format)
        console.print(f"[green]✓[/green] Dataset created: [bold]{result['name']}[/bold]")
        console.print(f"  ID: {result['id']}")
    
    asyncio.run(_create())


@dataset_app.command("upload")
def upload_dataset(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    file: Path = typer.Argument(..., exists=True, help="File to upload"),
):
    """Upload a file to a dataset."""
    client = get_client()
    
    async def _upload():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task(f"Uploading {file.name}...", total=None)
            result = await client.upload_dataset(dataset_id, str(file))
            progress.update(task, completed=True)
        
        console.print(f"[green]✓[/green] Uploaded: [bold]{file.name}[/bold]")
        console.print(f"  Version: {result['version_number']}")
        console.print(f"  Rows: {result['row_count']:,}")
        console.print(f"  Columns: {result['column_count']}")
    
    asyncio.run(_upload())


@dataset_app.command("profile")
def profile_dataset(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
):
    """Generate dataset profile."""
    client = get_client()
    
    async def _profile():
        with console.status("Profiling dataset..."):
            result = await client.profile_dataset(dataset_id)
        
        console.print(f"[green]✓[/green] Profile generated")
        console.print(f"  Rows: {result['row_count']:,}")
        console.print(f"  Columns: {result['column_count']}")
        
        table = Table(title="Column Summary")
        table.add_column("Column")
        table.add_column("Type")
        table.add_column("Missing %")
        table.add_column("Unique")
        
        for col in result.get("columns", [])[:20]:
            table.add_row(
                col["name"],
                col["dtype"],
                f"{col['null_percentage']:.1f}%",
                str(col["unique_count"]),
            )
        
        console.print(table)
    
    asyncio.run(_profile())


@dataset_app.command("validate")
def validate_dataset(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    rules_file: Optional[Path] = typer.Option(None, exists=True, help="JSON rules file"),
):
    """Validate dataset."""
    client = get_client()
    
    async def _validate():
        rules = None
        if rules_file:
            rules = json.loads(rules_file.read_text())
        
        with console.status("Validating..."):
            result = await client.validate_dataset(dataset_id, rules)
        
        status_color = "green" if result["is_valid"] else "red"
        console.print(f"Valid: [bold {status_color}]{result['is_valid']}[/bold {status_color}]")
        console.print(f"  Errors: {result['error_count']}")
        console.print(f"  Warnings: {result['warning_count']}")
        console.print(f"  Summary: {result['summary']}")
    
    asyncio.run(_validate())


@dataset_app.command("download")
def download_dataset(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    output: Path = typer.Option(Path("dataset_download"), help="Output file path"),
    version: Optional[int] = typer.Option(None, help="Specific version"),
):
    """Download dataset file."""
    client = get_client()
    
    async def _download():
        with console.status("Downloading..."):
            await client.download_dataset(dataset_id, str(output), version)
        console.print(f"[green]✓[/green] Downloaded to: [bold]{output}[/bold]")
    
    asyncio.run(_download())


@dataset_app.command("delete")
def delete_dataset(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a dataset."""
    if not force:
        confirm = typer.confirm(f"Delete dataset {dataset_id}? This cannot be undone.")
        if not confirm:
            raise typer.Abort()
    
    client = get_client()
    
    async def _delete():
        await client.delete_dataset(dataset_id)
        console.print(f"[green]✓[/green] Dataset deleted")
    
    asyncio.run(_delete())


# ============ Pipeline Commands ============

pipeline_app = typer.Typer(help="Pipeline management commands")
app.add_typer(pipeline_app, name="pipeline")


@pipeline_app.command("list")
def list_pipelines():
    """List all pipelines."""
    client = get_client()
    
    async def _list():
        pipelines = await client.list_pipelines()
        
        if not pipelines:
            console.print("[yellow]No pipelines found[/yellow]")
            return
        
        table = Table(title="Pipelines")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Status")
        table.add_column("Runs")
        table.add_column("Success Rate")
        
        for pl in pipelines:
            table.add_row(
                pl["id"][:8] + "...",
                pl["name"],
                pl["status"],
                str(pl.get("run_count", 0)),
                f"{pl.get('success_rate', 0):.1f}%",
            )
        
        console.print(table)
    
    asyncio.run(_list())


@pipeline_app.command("run")
def run_pipeline(
    pipeline_id: str = typer.Argument(..., help="Pipeline ID"),
    params_file: Optional[Path] = typer.Option(None, exists=True, help="JSON params file"),
    wait: bool = typer.Option(False, help="Wait for completion"),
):
    """Execute a pipeline."""
    client = get_client()
    
    async def _run():
        params = None
        if params_file:
            params = json.loads(params_file.read_text())
        
        with console.status("Starting pipeline..."):
            result = await client.run_pipeline(pipeline_id, params)
        
        console.print(f"[green]✓[/green] Pipeline started")
        console.print(f"  Run ID: {result['id']}")
        console.print(f"  Status: {result['status']}")
        
        if wait and result['status'] in ('running', 'pending'):
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
            ) as progress:
                task = progress.add_task("Waiting for completion...", total=None)
                while True:
                    await asyncio.sleep(2)
                    run = await client.get_pipeline_run(pipeline_id, result["id"])
                    if run["status"] in ("completed", "failed", "cancelled"):
                        progress.update(task, completed=True)
                        console.print(f"\nFinal status: [bold]{run['status']}[/bold]")
                        console.print(f"Duration: {run.get('duration_seconds', 0):.1f}s")
                        break
    
    asyncio.run(_run())


# ============ ML Commands ============

ml_app = typer.Typer(help="Machine Learning commands")
app.add_typer(ml_app, name="ml")


@ml_app.command("models")
def list_models():
    """List registered models."""
    client = get_client()
    
    async def _list():
        models = await client.list_models()
        
        if not models:
            console.print("[yellow]No models found[/yellow]")
            return
        
        table = Table(title="Model Registry")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("Status")
        table.add_column("Version")
        
        for m in models:
            version = m.get("current_version", {})
            table.add_row(
                m["id"][:8] + "...",
                m["name"],
                m["model_type"],
                m["status"],
                str(version.get("version", "N/A")),
            )
        
        console.print(table)
    
    asyncio.run(_list())


@ml_app.command("register")
def register_model(
    name: str = typer.Argument(..., help="Model name"),
    model_type: str = typer.Option("classification", help="Model type"),
    description: str = typer.Option("", help="Description"),
):
    """Register a new model."""
    client = get_client()
    
    async def _register():
        result = await client.register_model(name, model_type, description=description)
        console.print(f"[green]✓[/green] Model registered: [bold]{result['name']}[/bold]")
        console.print(f"  ID: {result['id']}")
        console.print(f"  Type: {result['model_type']}")
    
    asyncio.run(_register())


@ml_app.command("train")
def train_model(
    model_id: str = typer.Argument(..., help="Model ID"),
    config_file: Path = typer.Argument(..., exists=True, help="Training config JSON"),
    dataset_id: Optional[str] = typer.Option(None, help="Dataset ID"),
):
    """Train a model."""
    client = get_client()
    
    async def _train():
        config = json.loads(config_file.read_text())
        
        with console.status("Training model..."):
            result = await client.train_model(model_id, config)
        
        if result["status"] == "completed":
            console.print(f"[green]✓[/green] Training completed")
            console.print(f"  Duration: {result.get('duration_seconds', 0):.1f}s")
            console.print(f"  Metrics: {json.dumps(result.get('metrics', {}), indent=2)}")
        else:
            console.print(f"[red]✗ Training failed: {result.get('error_message', 'Unknown error')}[/red]")
    
    asyncio.run(_train())


@ml_app.command("predict")
def predict(
    model_id: str = typer.Argument(..., help="Model ID"),
    data_file: Path = typer.Argument(..., exists=True, help="JSON data file"),
):
    """Make predictions."""
    client = get_client()
    
    async def _predict():
        data = json.loads(data_file.read_text())
        instances = data if isinstance(data, list) else [data]
        
        with console.status("Predicting..."):
            result = await client.predict(model_id, instances)
        
        console.print(f"[green]✓[/green] Predictions generated")
        console.print(f"  Time: {result.get('prediction_time_ms', 0):.1f}ms")
        console.print(f"  Count: {len(result.get('predictions', []))}")
        
        # Show first few predictions
        for i, pred in enumerate(result.get("predictions", [])[:5]):
            console.print(f"  [{i}]: {pred}")
        if len(result.get("predictions", [])) > 5:
            console.print(f"  ... and {len(result['predictions']) - 5} more")
    
    asyncio.run(_predict())


@ml_app.command("deploy")
def deploy_model(
    model_id: str = typer.Argument(..., help="Model ID"),
    endpoint_name: Optional[str] = typer.Option(None, help="Endpoint name"),
):
    """Deploy a model for inference."""
    client = get_client()
    
    async def _deploy():
        # First deploy the model version
        deploy_result = await client.deploy_model(model_id)
        version = deploy_result.get("version", 1)
        
        # Create inference endpoint
        name = endpoint_name or f"model-{model_id[:8]}"
        endpoint = await client.create_endpoint(model_id, version, name)
        
        console.print(f"[green]✓[/green] Model deployed")
        console.print(f"  Version: {version}")
        console.print(f"  Endpoint: {endpoint['endpoint_path']}")
    
    asyncio.run(_deploy())


# ============ System Commands ============

@app.command()
def status():
    """Show platform status overview."""
    client = get_client()
    
    async def _status():
        health = await client.health_check()
        metrics = await client.get_system_metrics()
        
        console.print(Panel.fit(
            f"[bold]AgriMind Platform Status[/bold]\n"
            f"Version: {health['version']}\n"
            f"Status: {health['status'].upper()}\n"
            f"Uptime: {health['uptime_seconds']:.0f}s",
            title="Platform"
        ))
        
        console.print(f"CPU: {metrics.get('cpu_percent', 0):.1f}%")
        console.print(f"Memory: {metrics.get('memory_percent', 0):.1f}%")
        console.print(f"API Requests/min: {metrics.get('api_requests_per_minute', 0):.1f}")
        console.print(f"Avg Latency: {metrics.get('avg_latency_ms', 0):.1f}ms")
        console.print(f"Active Pipelines: {metrics.get('active_pipelines', 0)}")
        console.print(f"Models Deployed: {metrics.get('models_deployed', 0)}")
    
    asyncio.run(_status())


@app.command()
def generate_report(
    report_type: str = typer.Argument(..., help="Report type"),
    dataset_id: Optional[str] = typer.Option(None, help="Dataset ID"),
    model_id: Optional[str] = typer.Option(None, help="Model ID"),
    output: Path = typer.Option(Path("report.json"), help="Output file"),
):
    """Generate a report."""
    client = get_client()
    
    async def _gen():
        kwargs = {}
        if dataset_id:
            kwargs["dataset_id"] = dataset_id
        if model_id:
            kwargs["model_id"] = model_id
        
        with console.status(f"Generating {report_type} report..."):
            result = await client.generate_report(report_type, **kwargs)
        
        output.write_text(json.dumps(result, indent=2, default=str))
        console.print(f"[green]✓[/green] Report saved to: [bold]{output}[/bold]")
        console.print(f"  Summary: {result.get('summary', 'N/A')}")
    
    asyncio.run(_gen())


if __name__ == "__main__":
    app()