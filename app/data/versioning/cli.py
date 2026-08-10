"""
CLI interface for the versioning framework using Typer.
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import pandas as pd
from uuid import UUID

from .version_manager import VersionManager
from .models import SemanticVersion
from .exceptions import (
    VersionNotFoundError,
    DatasetNotFoundError,
    ArtifactNotFoundError,
    RollbackError,
    ChecksumMismatchError
)

app = typer.Typer(help="AgriMind AI Versioning Framework CLI")
console = Console()

# Global version manager instance
_version_manager: Optional[VersionManager] = None


def get_version_manager() -> VersionManager:
    """Get or create the version manager instance."""
    global _version_manager
    if _version_manager is None:
        base_path = Path.cwd()
        _version_manager = VersionManager(base_path)
    return _version_manager


@app.command()
def list(
    entity_type: str = typer.Option(
        "dataset",
        "--type",
        "-t",
        help="Entity type (dataset, artifact, feature, schema)"
    )
):
    """List all entities of a specific type."""
    vm = get_version_manager()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Fetching entities...", total=None)

        if entity_type == "dataset":
            entities = vm.list_datasets()
        elif entity_type == "artifact":
            entities = list(vm.artifact_registry._cache.keys())
        elif entity_type == "feature":
            entities = list(vm.feature_registry._cache.keys())
        elif entity_type == "schema":
            entities = list(vm.schema_registry._cache.keys())
        else:
            console.print(f"[red]Error:[/red] Unsupported entity type: {entity_type}")
            raise typer.Exit(1)

    if not entities:
        console.print(f"[yellow]No {entity_type}s found[/yellow]")
        return

    table = Table(title=f"{entity_type.capitalize()}s")
    table.add_column("Name", style="cyan")
    table.add_column("Versions", style="green")
    table.add_column("Latest", style="yellow")

    for entity in entities:
        if entity_type == "dataset":
            versions = vm.list_dataset_versions(entity)
        elif entity_type == "artifact":
            versions = list(vm.artifact_registry.list_versions(entity).keys())
        elif entity_type == "feature":
            versions = list(vm.feature_registry.list_versions(entity).keys())
        elif entity_type == "schema":
            versions = list(vm.schema_registry.list_versions(entity).keys())
        else:
            continue

        latest = str(max(versions)) if versions else "-"
        table.add_row(entity, str(len(versions)), latest)

    console.print(table)


@app.command()
def history(
    entity_name: str = typer.Argument(..., help="Name of the entity"),
    entity_type: str = typer.Option(
        "dataset",
        "--type",
        "-t",
        help="Entity type (dataset, artifact, feature, schema)"
    )
):
    """Show version history for an entity."""
    vm = get_version_manager()

    try:
        if entity_type == "dataset":
            history_data = vm.get_version_history(entity_type, entity_name)
        elif entity_type == "artifact":
            history_data = vm.get_version_history(entity_type, entity_name)
        else:
            console.print(f"[red]Error:[/red] History not supported for {entity_type}")
            raise typer.Exit(1)
    except (DatasetNotFoundError, ArtifactNotFoundError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if not history_data:
        console.print(f"[yellow]No history found for {entity_name}[/yellow]")
        return

    table = Table(title=f"Version History: {entity_name}")
    table.add_column("Version", style="cyan")
    table.add_column("Status", style="yellow")
    table.add_column("Created", style="green")
    table.add_column("Details", style="white")

    for entry in history_data:
        details = []
        if "rows" in entry:
            details.append(f"Rows: {entry['rows']}")
        if "columns" in entry:
            details.append(f"Cols: {entry['columns']}")
        if "artifact_type" in entry:
            details.append(f"Type: {entry['artifact_type']}")

        table.add_row(
            entry["version"],
            entry["status"],
            entry["created_at"][:19],
            ", ".join(details) if details else "-"
        )

    console.print(table)


@app.command()
def compare(
    entity_name: str = typer.Argument(..., help="Name of the entity"),
    version_a: str = typer.Argument(..., help="First version"),
    version_b: str = typer.Argument(..., help="Second version"),
    entity_type: str = typer.Option(
        "dataset",
        "--type",
        "-t",
        help="Entity type (dataset, artifact, feature, schema)"
    )
):
    """Compare two versions of an entity."""
    vm = get_version_manager()

    try:
        result = vm.compare_versions(
            entity_type,
            entity_name,
            version_a,
            version_b
        )
    except (VersionNotFoundError, DatasetNotFoundError, ArtifactNotFoundError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[bold cyan]Comparing {entity_name} v{version_a} vs v{version_b}[/bold cyan]\n")

    # Display comparison results
    if entity_type == "dataset":
        table = Table(title="Dataset Comparison")
        table.add_column("Metric", style="cyan")
        table.add_column(f"v{version_a}", style="yellow")
        table.add_column(f"v{version_b}", style="green")
        table.add_column("Difference", style="magenta")

        table.add_row("Rows", str(result["rows"]["a"]), str(result["rows"]["b"]), str(result["rows"]["difference"]))
        table.add_row("Columns", str(result["columns"]["a"]), str(result["columns"]["b"]), str(result["columns"]["difference"]))
        table.add_row("Checksum Match", str(result["checksum_match"]), "", "")

        console.print(table)

    elif entity_type == "artifact":
        table = Table(title="Artifact Comparison")
        table.add_column("Metric", style="cyan")
        table.add_column(f"v{version_a}", style="yellow")
        table.add_column(f"v{version_b}", style="green")

        for key, value in result.get("metrics_diffs", {}).items():
            table.add_row(
                key,
                str(value.get("version_a", "-")),
                str(value.get("version_b", "-"))
            )

        console.print(table)


@app.command()
def rollback(
    entity_name: str = typer.Argument(..., help="Name of the entity"),
    target_version: str = typer.Argument(..., help="Version to rollback to"),
    entity_type: str = typer.Option(
        "dataset",
        "--type",
        "-t",
        help="Entity type (dataset, artifact)"
    ),
    reason: Optional[str] = typer.Option(None, "--reason", "-r", help="Reason for rollback")
):
    """Rollback an entity to a previous version."""
    vm = get_version_manager()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Performing rollback...", total=None)

            if entity_type == "dataset":
                result = vm.rollback_dataset(entity_name, target_version, reason)
            elif entity_type == "artifact":
                result = vm.rollback_artifact(entity_name, target_version, reason)
            else:
                console.print(f"[red]Error:[/red] Rollback not supported for {entity_type}")
                raise typer.Exit(1)

        console.print(f"[green]✓[/green] Successfully rolled back {entity_name} to version {target_version}")
        console.print(f"[dim]New version: {result.version}[/dim]")

    except (RollbackError, VersionNotFoundError, DatasetNotFoundError, ArtifactNotFoundError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def lineage(
    entity_id: str = typer.Argument(..., help="Entity ID or name"),
    depth: int = typer.Option(3, "--depth", "-d", help="Depth of lineage to show"),
    entity_type: str = typer.Option(
        "dataset",
        "--type",
        "-t",
        help="Entity type (dataset, artifact)"
    )
):
    """Show lineage for an entity."""
    vm = get_version_manager()

    try:
        # Try to get entity by ID
        try:
            entity_uuid = UUID(entity_id)
        except ValueError:
            # If not a UUID, try to find by name
            if entity_type == "dataset":
                metadata = vm.get_dataset(entity_id)
                entity_uuid = metadata.id
            elif entity_type == "artifact":
                metadata = vm.get_artifact(entity_id)
                entity_uuid = metadata.id
            else:
                console.print(f"[red]Error:[/red] Unsupported entity type: {entity_type}")
                raise typer.Exit(1)

        lineage_data = vm.get_lineage(str(entity_uuid), depth=depth)

        console.print(f"[bold cyan]Lineage for: {lineage_data['entity']['entity_name']}[/bold cyan]")
        console.print(f"Version: {lineage_data['entity']['version']}\n")

        # Upstream dependencies
        if lineage_data.get("upstream"):
            console.print("[bold yellow]Upstream Dependencies:[/bold yellow]")
            for dep in lineage_data["upstream"]:
                console.print(f"  • {dep['entity_name']} (v{dep['version']}) - {dep['entity_type']}")

        # Downstream dependents
        if lineage_data.get("downstream"):
            console.print("\n[bold green]Downstream Dependents:[/bold green]")
            for dep in lineage_data["downstream"]:
                console.print(f"  • {dep['entity_name']} (v{dep['version']}) - {dep['entity_type']}")

        if not lineage_data.get("upstream") and not lineage_data.get("downstream"):
            console.print("[dim]No lineage relationships found[/dim]")

    except (VersionNotFoundError, DatasetNotFoundError, ArtifactNotFoundError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def report(
    entity_name: str = typer.Argument(..., help="Name of the entity"),
    entity_type: str = typer.Option(
        "dataset",
        "--type",
        "-t",
        help="Entity type (dataset, artifact)"
    ),
    output_format: str = typer.Option(
        "html",
        "--format",
        "-f",
        help="Output format (html, json, markdown)"
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path"
    )
):
    """Generate a version report."""
    vm = get_version_manager()

    try:
        if entity_type == "dataset":
            report_content = vm.generate_version_report(
                entity_name,
                "dataset",
                output_format
            )
        elif entity_type == "artifact":
            report_content = vm.generate_version_report(
                entity_name,
                "artifact",
                output_format
            )
        else:
            console.print(f"[red]Error:[/red] Report not supported for {entity_type}")
            raise typer.Exit(1)

        if output_file:
            output_file.write_text(report_content)
            console.print(f"[green]✓[/green] Report saved to {output_file}")
        else:
            console.print(report_content)

    except (VersionNotFoundError, DatasetNotFoundError, ArtifactNotFoundError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def checksum_verify(
    entity_name: str = typer.Argument(..., help="Name of the entity"),
    entity_type: str = typer.Option(
        "dataset",
        "--type",
        "-t",
        help="Entity type (dataset)"
    ),
    version: Optional[str] = typer.Option(
        None,
        "--version",
        "-v",
        help="Version to verify (defaults to latest)"
    )
):
    """Verify checksum of a dataset."""
    vm = get_version_manager()

    try:
        if entity_type == "dataset":
            version_obj = SemanticVersion.parse(version) if version else None
            is_valid = vm.validate_dataset_checksum(entity_name, version_obj)
        else:
            console.print(f"[red]Error:[/red] Checksum verification not supported for {entity_type}")
            raise typer.Exit(1)

        if is_valid:
            console.print(f"[green]✓[/green] Checksum verification passed for {entity_name}")
        else:
            console.print(f"[red]✗[/red] Checksum verification failed for {entity_name}")

    except (VersionNotFoundError, DatasetNotFoundError, ChecksumMismatchError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def summary():
    """Generate a registry summary report."""
    vm = get_version_manager()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Generating summary report...", total=None)
        report_content = vm.generate_registry_summary_report()

    console.print(report_content)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()