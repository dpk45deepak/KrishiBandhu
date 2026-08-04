"""AgriMind AI - Agricultural Intelligence Platform CLI."""

from __future__ import annotations

import typer
from loguru import logger

from app.cli.clean import register_clean_command
from app.cli.profile import register_profile_command
from app.cli.report import register_report_command
from app.cli.scan import register_scan_command
from app.cli.validate import register_validate_command
from app.config.config import load_config as load_project_config
from app.core.runtime import AgriMindRuntime
from app.logger.logger import setup_logger

app = typer.Typer(
    name="agrimind",
    help="AgriMind AI - Agricultural Intelligence Platform",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

runtime = AgriMindRuntime()


@app.callback()
def main_callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """AgriMind AI CLI - Dataset scanning, profiling, and reporting."""
    level = "DEBUG" if verbose else "INFO"
    load_project_config()
    setup_logger(level=level, colored=True)
    runtime.start()
    if verbose:
        logger.info("Verbose logging enabled")


register_scan_command(app)
register_profile_command(app)
register_report_command(app)
register_validate_command(app)
register_clean_command(app)


@app.command("health")
def health() -> None:
    """Show the current runtime health state."""
    print(runtime.health_check())


def main() -> None:
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
