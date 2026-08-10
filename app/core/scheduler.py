"""Background scheduler primitives for platform jobs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Scheduler:
    """A lightweight scheduler interface for periodic or delayed tasks."""

    def __init__(self) -> None:
        self._jobs: list[tuple[str, Callable[[], Any], float | None]] = []

    def add_job(self, name: str, job: Callable[[], Any], interval_seconds: float | None = None) -> None:
        self._jobs.append((name, job, interval_seconds))

    def list_jobs(self) -> list[tuple[str, Callable[[], Any], float | None]]:
        return list(self._jobs)

    def run_once(self, name: str) -> Any:
        for job_name, job, _ in self._jobs:
            if job_name == name:
                return job()
        raise KeyError(f"Job '{name}' is not registered")
