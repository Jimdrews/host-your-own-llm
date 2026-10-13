"""Locate the repository the CLI is run from."""

from pathlib import Path

from .errors import CliError


def find_project_root(start: Path | None = None) -> Path:
    start = start or Path.cwd()
    for directory in (start, *start.parents):
        if (directory / "docker-compose.yml").exists():
            return directory
    raise CliError("Run this from inside the host-your-own-llm repository.")
