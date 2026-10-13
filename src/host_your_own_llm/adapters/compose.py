"""`docker compose` for the project's stack."""

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from ..domain import ServiceState
from ..errors import CliError
from ..ui.console import console

Runner = Callable[..., subprocess.CompletedProcess]


def check_docker(runner: Runner = subprocess.run) -> bool:
    """Fail if Docker isn't usable. Return whether it lists the NVIDIA runtime."""
    if not shutil.which("docker"):
        raise CliError("Docker is required: https://docs.docker.com/get-docker/")
    runtimes = runner(["docker", "info", "--format", "{{json .Runtimes}}"], capture_output=True, text=True)
    if runtimes.returncode != 0:
        raise CliError("Docker isn't running. Start Docker and try again.")
    return "nvidia" in runtimes.stdout


class Compose:
    def __init__(self, root: Path, runner: Runner = subprocess.run):
        self.root = root
        self.runner = runner

    def run(self, *args: str, quiet: bool = False) -> str:
        """Run docker compose. With quiet, output is only shown if the command fails."""
        result = self.runner(["docker", "compose", "--progress", "quiet", *args], cwd=self.root, capture_output=quiet, text=True)
        if result.returncode != 0:
            if quiet:
                console.print(result.stdout + result.stderr, markup=False, highlight=False)
            raise CliError(f"`docker compose {' '.join(args)}` failed")
        return result.stdout if quiet else ""

    def build(self, service: str) -> None:
        self.run("build", "-q", service, quiet=True)

    def up(self, service: str, wait: bool = False) -> None:
        self.run("up", "-d", *(["--wait"] if wait else []), service, quiet=True)

    def down(self, volumes: bool = False) -> None:
        self.run("down", *(["-v"] if volumes else []), quiet=True)

    def logs(self, service: str, tail: int = 40) -> None:
        self.run("logs", "--tail", str(tail), service)

    def ps(self) -> list[ServiceState]:
        output = self.run("ps", "--format", "json", quiet=True)
        return [ServiceState.model_validate(json.loads(line)) for line in output.splitlines() if line]

    def is_running(self, service: str) -> bool:
        return bool(self.run("ps", "-q", "--status", "running", service, quiet=True).strip())

    def remove_container(self, name: str) -> None:
        self.runner(["docker", "rm", "-f", name], capture_output=True)
